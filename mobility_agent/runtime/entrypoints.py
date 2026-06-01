from __future__ import annotations

from typing import Any, Callable

from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field

from ..graph.state import ExternalEventRecord, normalize_external_event


class ExternalEventResumeRequest(BaseModel):
    workdir: str
    thread_id: str | None = None
    event: dict[str, Any] = Field(default_factory=dict)


class ExternalEventResumeResult(BaseModel):
    workdir: str
    thread_id: str | None = None
    event: dict[str, Any] = Field(default_factory=dict)
    outcome: dict[str, Any] = Field(default_factory=dict)


def build_batch_entrypoint(
    *,
    checkpointer: Any,
    store: Any,
    batch_init: Callable[[], dict[str, Any]],
    fetch_next: Callable[[dict[str, Any]], dict[str, Any]],
    prepare_item: Callable[[dict[str, Any]], dict[str, Any]],
    run_item: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_item: Callable[[dict[str, Any]], dict[str, Any]],
    finalize: Callable[[dict[str, Any]], dict[str, Any]],
):
    @task(name="batch_init")
    def _batch_init_task() -> dict[str, Any]:
        return dict(batch_init() or {})

    @task(name="batch_fetch_next")
    def _fetch_next_task(state: dict[str, Any]) -> dict[str, Any]:
        return dict(fetch_next(dict(state or {})) or {})

    @task(name="batch_prepare_item")
    def _prepare_item_task(state: dict[str, Any]) -> dict[str, Any]:
        return dict(prepare_item(dict(state or {})) or {})

    @task(name="batch_run_item")
    def _run_item_task(state: dict[str, Any]) -> dict[str, Any]:
        return dict(run_item(dict(state or {})) or {})

    @task(name="batch_aggregate_item")
    def _aggregate_item_task(state: dict[str, Any]) -> dict[str, Any]:
        return dict(aggregate_item(dict(state or {})) or {})

    @task(name="batch_finalize")
    def _finalize_task(state: dict[str, Any]) -> dict[str, Any]:
        return dict(finalize(dict(state or {})) or {})

    def _merged(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        merged = dict(state or {})
        merged.update(dict(updates or {}))
        return merged

    @entrypoint(checkpointer=checkpointer, store=store)
    def _batch_entrypoint(_: dict[str, Any] | None = None, *, previous: dict[str, Any] | None = None):
        state = dict(previous or {})
        if not state:
            state = _batch_init_task().result()
        while True:
            state = _merged(state, _fetch_next_task(state).result())
            if bool((state.get("batch", {}) or {}).get("done")):
                break
            state = _merged(state, _prepare_item_task(state).result())
            state = _merged(state, _run_item_task(state).result())
            state = _merged(state, _aggregate_item_task(state).result())
        state = _merged(state, _finalize_task(state).result())
        return entrypoint.final(value=state, save=state)

    return _batch_entrypoint


def build_external_event_resume_entrypoint(
    *,
    resume_material: Callable[[str, str | None, dict[str, Any]], dict[str, Any]],
):
    @task(name="normalize_external_event_request")
    def _normalize_request_task(payload: dict[str, Any]) -> dict[str, Any]:
        request = ExternalEventResumeRequest.model_validate(dict(payload or {}))
        event = normalize_external_event(request.event, default_thread_id=request.thread_id)
        return ExternalEventResumeRequest(
            workdir=request.workdir,
            thread_id=request.thread_id or event.thread_id,
            event=event.model_dump(mode="json"),
        ).model_dump(mode="json")

    @task(name="resume_material_external_event")
    def _resume_task(payload: dict[str, Any]) -> dict[str, Any]:
        request = ExternalEventResumeRequest.model_validate(dict(payload or {}))
        outcome = resume_material(request.workdir, request.thread_id, dict(request.event or {}))
        return ExternalEventResumeResult(
            workdir=request.workdir,
            thread_id=request.thread_id,
            event=dict(request.event or {}),
            outcome=dict(outcome or {}),
        ).model_dump(mode="json")

    @entrypoint()
    def _resume_entrypoint(payload: dict[str, Any]):
        normalized = _normalize_request_task(dict(payload or {})).result()
        result = _resume_task(normalized).result()
        return entrypoint.final(value=result, save=result)

    return _resume_entrypoint
