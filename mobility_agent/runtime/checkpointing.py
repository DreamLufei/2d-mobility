from __future__ import annotations

import json
import os
import pickle
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver

from .database import checkpoint_exists, delete_checkpoint_thread, open_checkpointer, redact_database_uri


def runtime_checkpoint_dir(workdir: str, checkpoint_subdir: str = ".runtime") -> Path:
    root = Path(workdir) / checkpoint_subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def runtime_checkpoint_metadata_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    return str(runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir) / "langgraph_checkpoint.json")


def runtime_sqlite_checkpoint_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    return runtime_checkpoint_metadata_path(workdir, checkpoint_subdir=checkpoint_subdir)


@contextmanager
def open_runtime_checkpointer(*, database_uri: str) -> Iterator[BaseCheckpointSaver]:
    with open_checkpointer(database_uri) as saver:
        yield saver


@contextmanager
def open_sqlite_checkpointer(*, workdir: str, checkpoint_subdir: str = ".runtime", database_uri: str | None = None) -> Iterator[BaseCheckpointSaver]:
    resolved_database_uri = str(database_uri or "").strip()
    if not resolved_database_uri:
        metadata = _read_json_file(runtime_checkpoint_metadata_path(workdir, checkpoint_subdir=checkpoint_subdir)) or {}
        raw_uri = str(metadata.get("database_uri_raw") or "").strip()
        if raw_uri:
            resolved_database_uri = raw_uri
        elif str(metadata.get("backend") or "") == "memory":
            resolved_database_uri = f"memory://{str(metadata.get('thread_id') or '').strip() or 'legacy-checkpointer'}"
    with open_runtime_checkpointer(database_uri=resolved_database_uri or "memory://legacy-checkpointer") as saver:
        yield saver


def langgraph_checkpoint_exists(*, database_uri: str, thread_id: str | None) -> bool:
    return checkpoint_exists(database_uri=database_uri, thread_id=thread_id)


def runtime_state_snapshot_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    """Review-oriented state export path.

    The canonical runtime state lives in the LangGraph checkpointer; this JSON export
    is kept only for inspection, debugging, and compatibility artifacts.
    """
    return str(runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir) / "shared_state.json")


def runtime_ui_state_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    return str(runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir) / "ui_state.json")


def runtime_ui_events_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    return str(runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir) / "ui_events.jsonl")


def runtime_thread_id_path(workdir: str, checkpoint_subdir: str = ".runtime") -> str:
    return str(runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir) / "thread_id.txt")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_text_atomic(path: str, payload: str) -> str:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)
    return path


def write_json_atomic(path: str, payload: Any) -> str:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _read_json_file(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _selected_action_payload(state: dict[str, Any]) -> dict[str, Any]:
    execution = dict(state.get("execution", {}) or {})
    current = dict(execution.get("current_action", {}) or {})
    if current:
        return current
    deliberation = dict(state.get("deliberation", {}) or {})
    selected = list(deliberation.get("selected_actions", []) or [])
    for item in reversed(selected):
        if isinstance(item, dict):
            action = dict(item.get("selected_action", {}) or {})
            if action:
                return action
            if any(key in item for key in ("action_family", "target_capability", "parameters")):
                return dict(item)
    return {}


def _latest_error_summary(state: dict[str, Any]) -> str:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    if diagnostics.get("last_error"):
        return str(diagnostics.get("last_error") or "")
    errors = list(diagnostics.get("errors", []) or [])
    if errors:
        return str(errors[-1] or "")
    workflow = dict(state.get("workflow", {}) or {})
    if workflow.get("termination_reason"):
        return str(workflow.get("termination_reason") or "")
    return ""


def _artifact_paths(state: dict[str, Any]) -> dict[str, str]:
    execution = dict(state.get("execution", {}) or {})
    merged = dict(execution.get("artifact_paths", {}) or {})
    merged.update(dict(execution.get("artifact_registry", {}) or {}))
    return {str(key): str(value) for key, value in merged.items() if str(value or "").strip()}


def build_ui_state_snapshot(*, state: dict[str, Any]) -> dict[str, Any]:
    task = dict(state.get("task", {}) or {})
    material = dict(state.get("material", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    services = dict(state.get("services", {}) or {})
    workflow_contract = dict(services.get("workflow_contract", {}) or {})
    execution_checkpoint = dict(execution.get("execution_checkpoint", {}) or {})
    selected_action = _selected_action_payload(state)
    hitl_pending = bool(
        workflow.get("run_status") == "needs_human"
        or workflow.get("pending_human_action")
        or workflow.get("pending_action_payload")
        or services.get("pending_human_payload")
    )
    return {
        "task_id": str(task.get("task_id") or ""),
        "material_id": str(material.get("material_id") or ""),
        "thread_id": str(execution.get("thread_id") or "") or None,
        "current_stage": str(workflow.get("current_stage") or ""),
        "runtime_run_status": str(workflow.get("run_status") or "pending"),
        "stage_status": dict(workflow.get("stage_status", {}) or {}),
        "selected_action": dict(selected_action or {}),
        "hitl_pending": hitl_pending,
        "wait_reason": str(workflow.get("wait_reason") or "") or None,
        "latest_error": _latest_error_summary(state) or None,
        "artifact_paths": _artifact_paths(state),
        "workflow_contract": {
            "contract_id": str(workflow_contract.get("contract_id") or "") or None,
            "version": workflow_contract.get("version"),
            "plan_status": workflow_contract.get("plan_status"),
            "current_focus": workflow_contract.get("current_focus"),
            "council_mode": workflow_contract.get("council_mode"),
            "planned_capabilities": list(workflow_contract.get("planned_capabilities", []) or []),
            "milestones": list(workflow_contract.get("milestones", []) or []),
            "deliberation_reason": workflow_contract.get("deliberation_reason"),
        },
        "execution_checkpoint": {
            "contract_id": str(execution_checkpoint.get("contract_id") or "") or None,
            "contract_version": execution_checkpoint.get("contract_version"),
            "current_capability": execution_checkpoint.get("current_capability"),
            "next_capability": execution_checkpoint.get("next_capability"),
            "completed_capabilities": list(execution_checkpoint.get("completed_capabilities", []) or []),
            "needs_deliberation": bool(execution_checkpoint.get("needs_deliberation", False)),
            "deliberation_reason": execution_checkpoint.get("deliberation_reason"),
        },
        "updated_at": str(task.get("updated_at") or _utc_now_iso()),
    }


def load_ui_state_snapshot(*, workdir: str, checkpoint_subdir: str = ".runtime") -> dict[str, Any] | None:
    return _read_json_file(runtime_ui_state_path(workdir, checkpoint_subdir=checkpoint_subdir))


def _selected_action_signature(ui_state: dict[str, Any] | None) -> tuple[str, str, str]:
    selected = dict((ui_state or {}).get("selected_action", {}) or {})
    return (
        str(selected.get("action_family") or ""),
        str(selected.get("target_capability") or selected.get("capability") or ""),
        json.dumps(dict(selected.get("parameters", {}) or {}), ensure_ascii=False, sort_keys=True),
    )


def _current_stage_status(ui_state: dict[str, Any] | None) -> str:
    payload = dict(ui_state or {})
    stage_status = dict(payload.get("stage_status", {}) or {})
    current_stage = str(payload.get("current_stage") or "")
    return str(stage_status.get(current_stage) or stage_status.get(current_stage.replace("01_", "")) or "")


def append_ui_event(
    *,
    workdir: str,
    event_type: str,
    checkpoint_subdir: str = ".runtime",
    state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_dir = runtime_checkpoint_dir(workdir, checkpoint_subdir=checkpoint_subdir)
    ui_state = build_ui_state_snapshot(state=state or {}) if state is not None else load_ui_state_snapshot(
        workdir=workdir,
        checkpoint_subdir=checkpoint_subdir,
    )
    if ui_state is None:
        ui_state = {}
    selected_action = dict(ui_state.get("selected_action", {}) or {})
    payload = {
        "timestamp": _utc_now_iso(),
        "event_type": str(event_type or "").strip() or "runtime_event",
        "current_stage": str(ui_state.get("current_stage") or ""),
        "runtime_run_status": str(ui_state.get("runtime_run_status") or "pending"),
        "selected_action_family": str(selected_action.get("action_family") or ""),
        "selected_capability": str(selected_action.get("target_capability") or selected_action.get("capability") or ""),
        "stage_status": _current_stage_status(ui_state),
        "hitl_pending": bool(ui_state.get("hitl_pending")),
        "wait_reason": ui_state.get("wait_reason"),
        "latest_error": ui_state.get("latest_error"),
        "latest_artifact_keys": sorted(dict(ui_state.get("artifact_paths", {}) or {}).keys()),
    }
    if extra:
        payload.update(dict(extra or {}))
    with (runtime_dir / "ui_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return payload


def _ui_state_events(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if previous is None:
        events: list[tuple[str, dict[str, Any]]] = [("state_initialized", {})]
        current_status = str(current.get("runtime_run_status") or "")
        if current_status and current_status != "pending":
            events.append(("run_status_changed", {"previous_runtime_run_status": "", "runtime_run_status": current_status}))
        current_stage_status = _current_stage_status(current)
        if current_stage_status:
            events.append(("stage_status_changed", {"previous_stage_status": "", "current_stage_status": current_stage_status}))
        if dict(current.get("selected_action", {}) or {}):
            events.append(("selected_action_changed", {"selected_action": dict(current.get("selected_action", {}) or {})}))
        current_artifacts = sorted(dict(current.get("artifact_paths", {}) or {}).keys())
        if current_artifacts:
            events.append(("artifacts_updated", {"latest_artifact_keys": current_artifacts}))
        return events
    events: list[tuple[str, dict[str, Any]]] = []
    if str(previous.get("current_stage") or "") != str(current.get("current_stage") or ""):
        events.append(
            (
                "stage_changed",
                {
                    "previous_stage": previous.get("current_stage"),
                    "current_stage": current.get("current_stage"),
                },
            )
        )
    if _current_stage_status(previous) != _current_stage_status(current):
        events.append(
            (
                "stage_status_changed",
                {
                    "previous_stage_status": _current_stage_status(previous),
                    "current_stage_status": _current_stage_status(current),
                },
            )
        )
    if _selected_action_signature(previous) != _selected_action_signature(current) and dict(current.get("selected_action", {}) or {}):
        events.append(("selected_action_changed", {"selected_action": dict(current.get("selected_action", {}) or {})}))
    previous_status = str(previous.get("runtime_run_status") or "")
    current_status = str(current.get("runtime_run_status") or "")
    if previous_status != current_status:
        events.append(
            (
                "run_status_changed",
                {
                    "previous_runtime_run_status": previous_status,
                    "runtime_run_status": current_status,
                },
            )
        )
    previous_artifacts = sorted(dict(previous.get("artifact_paths", {}) or {}).keys())
    current_artifacts = sorted(dict(current.get("artifact_paths", {}) or {}).keys())
    if previous_artifacts != current_artifacts:
        events.append(("artifacts_updated", {"latest_artifact_keys": current_artifacts}))
    return events


def save_ui_snapshot(*, workdir: str, state: dict[str, Any], checkpoint_subdir: str = ".runtime") -> str:
    previous = load_ui_state_snapshot(workdir=workdir, checkpoint_subdir=checkpoint_subdir)
    current = build_ui_state_snapshot(state=state)
    path = runtime_ui_state_path(workdir, checkpoint_subdir=checkpoint_subdir)
    write_json_atomic(path, current)
    for event_type, extra in _ui_state_events(previous, current):
        append_ui_event(
            workdir=workdir,
            event_type=event_type,
            checkpoint_subdir=checkpoint_subdir,
            extra=extra,
        )
    return path


def load_thread_id(*, workdir: str, checkpoint_subdir: str = ".runtime") -> str | None:
    path = runtime_thread_id_path(workdir, checkpoint_subdir=checkpoint_subdir)
    if not os.path.exists(path):
        return None
    value = Path(path).read_text(encoding="utf-8").strip()
    return value or None


def task_id_from_thread_id(thread_id: str | None) -> str | None:
    value = str(thread_id or "").strip()
    parts = value.split("::")
    if len(parts) >= 4 and parts[0] == "material":
        return parts[1] or None
    return None


def save_thread_id(*, workdir: str, thread_id: str, checkpoint_subdir: str = ".runtime") -> str:
    path = runtime_thread_id_path(workdir, checkpoint_subdir=checkpoint_subdir)
    return _write_text_atomic(path, f"{thread_id.strip()}\n")


def save_checkpoint_metadata(
    *,
    workdir: str,
    thread_id: str,
    database_uri: str,
    checkpoint_subdir: str = ".runtime",
) -> str:
    return write_json_atomic(
        runtime_checkpoint_metadata_path(workdir, checkpoint_subdir=checkpoint_subdir),
        {
            "thread_id": str(thread_id or "").strip(),
            "database_uri_raw": str(database_uri or "").strip(),
            "database_uri": redact_database_uri(database_uri),
            "backend": "memory" if str(database_uri).startswith("memory://") else "postgres",
            "updated_at": _utc_now_iso(),
        },
    )


def build_material_thread_id(*, task_id: str, material_id: str, run_id: str | None = None) -> str:
    return f"material::{task_id}::{material_id}::{run_id or uuid.uuid4().hex[:12]}"


def build_batch_thread_id(*, batch_id: str) -> str:
    return f"batch::{batch_id}"


def save_state_snapshot(*, workdir: str, state: dict[str, Any], checkpoint_subdir: str = ".runtime") -> str:
    """Export a review/debug snapshot without making it a parallel source of truth."""
    path = runtime_state_snapshot_path(workdir, checkpoint_subdir=checkpoint_subdir)
    write_json_atomic(path, state)
    save_ui_snapshot(workdir=workdir, state=state, checkpoint_subdir=checkpoint_subdir)
    return path


def load_state_snapshot(*, workdir: str, checkpoint_subdir: str = ".runtime") -> dict[str, Any] | None:
    return _read_json_file(runtime_state_snapshot_path(workdir, checkpoint_subdir=checkpoint_subdir))


def load_compatibility_checkpoint(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "shared_state" in payload:
        return dict(payload["shared_state"])
    if isinstance(payload, dict):
        return payload
    return None


def remove_checkpoints(
    *,
    workdir: str,
    checkpoint_subdir: str = ".runtime",
    database_uri: str | None = None,
    thread_id: str | None = None,
) -> None:
    snapshot = runtime_state_snapshot_path(workdir, checkpoint_subdir=checkpoint_subdir)
    thread_id_path = runtime_thread_id_path(workdir, checkpoint_subdir=checkpoint_subdir)
    checkpoint_metadata_path = runtime_checkpoint_metadata_path(workdir, checkpoint_subdir=checkpoint_subdir)
    resolved_thread_id = str(thread_id or "").strip() or load_thread_id(workdir=workdir, checkpoint_subdir=checkpoint_subdir)
    if database_uri and resolved_thread_id:
        delete_checkpoint_thread(database_uri=database_uri, thread_id=resolved_thread_id)
    for path in [
        snapshot,
        runtime_ui_state_path(workdir, checkpoint_subdir=checkpoint_subdir),
        runtime_ui_events_path(workdir, checkpoint_subdir=checkpoint_subdir),
        thread_id_path,
        checkpoint_metadata_path,
        os.path.join(workdir, "checkpoint.pkl"),
    ]:
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


def load_debug_export_state(*, workdir: str, checkpoint_subdir: str = ".runtime") -> dict[str, Any] | None:
    """Read exported snapshots for debugging or manual inspection only.

    This helper is intentionally not part of the canonical recovery path.
    """
    snapshot_state = load_state_snapshot(workdir=workdir, checkpoint_subdir=checkpoint_subdir)
    if snapshot_state is not None:
        return snapshot_state
    return load_compatibility_checkpoint(os.path.join(workdir, "checkpoint.pkl"))
