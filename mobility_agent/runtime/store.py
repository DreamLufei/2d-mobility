from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.store.base import BaseStore

from ..graph.stage_contracts import STAGE_ORDER
from .database import normalize_database_uri, open_kv_store


def memory_namespace(category: str, *scope: str) -> tuple[str, ...]:
    normalized = [str(category)]
    normalized.extend(str(item) for item in scope if str(item or "").strip())
    return tuple(normalized)


def memory_store_path(path: str) -> str:
    return normalize_database_uri(path, default_memory_name="mobility-store")


@contextmanager
def open_memory_store(path: str) -> Iterator[BaseStore]:
    target = memory_store_path(path)
    with open_kv_store(target) as store:
        yield store


def put_memory_item(
    store: BaseStore,
    *,
    category: str,
    item_key: str,
    payload: dict[str, Any],
    scope: tuple[str, ...] = (),
) -> None:
    store.put(memory_namespace(category, *scope), str(item_key), dict(payload))


def get_memory_item(
    store: BaseStore,
    *,
    category: str,
    item_key: str,
    scope: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    item = store.get(memory_namespace(category, *scope), str(item_key))
    if item is None:
        return None
    return dict(item.value)


def list_memory_items(
    store: BaseStore,
    *,
    category: str,
    scope: tuple[str, ...] = (),
    limit: int = 1000,
) -> list[dict[str, Any]]:
    return [dict(item.value) for item in store.search(memory_namespace(category, *scope), limit=int(limit))]


def record_recovery_case(store: BaseStore, *, task_id: str, payload: dict[str, Any]) -> None:
    stage = str(payload.get("stage") or "unknown")
    signature = str(payload.get("error_signature") or payload.get("error_type") or "unknown")
    put_memory_item(
        store,
        category="recovery_cases",
        item_key=f"{task_id}:{stage}:{signature}",
        payload=payload,
        scope=(stage,),
    )


def list_recovery_cases(store: BaseStore) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for stage in [*STAGE_ORDER, "unknown"]:
        items.extend(list_memory_items(store, category="recovery_cases", scope=(stage,), limit=100))
    return items


def find_recovery_cases(store: BaseStore, *, stage: str, error_signature: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    signature = str(error_signature or "").strip()
    matches: list[dict[str, Any]] = []
    for item in list_memory_items(store, category="recovery_cases", scope=(str(stage),), limit=limit):
        item_signature = str(item.get("error_signature") or item.get("error_type") or "").strip()
        if signature and item_signature and signature not in item_signature and item_signature not in signature:
            continue
        matches.append(dict(item))
        if len(matches) >= int(limit):
            break
    return matches


def record_validation_heuristic(store: BaseStore, *, heuristic_name: str, payload: dict[str, object]) -> None:
    put_memory_item(store, category="validation_heuristics", item_key=heuristic_name, payload=dict(payload))


def list_validation_heuristics(store: BaseStore) -> list[dict[str, object]]:
    return list_memory_items(store, category="validation_heuristics")


def find_validation_heuristics(
    store: BaseStore,
    *,
    anomaly_flags: list[str] | None = None,
    warnings: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, object]]:
    anomaly_flag_values = [str(item) for item in list(anomaly_flags or [])]
    warning_text = " ".join(str(item) for item in list(warnings or []))
    matches: list[dict[str, object]] = []
    for item in list_validation_heuristics(store):
        trigger_pattern = str(item.get("trigger_pattern") or "")
        if anomaly_flag_values and not any(flag in trigger_pattern for flag in anomaly_flag_values):
            if warning_text and warning_text not in trigger_pattern:
                continue
        matches.append(dict(item))
        if len(matches) >= int(limit):
            break
    return matches


def record_batch_statistics(store: BaseStore, *, collection_name: str, payload: dict[str, object]) -> None:
    put_memory_item(
        store,
        category="batch_statistics",
        item_key=collection_name,
        payload=dict(payload),
        scope=(str(collection_name),),
    )


def list_batch_statistics(store: BaseStore, *, collection_name: str | None = None) -> list[dict[str, object]]:
    if collection_name:
        return list_memory_items(store, category="batch_statistics", scope=(str(collection_name),))
    return list_memory_items(store, category="batch_statistics")


def record_skill_metadata(store: BaseStore, *, skill_name: str, payload: dict[str, object]) -> None:
    put_memory_item(store, category="skill_registry", item_key=skill_name, payload=dict(payload))


def list_skill_metadata(store: BaseStore) -> list[dict[str, object]]:
    return list_memory_items(store, category="skill_registry")
