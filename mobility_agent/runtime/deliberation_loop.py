from __future__ import annotations

from typing import Any

from .action_registry import capability_dependencies, capability_sequence


def build_initial_task_board() -> dict[str, list[dict[str, Any]]]:
    return {
        "pending_tasks": [
            {
                "task_id": f"capability::{capability}",
                "task_type": "capability",
                "capability": capability,
                "depends_on": capability_dependencies(capability),
                "status": "pending",
            }
            for capability in capability_sequence()
        ],
        "active_tasks": [],
        "completed_tasks": [],
        "blocked_tasks": [],
        "abandoned_tasks": [],
    }


def next_pending_task(state: dict[str, Any]) -> dict[str, Any] | None:
    task_board = dict(state.get("task_board", {}) or {})
    completed = {
        str(item.get("capability"))
        for item in list(task_board.get("completed_tasks", []) or [])
        if isinstance(item, dict) and item.get("capability")
    }
    for item in list(task_board.get("pending_tasks", []) or []):
        if not isinstance(item, dict):
            continue
        depends_on = [str(value) for value in list(item.get("depends_on", []) or [])]
        if all(dep in completed for dep in depends_on):
            return dict(item)
    return None


def move_task(board: dict[str, Any], *, capability: str, source_key: str, target_key: str, status: str) -> dict[str, Any]:
    updated = {
        "pending_tasks": list(board.get("pending_tasks", []) or []),
        "active_tasks": list(board.get("active_tasks", []) or []),
        "completed_tasks": list(board.get("completed_tasks", []) or []),
        "blocked_tasks": list(board.get("blocked_tasks", []) or []),
        "abandoned_tasks": list(board.get("abandoned_tasks", []) or []),
    }
    item = None
    source_items = []
    for current in list(updated.get(source_key, []) or []):
        if isinstance(current, dict) and str(current.get("capability") or "") == capability and item is None:
            item = {**current, "status": status}
            continue
        source_items.append(current)
    updated[source_key] = source_items
    if item is None:
        item = {
            "task_id": f"capability::{capability}",
            "task_type": "capability",
            "capability": capability,
            "depends_on": capability_dependencies(capability),
            "status": status,
        }
    updated[target_key] = list(updated.get(target_key, []) or []) + [item]
    return updated


def mark_task_started(board: dict[str, Any], capability: str) -> dict[str, Any]:
    return move_task(board, capability=capability, source_key="pending_tasks", target_key="active_tasks", status="active")


def mark_task_completed(board: dict[str, Any], capability: str) -> dict[str, Any]:
    source_key = "active_tasks"
    if not any(str((item or {}).get("capability") or "") == capability for item in list(board.get("active_tasks", []) or [])):
        source_key = "pending_tasks"
    return move_task(board, capability=capability, source_key=source_key, target_key="completed_tasks", status="completed")


def mark_task_blocked(board: dict[str, Any], capability: str, *, reason: str) -> dict[str, Any]:
    updated = move_task(board, capability=capability, source_key="active_tasks", target_key="blocked_tasks", status="blocked")
    if updated["blocked_tasks"]:
        updated["blocked_tasks"][-1]["reason"] = reason
    return updated


def mark_task_abandoned(board: dict[str, Any], capability: str, *, reason: str) -> dict[str, Any]:
    source_key = "active_tasks"
    if not any(str((item or {}).get("capability") or "") == capability for item in list(board.get("active_tasks", []) or [])):
        source_key = "pending_tasks"
    updated = move_task(board, capability=capability, source_key=source_key, target_key="abandoned_tasks", status="abandoned")
    if updated["abandoned_tasks"]:
        updated["abandoned_tasks"][-1]["reason"] = reason
    return updated


def all_tasks_resolved(state: dict[str, Any]) -> bool:
    task_board = dict(state.get("task_board", {}) or {})
    return (
        not list(task_board.get("pending_tasks", []) or [])
        and not list(task_board.get("active_tasks", []) or [])
        and not list(task_board.get("blocked_tasks", []) or [])
        and not list(task_board.get("abandoned_tasks", []) or [])
    )


def has_blocked_or_abandoned_tasks(state: dict[str, Any]) -> bool:
    task_board = dict(state.get("task_board", {}) or {})
    return bool(list(task_board.get("blocked_tasks", []) or []) or list(task_board.get("abandoned_tasks", []) or []))


def reset_from_capability(board: dict[str, Any], capability: str) -> dict[str, Any]:
    ordered = capability_sequence()
    if capability not in ordered:
        return dict(board)
    reset_index = ordered.index(capability)
    keep_completed: list[dict[str, Any]] = []
    requeued: list[dict[str, Any]] = []
    for item in list(board.get("completed_tasks", []) or []):
        if not isinstance(item, dict):
            continue
        cap = str(item.get("capability") or "")
        if cap in ordered and ordered.index(cap) < reset_index:
            keep_completed.append(item)
        else:
            requeued.append({**item, "status": "pending"})
    pending = list(board.get("pending_tasks", []) or []) + requeued
    active = [
        {**item, "status": "pending"}
        for item in list(board.get("active_tasks", []) or [])
        if isinstance(item, dict) and str(item.get("capability") or "") in ordered[reset_index:]
    ]
    pending_caps = {str((item or {}).get("capability") or "") for item in pending}
    for cap in ordered[reset_index:]:
        if cap not in pending_caps:
            pending.append(
                {
                    "task_id": f"capability::{cap}",
                    "task_type": "capability",
                    "capability": cap,
                    "depends_on": capability_dependencies(cap),
                    "status": "pending",
                }
            )
    return {
        "pending_tasks": pending + active,
        "active_tasks": [],
        "completed_tasks": keep_completed,
        "blocked_tasks": [item for item in list(board.get("blocked_tasks", []) or []) if str((item or {}).get("capability") or "") not in ordered[reset_index:]],
        "abandoned_tasks": [item for item in list(board.get("abandoned_tasks", []) or []) if str((item or {}).get("capability") or "") not in ordered[reset_index:]],
    }


def build_round_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_id": int((state.get("deliberation", {}) or {}).get("round_index", 0) or 0),
        "latest_observation": dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {}),
        "risk_flags": list((state.get("blackboard", {}) or {}).get("risk_flags", []) or []),
        "anomaly_flags": list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or []),
        "current_action": dict((state.get("execution", {}) or {}).get("current_action", {}) or {}),
        "run_status": str((state.get("workflow", {}) or {}).get("run_status") or ""),
        "wait_reason": str((state.get("workflow", {}) or {}).get("wait_reason") or "") or None,
        "task_board": dict(state.get("task_board", {}) or {}),
    }
