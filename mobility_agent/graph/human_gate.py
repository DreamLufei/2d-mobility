from __future__ import annotations

from typing import Any


def _resolve_recovery_stage(state: dict[str, Any]) -> str | None:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})

    current_action = dict(execution.get("current_action", {}) or {})
    stage = str(current_action.get("target_capability") or "").strip()
    if stage:
        return stage

    recovery_summary = dict(diagnostics.get("recovery_summary", {}) or {})
    stage = str(recovery_summary.get("stage") or recovery_summary.get("current_stage") or "").strip()
    if stage:
        return stage

    latest_observation = dict(execution.get("latest_execution_observation", {}) or {})
    stage = str(
        latest_observation.get("target_capability")
        or latest_observation.get("stage")
        or latest_observation.get("current_stage")
        or ""
    ).strip()
    if stage:
        return stage

    stage = str(workflow.get("current_stage") or "").strip()
    return stage or None


def build_human_escalation_payload(state: dict[str, Any], *, recommended_options: list[str]) -> dict[str, Any]:
    task = dict(state.get("task", {}) or {})
    material = dict(state.get("material", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    raw_evidence = dict(diagnostics.get("raw_evidence", {}) or {})
    log_paths: list[str] = []
    for evidence in raw_evidence.values():
        if not isinstance(evidence, dict):
            continue
        for path in evidence.get("log_paths", []) or []:
            text = str(path)
            if text and text not in log_paths:
                log_paths.append(text)
        for key in ["stdout_path", "stderr_path"]:
            value = evidence.get(key)
            if value:
                text = str(value)
                if text not in log_paths:
                    log_paths.append(text)
    effective_stage = _resolve_recovery_stage(state)
    return {
        "task_id": task.get("task_id"),
        "material_id": material.get("material_id"),
        "current_stage": effective_stage,
        "graph_node": workflow.get("current_stage"),
        "error_summary": diagnostics.get("last_error"),
        "recovery_history_summary": diagnostics.get("recovery_history", [])[-3:],
        "validation_summary": diagnostics.get("validation_report"),
        "recommended_options": recommended_options,
        "working_directory": execution.get("workdir"),
        "log_paths": log_paths,
        "timeout_policy": "skip_material_after_300s",
        "latest_observation_status": dict(execution.get("latest_execution_observation", {}) or {}).get("status"),
        "compute_completed": bool(dict((state.get("physics_results", {}) or {}).get("results", {}) or {}).get("results_by_direction")),
    }
