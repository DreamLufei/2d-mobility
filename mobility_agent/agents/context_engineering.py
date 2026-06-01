from __future__ import annotations

from typing import Any

from ..utils import dedupe_keep_order

_RECENT_ACTION_WINDOW = 2
_RECENT_ARBITRATION_WINDOW = 2
_RECENT_REFLECTION_WINDOW = 1
_RECENT_DISAGREEMENT_WINDOW = 1
_RECENT_MEMORY_WINDOW = 3
_RECENT_STAGE_WINDOW = 6
_RECENT_LIST_WINDOW = 4


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _recent(items: list[Any], *, limit: int) -> list[Any]:
    values = _as_list(items)
    return values[-limit:] if len(values) > limit else values


def _truncate_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[+{len(text) - limit} chars]"


def _compact_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, list):
        if len(value) <= _RECENT_LIST_WINDOW:
            return [_compact_scalar(item) for item in value]
        return [_compact_scalar(item) for item in value[:2]] + ["..."] + [_compact_scalar(item) for item in value[-1:]]
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in list(value.items())[:6]:
            compacted[str(key)] = _compact_scalar(item)
        if len(value) > 6:
            compacted["_truncated_keys"] = len(value) - 6
        return compacted
    return _truncate_text(value)


def task_capabilities(items: list[Any]) -> list[str]:
    caps: list[str] = []
    for item in _as_list(items):
        if isinstance(item, dict):
            cap = str(item.get("capability") or "").strip()
        else:
            cap = str(item or "").strip()
        if cap:
            caps.append(cap)
    return caps


def compact_latest_observation(state: dict[str, Any]) -> dict[str, Any]:
    latest = _as_dict(_as_dict(state.get("execution")).get("latest_execution_observation"))
    if not latest:
        latest = _as_dict(_as_dict(state.get("blackboard")).get("latest_execution_observation"))
    return {
        "status": latest.get("status"),
        "action_family": latest.get("action_family"),
        "target_capability": latest.get("target_capability"),
        "error_summary": _truncate_text(latest.get("error_summary"), limit=320) or None,
        "result_summary": _compact_scalar(_as_dict(latest.get("result_summary"))),
        "artifact_paths": _compact_scalar(_as_dict(latest.get("artifact_paths"))),
    }


def compact_results_by_direction(results_by_direction: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for direction, payload in _as_dict(results_by_direction).items():
        direction_payload = _as_dict(payload)
        electron = _as_dict(direction_payload.get("electron"))
        hole = _as_dict(direction_payload.get("hole"))
        summary[str(direction)] = {
            "n_points": direction_payload.get("n_points"),
            "elastic_modulus_C2D_J_m2": direction_payload.get("elastic_modulus_C2D_J_m2"),
            "electron_mobility_cm2_Vs": electron.get("mobility_cm2_Vs"),
            "hole_mobility_cm2_Vs": hole.get("mobility_cm2_Vs"),
            "electron_E1_fit_R2": electron.get("E1_fit_R2"),
            "hole_E1_fit_R2": hole.get("E1_fit_R2"),
        }
    return summary


def summarize_selected_actions(actions: list[Any], *, limit: int = _RECENT_ACTION_WINDOW) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for action in _recent(_as_list(actions), limit=limit):
        payload = _as_dict(action.model_dump(mode="json") if hasattr(action, "model_dump") else action)
        output.append(
            {
                "action_family": payload.get("action_family"),
                "target_capability": payload.get("target_capability"),
                "selected_skill": payload.get("selected_skill"),
                "parameters": _compact_scalar(_as_dict(payload.get("parameters"))),
                "rationale": _truncate_text(payload.get("rationale")),
                "fallback_if_failed": _recent(_as_list(payload.get("fallback_if_failed")), limit=3),
                "source_proposal_id": payload.get("source_proposal_id"),
            }
        )
    return output


def summarize_arbitrations(arbitrations: list[Any], *, limit: int = _RECENT_ARBITRATION_WINDOW) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _recent(_as_list(arbitrations), limit=limit):
        payload = _as_dict(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        output.append(
            {
                "selected_proposal_id": payload.get("selected_proposal_id"),
                "guardrail_notes": _recent(_as_list(payload.get("guardrail_notes")), limit=5),
                "rationale": _truncate_text(payload.get("rationale")),
                "disagreement_summary": _recent(_as_list(payload.get("disagreement_summary")), limit=5),
                "whether_noop": bool(payload.get("whether_noop")),
                "whether_waiting_external": bool(payload.get("whether_waiting_external")),
                "whether_ready_to_finalize": bool(payload.get("whether_ready_to_finalize")),
            }
        )
    return output


def summarize_reflections(reflections: list[Any], *, limit: int = _RECENT_REFLECTION_WINDOW) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _recent(_as_list(reflections), limit=limit):
        payload = _as_dict(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        output.append(
            {
                "selected_action": _compact_scalar(_as_dict(payload.get("selected_action"))),
                "tradeoff_summary": _recent(_as_list(payload.get("tradeoff_summary")), limit=4),
                "failure_pattern": _truncate_text(payload.get("failure_pattern"), limit=200) or None,
                "continue_deliberation": bool(payload.get("continue_deliberation", True)),
            }
        )
    return output


def summarize_disagreement_records(records: list[Any], *, limit: int = _RECENT_DISAGREEMENT_WINDOW) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _recent(_as_list(records), limit=limit):
        payload = _as_dict(item)
        output.append(
            {
                "round_id": payload.get("round_id"),
                "pattern": _truncate_text(payload.get("pattern"), limit=200),
                "selected_action": _compact_scalar(_as_dict(payload.get("selected_action"))),
            }
        )
    return output


def summarize_memory_patterns(patterns: list[Any], *, limit: int = _RECENT_MEMORY_WINDOW) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in _recent(_as_list(patterns), limit=limit):
        payload = _as_dict(item)
        output.append(
            {
                "stage": payload.get("stage"),
                "error_summary": _truncate_text(payload.get("error_summary"), limit=180),
                "decision": payload.get("decision"),
                "pattern": _truncate_text(payload.get("pattern"), limit=180),
            }
        )
    return output


def summarize_proposals(proposals: list[Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    values = _as_list(proposals)
    if limit is not None:
        values = _recent(values, limit=limit)
    for item in values:
        payload = _as_dict(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        content = _as_dict(payload.get("content"))
        output.append(
            {
                "proposal_id": payload.get("proposal_id"),
                "agent_name": payload.get("agent_name"),
                "action_family": payload.get("action_family"),
                "target_capability": payload.get("target_capability"),
                "selected_skill": payload.get("selected_skill"),
                "parameters": _compact_scalar(_as_dict(payload.get("parameters"))),
                "rationale": _truncate_text(payload.get("rationale"), limit=320),
                "expected_observation": _truncate_text(payload.get("expected_observation"), limit=220),
                "success_criteria": _recent(_as_list(payload.get("success_criteria")), limit=4),
                "fallback_if_failed": _recent(_as_list(payload.get("fallback_if_failed")), limit=3),
                "cost_estimate": payload.get("cost_estimate"),
                "risk_estimate": payload.get("risk_estimate"),
                "confidence": payload.get("confidence"),
                "cost_class": content.get("cost_class"),
                "risk_class": content.get("risk_class"),
                "submit_external_job": bool(payload.get("submit_external_job")),
                "wait_for_event_after_submission": bool(payload.get("wait_for_event_after_submission")),
            }
        )
    return output


def summarize_critiques(critiques: list[Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    values = _as_list(critiques)
    if limit is not None:
        values = _recent(values, limit=limit)
    for item in values:
        payload = _as_dict(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        output.append(
            {
                "proposal_id": payload.get("proposal_id"),
                "agent_name": payload.get("agent_name"),
                "stance": payload.get("stance"),
                "concerns": _recent(_as_list(payload.get("concerns")), limit=4),
                "recommendation": _truncate_text(payload.get("recommendation"), limit=220),
                "confidence": payload.get("confidence"),
            }
        )
    return output


def summarize_preferences(preferences: list[Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    values = _as_list(preferences)
    if limit is not None:
        values = _recent(values, limit=limit)
    for item in values:
        payload = _as_dict(item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
        output.append(
            {
                "preferred_proposal_id": payload.get("preferred_proposal_id"),
                "agent_name": payload.get("agent_name"),
                "preference_strength": payload.get("preference_strength"),
                "reason": _truncate_text(payload.get("reason"), limit=220),
                "confidence": payload.get("confidence"),
            }
        )
    return output


def summarize_guardrail_context(record: Any) -> dict[str, Any]:
    payload = _as_dict(record.model_dump(mode="json") if hasattr(record, "model_dump") else record)
    content = _as_dict(payload.get("content"))
    summarized_content = {
        "legal_proposal_ids": _recent(_as_list(content.get("legal_proposal_ids")), limit=8),
        "guardrail_preferred_proposal_id": content.get("guardrail_preferred_proposal_id"),
        "objection_counts": _compact_scalar(_as_dict(content.get("objection_counts"))),
        "support_counts": _compact_scalar(_as_dict(content.get("support_counts"))),
        "preference_strengths": _compact_scalar(_as_dict(content.get("preference_strengths"))),
        "supported_agent_opinions": _recent(_as_list(content.get("supported_agent_opinions")), limit=8),
    }
    return {
        "selected_proposal_id": payload.get("selected_proposal_id"),
        "guardrail_notes": _recent(_as_list(payload.get("guardrail_notes")), limit=8),
        "rejected_proposals": _recent(_as_list(payload.get("rejected_proposals")), limit=8),
        "rationale": _truncate_text(payload.get("rationale"), limit=240),
        "whether_noop": bool(payload.get("whether_noop")),
        "whether_waiting_external": bool(payload.get("whether_waiting_external")),
        "whether_ready_to_finalize": bool(payload.get("whether_ready_to_finalize")),
        "content": summarized_content,
        **summarized_content,
    }


def _strain_points_by_direction(strain_data: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _as_list(strain_data):
        payload = _as_dict(item)
        direction = str(payload.get("direction") or "").strip()
        if not direction:
            continue
        counts[direction] = int(counts.get(direction, 0) or 0) + 1
    return counts


def build_llm_context_summary(
    state: dict[str, Any],
    *,
    execution_status: dict[str, Any] | None = None,
    observation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task = _as_dict(state.get("task"))
    material = _as_dict(state.get("material"))
    workflow = _as_dict(state.get("workflow"))
    diagnostics = _as_dict(state.get("diagnostics"))
    physics = _as_dict(state.get("physics_results"))
    task_board = _as_dict(state.get("task_board"))
    deliberation = _as_dict(state.get("deliberation"))
    blackboard = _as_dict(state.get("blackboard"))
    memory = _as_dict(state.get("memory"))
    services = _as_dict(state.get("services"))
    status = _as_dict(execution_status)
    observation = _as_dict(observation_summary)
    validation_report = _as_dict(diagnostics.get("validation_report"))
    recovery_summary = _as_dict(diagnostics.get("recovery_summary"))
    results_by_direction = _as_dict(_as_dict(physics.get("results")).get("results_by_direction"))
    contract = _as_dict(services.get("workflow_contract"))

    return {
        "mission": {
            "task_id": task.get("task_id"),
            "task_type": task.get("task_type"),
            "user_goal": _truncate_text(task.get("user_goal"), limit=220),
            "material_id": material.get("material_id"),
            "composition": material.get("composition"),
            "dry_run": bool(task.get("dry_run")),
        },
        "workflow": {
            "current_stage": workflow.get("current_stage"),
            "run_status": workflow.get("run_status"),
            "wait_reason": workflow.get("wait_reason"),
            "termination_reason": workflow.get("termination_reason"),
            "completed_stages": _recent(_as_list(workflow.get("completed_stages")), limit=_RECENT_STAGE_WINDOW),
            "stage_status": _compact_scalar(_as_dict(workflow.get("stage_status"))),
            "retry_counts": _compact_scalar(_as_dict(workflow.get("retry_counts"))),
            "retry_budget": workflow.get("retry_budget"),
            "refinement_rounds": workflow.get("refinement_rounds"),
            "max_refinement_rounds": workflow.get("max_refinement_rounds"),
            "next_pending_capability": status.get("next_pending_capability"),
        },
        "task_board": {
            "pending_capabilities": task_capabilities(_as_list(task_board.get("pending_tasks"))),
            "active_capabilities": task_capabilities(_as_list(task_board.get("active_tasks"))),
            "blocked_capabilities": task_capabilities(_as_list(task_board.get("blocked_tasks"))),
            "completed_capabilities": task_capabilities(_as_list(task_board.get("completed_tasks"))),
            "abandoned_capabilities": task_capabilities(_as_list(task_board.get("abandoned_tasks"))),
        },
        "latest_execution_observation": compact_latest_observation(state),
        "physics_signals": {
            "accepted_channels": _as_list(physics.get("accepted_channels")),
            "rejected_channels": _as_list(physics.get("rejected_channels")),
            "fit_quality": observation.get("fit_quality"),
            "confidence_score": observation.get("confidence_score", diagnostics.get("confidence_score")),
            "anomaly_flags": dedupe_keep_order(
                _as_list(observation.get("anomaly_flags")) + _as_list(blackboard.get("anomaly_flags"))
            ),
            "risk_flags": dedupe_keep_order(_as_list(observation.get("risk_flags")) + _as_list(blackboard.get("risk_flags"))),
            "strain_points_by_direction": _strain_points_by_direction(_as_list(physics.get("strain_data"))),
            "results_by_direction": compact_results_by_direction(results_by_direction),
        },
        "diagnostics": {
            "last_error": _truncate_text(diagnostics.get("last_error"), limit=240) or None,
            "validation_decision": validation_report.get("decision"),
            "validation_failed_checks": _as_list(validation_report.get("failed_checks")),
            "validation_warning_count": len(_as_list(validation_report.get("warnings"))),
            "validation_fit_metrics": _compact_scalar(_as_dict(validation_report.get("fit_metrics"))),
            "recovery_summary": {
                "stage": recovery_summary.get("stage") or recovery_summary.get("current_stage"),
                "error_type": recovery_summary.get("error_type"),
                "error_summary": _truncate_text(recovery_summary.get("error_summary"), limit=220) or None,
            },
        },
        "contract_summary": {
            "contract_id": contract.get("contract_id"),
            "version": contract.get("version"),
            "council_mode": contract.get("council_mode"),
            "deliberation_reason": contract.get("deliberation_reason"),
            "current_focus": contract.get("current_focus"),
            "planned_capabilities": _recent(_as_list(contract.get("planned_capabilities")), limit=6),
            "milestones": _recent(_as_list(contract.get("milestones")), limit=4),
        },
        "recent_deliberation": {
            "selected_actions": summarize_selected_actions(_as_list(deliberation.get("selected_actions"))),
            "arbitrations": summarize_arbitrations(_as_list(deliberation.get("arbitrations"))),
            "reflections": summarize_reflections(_as_list(deliberation.get("reflections"))),
            "disagreement_records": summarize_disagreement_records(_as_list(deliberation.get("disagreement_records"))),
        },
        "memory_hints": {
            "recovered_case_patterns": summarize_memory_patterns(_as_list(memory.get("recovered_case_patterns"))),
            "validation_case_patterns": summarize_memory_patterns(_as_list(memory.get("validation_case_patterns"))),
            "historical_failures": summarize_memory_patterns(_as_list(memory.get("historical_failures"))),
        },
    }


def select_role_context(summary: dict[str, Any], *, role: str) -> dict[str, Any]:
    normalized = str(role or "specialist").strip().lower()
    fields_by_role = {
        "planner": ["mission", "workflow", "task_board", "latest_execution_observation", "physics_signals", "diagnostics", "contract_summary", "recent_deliberation"],
        "recovery": ["mission", "workflow", "task_board", "latest_execution_observation", "diagnostics", "contract_summary", "recent_deliberation", "memory_hints"],
        "critic": ["mission", "workflow", "latest_execution_observation", "diagnostics", "contract_summary", "recent_deliberation"],
        "physics_judge": ["mission", "workflow", "latest_execution_observation", "physics_signals", "diagnostics", "contract_summary", "recent_deliberation"],
        "cost_guardian": ["mission", "workflow", "latest_execution_observation", "diagnostics", "contract_summary", "recent_deliberation"],
        "orchestrator": ["mission", "workflow", "latest_execution_observation", "physics_signals", "diagnostics", "contract_summary", "recent_deliberation"],
        "reporter": ["mission", "workflow", "task_board", "latest_execution_observation", "physics_signals", "diagnostics", "contract_summary", "recent_deliberation"],
    }
    chosen = fields_by_role.get(normalized, ["mission", "workflow", "latest_execution_observation", "diagnostics"])
    return {field: _as_dict(summary.get(field)) for field in chosen}
