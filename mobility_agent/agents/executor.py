from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .schemas import ExecutionCommand, Proposal, SelectedAction
from ..runtime.deliberation_loop import next_pending_task


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for token in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


class ExecutorAgent(SkillAwareAgent):
    agent_name = "executor"
    llm_role = "executor"

    def propose(self, *, state: dict[str, Any], round_id: int) -> list[Proposal]:
        task_id = str(state.get("task", {}).get("task_id") or "")
        status = self.tool_gateway.call("query_execution_status", {"state": state})
        if str(status.get("run_status") or "") in {"waiting_external", "needs_human"}:
            return []
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        if latest_observation.get("status") == "failed":
            return []
        pending = next_pending_task(state)
        if not pending:
            return []
        capability = str(pending.get("capability") or "")
        metadata = self.tool_gateway.call(
            "query_capability_metadata",
            {"action_family": "run_capability", "capability": capability},
        )
        missing_inputs = [
            field_path
            for field_path in list(metadata.get("required_inputs", []) or [])
            if _is_missing(_nested_get(state, str(field_path)))
        ]
        if not missing_inputs:
            return []
        return [
            Proposal(
                agent_name=self.agent_name,
                round_id=round_id,
                target_task_id=task_id,
                proposal_id=f"executor::repair::{round_id}::{capability}",
                action_family="repair_execution_context",
                target_capability=capability,
                selected_skill="execution_feasibility",
                parameters={"repair_kind": "restore_required_inputs", "missing_inputs": missing_inputs},
                content={
                    "cost_class": "low",
                    "risk_class": "medium",
                    "required_inputs": list(metadata.get("required_inputs", []) or []),
                    "expected_artifacts": list(metadata.get("expected_artifacts", []) or []),
                },
                rationale=f"execution_inputs_missing_for:{capability}",
                expected_benefit="restore execution feasibility without blind rerun",
                expected_risk="context repair may still require human confirmation",
                expected_observation=f"required inputs restored for {capability}",
                success_criteria=["required inputs become available", "capability becomes executable"],
                fallback_if_failed=["escalate_human"],
                confidence=0.76,
            )
        ]

    def compile_selected_action(self, *, state: dict[str, Any], selected_action: SelectedAction, round_id: int) -> ExecutionCommand:
        capability = selected_action.target_capability
        metadata = self.tool_gateway.call(
            "query_capability_metadata",
            {"action_family": selected_action.action_family, "capability": capability},
        )
        expected_artifacts = list(metadata.get("expected_artifacts", []) or [])
        return ExecutionCommand(
            agent_name=self.agent_name,
            round_id=round_id,
            target_task_id=str(state.get("task", {}).get("task_id") or ""),
            action_family=selected_action.action_family,
            target_capability=capability,
            parameters=dict(selected_action.parameters or {}),
            expected_artifacts=expected_artifacts,
            dependency_snapshot={"depends_on": list(metadata.get("dependencies", []) or [])},
            submit_external_job=bool(selected_action.submit_external_job),
            wait_for_event_after_submission=bool(selected_action.wait_for_event_after_submission),
            content={"selected_action": selected_action.model_dump(mode="json")},
            confidence=0.95,
        )
