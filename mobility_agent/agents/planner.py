from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .base import SkillAwareAgent
from .context_engineering import build_llm_context_summary, select_role_context
from .schemas import Proposal, ProposalBundle
from ..runtime.deliberation_loop import all_tasks_resolved
from ..runtime.telemetry import emit_progress
from ..runtime.action_registry import list_action_families


class PlannerAgent(SkillAwareAgent):
    agent_name = "planner"
    llm_role = "planner"

    @staticmethod
    def _has_post_validation_context(state: dict[str, Any]) -> bool:
        diagnostics = dict(state.get("diagnostics", {}) or {})
        return bool(diagnostics.get("fit_diagnostics")) or bool(diagnostics.get("validation_report"))

    @staticmethod
    def _task_capabilities(items: list[Any]) -> list[str]:
        caps: list[str] = []
        for item in list(items or []):
            if isinstance(item, dict):
                cap = str(item.get("capability") or "").strip()
            else:
                cap = str(item or "").strip()
            if cap:
                caps.append(cap)
        return caps

    @staticmethod
    def _recent_items(items: list[Any], *, limit: int = 2) -> list[Any]:
        values = list(items or [])
        return values[-limit:] if len(values) > limit else values

    @staticmethod
    def _compact_latest_observation(state: dict[str, Any]) -> dict[str, Any]:
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        if not latest:
            latest = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        return {
            "status": latest.get("status"),
            "action_family": latest.get("action_family"),
            "target_capability": latest.get("target_capability"),
            "error_summary": latest.get("error_summary"),
            "result_summary": dict(latest.get("result_summary", {}) or {}),
            "artifact_paths": dict(latest.get("artifact_paths", {}) or {}),
        }

    @staticmethod
    def _compact_results_by_direction(results_by_direction: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for direction, payload in dict(results_by_direction or {}).items():
            direction_payload = dict(payload or {})
            electron = dict(direction_payload.get("electron", {}) or {})
            hole = dict(direction_payload.get("hole", {}) or {})
            summary[str(direction)] = {
                "n_points": direction_payload.get("n_points"),
                "elastic_modulus_C2D_J_m2": direction_payload.get("elastic_modulus_C2D_J_m2"),
                "electron_mobility_cm2_Vs": electron.get("mobility_cm2_Vs"),
                "hole_mobility_cm2_Vs": hole.get("mobility_cm2_Vs"),
                "electron_E1_fit_R2": electron.get("E1_fit_R2"),
                "hole_E1_fit_R2": hole.get("E1_fit_R2"),
            }
        return summary

    @staticmethod
    def _default_skill_for_capability(capability: str) -> str:
        return "physics_validation" if str(capability or "").strip() == "validation" else "single_material_mobility"

    def _default_capability_proposal(
        self,
        *,
        state: dict[str, Any],
        round_id: int,
        capability: str,
        rationale: str,
    ) -> Proposal:
        metadata = self.tool_gateway.call(
            "query_capability_metadata",
            {"action_family": "run_capability", "capability": capability},
        )
        task_id = str(state.get("task", {}).get("task_id") or "")
        return Proposal.model_validate(
            {
                "agent_name": self.agent_name,
                "round_id": round_id,
                "target_task_id": f"capability::{capability}" if capability else task_id,
                "proposal_id": f"{self.agent_name}::{round_id}::default::{capability}",
                "action_family": "run_capability",
                "target_capability": capability,
                "selected_skill": self._default_skill_for_capability(capability),
                "rationale": rationale,
                "expected_observation": f"{capability} capability executes with the currently available artifacts",
                "success_criteria": [f"{capability} stage completes successfully"],
                "fallback_if_failed": ["retry_capability", "escalate_human"],
                "content": {
                    "cost_class": metadata.get("cost_class", "medium"),
                    "risk_class": metadata.get("risk_class", "medium"),
                },
                "confidence": 0.9,
            }
        )

    def _ensure_default_next_capability_proposal(
        self,
        *,
        state: dict[str, Any],
        round_id: int,
        proposals: list[Proposal],
        default_next_capability: str,
    ) -> list[Proposal]:
        capability = str(default_next_capability or "").strip()
        if not capability:
            return proposals
        for item in proposals:
            if str(item.action_family or "") == "run_capability" and str(item.target_capability or "") == capability:
                return proposals
        return proposals + [
            self._default_capability_proposal(
                state=state,
                round_id=round_id,
                capability=capability,
                rationale=f"default_next_capability:{capability}",
            )
        ]

    def _visible_state_summary(self, *, state: dict[str, Any], round_id: int, hints: dict[str, Any]) -> dict[str, Any]:
        execution_status = self.tool_gateway.call("query_execution_status", {"state": state})
        observation_summary = self.tool_gateway.call("synthesize_observation", {"state": state})
        context_summary = select_role_context(
            dict((state.get("services", {}) or {}).get("llm_context_summary", {}) or {})
            or build_llm_context_summary(
                state,
                execution_status=execution_status,
                observation_summary=observation_summary,
            ),
            role=self.llm_role,
        )
        return {
            "round_id": round_id,
            "context_summary": context_summary,
            "observation_summary": observation_summary,
            "execution_status": execution_status,
            "planning_hints": hints,
        }

    def _planning_hints(self, *, state: dict[str, Any], round_id: int) -> dict[str, Any]:
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        status = self.tool_gateway.call("query_execution_status", {"state": state})
        pending_capability = str(status.get("next_pending_capability") or "")
        run_status = str(status.get("run_status") or "")
        task_board = dict(state.get("task_board", {}) or {})
        hints: dict[str, Any] = {
            "round_id": round_id,
            "run_status": run_status,
            "ready_to_finalize": run_status == "ready_to_finalize",
            "latest_observation_status": str(latest_observation.get("status") or ""),
            "default_next_capability": None,
            "task_board_summary": {
                "pending_capabilities": self._task_capabilities(task_board.get("pending_tasks", [])),
                "active_capabilities": self._task_capabilities(task_board.get("active_tasks", [])),
                "blocked_capabilities": self._task_capabilities(task_board.get("blocked_tasks", [])),
                "completed_capabilities": self._task_capabilities(task_board.get("completed_tasks", [])),
                "abandoned_capabilities": self._task_capabilities(task_board.get("abandoned_tasks", [])),
            },
            "dependency_hints": [],
            "mainline_reminder": (
                "Use the task board and capability sequence as context only. "
                "They suggest the default scientific next step but do not determine the action by themselves."
            ),
        }

        if run_status in {"waiting_external", "needs_human"}:
            return hints

        if pending_capability and latest_observation.get("status") != "failed":
            capability = pending_capability
            metadata = self.tool_gateway.call(
                "query_capability_metadata",
                {"action_family": "run_capability", "capability": capability},
            )
            hints["default_next_capability"] = {
                "capability": capability,
                "recommended_action": "run_capability",
                "selected_skill": "single_material_mobility",
                "cost_class": metadata.get("cost_class", "medium"),
                "risk_class": metadata.get("risk_class", "medium"),
                "dependencies": list(metadata.get("dependencies", []) or []),
                "expected_artifacts": list(metadata.get("expected_artifacts", []) or []),
            }
            hints["dependency_hints"] = list(metadata.get("dependencies", []) or [])
        return hints

    def propose(self, *, state: dict[str, Any], round_id: int) -> list[Proposal]:
        hints = self._planning_hints(state=state, round_id=round_id)
        if str(hints.get("run_status") or "") in {"waiting_external", "needs_human"}:
            return []
        payload = {
            "state_summary": self._visible_state_summary(state=state, round_id=round_id, hints=hints),
            "round_id": round_id,
            "planning_hints": hints,
            "allowed_actions": list_action_families(),
        }

        def _invoke(local_payload: dict[str, Any]) -> ProposalBundle:
            llm_result = self._call_llm_structured_with_tools(
                schema=ProposalBundle,
                task_type=str(state.get("task", {}).get("task_type") or "single_material"),
                stage=str(state.get("workflow", {}).get("current_stage") or "observe_state"),
                payload=local_payload,
                system_prompt=(
                    "You are the Planner Agent in an LLM-centered multi-agent scientific runtime. "
                    "Propose the next high-value actions. Use the default scientific mainline, task board, and dependency hints as scientific context, "
                    "not as an automatic workflow controller. The actual proposals must be your own structured planning judgment. "
                    "Use only the summarized state provided here; do not ask for or assume hidden full-state details."
                ),
                user_prompt=(
                    "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                    "STATE_AND_HINT_CONTEXT_JSON:\n{payload}\n\n"
                    "Return one or more structured proposals."
                ),
                explicit_skills=["recovery"],
                tool_names=self._tool_names_for_role(),
            )
            return ProposalBundle.model_validate(llm_result)

        default_next_capability = str(dict(hints.get("default_next_capability", {}) or {}).get("capability") or "").strip()
        current_run_status = str(hints.get("run_status") or "")
        requires_actionable_plan = bool(default_next_capability) or current_run_status in {
            "pending",
            "running",
        }
        latest_failed = str(self._compact_latest_observation(state).get("status") or "") == "failed"
        if current_run_status == "needs_recovery" or latest_failed:
            requires_actionable_plan = False
        if not default_next_capability and current_run_status in {"pending", "running"} and all_tasks_resolved(state):
            requires_actionable_plan = False
        task_id = str(state.get("task", {}).get("task_id") or "")
        task_board_summary = dict(hints.get("task_board_summary", {}) or {})
        capability_allowlist: list[str] = []
        for key in (
            "pending_capabilities",
            "active_capabilities",
            "blocked_capabilities",
            "completed_capabilities",
            "abandoned_capabilities",
        ):
            for item in list(task_board_summary.get(key, []) or []):
                capability = str(item or "").strip()
                if capability and capability not in capability_allowlist:
                    capability_allowlist.append(capability)
        if default_next_capability and default_next_capability not in capability_allowlist:
            capability_allowlist.append(default_next_capability)
        capability_required_actions = {
            "run_capability",
            "retry_capability",
            "rerun_from_capability",
            "repair_execution_context",
            "refine_sampling",
            "revalidate_result",
            "invalidate_channel",
            "skip_channel",
        }

        def _normalize(bundle: ProposalBundle) -> tuple[list[Proposal], list[dict[str, Any]]]:
            normalized: list[Proposal] = []
            dropped: list[dict[str, Any]] = []
            suppress_post_validation_actions = self._has_post_validation_context(state)
            for idx, item in enumerate(bundle.proposals, start=1):
                payload = item.model_dump(mode="json")
                action_family = str(payload.get("action_family") or "")
                target_capability = str(payload.get("target_capability") or "").strip() or None
                if suppress_post_validation_actions and action_family in {"refine_sampling", "finalize_material"}:
                    dropped.append(
                        {
                            "reason": "planner_post_validation_action_suppressed",
                            "action_family": action_family,
                            "target_capability": target_capability,
                        }
                    )
                    continue
                if action_family in capability_required_actions:
                    if not target_capability:
                        dropped.append({"reason": "missing_target_capability", "action_family": action_family})
                        continue
                    if capability_allowlist and target_capability not in capability_allowlist:
                        dropped.append(
                            {
                                "reason": "unsupported_target_capability",
                                "action_family": action_family,
                                "target_capability": target_capability,
                            }
                        )
                        continue
                target_task_id = f"capability::{target_capability}" if target_capability else task_id
                normalized.append(
                    Proposal.model_validate(
                        {
                            **payload,
                            "agent_name": self.agent_name,
                            "round_id": round_id,
                            "target_task_id": target_task_id,
                            "proposal_id": str(
                                payload.get("proposal_id")
                                or f"{self.agent_name}::{round_id}::{idx}::{action_family or 'unknown'}::{target_capability or 'none'}"
                            ),
                        }
                    )
                )
            return normalized, dropped

        try:
            parsed = _invoke(payload)
            normalized, dropped = _normalize(parsed)
            if requires_actionable_plan and not normalized and default_next_capability:
                normalized = self._ensure_default_next_capability_proposal(
                    state=state,
                    round_id=round_id,
                    proposals=normalized,
                    default_next_capability=default_next_capability,
                )
            if requires_actionable_plan and not normalized:
                emit_progress(
                    "planner returned empty or invalid proposals; retrying with stricter capability constraints",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "round_id": round_id,
                        "default_next_capability": default_next_capability or None,
                        "capability_allowlist": capability_allowlist,
                        "dropped_proposals": dropped,
                    },
                )
                parsed = _invoke(
                    {
                        **payload,
                        "planner_repair_hint": (
                            "Current context requires at least one actionable proposal. "
                            "Return a non-empty proposals list, using supported action families and supported capability targets only."
                        ),
                        "capability_allowlist": capability_allowlist,
                        "preferred_default_next_capability": default_next_capability or None,
                    }
                )
                normalized, dropped = _normalize(parsed)
                if requires_actionable_plan and not normalized and default_next_capability:
                    normalized = self._ensure_default_next_capability_proposal(
                        state=state,
                        round_id=round_id,
                        proposals=normalized,
                        default_next_capability=default_next_capability,
                    )
                if not normalized:
                    raise RuntimeError("planner_strict_agentic_failure:planner_invalid_or_empty_proposals_after_retry")
        except RuntimeError as exc:
            message = str(exc)
            if (
                "structured_output_invalid" in message
                or "request_failed" in message
                or "planner_invalid_or_empty_proposals_after_retry" in message
            ):
                emit_progress(
                    "planner llm structured output invalid; strict-agentic mode aborts this deliberation",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "round_id": round_id,
                        "error": message,
                    },
                )
                raise RuntimeError(f"planner_strict_agentic_failure:{message}") from exc
            raise
        except ValidationError as exc:
            message = f"planner_structured_output_invalid:{exc}"
            emit_progress(
                "planner llm payload validation failed; strict-agentic mode aborts this deliberation",
                channel="agent",
                details={
                    "agent": self.agent_name,
                    "role": self.llm_role,
                    "round_id": round_id,
                    "error": message,
                },
            )
            raise RuntimeError(f"planner_strict_agentic_failure:{message}") from exc
        if default_next_capability:
            normalized = self._ensure_default_next_capability_proposal(
                state=state,
                round_id=round_id,
                proposals=normalized,
                default_next_capability=default_next_capability,
            )
        return normalized
