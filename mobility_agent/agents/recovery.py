from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .base import SkillAwareAgent
from .context_engineering import build_llm_context_summary, select_role_context
from .schemas import Proposal, ProposalBundle, RecoveryDecision
from ..graph.escalation import should_escalate_recovery
from ..graph.stage_contracts import find_previous_stage
from ..policy.engine import AgenticPolicyEngine
from ..runtime.telemetry import emit_progress
from ..runtime.action_registry import list_action_families


class RecoveryAgent(SkillAwareAgent):
    agent_name = "recovery"
    llm_role = "recovery"

    def __init__(self, runtime, skills_root: str):
        super().__init__(runtime, skills_root)
        self.policy_engine = AgenticPolicyEngine(runtime)
        self.last_failure_diagnosis: dict[str, Any] = {}

    @staticmethod
    def _recent_items(items: list[Any], *, limit: int = 2) -> list[Any]:
        values = list(items or [])
        return values[-limit:] if len(values) > limit else values

    @staticmethod
    def _compact_latest_failure(state: dict[str, Any]) -> dict[str, Any]:
        latest = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        if str(latest.get("status") or "") != "failed":
            latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        return {
            "status": latest.get("status"),
            "stage": latest.get("target_capability") or latest.get("stage"),
            "action_family": latest.get("action_family"),
            "error_summary": latest.get("error_summary"),
            "error_category": latest.get("error_category"),
            "result_summary": dict(latest.get("result_summary", {}) or {}),
            "artifact_paths": dict(latest.get("artifact_paths", {}) or {}),
        }

    def _visible_recovery_summary(self, *, state: dict[str, Any], round_id: int, hints: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
        execution_status = self.tool_gateway.call("query_execution_status", {"state": state})
        observation_summary = self.tool_gateway.call("synthesize_observation", {"state": state})
        try:
            failure_evidence = self.tool_gateway.call(
                "retrieve_failure_evidence",
                {
                    "state": state,
                    "stage": str((hints.get("failure_context", {}) or {}).get("stage") or ""),
                    "top_k": max(1, int(self.runtime.policy_retrieval_top_k or 5)),
                },
            )
        except Exception as exc:
            failure_evidence = {"error": f"{type(exc).__name__}:{exc}", "items": []}
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
            "latest_failure": self._compact_latest_failure(state),
            "execution_status": execution_status,
            "observation_summary": observation_summary,
            "recovery_hints": hints,
            "failure_evidence": failure_evidence,
            "agentic_diagnosis": diagnosis,
        }

    def _recovery_hints(self, *, state: dict[str, Any], round_id: int) -> dict[str, Any]:
        workflow = dict(state.get("workflow", {}) or {})
        diagnostics = dict(state.get("diagnostics", {}) or {})
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        latest_status = str(latest_observation.get("status") or "")
        if latest_status != "failed":
            latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
            latest_status = str(latest_observation.get("status") or "")

        run_status = str(workflow.get("run_status") or "")
        synthetic_failure = False
        if latest_status != "failed":
            if run_status != "needs_recovery":
                return {}
            synthetic_failure = True

        status = self.tool_gateway.call("query_execution_status", {"state": state})
        current_stage = str(
            latest_observation.get("target_capability")
            or latest_observation.get("stage")
            or workflow.get("current_stage")
            or status.get("next_pending_capability")
            or ""
        )
        if not current_stage:
            return {}

        retries_used = int((status.get("retry_counts", {}) or {}).get(current_stage, 0) or 0)
        retry_budget = int(workflow.get("retry_budget", 2) or 2)
        previous_stage = find_previous_stage(current_stage) or current_stage
        retry_meta = self.tool_gateway.call(
            "query_capability_metadata",
            {"action_family": "retry_capability", "capability": current_stage},
        )
        rerun_meta = self.tool_gateway.call(
            "query_capability_metadata",
            {"action_family": "rerun_from_capability", "capability": previous_stage},
        )
        failure_status = "failed"
        fallback_error = str(diagnostics.get("last_error") or "")
        error_summary = str(latest_observation.get("error_summary") or fallback_error)
        error_category = str(latest_observation.get("error_category") or "")
        if synthetic_failure and not error_category and fallback_error:
            error_category = "framework_needs_recovery"
        safe_action_hints = [
            "retry_capability",
            "rerun_from_capability",
            "repair_execution_context",
            "escalate_human",
        ]
        if retries_used >= retry_budget or synthetic_failure:
            safe_action_hints.append("abort_material")
        return {
            "round_id": round_id,
            "failure_context": {
                "status": failure_status,
                "stage": current_stage,
                "error_summary": error_summary,
                "error_category": error_category,
            },
            "retry_context": {
                "retries_used": retries_used,
                "retry_budget": retry_budget,
                "retry_budget_remaining": max(0, retry_budget - retries_used),
                "safe_retry_allowed": retries_used < retry_budget,
            },
            "recovery_constraints": {
                "current_stage": current_stage,
                "previous_stage": previous_stage,
                "retry_required_evidence": list(retry_meta.get("required_evidence", []) or []),
                "rerun_required_evidence": list(rerun_meta.get("required_evidence", []) or []),
                "retry_cost_class": retry_meta.get("cost_class", "medium"),
                "rerun_cost_class": rerun_meta.get("cost_class", "high"),
                "retry_risk_class": retry_meta.get("risk_class", "medium"),
                "rerun_risk_class": rerun_meta.get("risk_class", "medium"),
            },
            "safe_action_hints": safe_action_hints,
            "recovery_context": {
                "synthetic_failure": synthetic_failure,
                "run_status": run_status,
                "latest_observation_status": latest_status,
            },
            "mainline_reminder": (
                "Use recovery to get back onto a credible scientific path. "
                "Do not merely repeat the default mainline if the failure evidence suggests a different repair strategy."
            ),
        }

    def propose(self, *, state: dict[str, Any], round_id: int) -> list[Proposal]:
        hints = self._recovery_hints(state=state, round_id=round_id)
        if not hints:
            return []
        latest_failure = self._compact_latest_failure(state)
        try:
            diagnosis_model = self.policy_engine.diagnose_failure(
                stage=str(latest_failure.get("stage") or hints.get("failure_context", {}).get("stage") or ""),
                state_payload=state,
                latest_failure=latest_failure,
                allowed_actions=list_action_families(),
            )
            diagnosis = diagnosis_model.model_dump(mode="json")
        except Exception as exc:
            diagnosis = {
                "stage": str(latest_failure.get("stage") or hints.get("failure_context", {}).get("stage") or ""),
                "source": "fallback",
                "hypotheses": [str(latest_failure.get("error_summary") or "unknown_failure")],
                "recommended_action": "retry_capability",
                "parameter_patch": {},
                "needs_human": False,
                "evidence_refs": [],
                "confidence": 0.2,
                "rationale": f"recovery_diagnosis_fallback:{type(exc).__name__}",
            }
        self.last_failure_diagnosis = diagnosis
        if (
            str(diagnosis.get("source") or "") == "deterministic_relax_failure_policy"
            and str(diagnosis.get("recommended_action") or "") == "escalate_human"
            and bool(diagnosis.get("needs_human"))
        ):
            stage = str(latest_failure.get("stage") or hints.get("failure_context", {}).get("stage") or "relax")
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": str(state.get("task", {}).get("task_id") or ""),
                        "proposal_id": f"{self.agent_name}::{round_id}::relax_failure::escalate_human",
                        "action_family": "escalate_human",
                        "target_capability": None,
                        "parameters": {
                            "recommended_options": [
                                "manual_fix_resume",
                                "retry_current_stage",
                                "rerun_previous_stage",
                                "skip_material",
                                "abort_task",
                            ]
                        },
                        "expected_benefit": "human reviews relaxation failure before any further automated retry",
                        "expected_risk": "workflow pauses for manual decision",
                        "rationale": (
                            f"Relaxation failure detected in {stage}; user policy requires human intervention "
                            "for RELAX_FAILED/relax_failed/CONTCAR relaxation-output failures only."
                        ),
                        "expected_observation": "human_escalation_payload is written and notification is sent",
                        "success_criteria": ["human escalation payload is available", "notification backend is invoked"],
                        "fallback_if_failed": ["abort_material"],
                        "confidence": float(diagnosis.get("confidence") or 0.98),
                    }
                )
            ]
        payload = {
            "state_summary": self._visible_recovery_summary(state=state, round_id=round_id, hints=hints, diagnosis=diagnosis),
            "round_id": round_id,
            "recovery_hints": hints,
            "agentic_diagnosis": diagnosis,
            "allowed_actions": list_action_families(),
        }
        try:
            llm_result = self._call_llm_structured_with_tools(
                schema=ProposalBundle,
                task_type=str(state.get("task", {}).get("task_type") or "single_material"),
                stage=str((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}).get("target_capability") or state.get("workflow", {}).get("current_stage") or "unknown"),
                payload=payload,
                system_prompt=(
                    "You are the Recovery Agent in an LLM-centered multi-agent scientific runtime. "
                    "When execution fails, propose multiple recovery strategies. Use failure summaries, retry limits, previous-stage context, "
                    "and recovery constraints as hints and guardrails, but generate the recovery plan yourself. "
                    "Use only the summarized state provided here; do not assume hidden full-state details."
                ),
                user_prompt=(
                    "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\n"
                    "AVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                    "STATE_AND_HINT_CONTEXT_JSON:\n{payload}\n\n"
                    "Return one or more structured recovery proposals."
                ),
                explicit_skills=["recovery"],
                tool_names=self._tool_names_for_role(),
            )
        except RuntimeError as exc:
            message = str(exc)
            if "structured_output_invalid" in message or "request_failed" in message:
                emit_progress(
                    "recovery llm structured output invalid; strict-agentic mode aborts this deliberation",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "round_id": round_id,
                        "error": message,
                    },
                )
                raise RuntimeError(f"recovery_strict_agentic_failure:{message}") from exc
            raise
        try:
            parsed = ProposalBundle.model_validate(llm_result)
        except ValidationError as exc:
            message = f"recovery_structured_output_invalid:{exc}"
            emit_progress(
                "recovery llm payload validation failed; strict-agentic mode aborts this deliberation",
                channel="agent",
                details={
                    "agent": self.agent_name,
                    "role": self.llm_role,
                    "round_id": round_id,
                    "error": message,
                },
            )
            raise RuntimeError(f"recovery_strict_agentic_failure:{message}") from exc
        task_id = str(state.get("task", {}).get("task_id") or "")
        normalized: list[Proposal] = []
        for idx, item in enumerate(parsed.proposals, start=1):
            target_task_id = f"capability::{item.target_capability}" if item.target_capability else task_id
            normalized.append(
                Proposal.model_validate(
                    {
                        **item.model_dump(mode="json"),
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": target_task_id,
                        "proposal_id": str(item.proposal_id or f"{self.agent_name}::{round_id}::{idx}::{item.action_family}::{item.target_capability or 'none'}"),
                    }
                )
            )
        return normalized

    def decide(self, *, state: dict[str, Any], summary: dict[str, Any], allowed_actions: list[str]) -> RecoveryDecision:
        stage = str(summary.get("stage") or summary.get("current_stage") or state.get("workflow", {}).get("current_stage") or "")
        error_type = str(summary.get("error_type") or "")
        retries_used = int(summary.get("retries_used", 0) or 0)
        has_contcar = bool(summary.get("has_contcar", False))
        historical_cases = list(summary.get("historical_cases", []) or [])
        decision = "skip_material"
        reason = f"default_recovery:{error_type or 'unknown_failure'}"
        confidence = 0.70
        parameter_updates: dict[str, Any] = {}
        return_stage = stage
        cleanup_policy = "retry_current_stage_only"

        for item in historical_cases:
            chosen_action = str((item or {}).get("chosen_action") or "")
            if chosen_action in allowed_actions and str((item or {}).get("success_or_failure") or "") in {"running", "completed"}:
                decision = chosen_action
                reason = f"historical_recovery_case:{chosen_action}"
                confidence = 0.80
                if chosen_action == "rerun_previous_stage":
                    return_stage = find_previous_stage(stage) or stage
                    cleanup_policy = "invalidate_downstream"
                break

        if "copy_contcar_to_poscar_and_retry" in allowed_actions and error_type in {"zbrent_fatal", "nonconverged", "rerun_with_smaller_ediff_or_copy_contcar"} and has_contcar:
            decision = "copy_contcar_to_poscar_and_retry"
            reason = f"continuation_recovery:{error_type}"
            confidence = 0.88
            parameter_updates = {"EDIFF": 1e-6}
        elif "modify_params_and_retry" in allowed_actions and error_type in {"zbrent_fatal", "nonconverged"}:
            decision = "modify_params_and_retry"
            reason = f"tighten_relax_parameters:{error_type}"
            confidence = 0.82
            parameter_updates = {"EDIFF": 1e-6}
        elif "retry_current_stage" in allowed_actions and error_type in {"missing_output", "nonzero_exit"}:
            decision = "retry_current_stage"
            reason = f"retry_stage_after:{error_type}"
            confidence = 0.72
        elif "rerun_previous_stage" in allowed_actions and stage != "relax":
            previous = find_previous_stage(stage) or stage
            decision = "rerun_previous_stage"
            reason = f"rerun_previous_stage:{previous}"
            confidence = 0.68
            return_stage = previous
            cleanup_policy = "invalidate_downstream"
        elif "manual_fix_resume" in allowed_actions and (error_type in {"unknown_failure", "missing_output"} or retries_used > 0):
            decision = "manual_fix_resume"
            reason = "manual_fix_recommended"
            confidence = 0.55
            cleanup_policy = "restart_from_stage"
        elif "skip_point" in allowed_actions and stage == "strain_loop":
            decision = "skip_point"
            reason = "skip_failed_strain_point"
            confidence = 0.65
        elif "abort_task" in allowed_actions and retries_used >= int(summary.get("max_retries", 2) or 2):
            decision = "abort_task"
            reason = "retry_budget_exhausted"
            confidence = 0.90

        rule = RecoveryDecision(
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            confidence=confidence,
            should_escalate=False,
            parameter_updates=parameter_updates,
            return_stage=return_stage,
            cleanup_policy=cleanup_policy,
        )
        llm = self._maybe_call_llm(
            kind="recovery",
            schema=RecoveryDecision,
            task_type=str(state.get("task", {}).get("task_type") or "single_material"),
            stage=stage,
            summary=summary,
            rule_payload=rule.model_dump(mode="json"),
            allowed_actions=allowed_actions,
            has_error=True,
        )
        merged = {**rule.model_dump(mode="json"), **(llm or {})}
        if merged.get("decision") not in allowed_actions:
            merged["decision"] = rule.decision
        merged["should_escalate"] = should_escalate_recovery(summary, merged)
        return RecoveryDecision.model_validate(merged)
