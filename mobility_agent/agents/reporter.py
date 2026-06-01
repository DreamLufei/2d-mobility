from __future__ import annotations

from typing import Any

from ..graph.state import (
    derive_compute_status_from_outcome_payload,
    resolve_outcome_scientific_decision,
    scientific_decision_bucket,
)
from ..runtime.telemetry import emit_progress
from .base import SkillAwareAgent
from .context_engineering import (
    build_llm_context_summary,
    select_role_context,
    summarize_disagreement_records,
    summarize_selected_actions,
)
from .schemas import BatchSummary, ReportSummary


class ReporterAgent(SkillAwareAgent):
    agent_name = "reporter"
    llm_role = "reporter"

    def _emit_reporter_fallback(self, *, stage: str, error: Exception) -> None:
        emit_progress(
            "reporter structured summary failed; using deterministic fallback",
            channel="agent",
            details={
                "agent": self.agent_name,
                "role": self.llm_role,
                "stage": stage,
                "error": f"{type(error).__name__}:{error}",
            },
        )

    def summarize_material(self, *, state: dict[str, Any]) -> dict[str, Any]:
        deliberation = dict(state.get("deliberation", {}) or {})
        diagnostics = dict(state.get("diagnostics", {}) or {})
        physics = dict(state.get("physics_results", {}) or {})
        workflow = dict(state.get("workflow", {}) or {})
        execution = dict(state.get("execution", {}) or {})
        summary = {
            "task_id": state.get("task", {}).get("task_id"),
            "material_id": state.get("material", {}).get("material_id"),
            "run_status": workflow.get("run_status"),
            "termination_reason": workflow.get("termination_reason"),
            "final_acceptance": diagnostics.get("validation_report", {}).get("decision"),
            "confidence_score": diagnostics.get("confidence_score"),
            "accepted_channels": list(physics.get("accepted_channels", []) or []),
            "rejected_channels": list(physics.get("rejected_channels", []) or []),
            "completed_capabilities": list(workflow.get("completed_stages", []) or []),
            "deliberation_rounds": int(deliberation.get("round_index", 0) or 0),
            "selected_actions": summarize_selected_actions(list(deliberation.get("selected_actions", []) or [])),
            "disagreement_records": summarize_disagreement_records(list(deliberation.get("disagreement_records", []) or [])),
            "artifact_paths": dict(execution.get("artifact_paths", {}) or {}),
        }
        context_summary = select_role_context(
            dict((state.get("services", {}) or {}).get("llm_context_summary", {}) or {})
            or build_llm_context_summary(state),
            role=self.llm_role,
        )
        try:
            llm_result = self._call_llm_structured_with_tools(
                schema=ReportSummary,
                task_type=str(state.get("task", {}).get("task_type") or "single_material"),
                stage="final_report",
                payload={"context_summary": context_summary, "summary_hints": summary},
                system_prompt=(
                    "You are the Reporter Agent in an LLM-centered multi-agent scientific runtime. "
                    "Produce a concise structured final summary that explains what was done, what succeeded, what was skipped, and why the run terminated. "
                    "Use the summary hints as context, but write the final structured summary yourself."
                ),
                user_prompt=(
                    "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                    "FINAL_CONTEXT_JSON:\n{payload}\n\n"
                    "Return a structured final summary."
                ),
                explicit_skills=["reporting"],
                tool_names=self._tool_names_for_role(),
            )
            parsed = ReportSummary.model_validate(llm_result)
        except Exception as exc:
            self._emit_reporter_fallback(stage="final_report", error=exc)
            return {
                **summary,
                "narrative_report": (
                    "Deterministic fallback summary used because the LLM reporter response "
                    "did not satisfy the structured summary schema."
                ),
                "report_generation_status": "fallback",
            }
        return {**summary, **dict(parsed.final_summary)}

    def summarize_batch(self, *, outcomes: list[dict[str, Any]]) -> BatchSummary:
        common_failure_stages: dict[str, int] = {}
        scientifically_passed = 0
        scientifically_warning = 0
        scientifically_failed = 0
        scientifically_unknown = 0
        for item in outcomes:
            status = derive_compute_status_from_outcome_payload(item)
            science_bucket = scientific_decision_bucket(resolve_outcome_scientific_decision(item))
            if science_bucket == "passed":
                scientifically_passed += 1
            elif science_bucket == "warning":
                scientifically_warning += 1
            elif science_bucket == "failed":
                scientifically_failed += 1
            else:
                scientifically_unknown += 1
            if status == "completed":
                continue
            stage_status = dict(item.get("stage_status", {}) or {})
            failed_stage = next((stage for stage, status in stage_status.items() if status == "failed"), None)
            failed_stage = str(failed_stage or item.get("termination_reason") or "unknown")
            common_failure_stages[failed_stage] = int(common_failure_stages.get(failed_stage, 0) or 0) + 1
        baseline = BatchSummary(
            processed=len(outcomes),
            succeeded=len([item for item in outcomes if derive_compute_status_from_outcome_payload(item) == "completed"]),
            failed=len(
                [
                    item
                    for item in outcomes
                    if derive_compute_status_from_outcome_payload(item) in {"failed", "aborted"}
                ]
            ),
            skipped=len([item for item in outcomes if derive_compute_status_from_outcome_payload(item) == "skipped"]),
            scientifically_passed=scientifically_passed,
            scientifically_warning=scientifically_warning,
            scientifically_failed=scientifically_failed,
            scientifically_unknown=scientifically_unknown,
            common_failure_stages=common_failure_stages,
            outcomes=outcomes,
        )
        try:
            llm_result = self._call_llm_structured_with_tools(
                schema=BatchSummary,
                task_type="batch_database",
                stage="batch_summary",
                payload={"outcomes": outcomes, "summary_hints": baseline.model_dump(mode="json")},
                system_prompt=(
                    "You are the Reporter Agent in an LLM-centered multi-agent scientific runtime. "
                    "Produce a structured batch summary that preserves counts and failure-stage aggregation. "
                    "Use the summary hints as context, but write the final structured batch summary yourself."
                ),
                user_prompt=(
                    "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                    "BATCH_CONTEXT_JSON:\n{payload}\n\n"
                    "Return a structured batch summary."
                ),
                explicit_skills=["reporting"],
                tool_names=self._tool_names_for_role(),
            )
            return BatchSummary.model_validate(llm_result)
        except Exception as exc:
            self._emit_reporter_fallback(stage="batch_summary", error=exc)
            return baseline
