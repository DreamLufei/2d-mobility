from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .schemas import Proposal, ValidationDecision
from ..graph.escalation import should_escalate_validation
from ..runtime.telemetry import emit_progress


class ValidationAgent(SkillAwareAgent):
    agent_name = "validation"
    llm_role = "validation"

    def _rule_decision(self, *, state: dict[str, Any], summary: dict[str, Any]) -> ValidationDecision:
        warnings = list(summary.get("warnings", []) or [])
        warnings = [item for item in warnings if str(item or "").strip() not in {"dry_run_mode"}]
        anomaly_flags = list(summary.get("anomaly_flags", []) or [])
        retained_subchannels = list(summary.get("retained_subchannels", []) or [])
        rejected_subchannels = list(summary.get("rejected_subchannels", []) or [])
        recommended_action = str(summary.get("recommended_action") or "finalize")
        historical_heuristics = list(summary.get("historical_heuristics", []) or [])
        effective_fit = float(summary.get("effective_fit_quality", 1.0) or 0.0)
        for item in historical_heuristics:
            description = str((item or {}).get("description") or "").strip()
            if description:
                warnings.append(f"historical_validation:{description}")
        if not retained_subchannels and recommended_action != "refine_sampling":
            return ValidationDecision(
                decision="fail",
                reason="no_retained_subchannels_remaining",
                confidence=0.99,
                failed_checks=rejected_subchannels,
                warnings=warnings,
            )
        if anomaly_flags:
            return ValidationDecision(
                decision="fail",
                reason="anomaly_flags_present",
                confidence=0.65,
                failed_checks=anomaly_flags,
                warnings=warnings,
            )
        if recommended_action == "refine_sampling":
            return ValidationDecision(
                decision="pass_with_warning",
                reason="validation_recommends_refinement",
                confidence=0.74,
                warnings=warnings + [f"fit_r2:{effective_fit:.4f}"],
            )
        if effective_fit < float(summary.get("fit_r2_threshold", 0.90) or 0.90):
            return ValidationDecision(
                decision="pass_with_warning",
                reason="fit_quality_below_threshold",
                confidence=0.70,
                warnings=warnings + [f"fit_r2:{effective_fit:.4f}"],
            )
        if warnings or rejected_subchannels:
            return ValidationDecision(
                decision="pass_with_warning",
                reason="validation_warnings_present" if warnings else "partial_channel_rejection_recorded",
                confidence=0.82,
                warnings=warnings,
            )
        return ValidationDecision(decision="pass", reason="validation_passed", confidence=0.92)

    @staticmethod
    def _build_summary(state: dict[str, Any]) -> dict[str, Any]:
        diagnostics = dict(state.get("diagnostics", {}) or {})
        validation_report = dict(diagnostics.get("validation_report", {}) or {})
        fit_diagnostics = dict(diagnostics.get("fit_diagnostics", {}) or {})
        memory = dict(state.get("memory", {}) or {})
        fit_metrics = dict(validation_report.get("fit_metrics", {}) or {})
        effective_fit = fit_metrics.get("effective_fit_quality", fit_diagnostics.get("fit_r2_min", 1.0))
        return {
            "warnings": list(validation_report.get("warnings", []) or []),
            "anomaly_flags": list(validation_report.get("failed_checks", []) or []),
            "historical_heuristics": list(memory.get("validation_case_patterns", []) or []),
            "effective_fit_quality": float(effective_fit or 0.0),
            "fit_r2_threshold": float((state.get("services", {}) or {}).get("fit_r2_threshold") or 0.90),
            "validation_decision": str(validation_report.get("decision") or ""),
            "recommended_action": str(validation_report.get("recommended_action") or ""),
            "refinement_targets": list(validation_report.get("refinement_targets", []) or []),
            "refinement_preview": dict(validation_report.get("refinement_preview", {}) or {}),
            "retained_subchannels": list(validation_report.get("retained_subchannels", []) or []),
            "rejected_subchannels": list(validation_report.get("rejected_subchannels", []) or []),
        }

    def decide(self, *, state: dict[str, Any], summary: dict[str, Any]) -> ValidationDecision:
        rule = self._rule_decision(state=state, summary=summary)
        anomaly_flags = list(summary.get("anomaly_flags", []) or [])
        effective_fit = float(summary.get("effective_fit_quality", 1.0) or 0.0)
        llm = self._maybe_call_llm(
            kind="validation",
            schema=ValidationDecision,
            task_type=str(state.get("task", {}).get("task_type") or "single_material"),
            stage="validation",
            summary=summary,
            rule_payload=rule.model_dump(mode="json"),
            allowed_actions=["pass", "pass_with_warning", "fail", "escalate"],
            has_error=bool(anomaly_flags) or effective_fit < float(summary.get("fit_r2_threshold", 0.90) or 0.90),
            explicit_skills=["validation", "physics_validation"],
        )
        merged = {**rule.model_dump(mode="json"), **(llm or {})}
        if should_escalate_validation(merged):
            merged["decision"] = "escalate"
        return ValidationDecision.model_validate(merged)

    def propose(self, *, state: dict[str, Any], round_id: int) -> list[Proposal]:
        validation_report = dict((state.get("diagnostics", {}) or {}).get("validation_report", {}) or {})
        if not validation_report:
            return []
        summary = self._build_summary(state)
        try:
            decision = self.decide(state=state, summary=summary)
        except Exception as exc:
            emit_progress(
                "validation decide failed; using bounded rule decision",
                channel="agent",
                details={"agent": self.agent_name, "round_id": round_id, "error": f"{type(exc).__name__}:{exc}"},
            )
            decision = self._rule_decision(state=state, summary=summary)
        task_id = str(state.get("task", {}).get("task_id") or "")
        recommended_action = str(summary.get("recommended_action") or "finalize")
        refinement_targets = [str(item) for item in list(summary.get("refinement_targets", []) or []) if str(item).strip()]
        refinement_preview = dict(summary.get("refinement_preview", {}) or {})
        if recommended_action == "refine_sampling" and refinement_targets:
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": "capability::strain_loop",
                        "proposal_id": f"{self.agent_name}::{round_id}::refine_sampling::strain_loop",
                        "action_family": "refine_sampling",
                        "target_capability": "strain_loop",
                        "selected_skill": "strain_refinement",
                        "parameters": {
                            "target_channels": refinement_targets,
                            "suggested_points": dict(refinement_preview.get("suggested_points", {}) or {}),
                            "verify_non_redundancy": True,
                        },
                        "content": {"cost_class": "high", "risk_class": "medium"},
                        "rationale": decision.reason or "validation_recommends_refinement",
                        "expected_observation": "selected directions are resampled with fresh strain points before revalidation",
                        "success_criteria": ["new strain points are injected", "mobility is recomputed for the refined directions"],
                        "fallback_if_failed": ["revalidate_result", "finalize_material", "abort_material"],
                        "confidence": float(decision.confidence or 0.74),
                    }
                )
            ]
        if decision.decision in {"pass", "pass_with_warning"}:
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "proposal_id": f"{self.agent_name}::{round_id}::finalize_material",
                        "action_family": "finalize_material",
                        "selected_skill": "validation",
                        "content": {"cost_class": "low", "risk_class": "low"},
                        "rationale": decision.reason or "validation_finalize_material",
                        "expected_observation": "material outcome finalized after validation review",
                        "success_criteria": ["final report is generated", "material outcome is persisted"],
                        "fallback_if_failed": ["revalidate_result", "escalate_human"],
                        "confidence": float(decision.confidence or 0.8),
                    }
                )
            ]
        if decision.decision == "fail":
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "proposal_id": f"{self.agent_name}::{round_id}::finalize_material_failed_validation",
                        "action_family": "finalize_material",
                        "selected_skill": "validation",
                        "content": {"cost_class": "low", "risk_class": "low"},
                        "rationale": decision.reason or "validation_failed_finalize_material",
                        "expected_observation": "material outcome finalized as not retained after validation review",
                        "success_criteria": ["validation failure is persisted", "final report records retained/rejected channels"],
                        "fallback_if_failed": ["abort_material"],
                        "confidence": float(decision.confidence or 0.8),
                    }
                )
            ]
        if decision.decision == "escalate":
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "proposal_id": f"{self.agent_name}::{round_id}::revalidate_result",
                        "action_family": "revalidate_result",
                        "target_capability": "validation",
                        "selected_skill": "validation",
                        "rationale": decision.reason or "validation_recheck_requested",
                        "expected_observation": "validation reruns with the latest accepted channels and metrics",
                        "success_criteria": ["validation report refreshed"],
                        "fallback_if_failed": ["escalate_human", "abort_material"],
                        "confidence": float(decision.confidence or 0.7),
                    }
                )
            ]
        return []
