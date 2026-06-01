from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .schemas import Proposal, RefinementDecision
from ..runtime.telemetry import emit_progress


class RefinementAgent(SkillAwareAgent):
    agent_name = "refinement"
    llm_role = "recovery"

    @staticmethod
    def _default_fit_threshold(state: dict[str, Any]) -> float:
        try:
            return float((state.get("services", {}) or {}).get("fit_r2_threshold") or 0.90)
        except Exception:
            return 0.90

    @staticmethod
    def _build_refinement_summary(state: dict[str, Any]) -> dict[str, Any]:
        diagnostics = dict(state.get("diagnostics", {}) or {})
        fit = dict(diagnostics.get("fit_diagnostics", {}) or {})
        per_direction_raw = dict(fit.get("per_direction", {}) or {})
        per_direction: dict[str, dict[str, float]] = {}
        for direction in ["x", "y"]:
            payload = dict(per_direction_raw.get(direction, {}) or {})
            quality = payload.get("effective_fit_quality", payload.get("edge_fit_r2", fit.get("fit_r2_min", 1.0)))
            per_direction[direction] = {
                "overall_fit_quality": float(quality or 1.0),
                "edge_fit_r2": float(payload.get("edge_fit_r2", 1.0) or 1.0),
                "energy_fit_r2": float(payload.get("energy_fit_r2", 1.0) or 1.0),
            }
        strain_summary = dict(diagnostics.get("strain_summary", {}) or {})
        anomaly_flags = [str(item) for item in list(fit.get("anomaly_flags", []) or [])]
        valley_switch = any("valley_switch" in item.lower() for item in anomaly_flags)
        return {
            "per_direction": per_direction,
            "failed_points": int(strain_summary.get("failed_points", 0) or 0),
            "current_refinement_rounds": int((state.get("workflow", {}) or {}).get("refinement_rounds", 0) or 0),
            "max_refinement_rounds": int((state.get("workflow", {}) or {}).get("max_refinement_rounds", 1) or 1),
            "fit_r2_threshold": RefinementAgent._default_fit_threshold(state),
            "valley_switch_detected": valley_switch,
        }

    @staticmethod
    def _default_suggested_points(state: dict[str, Any], channels: list[str]) -> dict[str, list[float]]:
        plan = dict((state.get("physics_results", {}) or {}).get("strain_plan_by_direction", {}) or {})
        base_candidates = [-0.015, -0.005, 0.005, 0.015]
        output: dict[str, list[float]] = {}
        for channel in channels:
            existing = {round(float(v), 6) for v in list(plan.get(channel, []) or [])}
            fresh = [float(v) for v in base_candidates if round(float(v), 6) not in existing]
            if fresh:
                output[channel] = fresh
        return output

    def decide(self, *, state: dict[str, Any], summary: dict[str, Any]) -> RefinementDecision:
        return self._rule_or_llm_decision(state=state, summary=summary)

    def _rule_decision(self, *, state: dict[str, Any], summary: dict[str, Any]) -> RefinementDecision:
        per_direction = dict(summary.get("per_direction", {}) or {})
        current_rounds = int(summary.get("current_refinement_rounds", 0) or 0)
        max_rounds = int(summary.get("max_refinement_rounds", 1) or 1)
        accepted = list(state.get("physics_results", {}).get("accepted_channels", ["x", "y"]))
        weak_dirs = [
            direction
            for direction, payload in per_direction.items()
            if float((payload or {}).get("overall_fit_quality", 1.0) or 0.0) < float(summary.get("fit_r2_threshold", 0.90))
        ]
        if bool(summary.get("valley_switch_detected", False)) or int(summary.get("failed_points", 0) or 0) > 0:
            reject_dirs = weak_dirs or [direction for direction in ["x", "y"] if direction in accepted]
            remaining = [direction for direction in accepted if direction not in reject_dirs]
            rule = (
                RefinementDecision(
                    decision="reject_channel",
                    reason="channel_specific_instability_detected",
                    confidence=0.85,
                    target_channels=reject_dirs,
                )
                if remaining
                else RefinementDecision(
                    decision="terminate",
                    reason="all_channels_rejected",
                    confidence=0.90,
                    target_channels=reject_dirs,
                )
            )
        elif weak_dirs and current_rounds < max_rounds:
            suggested = {}
            for direction in weak_dirs:
                points = list((state.get("physics_results", {}).get("strain_data_summary", {}) or {}).get(direction, []) or [])
                suggested[direction] = points
            rule = RefinementDecision(
                decision="refine_more_points",
                reason="fit_quality_below_threshold",
                confidence=0.78,
                target_channels=weak_dirs,
                suggested_points=suggested,
            )
        elif weak_dirs:
            rule = RefinementDecision(
                decision="terminate",
                reason="refinement_budget_exhausted",
                confidence=0.82,
                target_channels=weak_dirs,
            )
        else:
            rule = RefinementDecision(
                decision="accept",
                reason="strain_quality_acceptable",
                confidence=0.92,
                target_channels=accepted,
            )
        return rule

    def _rule_or_llm_decision(self, *, state: dict[str, Any], summary: dict[str, Any]) -> RefinementDecision:
        rule = self._rule_decision(state=state, summary=summary)
        per_direction = dict(summary.get("per_direction", {}) or {})
        weak_dirs = [
            direction
            for direction, payload in per_direction.items()
            if float((payload or {}).get("overall_fit_quality", 1.0) or 0.0) < float(summary.get("fit_r2_threshold", 0.90))
        ]

        llm = self._maybe_call_llm(
            kind="refinement",
            schema=RefinementDecision,
            task_type=str(state.get("task", {}).get("task_type") or "single_material"),
            stage="refinement",
            summary=summary,
            rule_payload=rule.model_dump(mode="json"),
            allowed_actions=["accept", "refine_more_points", "reject_channel", "terminate", "escalate"],
            has_error=bool(weak_dirs) or bool(summary.get("valley_switch_detected", False)) or int(summary.get("failed_points", 0) or 0) > 0,
            explicit_skills=["strain_refinement", "physics_validation"],
        )
        merged = {**rule.model_dump(mode="json"), **(llm or {})}
        return RefinementDecision.model_validate(merged)

    def propose(self, *, state: dict[str, Any], round_id: int) -> list[Proposal]:
        fit = dict((state.get("diagnostics", {}) or {}).get("fit_diagnostics", {}) or {})
        if not fit:
            return []
        summary = self._build_refinement_summary(state)
        try:
            decision = self._rule_or_llm_decision(state=state, summary=summary)
        except Exception as exc:
            emit_progress(
                "refinement decide failed; using bounded rule decision",
                channel="agent",
                details={"agent": self.agent_name, "round_id": round_id, "error": f"{type(exc).__name__}:{exc}"},
            )
            decision = self._rule_decision(state=state, summary=summary)

        task_id = str(state.get("task", {}).get("task_id") or "")
        cap_task_id = "capability::strain_loop"
        if decision.decision == "refine_more_points":
            channels = [str(c) for c in list(decision.target_channels or []) if str(c).strip()]
            if not channels:
                channels = ["x", "y"]
            suggested = {
                str(k): [float(v) for v in list(vals or [])]
                for k, vals in dict(decision.suggested_points or {}).items()
                if str(k).strip()
            }
            suggested = {
                key: [float(v) for v in values]
                for key, values in suggested.items()
                if [float(v) for v in values]
            }
            if not suggested:
                suggested = self._default_suggested_points(state, channels)
            if not suggested:
                return []
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": cap_task_id,
                        "proposal_id": f"{self.agent_name}::{round_id}::refine_sampling::strain_loop",
                        "action_family": "refine_sampling",
                        "target_capability": "strain_loop",
                        "selected_skill": "strain_refinement",
                        "parameters": {
                            "target_channels": channels,
                            "suggested_points": suggested,
                            "verify_non_redundancy": True,
                        },
                        "rationale": decision.reason or "refine_more_points_by_llm_decision",
                        "expected_observation": "strain_loop reruns with additional sampling points",
                        "success_criteria": [
                            "new strain points are injected into the plan",
                            "strain_loop recomputes only missing points",
                        ],
                        "fallback_if_failed": ["skip_channel", "escalate_human", "abort_material"],
                    }
                )
            ]

        if decision.decision == "reject_channel":
            channels = [str(c) for c in list(decision.target_channels or []) if str(c).strip()]
            if not channels:
                return []
            return [
                Proposal.model_validate(
                    {
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "proposal_id": f"{self.agent_name}::{round_id}::skip_channel",
                        "action_family": "skip_channel",
                        "parameters": {"target_channels": channels},
                        "rationale": decision.reason or "reject_unstable_channels",
                        "expected_observation": "unstable channels are removed from accepted_channels",
                        "success_criteria": ["accepted_channels updated", "rejected_channels updated"],
                        "fallback_if_failed": ["escalate_human", "abort_material"],
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
                        "proposal_id": f"{self.agent_name}::{round_id}::escalate_human",
                        "action_family": "escalate_human",
                        "rationale": decision.reason or "refinement_escalation",
                        "expected_observation": "human escalation payload generated",
                        "success_criteria": ["human escalation requested"],
                        "fallback_if_failed": ["abort_material"],
                    }
                )
            ]

        if decision.decision == "terminate":
            # Refinement-budget exhaustion is a scientific quality outcome, not a
            # runtime failure. Returning no proposal lets validation finalize the
            # material with labels instead of paging a human.
            return []

        return []
