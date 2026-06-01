from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .context_engineering import build_llm_context_summary, select_role_context, summarize_proposals
from .schemas import Critique, Preference, Proposal, ReviewBundle


class PhysicsJudgeAgent(SkillAwareAgent):
    agent_name = "physics_judge"
    llm_role = "physics_judge"

    def _review_hints(self, *, state: dict[str, Any], proposals: list[Proposal], round_id: int) -> dict[str, Any]:
        observation = self.tool_gateway.call("synthesize_observation", {"state": state})
        anomaly_flags = list(observation.get("anomaly_flags", []) or [])
        accepted_channels = list(observation.get("accepted_channels", []) or [])
        fit_quality = float(observation.get("fit_quality", 1.0) or 1.0)
        proposal_hints: list[dict[str, Any]] = []
        for proposal in proposals:
            proposal_hints.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "action_family": proposal.action_family,
                    "target_capability": proposal.target_capability,
                    "finalize_risk": proposal.action_family == "finalize_material"
                    and bool(anomaly_flags or fit_quality < 0.9 or not accepted_channels),
                    "refinement_opportunity": proposal.action_family in {"refine_sampling", "revalidate_result", "rerun_from_capability"},
                }
            )
        return {
            "round_id": round_id,
            "physics_hints": {
                "anomaly_flags": dedupe(anomaly_flags),
                "accepted_channels": accepted_channels,
                "fit_quality": fit_quality,
                "has_accepted_channels": bool(accepted_channels),
                "physics_warning_tags": dedupe(
                    anomaly_flags
                    + ([] if accepted_channels else ["no_accepted_channels"])
                    + ([f"fit_quality:{fit_quality:.3f}"] if fit_quality < 0.9 else [])
                ),
            },
            "proposal_hints": proposal_hints,
            "physics_policy": (
                "Use these physics hints as guardrails and anomaly reminders. "
                "You must produce the actual physics judgment yourself."
            ),
        }

    def review(self, *, state: dict[str, Any], proposals: list[Proposal], round_id: int) -> tuple[list[Critique], list[Preference]]:
        if not proposals:
            return [], []
        context_summary = select_role_context(
            dict((state.get("services", {}) or {}).get("llm_context_summary", {}) or {})
            or build_llm_context_summary(state),
            role=self.llm_role,
        )
        payload = {
            "context_summary": context_summary,
            "round_id": round_id,
            "proposals": summarize_proposals(proposals),
            "review_hints": self._review_hints(state=state, proposals=proposals, round_id=round_id),
        }
        llm_result = self._call_llm_structured_with_tools(
            schema=ReviewBundle,
            task_type=str(state.get("task", {}).get("task_type") or "single_material"),
            stage=str(state.get("workflow", {}).get("current_stage") or "critique_phase"),
            payload=payload,
            system_prompt=(
                "You are the Physics Judge Agent in an LLM-centered multi-agent scientific runtime. "
                "Evaluate proposals for physics credibility, fit stability, channel validity, and mobility consistency. "
                "Use the supplied physics hints as guardrails, not as a precomputed decision."
            ),
            user_prompt=(
                "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                "STATE_PROPOSAL_AND_HINT_CONTEXT_JSON:\n{payload}\n\n"
                "Return structured critiques and preferences."
            ),
            explicit_skills=["physics_validation", "strain_refinement"],
            tool_names=self._tool_names_for_role(),
        )
        parsed = ReviewBundle.model_validate(llm_result)
        task_id = str(state.get("task", {}).get("task_id") or "")
        proposal_map = {item.proposal_id: item for item in proposals}
        fallback_proposal_id = proposals[0].proposal_id if len(proposals) == 1 else ""

        critiques: list[Critique] = []
        for item in parsed.critiques:
            proposal_id = item.proposal_id if item.proposal_id in proposal_map else fallback_proposal_id
            target_task_id = (
                f"capability::{proposal_map[proposal_id].target_capability}"
                if proposal_id and proposal_id in proposal_map and proposal_map[proposal_id].target_capability
                else task_id
            )
            critiques.append(
                Critique.model_validate(
                    {
                        **item.model_dump(mode="json"),
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": target_task_id,
                        "proposal_id": proposal_id,
                    }
                )
            )

        preferences: list[Preference] = []
        for item in parsed.preferences:
            preferred_proposal_id = item.preferred_proposal_id if item.preferred_proposal_id in proposal_map else fallback_proposal_id
            target_task_id = (
                f"capability::{proposal_map[preferred_proposal_id].target_capability}"
                if preferred_proposal_id and preferred_proposal_id in proposal_map and proposal_map[preferred_proposal_id].target_capability
                else task_id
            )
            preferences.append(
                Preference.model_validate(
                    {
                        **item.model_dump(mode="json"),
                        "agent_name": self.agent_name,
                        "round_id": round_id,
                        "target_task_id": target_task_id,
                        "preferred_proposal_id": preferred_proposal_id,
                    }
                )
            )
        return critiques, preferences


def dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.append(value)
    return seen
