from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .context_engineering import build_llm_context_summary, select_role_context, summarize_proposals
from .schemas import Critique, Preference, Proposal, ReviewBundle


class CriticAgent(SkillAwareAgent):
    agent_name = "critic"
    llm_role = "critic"

    def _review_hints(self, *, state: dict[str, Any], proposals: list[Proposal], round_id: int) -> dict[str, Any]:
        proposal_hints: list[dict[str, Any]] = []
        for proposal in proposals:
            legality = self.tool_gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": proposal.action_family,
                    "target_capability": proposal.target_capability,
                    "parameters": dict(proposal.parameters or {}),
                },
            )
            obvious_concerns: list[str] = []
            if proposal.action_family == "rerun_from_capability":
                obvious_concerns.append("recompute_cost_high")
            if proposal.action_family == "abort_material":
                obvious_concerns.append("termination_is_irreversible")
            proposal_hints.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "action_family": proposal.action_family,
                    "target_capability": proposal.target_capability,
                    "allowed": bool(legality.get("allowed", False)),
                    "refusal_reasons": list(legality.get("refusal_reasons", []) or []),
                    "fallback_action": legality.get("fallback_action"),
                    "obvious_concerns": obvious_concerns,
                }
            )
        conservative_choice = None
        if proposals:
            conservative_choice = min(proposals, key=lambda item: float(item.risk_estimate) + float(item.cost_estimate)).proposal_id
        return {
            "round_id": round_id,
            "proposal_hints": proposal_hints,
            "conservative_preference_hint": conservative_choice,
            "critic_policy": (
                "Use legality and obvious inconsistency notes as guardrails. "
                "They are not the full critique. You must generate the actual critique and preference judgments."
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
                "You are the Critic Agent in an LLM-centered multi-agent scientific runtime. "
                "Challenge weak, under-evidenced, illegal, or risky proposals. "
                "Use the review hints only as guardrails and conservative reminders; the actual critique must be your own judgment."
            ),
            user_prompt=(
                "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                "STATE_PROPOSAL_AND_HINT_CONTEXT_JSON:\n{payload}\n\n"
                "Return structured critiques and preferences."
            ),
            explicit_skills=["physics_validation"],
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
