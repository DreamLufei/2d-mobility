from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .context_engineering import build_llm_context_summary, select_role_context, summarize_proposals
from .schemas import Critique, Preference, Proposal, ReviewBundle


class CostGuardianAgent(SkillAwareAgent):
    agent_name = "cost_guardian"
    llm_role = "cost_guardian"

    def _review_hints(self, *, state: dict[str, Any], proposals: list[Proposal], round_id: int) -> dict[str, Any]:
        status = self.tool_gateway.call("query_execution_status", {"state": state})
        retries = dict(status.get("retry_counts", {}) or {})
        refinement_rounds = int(status.get("refinement_rounds", 0) or 0)
        max_refinements = int(status.get("max_refinement_rounds", 1) or 1)
        proposal_hints: list[dict[str, Any]] = []

        for proposal in proposals:
            target = str(proposal.target_capability or "")
            retry_count = int(retries.get(target, 0) or 0)
            metadata = self.tool_gateway.call(
                "query_capability_metadata",
                {"action_family": proposal.action_family, "capability": proposal.target_capability},
            )
            proposal_hints.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "action_family": proposal.action_family,
                    "target_capability": proposal.target_capability,
                    "retry_count": retry_count,
                    "cost_class": metadata.get("cost_class", proposal.content.get("cost_class", "medium")),
                    "risk_class": metadata.get("risk_class", proposal.content.get("risk_class", "medium")),
                    "budget_flags": (
                        ([f"already_retried:{target}:{retry_count}"] if retry_count >= 1 else [])
                        + (
                            ["refinement_budget_exhausted"]
                            if proposal.action_family == "refine_sampling" and refinement_rounds >= max_refinements
                            else []
                        )
                    ),
                }
            )
        low_cost = next((item for item in proposals if item.action_family in {"run_capability", "revalidate_result", "finalize_material"}), None)
        return {
            "round_id": round_id,
            "proposal_hints": proposal_hints,
            "refinement_budget": {
                "used": refinement_rounds,
                "max": max_refinements,
            },
            "preferred_low_cost_proposal_id": low_cost.proposal_id if low_cost is not None else None,
            "cost_policy": (
                "Use retry ceilings, refinement budgets, and cost classes as constraints and hints. "
                "They do not decide the action by themselves."
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
                "You are the Cost Guardian Agent in an LLM-centered multi-agent scientific runtime. "
                "Evaluate whether proposals justify their retry, refinement, or recompute cost. "
                "Use cost hints and retry ceilings as guardrails, not as an automatic answer."
            ),
            user_prompt=(
                "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                "STATE_PROPOSAL_AND_HINT_CONTEXT_JSON:\n{payload}\n\n"
                "Return structured critiques and preferences."
            ),
            explicit_skills=["recovery"],
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
