from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import replace
from json import JSONDecoder
from typing import Any, Iterator
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.tool import tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from mobility_agent.config_runtime import AgentRuntimeConfig


TEST_LLM_ENV = {
    "MOBILITY_DB_URI": "memory://llm-test-runtime",
    "EMBEDDING_MODEL": "test-embedding-model",
    "LLM_PROVIDER": "openai",
    "LLM_BASE_URL": "http://127.0.0.1:9/v1",
    "LLM_API_KEY": "test-key",
    "LLM_MODEL": "test-model",
    "PLANNER_MODE": "",
    "LLM_ENABLED": "",
}

_AGENT_TOOL_NAMES = {
    "inspect_workspace",
    "inspect_artifacts",
    "retrieve_policy_evidence",
    "retrieve_failure_evidence",
    "query_capability_metadata",
    "query_execution_status",
    "synthesize_observation",
    "check_action_legality",
    "query_memory_hits",
    "write_memory_reflection",
    "inspect_hitl_state",
    "write_runtime_artifacts",
    "summarize_batch_outcomes",
}


def build_test_agent_runtime(**overrides: Any) -> AgentRuntimeConfig:
    with patch.dict(os.environ, TEST_LLM_ENV, clear=False):
        base = AgentRuntimeConfig.from_env()
    return replace(base, **overrides)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def _extract_json_after_marker(text: str, marker: str) -> dict[str, Any]:
    if marker not in text:
        return {}
    candidate = text.split(marker, 1)[1].strip()
    start_positions = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0]
    if not start_positions:
        return {}
    start = min(start_positions)
    decoder = JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(candidate[start:])
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _extract_payload(messages: list[BaseMessage]) -> dict[str, Any]:
    markers = (
        "DELIBERATION_CONTEXT_JSON:",
        "DELIBERATION_AND_GUARDRAIL_CONTEXT_JSON:",
        "STATE_AND_PROPOSAL_CONTEXT_JSON:",
        "STATE_PROPOSAL_AND_HINT_CONTEXT_JSON:",
        "STATE_AND_BASELINE_CONTEXT_JSON:",
        "STATE_AND_HINT_CONTEXT_JSON:",
        "STATE_AND_RULE_CONTEXT_JSON:",
        "FINAL_CONTEXT_JSON:",
        "BATCH_CONTEXT_JSON:",
        "RULE_PAYLOAD_JSON:",
        "VISIBLE_STATE_JSON:",
    )
    for message in reversed(messages):
        content = _message_text(message)
        for marker in markers:
            payload = _extract_json_after_marker(content, marker)
            if payload:
                return payload
    return {}


class FakeRoleAwareChatModel(BaseChatModel):
    role: str = "specialist"

    @property
    def _llm_type(self) -> str:
        return "fake-role-aware-chat-model"

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=openai_tools, tool_choice=tool_choice, **kwargs)

    def _planner_proposals(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        state = dict(payload.get("state") or {})
        hints = dict(payload.get("planning_hints") or {})
        task_id = str((state.get("task", {}) or {}).get("task_id") or "")
        round_id = int(payload.get("round_id", 0) or 0)
        allowed = set(payload.get("allowed_actions") or [])
        proposals: list[dict[str, Any]] = []
        if str(hints.get("run_status") or "") in {"waiting_external", "needs_human"}:
            return proposals
        if bool(hints.get("ready_to_finalize")) and "finalize_material" in allowed:
            proposals.append(
                {
                    "agent_name": "planner",
                    "round_id": round_id,
                    "target_task_id": task_id,
                    "proposal_id": f"planner::finalize::{round_id}",
                    "action_family": "finalize_material",
                    "selected_skill": "reporting",
                    "content": {"cost_class": "low", "risk_class": "low"},
                    "rationale": "llm_finalize_after_ready_to_finalize",
                    "expected_benefit": "finalize material outcome",
                    "expected_risk": "minimal",
                    "expected_observation": "final outcome written",
                    "success_criteria": ["final summary written", "material outcome persisted"],
                    "fallback_if_failed": ["revalidate_result"],
                    "confidence": 0.95,
                }
            )
            return proposals
        default_next = dict(hints.get("default_next_capability") or {})
        capability = str(default_next.get("capability") or "")
        if capability and "run_capability" in allowed:
            proposals.append(
                {
                    "agent_name": "planner",
                    "round_id": round_id,
                    "target_task_id": task_id,
                    "proposal_id": f"planner::run::{round_id}::{capability}",
                    "action_family": "run_capability",
                    "target_capability": capability,
                    "selected_skill": str(default_next.get("selected_skill") or "single_material_mobility"),
                    "content": {
                        "cost_class": default_next.get("cost_class", "medium"),
                        "risk_class": default_next.get("risk_class", "medium"),
                        "dependencies": list(default_next.get("dependencies", []) or []),
                        "expected_artifacts": list(default_next.get("expected_artifacts", []) or []),
                    },
                    "rationale": f"llm_follow_default_mainline_with:{capability}",
                    "expected_benefit": "advance workflow closure",
                    "expected_risk": "standard stage execution risk",
                    "expected_observation": f"stage observation for {capability}",
                    "success_criteria": [f"{capability} completes successfully"],
                    "fallback_if_failed": ["retry_capability", "rerun_from_capability", "escalate_human"],
                    "confidence": 0.84,
                }
            )
        return proposals

    def _recovery_proposals(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        state = dict(payload.get("state") or {})
        hints = dict(payload.get("recovery_hints") or {})
        failure_context = dict(hints.get("failure_context") or {})
        if str(failure_context.get("status") or "") != "failed":
            return []
        constraints = dict(hints.get("recovery_constraints") or {})
        retry_ctx = dict(hints.get("retry_context") or {})
        task_id = str((state.get("task", {}) or {}).get("task_id") or "")
        round_id = int(payload.get("round_id", 0) or 0)
        current_stage = str(constraints.get("current_stage") or failure_context.get("stage") or "")
        previous_stage = str(constraints.get("previous_stage") or current_stage)
        retry_budget_remaining = int(retry_ctx.get("retry_budget_remaining", 0) or 0)
        proposals = [
            {
                "agent_name": "recovery",
                "round_id": round_id,
                "target_task_id": task_id,
                "proposal_id": f"recovery::retry::{round_id}::{current_stage}",
                "action_family": "retry_capability",
                "target_capability": current_stage,
                "content": {
                    "cost_class": constraints.get("retry_cost_class", "medium"),
                    "risk_class": constraints.get("retry_risk_class", "medium"),
                    "required_evidence": list(constraints.get("retry_required_evidence", []) or []),
                },
                "rationale": f"llm_recovery_retry:{current_stage}",
                "expected_benefit": "fastest recovery path",
                "expected_risk": "repeat failure possible",
                "confidence": 0.66 if retry_budget_remaining > 0 else 0.3,
            },
            {
                "agent_name": "recovery",
                "round_id": round_id,
                "target_task_id": task_id,
                "proposal_id": f"recovery::rerun::{round_id}::{previous_stage}",
                "action_family": "rerun_from_capability",
                "target_capability": previous_stage,
                "content": {
                    "cost_class": constraints.get("rerun_cost_class", "high"),
                    "risk_class": constraints.get("rerun_risk_class", "medium"),
                    "required_evidence": list(constraints.get("rerun_required_evidence", []) or []),
                },
                "rationale": f"llm_recovery_rebuild:{previous_stage}",
                "expected_benefit": "repair broken dependency chain",
                "expected_risk": "higher recompute cost",
                "confidence": 0.64,
            },
            {
                "agent_name": "recovery",
                "round_id": round_id,
                "target_task_id": task_id,
                "proposal_id": f"recovery::human::{round_id}::{current_stage}",
                "action_family": "escalate_human",
                "target_capability": current_stage,
                "parameters": {"recommended_options": ["manual_fix_resume", "retry_current_stage", "skip_material", "abort_task"]},
                "content": {"cost_class": "low", "risk_class": "low"},
                "rationale": f"llm_recovery_human:{current_stage}",
                "expected_benefit": "unlock blocked execution path",
                "expected_risk": "requires manual attention",
                "confidence": 0.88,
            },
        ]
        if retry_budget_remaining <= 0:
            proposals.append(
                {
                    "agent_name": "recovery",
                    "round_id": round_id,
                    "target_task_id": task_id,
                    "proposal_id": f"recovery::abort::{round_id}::{current_stage}",
                    "action_family": "abort_material",
                    "target_capability": current_stage,
                    "content": {"cost_class": "low", "risk_class": "high"},
                    "rationale": "llm_retry_budget_exhausted",
                    "expected_benefit": "cap further wasted cost",
                    "expected_risk": "task termination",
                    "confidence": 0.9,
                }
            )
        return proposals

    def _review_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposals = list(payload.get("proposals") or [])
        hints = dict(payload.get("review_hints") or {})
        state = dict(payload.get("state") or {})
        task_id = str((state.get("task", {}) or {}).get("task_id") or "")
        round_id = int(payload.get("round_id", 0) or 0)
        critiques: list[dict[str, Any]] = []
        preferences: list[dict[str, Any]] = []
        if self.role == "critic":
            hint_index = {str(item.get("proposal_id") or ""): dict(item) for item in list(hints.get("proposal_hints", []) or [])}
            for proposal in proposals:
                proposal_id = str(proposal.get("proposal_id") or "")
                item = hint_index.get(proposal_id, {})
                concerns = list(item.get("refusal_reasons", []) or []) + list(item.get("obvious_concerns", []) or [])
                if item and not bool(item.get("allowed", True)):
                    critiques.append(
                        {
                            "agent_name": "critic",
                            "message_type": "objection",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "proposal_id": proposal_id,
                            "content": {"legality": item},
                            "concerns": concerns,
                            "recommendation": f"prefer_{item.get('fallback_action') or 'safer_option'}",
                            "confidence": 0.92,
                            "risk_estimate": 0.9,
                        }
                    )
                else:
                    critiques.append(
                        {
                            "agent_name": "critic",
                            "message_type": "critique" if concerns else "support",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "proposal_id": proposal_id,
                            "content": {"proposal_action": proposal.get("action_family")},
                            "concerns": concerns,
                            "recommendation": "acceptable" if not concerns else "only_if_better_options_fail",
                            "confidence": 0.72 if concerns else 0.8,
                            "risk_estimate": 0.4 if concerns else 0.2,
                        }
                    )
            preferred = str(hints.get("conservative_preference_hint") or "")
            if preferred:
                preferences.append(
                    {
                        "agent_name": "critic",
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "preferred_proposal_id": preferred,
                        "reason": "prefer_conservative_option",
                        "preference_strength": 0.62,
                        "confidence": 0.75,
                    }
                )
        elif self.role == "physics_judge":
            physics = dict(hints.get("physics_hints") or {})
            anomaly_flags = list(physics.get("anomaly_flags", []) or [])
            accepted_channels = list(physics.get("accepted_channels", []) or [])
            fit_quality = float(physics.get("fit_quality", 1.0) or 1.0)
            for proposal in proposals:
                proposal_id = str(proposal.get("proposal_id") or "")
                action_family = str(proposal.get("action_family") or "")
                if action_family == "finalize_material" and (anomaly_flags or fit_quality < 0.9 or not accepted_channels):
                    critiques.append(
                        {
                            "agent_name": "physics_judge",
                            "message_type": "objection",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "proposal_id": proposal_id,
                            "concerns": list(physics.get("physics_warning_tags", []) or []),
                            "recommendation": "prefer_revalidate_or_recompute",
                            "confidence": 0.91,
                            "risk_estimate": 0.8,
                        }
                    )
                elif action_family in {"refine_sampling", "revalidate_result"}:
                    critiques.append(
                        {
                            "agent_name": "physics_judge",
                            "message_type": "support",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "proposal_id": proposal_id,
                            "concerns": [],
                            "recommendation": "physics_consistency_improvement",
                            "confidence": 0.82,
                            "risk_estimate": 0.2,
                        }
                    )
            if proposals and (anomaly_flags or fit_quality < 0.9):
                recheck = next(
                    (item for item in proposals if str(item.get("action_family") or "") in {"refine_sampling", "revalidate_result", "rerun_from_capability"}),
                    None,
                )
                if recheck is not None:
                    preferences.append(
                        {
                            "agent_name": "physics_judge",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "preferred_proposal_id": str(recheck.get("proposal_id") or ""),
                            "reason": "prefer_physics_confidence_improvement",
                            "preference_strength": 0.85,
                            "confidence": 0.86,
                        }
                    )
        elif self.role == "cost_guardian":
            hint_index = {str(item.get("proposal_id") or ""): dict(item) for item in list(hints.get("proposal_hints", []) or [])}
            for proposal in proposals:
                proposal_id = str(proposal.get("proposal_id") or "")
                item = hint_index.get(proposal_id, {})
                action_family = str(proposal.get("action_family") or "")
                if action_family in {"rerun_from_capability", "refine_sampling"}:
                    concerns = list(item.get("budget_flags", []) or [])
                    if not concerns:
                        concerns = [f"high_cost_action:{item.get('cost_class', 'high')}"]
                    critiques.append(
                        {
                            "agent_name": "cost_guardian",
                            "message_type": "objection" if list(item.get("budget_flags", []) or []) else "critique",
                            "round_id": round_id,
                            "target_task_id": task_id,
                            "proposal_id": proposal_id,
                            "concerns": concerns,
                            "recommendation": "choose_lower_cost_option_if_available",
                            "confidence": 0.8,
                            "cost_estimate": 0.8 if item.get("cost_class") == "high" else 0.6,
                            "risk_estimate": 0.5,
                        }
                    )
            preferred = str(hints.get("preferred_low_cost_proposal_id") or "")
            if preferred:
                preferences.append(
                    {
                        "agent_name": "cost_guardian",
                        "round_id": round_id,
                        "target_task_id": task_id,
                        "preferred_proposal_id": preferred,
                        "reason": "prefer_lower_cost_option",
                        "preference_strength": 0.7,
                        "confidence": 0.8,
                    }
                )
        return {"critiques": critiques, "preferences": preferences}

    def _arbitration_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        proposals = list(payload.get("proposals") or [])
        state = dict(payload.get("state") or {})
        guardrail = dict(payload.get("guardrail_context") or {})
        guardrail_content = dict(guardrail.get("content") or {})
        legal_ids = list(guardrail_content.get("legal_proposal_ids", []) or [])
        preferred = str(guardrail_content.get("guardrail_preferred_proposal_id") or "")
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        failure_status = str(latest_observation.get("status") or "")
        if failure_status == "failed":
            human_option = next(
                (
                    str(item.get("proposal_id") or "")
                    for item in proposals
                    if str(item.get("proposal_id") or "") in legal_ids and str(item.get("action_family") or "") == "escalate_human"
                ),
                "",
            )
            selected_proposal_id = human_option or (preferred if preferred in legal_ids else (legal_ids[0] if legal_ids else None))
        else:
            selected_proposal_id = preferred if preferred in legal_ids else (legal_ids[0] if legal_ids else None)
        selected = next((item for item in proposals if str(item.get("proposal_id") or "") == selected_proposal_id), None)
        if selected is None:
            return {
                "selected_proposal_id": None,
                "selected_action": None,
                "rejected_proposal_ids": [str(item.get("proposal_id") or "") for item in proposals],
                "guardrail_notes": list(guardrail.get("guardrail_notes", []) or []),
                "rationale": "no_legal_proposal_available_for_llm_selection",
                "disagreement_summary": [],
                "whether_noop": True,
                "whether_waiting_external": bool(guardrail.get("whether_waiting_external", False)),
                "whether_ready_to_finalize": bool(guardrail.get("whether_ready_to_finalize", False)),
            }
        return {
            "selected_proposal_id": selected_proposal_id,
            "selected_action": {
                "action_family": selected.get("action_family"),
                "target_capability": selected.get("target_capability"),
                "selected_skill": selected.get("selected_skill"),
                "parameters": dict(selected.get("parameters", {}) or {}),
                "source_proposal_id": selected_proposal_id,
                "rationale": str(selected.get("rationale") or "llm_selected_proposal"),
                "cost_class": str((selected.get("content") or {}).get("cost_class") or "medium"),
                "risk_class": str((selected.get("content") or {}).get("risk_class") or "medium"),
                "expected_observation": str(selected.get("expected_observation") or ""),
                "success_criteria": list(selected.get("success_criteria", []) or []),
                "fallback_if_failed": list(selected.get("fallback_if_failed", []) or []),
                "submit_external_job": bool(selected.get("submit_external_job", False)),
                "wait_for_event_after_submission": bool(selected.get("wait_for_event_after_submission", False)),
            },
            "rejected_proposal_ids": [str(item.get("proposal_id") or "") for item in proposals if str(item.get("proposal_id") or "") != selected_proposal_id],
            "guardrail_notes": list(guardrail.get("guardrail_notes", []) or []),
            "rationale": f"llm_selected_legal_proposal:{selected_proposal_id}",
            "disagreement_summary": [],
            "whether_noop": False,
            "whether_waiting_external": False,
            "whether_ready_to_finalize": False,
        }

    def _structured_args(self, messages: list[BaseMessage], tool_name: str) -> dict[str, Any]:
        payload = _extract_payload(messages)
        if tool_name.endswith("ProposalBundle"):
            if payload.get("planning_hints") is not None:
                return {"proposals": self._planner_proposals(payload)}
            if payload.get("recovery_hints") is not None:
                return {"proposals": self._recovery_proposals(payload)}
            return {"proposals": list(payload.get("baseline_proposals") or payload.get("rule_proposals") or [])}
        if tool_name.endswith("ReviewBundle"):
            if payload.get("review_hints") is not None:
                return self._review_bundle(payload)
            return {
                "critiques": list(payload.get("baseline_critiques") or payload.get("rule_critiques") or []),
                "preferences": list(payload.get("baseline_preferences") or payload.get("rule_preferences") or []),
            }
        if tool_name.endswith("ArbitrationDecisionPayload"):
            if payload.get("guardrail_context") is not None:
                return self._arbitration_payload(payload)
            baseline = dict(payload.get("baseline_arbitration") or payload.get("fallback_arbitration") or {})
            return {
                "selected_proposal_id": baseline.get("selected_proposal_id"),
                "selected_action": baseline.get("selected_action"),
                "rejected_proposal_ids": list(baseline.get("rejected_proposals", []) or []),
                "guardrail_notes": list(baseline.get("guardrail_notes", []) or []),
                "rationale": baseline.get("rationale", f"{self.role}_baseline_arbitration"),
                "disagreement_summary": list(baseline.get("disagreement_summary", []) or []),
                "whether_noop": bool(baseline.get("whether_noop", False)),
                "whether_waiting_external": bool(baseline.get("whether_waiting_external", False)),
                "whether_ready_to_finalize": bool(baseline.get("whether_ready_to_finalize", False)),
            }
        if tool_name.endswith("ReportSummary"):
            baseline = dict(payload.get("summary_hints") or payload.get("baseline_summary") or payload.get("fallback_summary") or {})
            return {
                "reason": "llm_report_generated",
                "confidence": 0.92,
                "final_summary": baseline,
                "key_findings": [
                    f"run_status:{baseline.get('run_status', 'unknown')}",
                    f"termination_reason:{baseline.get('termination_reason', 'unknown')}",
                ],
                "artifact_references": dict(baseline.get("artifact_paths", {}) or {}),
            }
        if tool_name.endswith("BatchSummary"):
            baseline = dict(payload.get("summary_hints") or payload.get("baseline_summary") or {})
            return baseline
        return payload

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        tools = list(kwargs.get("tools", []) or [])
        if not tools:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", response_metadata={"provider": "test"}))])

        tool_schema = dict(tools[0] or {})
        function = dict(tool_schema.get("function", {}) or {})
        tool_name = str(function.get("name") or "StructuredResponse")

        if tool_name in _AGENT_TOOL_NAMES:
            message = AIMessage(content="", response_metadata={"provider": "test", "role": self.role})
            return ChatResult(generations=[ChatGeneration(message=message)])

        arguments = self._structured_args(messages, tool_name)
        raw_tool_call = {
            "id": f"fake-{self.role}-tool-call",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        message = AIMessage(
            content="",
            tool_calls=[tool_call(name=tool_name, args=arguments, id=raw_tool_call["id"])],
            additional_kwargs={"tool_calls": [raw_tool_call]},
            response_metadata={"provider": "test", "role": self.role},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@contextmanager
def patch_test_llm_clients() -> Iterator[None]:
    def _build(_runtime: AgentRuntimeConfig, *, role: str | None = None, require_real: bool = False):
        del _runtime, require_real
        return FakeRoleAwareChatModel(role=str(role or "specialist")), None

    with (
        patch.dict(os.environ, TEST_LLM_ENV, clear=False),
        patch("mobility_agent.agents.base.build_llm_client", side_effect=_build),
        patch("mobility_agent.policy.engine.build_llm_client", side_effect=_build),
    ):
        yield
