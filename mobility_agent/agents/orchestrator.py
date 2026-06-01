from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .context_engineering import (
    build_llm_context_summary,
    select_role_context,
    summarize_critiques,
    summarize_guardrail_context,
    summarize_preferences,
    summarize_proposals,
)
from .schemas import ArbitrationDecisionPayload, ArbitrationRecord, Critique, Preference, Proposal, SelectedAction
from ..runtime.deliberation_loop import all_tasks_resolved, has_blocked_or_abandoned_tasks


class OrchestratorAgent(SkillAwareAgent):
    agent_name = "orchestrator"
    llm_role = "orchestrator"
    _LOCAL_EXECUTION_ACTION_FAMILIES = frozenset(
        {
            "run_capability",
            "retry_capability",
            "rerun_from_capability",
            "refine_sampling",
            "revalidate_result",
            "finalize_material",
            "abort_material",
            "skip_channel",
            "invalidate_channel",
        }
    )

    @staticmethod
    def _canonicalize_action_parameters(action_family: str, parameters: dict[str, Any] | None) -> tuple[dict[str, Any], bool]:
        params = dict(parameters or {})
        normalized = dict(params)
        family = str(action_family or "")

        if family not in {"invalidate_channel", "skip_channel"}:
            return normalized, normalized != params

        alias_keys = (
            ["channels_to_invalidate", "channels", "target_directions", "directions"]
            if family == "invalidate_channel"
            else ["channels_to_skip", "channels", "target_directions", "directions"]
        )

        raw = normalized.get("target_channels")
        if raw is None:
            for key in alias_keys:
                if key in normalized and normalized.get(key) is not None:
                    raw = normalized.get(key)
                    break
        if raw is None:
            single = normalized.get("channel") or normalized.get("target_channel")
            raw = [single] if single else []
        if isinstance(raw, str):
            raw = [raw]
        target_channels = [str(item).strip() for item in list(raw or []) if str(item).strip()]
        if target_channels:
            normalized["target_channels"] = target_channels
        for key in ("channels_to_invalidate", "channels_to_skip", "channels", "target_directions", "directions"):
            if key in normalized:
                normalized.pop(key, None)

        return normalized, normalized != params

    @classmethod
    def _canonicalize_proposal_execution_semantics(cls, proposal: Proposal) -> tuple[Proposal, list[str]]:
        if str(proposal.action_family or "") not in cls._LOCAL_EXECUTION_ACTION_FAMILIES:
            return proposal, []
        update: dict[str, Any] = {}
        notes: list[str] = []
        if bool(proposal.submit_external_job):
            update["submit_external_job"] = False
            notes.append("canonicalized_proposal:submit_external_job")
        if bool(proposal.wait_for_event_after_submission):
            update["wait_for_event_after_submission"] = False
            notes.append("canonicalized_proposal:wait_for_event_after_submission")
        normalized_parameters, changed = cls._canonicalize_action_parameters(
            str(proposal.action_family or ""),
            dict(proposal.parameters or {}),
        )
        if changed:
            update["parameters"] = normalized_parameters
            notes.append("canonicalized_proposal:parameters")
        if not update:
            return proposal, []
        return proposal.model_copy(update=update), notes

    @staticmethod
    def _selected_action_from_proposal(proposal: Proposal, *, supporting_opinions: list[str]) -> SelectedAction:
        normalized_parameters, _ = OrchestratorAgent._canonicalize_action_parameters(
            str(proposal.action_family or ""),
            dict(proposal.parameters or {}),
        )
        return SelectedAction(
            action_family=proposal.action_family,
            target_capability=proposal.target_capability,
            selected_skill=proposal.selected_skill,
            parameters=normalized_parameters,
            source_proposal_id=proposal.proposal_id,
            rationale=proposal.rationale,
            cost_class=str(proposal.content.get("cost_class") or "medium"),
            risk_class=str(proposal.content.get("risk_class") or "medium"),
            expected_observation=proposal.expected_observation,
            success_criteria=list(proposal.success_criteria),
            fallback_if_failed=list(proposal.fallback_if_failed),
            submit_external_job=bool(proposal.submit_external_job),
            wait_for_event_after_submission=bool(proposal.wait_for_event_after_submission),
            supporting_agent_opinions=supporting_opinions,
        )

    def _canonicalize_selected_action(
        self,
        *,
        proposals: list[Proposal],
        selected_proposal_id: str,
        parsed_action: SelectedAction | None,
        supporting_opinions: list[str],
    ) -> tuple[SelectedAction, list[str]]:
        source = next(item for item in proposals if item.proposal_id == selected_proposal_id)
        canonical = self._selected_action_from_proposal(source, supporting_opinions=supporting_opinions)
        if parsed_action is None:
            return canonical, []

        normalized_parsed_parameters, parameters_changed = self._canonicalize_action_parameters(
            str(canonical.action_family or ""),
            dict(parsed_action.parameters or {}),
        )
        if parameters_changed:
            parsed_action = parsed_action.model_copy(update={"parameters": normalized_parsed_parameters})

        notes: list[str] = []
        if str(parsed_action.source_proposal_id or "").strip() not in {"", selected_proposal_id}:
            notes.append("canonicalized_selected_action:source_proposal_id")
        if str(parsed_action.action_family or "") != str(canonical.action_family or ""):
            notes.append("canonicalized_selected_action:action_family")
        if str(parsed_action.target_capability or "") != str(canonical.target_capability or ""):
            notes.append("canonicalized_selected_action:target_capability")
        if str(parsed_action.selected_skill or "") != str(canonical.selected_skill or ""):
            notes.append("canonicalized_selected_action:selected_skill")
        if dict(parsed_action.parameters or {}) != dict(canonical.parameters or {}):
            notes.append("canonicalized_selected_action:parameters")
        if bool(parsed_action.submit_external_job) != bool(canonical.submit_external_job):
            notes.append("canonicalized_selected_action:submit_external_job")
        if bool(parsed_action.wait_for_event_after_submission) != bool(canonical.wait_for_event_after_submission):
            notes.append("canonicalized_selected_action:wait_for_event_after_submission")

        overlay: dict[str, Any] = {"supporting_agent_opinions": supporting_opinions}
        if str(parsed_action.rationale or "").strip():
            overlay["rationale"] = str(parsed_action.rationale or "").strip()
        if str(parsed_action.expected_observation or "").strip():
            overlay["expected_observation"] = str(parsed_action.expected_observation or "").strip()
        if list(parsed_action.success_criteria or []):
            overlay["success_criteria"] = list(parsed_action.success_criteria or [])
        if list(parsed_action.fallback_if_failed or []):
            overlay["fallback_if_failed"] = list(parsed_action.fallback_if_failed or [])
        if str(parsed_action.cost_class or "").strip():
            overlay["cost_class"] = str(parsed_action.cost_class or "").strip()
        if str(parsed_action.risk_class or "").strip():
            overlay["risk_class"] = str(parsed_action.risk_class or "").strip()
        return canonical.model_copy(update=overlay), notes

    @staticmethod
    def _is_explicit_override_proposal_id(selected_proposal_id: str | None) -> bool:
        return str(selected_proposal_id or "").strip().startswith("ORCHESTRATOR_OVERRIDE:")

    def _canonicalize_override_selected_action(
        self,
        *,
        selected_action: SelectedAction,
        supporting_opinions: list[str],
    ) -> tuple[SelectedAction, list[str]]:
        notes: list[str] = []
        overlay: dict[str, Any] = {"supporting_agent_opinions": supporting_opinions}
        if str(selected_action.action_family or "") in self._LOCAL_EXECUTION_ACTION_FAMILIES:
            if bool(selected_action.submit_external_job):
                overlay["submit_external_job"] = False
                notes.append("canonicalized_override_selected_action:submit_external_job")
            if bool(selected_action.wait_for_event_after_submission):
                overlay["wait_for_event_after_submission"] = False
                notes.append("canonicalized_override_selected_action:wait_for_event_after_submission")
            normalized_parameters, changed = self._canonicalize_action_parameters(
                str(selected_action.action_family or ""),
                dict(selected_action.parameters or {}),
            )
            if changed:
                overlay["parameters"] = normalized_parameters
                notes.append("canonicalized_override_selected_action:parameters")
        return selected_action.model_copy(update=overlay), notes

    def _noop_record(
        self,
        *,
        state: dict[str, Any],
        round_id: int,
        proposals: list[Proposal],
        rationale: str,
        guardrail_notes: list[str] | None = None,
        disagreement: list[str] | None = None,
        waiting_external: bool = False,
        ready_to_finalize: bool = False,
        extra_content: dict[str, Any] | None = None,
    ) -> ArbitrationRecord:
        return ArbitrationRecord(
            agent_name=self.agent_name,
            round_id=round_id,
            target_task_id=str(state.get("task", {}).get("task_id") or ""),
            selected_proposal_id=None,
            selected_action=None,
            rejected_proposals=[item.proposal_id for item in proposals],
            guardrail_notes=list(guardrail_notes or []),
            rationale=rationale,
            disagreement_summary=list(disagreement or []),
            whether_noop=True,
            whether_waiting_external=waiting_external,
            whether_ready_to_finalize=ready_to_finalize,
            content=dict(extra_content or {}),
            confidence=0.8 if (waiting_external or ready_to_finalize) else 0.7,
        )

    def _guardrail_arbitration(
        self,
        *,
        state: dict[str, Any],
        proposals: list[Proposal],
        critiques: list[Critique],
        preferences: list[Preference],
        round_id: int,
    ) -> ArbitrationRecord:
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        memory_hits = dict(state.get("memory", {}) or {})
        status = self.tool_gateway.call("query_execution_status", {"state": state})
        current_run_status = str(state.get("workflow", {}).get("run_status") or status.get("run_status") or "")
        task_id = str(state.get("task", {}).get("task_id") or "")
        failing_capability = str(
            latest_observation.get("target_capability")
            or status.get("next_pending_capability")
            or ""
        ).strip()
        retry_counts = dict(status.get("retry_counts", {}) or {})
        retry_budget = int((state.get("workflow", {}) or {}).get("retry_budget", 2) or 2)
        retries_used = int(retry_counts.get(failing_capability, 0) or 0) if failing_capability else 0
        supported_opinions = [
            f"{item.agent_name}:{item.stance if hasattr(item, 'stance') else 'preference'}"
            for item in critiques + preferences
        ]

        if (
            failing_capability
            and str(latest_observation.get("status") or "") == "failed"
            and retry_budget > 0
            and retries_used >= retry_budget
        ):
            return self._noop_record(
                state=state,
                round_id=round_id,
                proposals=proposals,
                rationale="retry_budget_exhausted_for_failing_capability_without_auto_fallback",
                guardrail_notes=[f"retry_budget_exhausted:{failing_capability}:{retries_used}/{retry_budget}"],
                extra_content={"supported_agent_opinions": supported_opinions},
            )

        if not proposals:
            if current_run_status == "waiting_external":
                return self._noop_record(
                    state=state,
                    round_id=round_id,
                    proposals=[],
                    rationale="no_proposals_while_waiting_for_external_event",
                    guardrail_notes=["waiting_external"],
                    waiting_external=True,
                    extra_content={"framework_status": "waiting_external", "wait_reason": status.get("wait_reason")},
                )
            if all_tasks_resolved(state):
                return self._noop_record(
                    state=state,
                    round_id=round_id,
                    proposals=[],
                    rationale="all_tasks_resolved_waiting_for_deliberate_finalization",
                    guardrail_notes=["ready_to_finalize"],
                    ready_to_finalize=True,
                    extra_content={"framework_status": "ready_to_finalize"},
                )
            if latest_observation.get("status") == "failed" or has_blocked_or_abandoned_tasks(state):
                return self._noop_record(
                    state=state,
                    round_id=round_id,
                    proposals=[],
                    rationale="no_viable_recovery_proposals_after_failure_or_blocked_state_without_auto_fallback",
                    guardrail_notes=["failure_without_viable_proposals"],
                    extra_content={"supported_agent_opinions": supported_opinions},
                )
            return self._noop_record(
                state=state,
                round_id=round_id,
                proposals=[],
                rationale="no_proposals_and_state_requires_additional_observation",
                guardrail_notes=["insufficient_proposals"],
                extra_content={"framework_status": "running", "supported_agent_opinions": supported_opinions},
            )

        critique_index: dict[str, list[Critique]] = {}
        legality_index: dict[str, dict[str, Any]] = {}
        viable_proposals: list[Proposal] = []
        objection_counts: dict[str, int] = {}
        support_counts: dict[str, int] = {}
        preference_strengths: dict[str, float] = {}
        guardrail_notes: list[str] = []
        for original_proposal in proposals:
            proposal, canonicalization_notes = self._canonicalize_proposal_execution_semantics(original_proposal)
            if canonicalization_notes:
                guardrail_notes.extend(
                    [f"{proposal.proposal_id}:{note}" for note in canonicalization_notes]
                )
            legality = self.tool_gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": proposal.action_family,
                    "target_capability": proposal.target_capability,
                    "parameters": dict(proposal.parameters or {}),
                    "submit_external_job": bool(proposal.submit_external_job),
                    "wait_for_event_after_submission": bool(proposal.wait_for_event_after_submission),
                },
            )
            legality_index[proposal.proposal_id] = legality
            if not legality.get("allowed", False):
                guardrail_notes.extend(
                    [f"{proposal.proposal_id}:illegal:{reason}" for reason in list(legality.get("refusal_reasons", []) or [])]
                )
            else:
                viable_proposals.append(proposal)
            critique_index[proposal.proposal_id] = []
            objection_counts[proposal.proposal_id] = 0
            support_counts[proposal.proposal_id] = 0
            preference_strengths[proposal.proposal_id] = 0.0
        for critique in critiques:
            critique_index.setdefault(critique.proposal_id, []).append(critique)
            if critique.stance == "support":
                support_counts[critique.proposal_id] = support_counts.get(critique.proposal_id, 0) + 1
            elif critique.stance == "objection":
                objection_counts[critique.proposal_id] = objection_counts.get(critique.proposal_id, 0) + 1
                guardrail_notes.extend(
                    [f"{critique.proposal_id}:objection:{critique.agent_name}:{concern}" for concern in list(critique.concerns or [])]
                )
        for preference in preferences:
            preference_strengths[preference.preferred_proposal_id] = preference_strengths.get(
                preference.preferred_proposal_id, 0.0
            ) + float(preference.preference_strength)
        if not viable_proposals:
            if current_run_status == "waiting_external":
                return self._noop_record(
                    state=state,
                    round_id=round_id,
                    proposals=proposals,
                    rationale="all_proposals_rejected_while_waiting_for_external_event",
                    guardrail_notes=guardrail_notes or ["waiting_external"],
                    waiting_external=True,
                    extra_content={"legality": legality_index},
                )
            if all_tasks_resolved(state):
                return self._noop_record(
                    state=state,
                    round_id=round_id,
                    proposals=proposals,
                    rationale="all_proposals_rejected_but_task_board_is_ready_to_finalize",
                    guardrail_notes=guardrail_notes or ["ready_to_finalize"],
                    ready_to_finalize=True,
                    extra_content={"legality": legality_index},
                )
            return self._noop_record(
                state=state,
                round_id=round_id,
                proposals=proposals,
                rationale="no_legal_proposals_survived_deliberation_without_auto_fallback",
                guardrail_notes=guardrail_notes or ["no_legal_proposals"],
                extra_content={"legality": legality_index, "supported_agent_opinions": supported_opinions},
            )
        legal_sorted = sorted(
            viable_proposals,
            key=lambda item: (
                objection_counts.get(item.proposal_id, 0),
                -(support_counts.get(item.proposal_id, 0)),
                -(preference_strengths.get(item.proposal_id, 0.0)),
                float(item.risk_estimate),
                float(item.cost_estimate),
            ),
        )
        preferred_proposal_id = legal_sorted[0].proposal_id if legal_sorted else None
        return ArbitrationRecord(
            agent_name=self.agent_name,
            round_id=round_id,
            target_task_id=task_id,
            selected_proposal_id=None,
            selected_action=None,
            rejected_proposals=[item.proposal_id for item in proposals if item.proposal_id not in {proposal.proposal_id for proposal in viable_proposals}],
            guardrail_notes=guardrail_notes,
            rationale="legal_proposals_available_for_llm_arbitration",
            disagreement_summary=[],
            content={
                "legality": legality_index,
                "legal_proposal_ids": [item.proposal_id for item in viable_proposals],
                "guardrail_preferred_proposal_id": preferred_proposal_id,
                "objection_counts": objection_counts,
                "support_counts": support_counts,
                "preference_strengths": preference_strengths,
                "memory_hints": {
                    "recovered_case_patterns": len(list(memory_hits.get("recovered_case_patterns", []) or [])),
                    "validation_case_patterns": len(list(memory_hits.get("validation_case_patterns", []) or [])),
                    "historical_failures": len(list(memory_hits.get("historical_failures", []) or [])),
                },
                "supported_agent_opinions": supported_opinions,
            },
            confidence=0.75,
        )

    def arbitrate(
        self,
        *,
        state: dict[str, Any],
        proposals: list[Proposal],
        critiques: list[Critique],
        preferences: list[Preference],
        round_id: int,
    ) -> ArbitrationRecord:
        normalized_proposals: list[Proposal] = []
        normalization_notes: list[str] = []
        for item in proposals:
            normalized, notes = self._canonicalize_proposal_execution_semantics(item)
            normalized_proposals.append(normalized)
            normalization_notes.extend([f"{normalized.proposal_id}:{note}" for note in notes])
        guardrail = self._guardrail_arbitration(
            state=state,
            proposals=normalized_proposals,
            critiques=critiques,
            preferences=preferences,
            round_id=round_id,
        )
        if normalization_notes:
            guardrail.guardrail_notes = list(guardrail.guardrail_notes or []) + normalization_notes
        if not normalized_proposals:
            return guardrail
        latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        status = self.tool_gateway.call("query_execution_status", {"state": state})
        legal_ids = set(list((guardrail.content or {}).get("legal_proposal_ids", []) or []))
        if not legal_ids:
            return guardrail
        context_summary = select_role_context(
            dict((state.get("services", {}) or {}).get("llm_context_summary", {}) or {})
            or build_llm_context_summary(state, execution_status=status),
            role=self.llm_role,
        )
        payload = {
            "state": {
                "task": {"task_id": str(state.get("task", {}).get("task_id") or "")},
                "workflow": {
                    "current_stage": str(state.get("workflow", {}).get("current_stage") or ""),
                    "run_status": str(status.get("run_status") or state.get("workflow", {}).get("run_status") or ""),
                },
                "blackboard": {
                    "latest_execution_observation": latest_observation,
                },
            },
            "context_summary": context_summary,
            "round_id": round_id,
            "proposals": summarize_proposals(normalized_proposals),
            "critiques": summarize_critiques(critiques),
            "preferences": summarize_preferences(preferences),
            "guardrail_context": summarize_guardrail_context(guardrail),
        }
        try:
            llm_result = self._call_llm_structured_with_tools(
                schema=ArbitrationDecisionPayload,
                task_type=str(state.get("task", {}).get("task_type") or "single_material"),
                stage=str(state.get("workflow", {}).get("current_stage") or "arbitration_phase"),
                payload=payload,
                system_prompt=(
                    "You are the Orchestrator Agent, the chief LLM decision-maker in an LLM-centered multi-agent scientific runtime. "
                    "Integrate planner, recovery, critic, physics, and cost opinions into exactly one selected action. "
                    "Legality is a hard guardrail. Task board and the default scientific mainline are context, not an automatic controller."
                ),
                user_prompt=(
                    "TASK_TYPE: {task_type}\nCURRENT_STAGE: {stage}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\nAVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "TOOL_EVIDENCE_JSON:\n{tool_evidence_json}\n\n"
                    "DELIBERATION_AND_GUARDRAIL_CONTEXT_JSON:\n{payload}\n\n"
                    "Return a structured arbitration decision."
                ),
                explicit_skills=["recovery", "physics_validation", "reporting"],
                tool_names=self._tool_names_for_role(),
            )
            parsed = ArbitrationDecisionPayload.model_validate(llm_result)
            selected_proposal_id = str(parsed.selected_proposal_id or "").strip()
            if not selected_proposal_id and parsed.selected_action is not None:
                selected_proposal_id = str(parsed.selected_action.source_proposal_id or "").strip()
            supported = [
                f"{item.agent_name}:{item.stance if hasattr(item, 'stance') else 'preference'}"
                for item in critiques + preferences
            ]
            extra_canonicalization_notes: list[str] = []
            selected_action = None
            canonicalization_notes: list[str] = []
            if parsed.selected_action is not None and self._is_explicit_override_proposal_id(selected_proposal_id):
                override_action, override_notes = self._canonicalize_override_selected_action(
                    selected_action=parsed.selected_action,
                    supporting_opinions=supported,
                )
                override_legality = self.tool_gateway.call(
                    "check_action_legality",
                    {
                        "state": state,
                        "action_family": override_action.action_family,
                        "target_capability": override_action.target_capability,
                        "parameters": dict(override_action.parameters or {}),
                        "submit_external_job": bool(override_action.submit_external_job),
                        "wait_for_event_after_submission": bool(override_action.wait_for_event_after_submission),
                    },
                )
                if override_legality.get("allowed", False):
                    selected_action = override_action
                    extra_canonicalization_notes.extend([f"{selected_proposal_id}:{note}" for note in override_notes])
                    extra_canonicalization_notes.append(f"{selected_proposal_id}:explicit_legal_override")
                else:
                    extra_canonicalization_notes.extend(
                        [
                            f"{selected_proposal_id}:illegal_override:{reason}"
                            for reason in list(override_legality.get("refusal_reasons", []) or [])
                        ]
                    )
            fallback_proposal_id = str((guardrail.content or {}).get("guardrail_preferred_proposal_id") or "").strip()
            if selected_action is None and not selected_proposal_id:
                if fallback_proposal_id in legal_ids:
                    extra_canonicalization_notes.append(
                        "canonicalized_selected_action:missing_selection_fell_back_to_guardrail_preferred"
                    )
                    selected_proposal_id = fallback_proposal_id
                elif len(legal_ids) == 1:
                    selected_proposal_id = next(iter(legal_ids))
                    extra_canonicalization_notes.append(
                        f"canonicalized_selected_action:missing_selection_fell_back_to_only_legal:{selected_proposal_id}"
                    )
            if selected_action is None and parsed.selected_action is not None and selected_proposal_id not in legal_ids:
                if fallback_proposal_id in legal_ids:
                    extra_canonicalization_notes.append(
                        f"canonicalized_selected_action:illegal_or_unknown_override_fell_back_to:{fallback_proposal_id}"
                    )
                    selected_proposal_id = fallback_proposal_id
                else:
                    raise RuntimeError(f"orchestrator_selected_illegal_or_unknown_proposal:{selected_proposal_id or 'unset'}")
            if selected_proposal_id in legal_ids:
                selected_action, canonicalization_notes = self._canonicalize_selected_action(
                    proposals=normalized_proposals,
                    selected_proposal_id=selected_proposal_id,
                    parsed_action=(
                        parsed.selected_action
                        if parsed.selected_action is not None
                        and str(parsed.selected_action.source_proposal_id or "").strip() == selected_proposal_id
                        else None
                    ),
                    supporting_opinions=supported,
                )
            selected_proposal_id = selected_proposal_id or (selected_action.source_proposal_id if selected_action else None)
            rejected_ids = list(
                parsed.rejected_proposal_ids
                or [item.proposal_id for item in normalized_proposals if item.proposal_id != selected_proposal_id]
            )
            return ArbitrationRecord(
                agent_name=self.agent_name,
                round_id=round_id,
                target_task_id=str(state.get("task", {}).get("task_id") or ""),
                selected_proposal_id=selected_proposal_id,
                selected_action=selected_action,
                rejected_proposals=rejected_ids,
                guardrail_notes=list(parsed.guardrail_notes or guardrail.guardrail_notes) + extra_canonicalization_notes + canonicalization_notes,
                rationale=parsed.rationale or guardrail.rationale,
                disagreement_summary=list(parsed.disagreement_summary or guardrail.disagreement_summary),
                whether_noop=bool(parsed.whether_noop) if selected_action is None else False,
                whether_waiting_external=bool(selected_action.wait_for_event_after_submission) if selected_action is not None else bool(parsed.whether_waiting_external),
                whether_ready_to_finalize=bool(parsed.whether_ready_to_finalize) if selected_action is None else False,
                content={"guardrail_context": guardrail.model_dump(mode="json"), "tool_evidence": llm_result.get("tool_evidence", [])},
                confidence=max(
                    0.5,
                    min(
                        1.0,
                        float(
                            selected_action.parameters.get("confidence", guardrail.confidence)
                            if selected_action is not None and isinstance(selected_action.parameters, dict)
                            else guardrail.confidence
                        ),
                    ),
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"orchestrator_strict_agentic_failure:{type(exc).__name__}:{exc}") from exc
