from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
from typing import Any

from ..agents.context_engineering import (
    build_llm_context_summary,
    select_role_context,
    summarize_critiques,
    summarize_preferences,
    summarize_proposals,
)
from ..graph import build_material_graph
from ..agents import ManualFixInstruction
from ..agents.cost_guardian import CostGuardianAgent
from ..agents.critic import CriticAgent
from ..agents.executor import ExecutorAgent
from ..agents.orchestrator import OrchestratorAgent
from ..agents.physics_judge import PhysicsJudgeAgent
from ..agents.planner import PlannerAgent
from ..agents.refinement import RefinementAgent
from ..agents.recovery import RecoveryAgent
from ..agents.validation import ValidationAgent
from ..agents.base import _is_connection_error, _is_rate_limit_error
from ..agents.llm_client import runtime_uses_serialized_official_provider
from ..agents.schemas import ArbitrationRecord, Critique, Preference, Proposal, SelectedAction
from ..graph.human_gate import build_human_escalation_payload
from ..graph.runtime_nodes import (
    _append_decision_trace,
    _append_framework_diagnostic,
    _apply_resume_strategy,
    _build_execution_observation,
    _consume_pending_external_event,
    _instantiate_tools,
    _validate_result_capability,
    make_arbitration_phase_node,
    make_check_termination_node,
    make_execute_selected_action_node,
    make_final_report_node,
    make_observe_state_node,
    make_proposal_phase_node,
    make_reflect_round_node,
    make_critique_phase_node,
)
from ..graph.stage_contracts import find_previous_stage
from ..graph.state import (
    CAPABILITY_SEQUENCE,
    TERMINAL_RUN_STATUSES,
    WAITING_RUN_STATUSES,
    ExternalEventRecord,
    MaterialRunOutcome,
    MaterialTaskState,
    apply_state_patch,
    build_material_outcome,
    has_completed_compute_state_payload,
    derive_compute_status_from_state_payload,
    make_initial_material_state,
    normalize_external_event,
    utc_now_iso,
    validation_report_supports_finalize,
)
from ..hitl.cleanup import apply_cleanup, preview_cleanup
from ..hitl.escalation import notify_escalation, resolve_human_decision, write_escalation_payload
from ..policy.engine import has_relax_failure_signature
from ..runtime.channel_utils import derive_direction_acceptance
from ..runtime.deliberation_loop import all_tasks_resolved, build_initial_task_board
from ..runtime.store import open_memory_store
from ..runtime.telemetry import emit_progress
from ..tools.errors import CheckpointRestoreError
from ..utils import dedupe_keep_order
from .capability_registry import capability_registry_payload, next_capability_after
from .checkpointing import (
    append_ui_event,
    build_material_thread_id,
    langgraph_checkpoint_exists,
    load_state_snapshot,
    open_runtime_checkpointer,
    remove_checkpoints,
    save_checkpoint_metadata,
    save_state_snapshot,
    save_thread_id,
)
from .context import RuntimeContext
from .contracts import CapabilityDecision, DecisionLedgerEntry, ExecutionCheckpoint, WorkflowContract


class CouncilRoleFailure(RuntimeError):
    def __init__(
        self,
        *,
        agent_name: str,
        phase: str,
        critical: bool,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.phase = phase
        self.critical = bool(critical)
        self.error = error
        self.metadata = dict(metadata or {})
        super().__init__(f"{phase}_failed:{agent_name}:{type(error).__name__}:{error}")


def _state_dict(payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
    return MaterialTaskState.from_dict(payload).to_dict()


def _selected_action_dict(action: SelectedAction | dict[str, Any] | None) -> dict[str, Any]:
    if action is None:
        return {}
    if isinstance(action, SelectedAction):
        return action.model_dump(mode="json")
    return dict(action or {})


def _retry_counts_snapshot(state: dict[str, Any]) -> dict[str, int]:
    workflow = dict(state.get("workflow", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    counts: dict[str, int] = {}
    for source in (execution.get("retry_counts", {}), workflow.get("retry_counts", {})):
        for key, value in dict(source or {}).items():
            capability = str(key or "").strip()
            if capability:
                counts[capability] = max(int(counts.get(capability, 0) or 0), int(value or 0))
    return counts


def _ensure_selected_retry_accounted_after_execution(
    state: dict[str, Any],
    *,
    selected_action: dict[str, Any],
    before_retry_counts: dict[str, int],
) -> dict[str, Any]:
    action_family = str(selected_action.get("action_family") or "")
    if action_family not in {"retry_capability", "rerun_from_capability"}:
        return state
    target_capability = str(selected_action.get("target_capability") or "").strip()
    if not target_capability:
        return state
    latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    error_summary = str(latest_observation.get("error_summary") or "")
    if error_summary.startswith("illegal_selected_action"):
        return state
    required_count = int(before_retry_counts.get(target_capability, 0) or 0) + 1
    retry_counts = _retry_counts_snapshot(state)
    if int(retry_counts.get(target_capability, 0) or 0) >= required_count:
        return state
    retry_counts[target_capability] = required_count
    execution = dict(state.get("execution", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    execution["retry_counts"] = dict(retry_counts)
    workflow["retry_counts"] = dict(retry_counts)
    state["execution"] = execution
    state["workflow"] = workflow
    emit_progress(
        "retry counter reconciled after selected action execution",
        workdir=execution.get("workdir"),
        channel="runtime",
        details={
            "action_family": action_family,
            "target_capability": target_capability,
            "retry_count": required_count,
        },
    )
    return state


def _terminal_or_finalizable(state: dict[str, Any]) -> bool:
    workflow = dict(state.get("workflow", {}) or {})
    return bool(state.get("services", {}).get("termination_requested")) or str(workflow.get("run_status") or "") in {
        "ready_to_finalize",
        *TERMINAL_RUN_STATUSES,
    }


def _has_completed_mobility_without_runtime_errors(state: dict[str, Any]) -> bool:
    return has_completed_compute_state_payload(state)


def _validation_finalize_ready(state: dict[str, Any]) -> bool:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    validation_report = dict(diagnostics.get("validation_report", {}) or {})
    validation_stage_status = str((workflow.get("stage_status", {}) or {}).get("validation") or "")
    latest_observation = dict(execution.get("latest_execution_observation", {}) or {})
    return _has_completed_mobility_without_runtime_errors(state) and validation_report_supports_finalize(
        validation_report=validation_report,
        validation_stage_status=validation_stage_status,
        latest_observation=latest_observation,
    )


def _ensure_validation_finalize_ready(state: dict[str, Any]) -> bool:
    if _validation_finalize_ready(state):
        return True
    if not _has_completed_mobility_without_runtime_errors(state):
        return False

    validation = _validate_result_capability(state)
    validation_report = dict(validation.get("validation_report", {}) or {})
    if not validation_report:
        return False

    state["diagnostics"]["validation_report"] = validation_report
    if validation.get("confidence_score") is not None:
        state["diagnostics"]["confidence_score"] = float(validation["confidence_score"])
    state["blackboard"]["anomaly_flags"] = dedupe_keep_order(
        list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or [])
        + list(validation.get("anomaly_flags", []) or [])
    )

    retained_subchannels = list(validation_report.get("retained_subchannels", []) or [])
    rejected_subchannels = list(validation_report.get("rejected_subchannels", []) or [])
    accepted_directions, rejected_directions = derive_direction_acceptance(retained_subchannels, rejected_subchannels)
    state["physics_results"]["accepted_channels"] = accepted_directions
    state["physics_results"]["rejected_channels"] = rejected_directions
    state["diagnostics"]["validation_report"]["accepted_channels"] = accepted_directions
    state["diagnostics"]["validation_report"]["rejected_channels"] = rejected_directions

    if validation_report.get("decision") == "fail":
        state["material"]["warnings"] = dedupe_keep_order(
            list((state.get("material", {}) or {}).get("warnings", []) or []) + list(validation.get("anomaly_flags", []) or [])
        )

    state["workflow"]["stage_status"]["validation"] = "success"
    state["workflow"]["completed_stages"] = dedupe_keep_order(
        list((state.get("workflow", {}) or {}).get("completed_stages", []) or []) + ["validation"]
    )
    return _validation_finalize_ready(state)


def _should_notify_human_for_escalation(state: dict[str, Any], *, target_capability: str | None) -> bool:
    workflow = dict(state.get("workflow", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    latest = dict(execution.get("latest_execution_observation", {}) or {})
    stage_status = dict(workflow.get("stage_status", {}) or {})
    if str(latest.get("status") or "") == "failed":
        return True
    if str(workflow.get("run_status") or "") in {"failed", "needs_recovery"}:
        return True
    if target_capability and str(stage_status.get(target_capability) or "") == "failed":
        return True
    return False


class AgenticMaterialController:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        self.skills_root = os.path.abspath(runtime.skills_root)
        self.tools = _instantiate_tools(runtime)
        self.observe_node = make_observe_state_node(runtime, skills_root=self.skills_root)
        self.proposal_node = make_proposal_phase_node(runtime, skills_root=self.skills_root)
        self.critique_node = make_critique_phase_node(runtime, skills_root=self.skills_root)
        self.arbitration_node = make_arbitration_phase_node(runtime, skills_root=self.skills_root)
        self.execute_node = make_execute_selected_action_node(
            runtime,
            skills_root=self.skills_root,
            tools=self.tools,
        )
        self.reflect_node = make_reflect_round_node(runtime)
        self.check_termination_node = make_check_termination_node(runtime)
        self.final_report_node = make_final_report_node(runtime, skills_root=self.skills_root)
        self.planner = PlannerAgent(runtime, self.skills_root)
        self.recovery = RecoveryAgent(runtime, self.skills_root)
        self.refinement = RefinementAgent(runtime, self.skills_root)
        self.validation = ValidationAgent(runtime, self.skills_root)
        self.executor = ExecutorAgent(runtime, self.skills_root)
        self.physics_judge = PhysicsJudgeAgent(runtime, self.skills_root)
        self.cost_guardian = CostGuardianAgent(runtime, self.skills_root)
        self.critic = CriticAgent(runtime, self.skills_root)
        self.orchestrator = OrchestratorAgent(runtime, self.skills_root)
        self._compat_checkpointer = None
        self._compat_store = None
        self._compat_app = None

    def _uses_serialized_official_provider(self, *, role: str | None = None) -> bool:
        return runtime_uses_serialized_official_provider(self.runtime.agent_runtime, role=role)

    def _open_compatibility_app(self) -> None:
        if self._compat_app is not None:
            return
        self._compat_checkpointer = open_runtime_checkpointer(database_uri=self.runtime.resolved_db_uri)
        checkpointer = self._compat_checkpointer.__enter__()
        self._compat_store = open_memory_store(self.runtime.store_path)
        store = self._compat_store.__enter__()
        graph = build_material_graph(
            {
                "observe_state": self.observe_node,
                "proposal_phase": self.proposal_node,
                "critique_phase": self.critique_node,
                "arbitration_phase": self.arbitration_node,
                "execute_selected_action": self.execute_node,
                "reflect_round": self.reflect_node,
                "check_termination": self.check_termination_node,
                "final_report": self.final_report_node,
            }
        )
        self._compat_app = graph.compile(checkpointer=checkpointer, store=store)

    def _close_compatibility_app(self) -> None:
        if self._compat_store is not None:
            self._compat_store.__exit__(None, None, None)
        if self._compat_checkpointer is not None:
            self._compat_checkpointer.__exit__(None, None, None)
        self._compat_app = None
        self._compat_store = None
        self._compat_checkpointer = None

    def _sync_durable_checkpoint(self, state: dict[str, Any]) -> None:
        thread_id = str((state.get("execution", {}) or {}).get("thread_id") or "").strip()
        if not thread_id:
            return
        self._open_compatibility_app()
        assert self._compat_app is not None
        self._compat_app.update_state(
            {"configurable": {"thread_id": thread_id}},
            MaterialTaskState.from_dict(state).to_dict(),
            as_node="observe_state",
        )

    def _load_durable_state(self, thread_id: str) -> dict[str, Any] | None:
        resolved_thread_id = str(thread_id or "").strip()
        if not resolved_thread_id:
            return None
        self._open_compatibility_app()
        assert self._compat_app is not None
        snapshot = self._compat_app.get_state({"configurable": {"thread_id": resolved_thread_id}})
        raw_values = getattr(snapshot, "values", None)
        if not raw_values:
            return None
        return MaterialTaskState.from_dict(raw_values).to_dict()

    def _persist_state(
        self,
        state: dict[str, Any],
        *,
        event_type: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._sync_material_job_state(state)
        normalized = _state_dict(state)
        save_state_snapshot(
            workdir=str(normalized.get("execution", {}).get("workdir") or ""),
            state=normalized,
            checkpoint_subdir=self.runtime.checkpoint_subdir,
        )
        if event_type:
            append_ui_event(
                workdir=str(normalized.get("execution", {}).get("workdir") or ""),
                event_type=event_type,
                checkpoint_subdir=self.runtime.checkpoint_subdir,
                state=normalized,
                extra=extra,
            )
        self._sync_durable_checkpoint(normalized)
        return normalized

    def _append_decision_ledger(
        self,
        state: dict[str, Any],
        *,
        entry_type: str,
        reason: str,
        round_id: int = 0,
        contract: WorkflowContract | None = None,
        agent_names: list[str] | None = None,
        selected_action: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = DecisionLedgerEntry(
            entry_type=entry_type,
            reason=reason,
            round_id=round_id,
            contract_id=(contract.contract_id if contract is not None else None),
            contract_version=(contract.version if contract is not None else None),
            agent_names=list(agent_names or []),
            selected_action=dict(selected_action or {}),
            summary=dict(summary or {}),
        )
        services = dict(state.get("services", {}) or {})
        services["decision_ledger"] = dedupe_keep_order(list(services.get("decision_ledger", []) or []) + [entry.model_dump(mode="json")])
        state["services"] = services
        return state

    def _execution_checkpoint(self, state: dict[str, Any]) -> ExecutionCheckpoint:
        payload = dict((state.get("execution", {}) or {}).get("execution_checkpoint", {}) or {})
        return ExecutionCheckpoint.model_validate(payload)

    def _workflow_contract(self, state: dict[str, Any]) -> WorkflowContract | None:
        payload = dict((state.get("services", {}) or {}).get("workflow_contract", {}) or {})
        if not payload:
            return None
        return WorkflowContract.model_validate(payload)

    def _sync_execution_checkpoint(
        self,
        state: dict[str, Any],
        *,
        contract: WorkflowContract | None,
        current_capability: str | None = None,
        next_capability: str | None = None,
        needs_deliberation: bool,
        deliberation_reason: str | None,
    ) -> dict[str, Any]:
        completed = list((state.get("workflow", {}) or {}).get("completed_stages", []) or [])
        latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        checkpoint = ExecutionCheckpoint(
            contract_id=(contract.contract_id if contract is not None else None),
            contract_version=(contract.version if contract is not None else 0),
            current_capability=current_capability,
            next_capability=next_capability,
            completed_capabilities=completed,
            last_observation=latest_observation,
            needs_deliberation=needs_deliberation,
            deliberation_reason=deliberation_reason,
        )
        state["execution"]["execution_checkpoint"] = checkpoint.model_dump(mode="json")
        return state

    @staticmethod
    def _count_timeout_auto_attempts(state: dict[str, Any], *, stage: str | None) -> int:
        target = str(stage or "").strip()
        attempts = 0
        for item in list((state.get("diagnostics", {}) or {}).get("recovery_history", []) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("origin") or "") != "timeout_auto":
                continue
            entry_stage = str(item.get("target_stage") or item.get("stage") or "").strip()
            if target and entry_stage and entry_stage != target:
                continue
            attempts += 1
        return attempts

    @staticmethod
    def _timeout_default_action_for_risk(*, risk_class: str, prior_attempts: int) -> str:
        return "skip_material"

    def _sync_material_job_state(self, state: dict[str, Any]) -> dict[str, Any]:
        task = dict(state.get("task", {}) or {})
        material = dict(state.get("material", {}) or {})
        workflow = dict(state.get("workflow", {}) or {})
        execution = dict(state.get("execution", {}) or {})
        diagnostics = dict(state.get("diagnostics", {}) or {})
        services = dict(state.get("services", {}) or {})
        latest = dict(execution.get("latest_execution_observation", {}) or {})
        state["services"]["material_job"] = {
            "job_id": str(task.get("task_id") or ""),
            "material_id": str(material.get("material_id") or ""),
            "current_stage": str(workflow.get("current_stage") or ""),
            "latest_capability": str(latest.get("target_capability") or workflow.get("current_stage") or ""),
            "run_status": str(workflow.get("run_status") or ""),
            "retry_counts": dict(workflow.get("retry_counts", {}) or execution.get("retry_counts", {}) or {}),
            "error_history": list(diagnostics.get("errors", []) or []),
            "latest_error": diagnostics.get("last_error"),
            "human_state": {
                "pending": bool(services.get("pending_human_payload")),
                "window_seconds": int(self.runtime.agent_runtime.human_review_timeout_seconds or 300),
                "latest_decision": dict(services.get("latest_human_decision", {}) or {}),
            },
            "result_cache": {
                "stage_status": dict(workflow.get("stage_status", {}) or {}),
                "completed_stages": list(workflow.get("completed_stages", []) or []),
                "validation_report": dict(diagnostics.get("validation_report", {}) or {}),
                "quality_grade": diagnostics.get("quality_grade"),
            },
            "execution_checkpoint": dict(execution.get("execution_checkpoint", {}) or {}),
            "updated_at": utc_now_iso(),
        }
        return state

    def _update_contract_state(self, state: dict[str, Any], contract: WorkflowContract) -> dict[str, Any]:
        services = dict(state.get("services", {}) or {})
        history = list(services.get("workflow_contract_history", []) or [])
        history.append(contract.model_dump(mode="json"))
        services["workflow_contract"] = contract.model_dump(mode="json")
        services["workflow_contract_history"] = dedupe_keep_order(history)
        services["capability_registry"] = capability_registry_payload()
        state["services"] = services
        return state

    @staticmethod
    def _stable_hash(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _council_output_cache(self, state: dict[str, Any]) -> dict[str, Any]:
        services = dict(state.get("services", {}) or {})
        cache = dict(services.get("council_output_cache", {}) or {})
        services["council_output_cache"] = cache
        state["services"] = services
        return cache

    def _role_payload_hash(
        self,
        state: dict[str, Any],
        *,
        agent_name: str,
        phase: str,
        council_mode: str,
        reason: str,
        proposals: list[Proposal] | None = None,
        critiques: list[Critique] | None = None,
        preferences: list[Preference] | None = None,
    ) -> str:
        execution_status = self.executor.tool_gateway.call("query_execution_status", {"state": state})
        summary = build_llm_context_summary(state, execution_status=execution_status)
        agent = getattr(self, agent_name)
        payload = {
            "agent_name": agent_name,
            "phase": phase,
            "council_mode": council_mode,
            "reason": reason,
            "role_context": select_role_context(summary, role=str(getattr(agent, "llm_role", agent_name) or agent_name)),
            "proposals": summarize_proposals(proposals or []),
            "critiques": summarize_critiques(critiques or []),
            "preferences": summarize_preferences(preferences or []),
        }
        return self._stable_hash(payload)

    def _deserialize_cached_role_output(self, phase: str, payload: dict[str, Any]) -> Any:
        if phase == "proposal":
            return [Proposal.model_validate(item) for item in list(payload.get("proposals", []) or [])]
        if phase == "review":
            return (
                [Critique.model_validate(item) for item in list(payload.get("critiques", []) or [])],
                [Preference.model_validate(item) for item in list(payload.get("preferences", []) or [])],
            )
        if phase == "arbitration":
            return ArbitrationRecord.model_validate(dict(payload.get("arbitration", {}) or {}))
        raise RuntimeError(f"unsupported_council_phase:{phase}")

    def _serialize_role_output(self, phase: str, output: Any) -> dict[str, Any]:
        if phase == "proposal":
            return {"proposals": [item.model_dump(mode="json") for item in list(output or [])]}
        if phase == "review":
            critiques, preferences = output
            return {
                "critiques": [item.model_dump(mode="json") for item in list(critiques or [])],
                "preferences": [item.model_dump(mode="json") for item in list(preferences or [])],
            }
        if phase == "arbitration":
            arbitration = ArbitrationRecord.model_validate(output)
            return {"arbitration": arbitration.model_dump(mode="json")}
        raise RuntimeError(f"unsupported_council_phase:{phase}")

    def _run_role(
        self,
        state: dict[str, Any],
        *,
        agent_name: str,
        phase: str,
        council_mode: str,
        reason: str,
        critical: bool,
        fn,
        proposals: list[Proposal] | None = None,
        critiques: list[Critique] | None = None,
        preferences: list[Preference] | None = None,
        reused_roles: list[str] | None = None,
    ) -> Any:
        payload_hash = self._role_payload_hash(
            state,
            agent_name=agent_name,
            phase=phase,
            council_mode=council_mode,
            reason=reason,
            proposals=proposals,
            critiques=critiques,
            preferences=preferences,
        )
        cache_key = f"{phase}:{agent_name}:{payload_hash}"
        cache = self._council_output_cache(state)
        cached = dict(cache.get(cache_key, {}) or {})
        if cached:
            if reused_roles is not None:
                reused_roles.append(agent_name)
            getattr(self, agent_name).last_llm_call_metadata = dict(cached.get("metadata", {}) or {})
            return self._deserialize_cached_role_output(phase, cached)
        agent = getattr(self, agent_name)
        try:
            output = fn()
        except Exception as exc:
            raise CouncilRoleFailure(
                agent_name=agent_name,
                phase=phase,
                critical=critical,
                error=exc,
                metadata=dict(getattr(agent, "last_llm_call_metadata", {}) or {}),
            ) from exc
        cache[cache_key] = {
            **self._serialize_role_output(phase, output),
            "metadata": dict(getattr(agent, "last_llm_call_metadata", {}) or {}),
        }
        return output

    def _run_roles_parallel(
        self,
        state: dict[str, Any],
        *,
        specs: list[dict[str, Any]],
        phase: str,
        council_mode: str,
        reason: str,
        proposals: list[Proposal] | None = None,
        critiques: list[Critique] | None = None,
        preferences: list[Preference] | None = None,
        reused_roles: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[CouncilRoleFailure]]:
        if not specs:
            return [], []

        cache = self._council_output_cache(state)
        ordered_results: list[dict[str, Any]] = []
        failures: list[CouncilRoleFailure] = []
        pending: list[dict[str, Any]] = []

        for index, spec in enumerate(specs):
            agent_name = str(spec["agent_name"])
            payload_hash = self._role_payload_hash(
                state,
                agent_name=agent_name,
                phase=phase,
                council_mode=council_mode,
                reason=reason,
                proposals=proposals,
                critiques=critiques,
                preferences=preferences,
            )
            cache_key = f"{phase}:{agent_name}:{payload_hash}"
            cached = dict(cache.get(cache_key, {}) or {})
            if cached:
                if reused_roles is not None:
                    reused_roles.append(agent_name)
                getattr(self, agent_name).last_llm_call_metadata = dict(cached.get("metadata", {}) or {})
                ordered_results.append(
                    {
                        "index": index,
                        "agent_name": agent_name,
                        "critical": bool(spec["critical"]),
                        "output": self._deserialize_cached_role_output(phase, cached),
                        "reused": True,
                    }
                )
                continue
            pending.append(
                {
                    "index": index,
                    "agent_name": agent_name,
                    "critical": bool(spec["critical"]),
                    "cache_key": cache_key,
                    "invoke": spec["invoke"],
                }
            )

        def _worker(entry: dict[str, Any]) -> dict[str, Any]:
            agent_name = str(entry["agent_name"])
            agent = type(getattr(self, agent_name))(self.runtime, self.skills_root)
            local_state = _state_dict(state)
            try:
                output = entry["invoke"](agent, local_state)
            except Exception as exc:
                return {
                    "index": int(entry["index"]),
                    "agent_name": agent_name,
                    "critical": bool(entry["critical"]),
                    "error": exc,
                    "metadata": dict(getattr(agent, "last_llm_call_metadata", {}) or {}),
                }
            return {
                "index": int(entry["index"]),
                "agent_name": agent_name,
                "critical": bool(entry["critical"]),
                "output": output,
                "metadata": dict(getattr(agent, "last_llm_call_metadata", {}) or {}),
                "serialized": self._serialize_role_output(phase, output),
                "recovery_diagnosis": (
                    dict(getattr(agent, "last_failure_diagnosis", {}) or {})
                    if agent_name == "recovery"
                    else None
                ),
            }

        if pending:
            max_workers = max(1, len(pending))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_worker, entry): entry
                    for entry in pending
                }
                worker_results: dict[int, dict[str, Any]] = {}
                for future in as_completed(future_map):
                    entry = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        result = {
                            "index": int(entry["index"]),
                            "agent_name": str(entry["agent_name"]),
                            "critical": bool(entry["critical"]),
                            "error": exc,
                            "metadata": {},
                        }
                    worker_results[int(result["index"])] = result

            for entry in pending:
                result = dict(worker_results.get(int(entry["index"]), {}) or {})
                agent_name = str(entry["agent_name"])
                agent = getattr(self, agent_name)
                agent.last_llm_call_metadata = dict(result.get("metadata", {}) or {})
                if "error" in result:
                    failures.append(
                        CouncilRoleFailure(
                            agent_name=agent_name,
                            phase=phase,
                            critical=bool(entry["critical"]),
                            error=result["error"],
                            metadata=dict(result.get("metadata", {}) or {}),
                        )
                    )
                    continue
                cache[str(entry["cache_key"])] = {
                    **dict(result.get("serialized", {}) or {}),
                    "metadata": dict(result.get("metadata", {}) or {}),
                }
                if agent_name == "recovery":
                    self.recovery.last_failure_diagnosis = dict(result.get("recovery_diagnosis", {}) or {})
                ordered_results.append(
                    {
                        "index": int(entry["index"]),
                        "agent_name": agent_name,
                        "critical": bool(entry["critical"]),
                        "output": result.get("output"),
                        "reused": False,
                    }
                )

        ordered_results.sort(key=lambda item: int(item["index"]))
        failures.sort(key=lambda item: next((idx for idx, spec in enumerate(specs) if str(spec["agent_name"]) == item.agent_name), 0))
        return ordered_results, failures

    def _run_reviewer_roles(
        self,
        working_state: dict[str, Any],
        *,
        reviewer_specs: list[dict[str, Any]],
        round_id: int,
        council_mode: str,
        reason: str,
        proposals: list[Proposal],
        reused_roles: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[CouncilRoleFailure]]:
        if self._uses_serialized_official_provider(role="physics_judge"):
            emit_progress(
                "reviewer sequentialized",
                workdir=working_state["execution"].get("workdir"),
                channel="runtime",
                details={
                    "phase": "review",
                    "round_id": round_id,
                    "reason": reason,
                    "agent_names": [str(item.get("agent_name") or "") for item in reviewer_specs],
                },
            )
            reviewer_results: list[dict[str, Any]] = []
            reviewer_failures: list[CouncilRoleFailure] = []
            for index, spec in enumerate(reviewer_specs):
                agent_name = str(spec["agent_name"])
                try:
                    output = self._run_role(
                        working_state,
                        agent_name=agent_name,
                        phase="review",
                        council_mode=council_mode,
                        reason=reason,
                        critical=bool(spec["critical"]),
                        fn=lambda agent_name=agent_name, invoke=spec["invoke"]: invoke(getattr(self, agent_name), working_state),
                        proposals=proposals,
                        reused_roles=reused_roles,
                    )
                    reviewer_results.append(
                        {
                            "index": index,
                            "agent_name": agent_name,
                            "critical": bool(spec["critical"]),
                            "output": output,
                            "reused": agent_name in list(reused_roles or []),
                        }
                    )
                except CouncilRoleFailure as exc:
                    reviewer_failures.append(exc)
            return reviewer_results, reviewer_failures
        return self._run_roles_parallel(
            working_state,
            specs=reviewer_specs,
            phase="review",
            council_mode=council_mode,
            reason=reason,
            proposals=proposals,
            reused_roles=reused_roles,
        )

    def _apply_council_reopen_cooldown_if_needed(
        self,
        *,
        state: dict[str, Any],
        round_id: int,
        exc: CouncilRoleFailure,
    ) -> None:
        reopen_cooldown_s: float | None = None
        reason = ""
        if self._uses_serialized_official_provider(role=exc.agent_name) and _is_rate_limit_error(exc.error):
            reopen_cooldown_s = 30.0
            reason = "serialized_official_provider_council_reopen"
        elif _is_connection_error(exc.error):
            reopen_cooldown_s = 15.0
            reason = "transient_llm_connection_error"
        if reopen_cooldown_s is None:
            return
        emit_progress(
            "council reopen cooldown applied",
            workdir=state["execution"].get("workdir"),
            channel="runtime",
            details={
                "agent": exc.agent_name,
                "phase": exc.phase,
                "round_id": round_id,
                "cooldown_s": reopen_cooldown_s,
                "reason": reason,
            },
        )
        time.sleep(reopen_cooldown_s)

    @staticmethod
    def _segment_from_capability(start_capability: str | None, *, end_capability: str) -> list[str]:
        start = str(start_capability or "").strip()
        if not start or start not in CAPABILITY_SEQUENCE or end_capability not in CAPABILITY_SEQUENCE:
            return []
        start_index = CAPABILITY_SEQUENCE.index(start)
        end_index = CAPABILITY_SEQUENCE.index(end_capability)
        if start_index > end_index:
            return [start]
        return list(CAPABILITY_SEQUENCE[start_index : end_index + 1])

    @staticmethod
    def _distinct_proposal_count(proposals: list[Proposal]) -> int:
        keys = {
            (
                str(item.action_family or ""),
                str(item.target_capability or ""),
                json.dumps(dict(item.parameters or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
            for item in list(proposals or [])
        }
        return len(keys)

    def _council_mode(self, state: dict[str, Any], *, reason: str) -> str:
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        if reason.startswith("failure"):
            return "failure_council"
        if reason in {"validation_followup", "post_mobility_quality_review"} or reason.startswith("anomaly:validation"):
            return "validation_followup_council"
        if reason.startswith("resume"):
            if str(latest.get("status") or "") == "failed":
                return "failure_council"
            if str(latest.get("target_capability") or "") == "validation":
                return "validation_followup_council"
        return "segment_council"

    def _planned_capabilities_for_selection(
        self,
        *,
        reason: str,
        council_mode: str,
        action_family: str,
        target_capability: str,
    ) -> tuple[list[str], list[str]]:
        policy_mode = str(self.runtime.council_policy_mode or "balanced").strip().lower() or "balanced"
        target = str(target_capability or "").strip()
        if policy_mode == "strict":
            if action_family == "refine_sampling":
                return ["strain_loop", "mobility", "validation"], ["validation"]
            if action_family in {"finalize_material", "abort_material", "escalate_human"}:
                return [], []
            if council_mode == "validation_followup_council" and action_family == "revalidate_result":
                return ["validation"], ["validation"]
            if target:
                return [target], [target]
            if reason == "initial_plan":
                return ["prepare"], ["prepare"]
            return [], []
        if action_family == "refine_sampling":
            return ["strain_loop", "mobility", "validation"], ["validation"]
        if action_family in {"finalize_material", "abort_material", "escalate_human"}:
            return [], []
        if council_mode == "validation_followup_council":
            if action_family == "revalidate_result":
                return ["validation"], ["validation"]
            return [], []
        if target in {"prepare", "relax", "scf", "band", "effective_mass"}:
            return self._segment_from_capability(target, end_capability="strain_loop"), ["strain_loop"]
        if target == "strain_loop":
            return ["strain_loop"], ["strain_loop"]
        if target == "mobility":
            return ["mobility", "validation"], ["validation"]
        if target == "validation":
            return ["validation"], ["validation"]
        if reason == "initial_plan":
            return self._segment_from_capability("prepare", end_capability="strain_loop"), ["strain_loop"]
        return [], []

    def _requires_cost_guardian(self, council_mode: str, proposals: list[Proposal]) -> bool:
        policy_mode = str(self.runtime.council_policy_mode or "balanced").strip().lower() or "balanced"
        if policy_mode == "strict":
            return bool(list(proposals or []))
        if council_mode == "validation_followup_council":
            return True
        expensive_actions = {
            "refine_sampling",
            "retry_capability",
            "rerun_from_capability",
            "skip_channel",
            "invalidate_channel",
            "finalize_material",
        }
        medium_costs = {"medium", "high"} if policy_mode != "permissive" else {"high"}
        if council_mode == "failure_council":
            return any(str(item.action_family or "") in expensive_actions for item in list(proposals or []))
        for item in list(proposals or []):
            action_family = str(item.action_family or "")
            cost_class = str((item.content or {}).get("cost_class") or "").strip().lower()
            if action_family in expensive_actions or cost_class in medium_costs:
                return True
        return False

    def _requires_critic(self, council_mode: str, proposals: list[Proposal], critiques: list[Critique]) -> bool:
        policy_mode = str(self.runtime.council_policy_mode or "balanced").strip().lower() or "balanced"
        if council_mode == "validation_followup_council":
            return True
        if policy_mode == "strict":
            return bool(list(proposals or []))
        if any(str(item.stance or "") == "objection" for item in list(critiques or [])):
            return True
        distinct = self._distinct_proposal_count(proposals)
        if policy_mode == "permissive":
            return distinct >= 3
        return distinct >= 2

    def _should_include_recovery(self, state: dict[str, Any], reason: str) -> bool:
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        return str(latest.get("status") or "") == "failed" or reason.startswith("failure") or reason.startswith("resume")

    def _requires_post_mobility_quality_review(
        self,
        state: dict[str, Any],
        *,
        latest: dict[str, Any],
    ) -> bool:
        if str(latest.get("target_capability") or "") != "mobility":
            return False
        if str(latest.get("status") or "") not in {"success", "completed"}:
            return False
        if list(latest.get("risk_flags", []) or []):
            return True
        diagnostics = dict(state.get("diagnostics", {}) or {})
        fit = dict(diagnostics.get("fit_diagnostics", {}) or {})
        try:
            effective_fit = float(fit.get("effective_fit_quality", fit.get("fit_r2_min", 1.0)) or 0.0)
        except Exception:
            effective_fit = 1.0
        try:
            threshold = float((state.get("services", {}) or {}).get("fit_r2_threshold") or 0.90)
        except Exception:
            threshold = 0.90
        return effective_fit < threshold

    def _deliberation_reason(self, state: dict[str, Any], contract: WorkflowContract | None) -> str | None:
        workflow = dict(state.get("workflow", {}) or {})
        execution = dict(state.get("execution", {}) or {})
        latest = dict(execution.get("latest_execution_observation", {}) or {})
        checkpoint = self._execution_checkpoint(state)
        if contract is None:
            return "initial_plan"
        if list(execution.get("pending_events", []) or []):
            return "resume_event"
        if str(workflow.get("run_status") or "") in {"waiting_external", "needs_human"} and not list(execution.get("pending_events", []) or []):
            return None
        if checkpoint.needs_deliberation and checkpoint.deliberation_reason:
            return checkpoint.deliberation_reason
        if str(latest.get("status") or "") == "failed" or str(workflow.get("run_status") or "") == "needs_recovery":
            target = str(latest.get("target_capability") or workflow.get("current_stage") or "unknown")
            return f"failure:{target}"
        target = str(latest.get("target_capability") or "")
        if self._requires_post_mobility_quality_review(state, latest=latest):
            return "post_mobility_quality_review"
        if list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or []) and target in {"strain_loop", "mobility", "validation"}:
            return f"anomaly:{target}"
        if target == "validation" and str(latest.get("status") or "") in {"success", "completed"}:
            return "validation_followup"
        if target == "strain_loop" and str(latest.get("status") or "") in {"success", "completed"}:
            return "milestone:strain_loop"
        if contract is not None and next_capability_after(
            list((state.get("workflow", {}) or {}).get("completed_stages", []) or []),
            list(contract.planned_capabilities or []),
        ) is None:
            return "contract_completed"
        if all_tasks_resolved(state) or str(workflow.get("run_status") or "") == "ready_to_finalize":
            return "contract_completed"
        return None

    def _approved_agent_names(
        self,
        proposals: list[Proposal],
        critiques: list[Critique],
        preferences: list[Preference],
        arbitration: ArbitrationRecord,
    ) -> list[str]:
        return dedupe_keep_order(
            [item.agent_name for item in proposals]
            + [item.agent_name for item in critiques]
            + [item.agent_name for item in preferences]
            + [arbitration.agent_name]
        )

    def _build_contract(
        self,
        state: dict[str, Any],
        *,
        reason: str,
        proposals: list[Proposal],
        critiques: list[Critique],
        preferences: list[Preference],
        arbitration: ArbitrationRecord,
    ) -> WorkflowContract:
        previous = self._workflow_contract(state)
        selected_action = arbitration.selected_action
        action_family = str((selected_action.action_family if selected_action is not None else "") or "")
        target_capability = str((selected_action.target_capability if selected_action is not None else "") or "")
        council_mode = self._council_mode(state, reason=reason)
        planned_capabilities, milestones = self._planned_capabilities_for_selection(
            reason=reason,
            council_mode=council_mode,
            action_family=action_family,
            target_capability=target_capability,
        )
        supporting_agents = [
            item.agent_name
            for item in critiques
            if item.stance == "support"
        ] + [item.agent_name for item in preferences]
        opposing_agents = [item.agent_name for item in critiques if item.stance == "objection"]
        agent_names = self._approved_agent_names(proposals, critiques, preferences, arbitration)
        contract_input_hash = self._stable_hash(
            {
                "reason": reason,
                "council_mode": council_mode,
                "selected_action": _selected_action_dict(selected_action),
                "proposals": summarize_proposals(proposals),
                "critiques": summarize_critiques(critiques),
                "preferences": summarize_preferences(preferences),
            }
        )
        capability_decisions: list[CapabilityDecision] = []
        for index, capability in enumerate(planned_capabilities):
            capability_decisions.append(
                CapabilityDecision(
                    capability=capability,
                    action_family=action_family or "run_capability",
                    source_agents=agent_names,
                    supporting_agents=supporting_agents,
                    opposing_agents=opposing_agents if index == 0 else [],
                    rationale=(
                        arbitration.rationale
                        if index == 0
                        else f"council_authorized_downstream_mainline_after:{target_capability or capability}"
                    ),
                    confidence=float(arbitration.confidence or 0.75) if index == 0 else max(0.55, float(arbitration.confidence or 0.75) - 0.1),
                    fallback_actions=list((selected_action.fallback_if_failed if selected_action is not None else []) or []),
                )
            )
        contract = WorkflowContract(
            contract_id=(previous.contract_id if previous is not None else None) or WorkflowContract().contract_id,
            version=(int(previous.version) + 1 if previous is not None else 1),
            deliberation_reason=reason,
            council_mode=council_mode,
            approved_by_agents=agent_names,
            current_focus=target_capability or (planned_capabilities[0] if planned_capabilities else None),
            planned_capabilities=planned_capabilities,
            milestones=milestones,
            input_hash=contract_input_hash,
            revisit_triggers={
                "on_failure": True,
                "on_contract_completion": True,
                "milestone_capabilities": milestones,
                "anomaly_capabilities": ["strain_loop", "mobility", "validation"],
            },
            allowed_branches=dedupe_keep_order(
                list((selected_action.fallback_if_failed if selected_action is not None else []) or [])
                + [str((selected_action.action_family if selected_action is not None else "") or "")]
            ),
            decision_rationale=str(arbitration.rationale or "") or f"agentic_contract:{reason}",
            evidence_summary={
                "latest_observation": dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {}),
                "risk_flags": list((state.get("blackboard", {}) or {}).get("risk_flags", []) or []),
                "anomaly_flags": list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or []),
            },
            capability_decisions=capability_decisions,
            reuse_metadata={
                "cached_role_outputs": len(dict((state.get("services", {}) or {}).get("council_output_cache", {}) or {})),
                "council_round_metrics": len(list((state.get("services", {}) or {}).get("council_round_metrics", []) or [])),
            },
        )
        return contract

    def _record_council_outputs(
        self,
        state: dict[str, Any],
        *,
        proposals: list[Proposal],
        critiques: list[Critique],
        preferences: list[Preference],
        arbitration: ArbitrationRecord,
    ) -> dict[str, Any]:
        state["deliberation"]["proposals"] = dedupe_keep_order(
            list(state["deliberation"].get("proposals", []) or []) + [item.model_dump(mode="json") for item in proposals]
        )
        state["deliberation"]["critiques"] = dedupe_keep_order(
            list(state["deliberation"].get("critiques", []) or []) + [item.model_dump(mode="json") for item in critiques]
        )
        state["deliberation"]["preferences"] = dedupe_keep_order(
            list(state["deliberation"].get("preferences", []) or []) + [item.model_dump(mode="json") for item in preferences]
        )
        state["deliberation"]["arbitrations"] = dedupe_keep_order(
            list(state["deliberation"].get("arbitrations", []) or []) + [arbitration.model_dump(mode="json")]
        )
        if arbitration.selected_action is not None:
            state["deliberation"]["selected_actions"] = dedupe_keep_order(
                list(state["deliberation"].get("selected_actions", []) or []) + [arbitration.selected_action.model_dump(mode="json")]
            )
        state["deliberation"]["rationale_history"] = dedupe_keep_order(
            list(state["deliberation"].get("rationale_history", []) or []) + [arbitration.rationale]
        )
        for item in proposals:
            _append_decision_trace(state, {"node": "agentic_planning", "message_type": "proposal", **item.model_dump(mode="json")})
        for item in critiques + preferences:
            _append_decision_trace(state, {"node": "agentic_planning", "message_type": item.message_type, **item.model_dump(mode="json")})
        _append_decision_trace(state, {"node": "agentic_planning", "message_type": "arbitration", **arbitration.model_dump(mode="json")})
        return state

    def _deliberate(self, state: dict[str, Any], *, reason: str) -> dict[str, Any]:
        state = apply_state_patch(state, self.observe_node(state))
        round_id = int((state.get("deliberation", {}) or {}).get("round_index", 0) or 0)
        council_mode = self._council_mode(state, reason=reason)
        services = dict(state.get("services", {}) or {})
        services["runtime_strategy"] = {
            "council_policy_mode": str(self.runtime.council_policy_mode or "balanced"),
            "council_mode": council_mode,
            "deliberation_reason": reason,
        }
        state["services"] = services

        proposals: list[Proposal] = []
        critiques: list[Critique] = []
        preferences: list[Preference] = []
        reused_roles: list[str] = []
        omitted_roles: list[dict[str, Any]] = []
        role_metrics: dict[str, Any] = {}
        snapshot = _state_dict(state)
        cache_store = dict((state.get("services", {}) or {}).get("council_output_cache", {}) or {})
        arbitration: ArbitrationRecord | None = None

        def _record_role(agent_name: str, *, phase: str, critical: bool, reused: bool = False) -> None:
            agent = getattr(self, agent_name)
            role_metrics[agent_name] = {
                "phase": phase,
                "critical": critical,
                "reused": reused,
                **dict(getattr(agent, "last_llm_call_metadata", {}) or {}),
            }

        def _omit_role(exc: CouncilRoleFailure) -> None:
            omitted_roles.append(
                {
                    "agent_name": exc.agent_name,
                    "phase": exc.phase,
                    "reason": f"{type(exc.error).__name__}:{exc.error}",
                }
            )
            role_metrics[exc.agent_name] = {
                "phase": exc.phase,
                "critical": False,
                "omitted": True,
                **dict(exc.metadata or {}),
            }
            emit_progress(
                "non-critical council role omitted after bounded retries",
                workdir=state["execution"].get("workdir"),
                channel="runtime",
                details={"agent": exc.agent_name, "phase": exc.phase, "round_id": round_id, "error": f"{type(exc.error).__name__}:{exc.error}"},
            )

        for council_attempt in range(1, 3):
            proposals = []
            critiques = []
            preferences = []
            working_state = _state_dict(snapshot)
            working_services = dict(working_state.get("services", {}) or {})
            working_services["council_output_cache"] = dict(cache_store)
            working_state["services"] = working_services
            try:
                proposal_specs: list[tuple[str, bool, Any]] = []
                if council_mode == "failure_council":
                    proposal_specs = [
                        {
                            "agent_name": "recovery",
                            "critical": True,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                        {
                            "agent_name": "planner",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                        {
                            "agent_name": "executor",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                    ]
                elif council_mode == "validation_followup_council":
                    proposal_specs = [
                        {
                            "agent_name": "validation",
                            "critical": True,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                        {
                            "agent_name": "refinement",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                        {
                            "agent_name": "executor",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                    ]
                else:
                    proposal_specs = [
                        {
                            "agent_name": "planner",
                            "critical": True,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                        {
                            "agent_name": "executor",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.propose(state=local_state, round_id=round_id),
                        },
                    ]
                proposal_results, proposal_failures = self._run_roles_parallel(
                    working_state,
                    specs=proposal_specs,
                    phase="proposal",
                    council_mode=council_mode,
                    reason=reason,
                    reused_roles=reused_roles,
                )
                critical_proposal_failure = next((item for item in proposal_failures if item.critical), None)
                for exc in proposal_failures:
                    if exc.critical:
                        continue
                    _omit_role(exc)
                for item in proposal_results:
                    agent_name = str(item["agent_name"])
                    critical = bool(item["critical"])
                    proposals.extend(list(item.get("output") or []))
                    _record_role(agent_name, phase="proposal", critical=critical, reused=bool(item.get("reused")))
                if critical_proposal_failure is not None:
                    raise critical_proposal_failure
                if self.recovery.last_failure_diagnosis:
                    working_state["diagnostics"]["recovery_diagnosis"] = dict(self.recovery.last_failure_diagnosis)

                reviewer_specs: list[dict[str, Any]] = [
                    {
                        "agent_name": "physics_judge",
                        "critical": True,
                        "invoke": lambda agent, local_state: agent.review(
                            state=local_state,
                            proposals=proposals,
                            round_id=round_id,
                        ),
                    }
                ]
                if self._requires_cost_guardian(council_mode, proposals):
                    reviewer_specs.append(
                        {
                            "agent_name": "cost_guardian",
                            "critical": False,
                            "invoke": lambda agent, local_state: agent.review(
                                state=local_state,
                                proposals=proposals,
                                round_id=round_id,
                            ),
                        }
                    )
                reviewer_results, reviewer_failures = self._run_reviewer_roles(
                    working_state,
                    reviewer_specs=reviewer_specs,
                    round_id=round_id,
                    council_mode=council_mode,
                    reason=reason,
                    proposals=proposals,
                    reused_roles=reused_roles,
                )
                critical_reviewer_failure = next((item for item in reviewer_failures if item.critical), None)
                for exc in reviewer_failures:
                    if exc.critical:
                        continue
                    _omit_role(exc)
                for item in reviewer_results:
                    agent_name = str(item["agent_name"])
                    critical = bool(item["critical"])
                    agent_critiques, agent_preferences = item.get("output") or ([], [])
                    critiques.extend(agent_critiques)
                    preferences.extend(agent_preferences)
                    _record_role(agent_name, phase="review", critical=critical, reused=bool(item.get("reused")))
                if critical_reviewer_failure is not None:
                    raise critical_reviewer_failure

                if self._requires_critic(council_mode, proposals, critiques):
                    try:
                        agent_critiques, agent_preferences = self._run_role(
                            working_state,
                            agent_name="critic",
                            phase="review",
                            council_mode=council_mode,
                            reason=reason,
                            critical=False,
                            fn=lambda: self.critic.review(state=working_state, proposals=proposals, round_id=round_id),
                            proposals=proposals,
                            reused_roles=reused_roles,
                        )
                        critiques.extend(agent_critiques)
                        preferences.extend(agent_preferences)
                        _record_role("critic", phase="review", critical=False, reused="critic" in reused_roles)
                    except CouncilRoleFailure as exc:
                        _omit_role(exc)

                arbitration = self._run_role(
                    working_state,
                    agent_name="orchestrator",
                    phase="arbitration",
                    council_mode=council_mode,
                    reason=reason,
                    critical=True,
                    fn=lambda: self.orchestrator.arbitrate(
                        state=working_state,
                        proposals=proposals,
                        critiques=critiques,
                        preferences=preferences,
                        round_id=round_id,
                    ),
                    proposals=proposals,
                    critiques=critiques,
                    preferences=preferences,
                    reused_roles=reused_roles,
                )
                _record_role("orchestrator", phase="arbitration", critical=True, reused="orchestrator" in reused_roles)
                cache_store = dict((working_state.get("services", {}) or {}).get("council_output_cache", {}) or {})
                state = working_state
                break
            except CouncilRoleFailure as exc:
                cache_store = dict((working_state.get("services", {}) or {}).get("council_output_cache", {}) or {})
                if council_attempt < 2 and exc.critical:
                    emit_progress(
                        "critical council role failed; reopening the same council with cached successful outputs",
                        workdir=state["execution"].get("workdir"),
                        channel="runtime",
                        details={"agent": exc.agent_name, "phase": exc.phase, "round_id": round_id, "error": f"{type(exc.error).__name__}:{exc.error}"},
                    )
                    self._apply_council_reopen_cooldown_if_needed(
                        state=state,
                        round_id=round_id,
                        exc=exc,
                    )
                    continue
                raise RuntimeError(
                    f"agentic_council_role_failed:{exc.agent_name}:{type(exc.error).__name__}:{exc.error}"
                ) from exc
        if arbitration is None:
            raise RuntimeError(f"agentic_no_arbitration_record:{reason}")
        arbitration.content = {
            **dict(arbitration.content or {}),
            "council_mode": council_mode,
            "omitted_roles": omitted_roles,
            "reused_roles": dedupe_keep_order(reused_roles),
            "role_metrics": role_metrics,
        }
        if (
            arbitration.selected_action is None
            and council_mode == "failure_council"
            and has_relax_failure_signature(
                stage=str(reason.split(":", 1)[1] if ":" in reason else ""),
                latest_failure=dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {}),
                state_payload=state,
            )
        ):
            arbitration.selected_action = SelectedAction(
                action_family="escalate_human",
                target_capability=None,
                parameters={
                    "recommended_options": [
                        "manual_fix_resume",
                        "retry_current_stage",
                        "rerun_previous_stage",
                        "skip_material",
                        "abort_task",
                    ]
                },
                source_proposal_id="deterministic::relax_failure::escalate_human",
                rationale=(
                    "Relaxation failure evidence was found after failure-council no-op; "
                    "user policy requires human intervention for relax failures."
                ),
                cost_class="low",
                risk_class="low",
                expected_observation="human_escalation_payload is written and notification is sent",
                success_criteria=["human escalation payload is available", "notification backend is invoked"],
                fallback_if_failed=["abort_material"],
            )
            arbitration.selected_proposal_id = "deterministic::relax_failure::escalate_human"
            arbitration.whether_noop = False
            arbitration.rationale = arbitration.rationale or arbitration.selected_action.rationale
            arbitration.guardrail_notes = dedupe_keep_order(
                list(arbitration.guardrail_notes or []) + ["deterministic_relax_failure_escalation"]
            )
        state = self._record_council_outputs(
            state,
            proposals=proposals,
            critiques=critiques,
            preferences=preferences,
            arbitration=arbitration,
        )
        approved_agents = self._approved_agent_names(proposals, critiques, preferences, arbitration)
        if arbitration.selected_action is None:
            state["execution"]["current_action"] = {}
            state["execution"]["action_status"] = "noop"
            state["workflow"]["next_action"] = None
            state["services"]["selected_action_requires_execution"] = False
            state = _append_framework_diagnostic(
                state,
                code="agentic_no_selected_action_after_arbitration",
                detail={
                    "reason": reason,
                    "council_mode": council_mode,
                    "whether_noop": bool(arbitration.whether_noop),
                    "whether_waiting_external": bool(arbitration.whether_waiting_external),
                    "whether_ready_to_finalize": bool(arbitration.whether_ready_to_finalize),
                    "guardrail_notes": list(arbitration.guardrail_notes or []),
                    "rationale": str(arbitration.rationale or ""),
                },
            )
            if arbitration.whether_waiting_external:
                state["workflow"]["run_status"] = "waiting_external"
                if not str((state.get("workflow", {}) or {}).get("wait_reason") or ""):
                    state["workflow"]["wait_reason"] = "awaiting_external_event:arbitration_noop"
            elif arbitration.whether_ready_to_finalize:
                state["workflow"]["run_status"] = "ready_to_finalize"
                state["workflow"]["wait_reason"] = None
            elif reason in {"validation_followup", "post_mobility_quality_review"} or reason.startswith("anomaly:validation"):
                if _ensure_validation_finalize_ready(state):
                    state["workflow"]["run_status"] = "ready_to_finalize"
                    state["workflow"]["wait_reason"] = None
                    state["workflow"]["termination_reason"] = "validation_finalized_without_followup_action"
                else:
                    state["services"]["termination_requested"] = True
                    state["workflow"]["termination_reason"] = "validation_followup_no_viable_action"
                    state["diagnostics"]["last_error"] = f"validation_followup_no_viable_action:{reason}"
                    state["workflow"]["run_status"] = "skipped"
            else:
                state["services"]["termination_requested"] = True
                state["workflow"]["run_status"] = "failed"
                state["workflow"]["termination_reason"] = "agentic_no_viable_action_after_arbitration"
                state["diagnostics"]["last_error"] = f"agentic_no_viable_action_after_arbitration:{reason}"
            terminal_like_statuses = set(TERMINAL_RUN_STATUSES) | set(WAITING_RUN_STATUSES) | {"ready_to_finalize"}
            checkpoint_reason = (
                None
                if str((state.get("workflow", {}) or {}).get("run_status") or "") in terminal_like_statuses
                else "missing_selected_action"
            )
            contract = self._workflow_contract(state)
            state = self._sync_execution_checkpoint(
                state,
                contract=contract,
                current_capability=None,
                next_capability=None,
                needs_deliberation=bool(checkpoint_reason),
                deliberation_reason=checkpoint_reason,
            )
            state = self._append_decision_ledger(
                state,
                entry_type="arbitration_noop",
                reason=reason,
                round_id=round_id,
                contract=contract,
                agent_names=approved_agents,
                selected_action={},
                summary={
                    "council_mode": council_mode,
                    "omitted_roles": omitted_roles,
                    "reused_roles": dedupe_keep_order(reused_roles),
                    "guardrail_notes": list(arbitration.guardrail_notes or []),
                    "whether_noop": bool(arbitration.whether_noop),
                    "whether_waiting_external": bool(arbitration.whether_waiting_external),
                    "whether_ready_to_finalize": bool(arbitration.whether_ready_to_finalize),
                },
            )
            return self._persist_state(
                state,
                event_type="arbitration_noop",
                extra={
                    "deliberation_reason": reason,
                    "council_mode": council_mode,
                    "run_status": str((state.get("workflow", {}) or {}).get("run_status") or ""),
                },
            )
        selected_action_dict = _selected_action_dict(arbitration.selected_action)
        state["execution"]["current_action"] = selected_action_dict
        state["execution"]["action_status"] = "selected" if selected_action_dict else "noop"
        state["workflow"]["next_action"] = str(selected_action_dict.get("action_family") or "") or None
        state["services"]["selected_action_requires_execution"] = bool(selected_action_dict)
        contract = self._build_contract(
            state,
            reason=reason,
            proposals=proposals,
            critiques=critiques,
            preferences=preferences,
            arbitration=arbitration,
        )
        state = self._update_contract_state(state, contract)
        state = self._sync_execution_checkpoint(
            state,
            contract=contract,
            current_capability=str(selected_action_dict.get("target_capability") or "") or None,
            next_capability=str(selected_action_dict.get("target_capability") or next_capability_after(list(state.get("workflow", {}).get("completed_stages", []) or []), contract.planned_capabilities) or "") or None,
            needs_deliberation=False,
            deliberation_reason=None,
        )
        state = self._append_decision_ledger(
            state,
            entry_type="workflow_contract_updated",
            reason=reason,
            round_id=round_id,
            contract=contract,
            agent_names=approved_agents,
            selected_action=selected_action_dict,
            summary={
                "council_mode": council_mode,
                "planned_capabilities": list(contract.planned_capabilities),
                "milestones": list(contract.milestones),
                "current_focus": contract.current_focus,
                "omitted_roles": omitted_roles,
                "reused_roles": dedupe_keep_order(reused_roles),
            },
        )
        services = dict(state.get("services", {}) or {})
        services["council_round_metrics"] = dedupe_keep_order(
            list(services.get("council_round_metrics", []) or [])
            + [{
                "round_id": round_id,
                "council_mode": council_mode,
                "reason": reason,
                "role_metrics": role_metrics,
                "omitted_roles": omitted_roles,
                "reused_roles": dedupe_keep_order(reused_roles),
                "proposal_count": len(proposals),
                "critique_count": len(critiques),
                "preference_count": len(preferences),
            }]
        )
        state["services"] = services
        state = self._persist_state(
            state,
            event_type="workflow_contract_updated",
            extra={
                "contract_id": contract.contract_id,
                "contract_version": contract.version,
                "deliberation_reason": reason,
                "council_mode": council_mode,
                "planned_capabilities": list(contract.planned_capabilities),
                "selected_capability": selected_action_dict.get("target_capability"),
            },
        )
        return state

    def _activate_contract_action(self, state: dict[str, Any]) -> dict[str, Any]:
        contract = self._workflow_contract(state)
        checkpoint = self._execution_checkpoint(state)
        if contract is None:
            return state
        next_capability = next_capability_after(
            checkpoint.completed_capabilities or list((state.get("workflow", {}) or {}).get("completed_stages", []) or []),
            contract.planned_capabilities,
        )
        if next_capability is None:
            state["execution"]["current_action"] = {}
            state["execution"]["action_status"] = "waiting_council"
            state["workflow"]["next_action"] = None
            state["services"]["selected_action_requires_execution"] = False
            state = self._sync_execution_checkpoint(
                state,
                contract=contract,
                current_capability=None,
                next_capability=None,
                needs_deliberation=True,
                deliberation_reason="contract_completed",
            )
            return self._persist_state(
                state,
                event_type="contract_boundary_waiting_for_council",
                extra={
                    "contract_id": contract.contract_id,
                    "contract_version": contract.version,
                    "deliberation_reason": "contract_completed",
                },
            )
        selected = SelectedAction(
            action_family="run_capability",
            target_capability=next_capability,
            selected_skill="single_material_mobility",
            rationale=f"workflow_contract_v{contract.version}_continue",
            cost_class="medium",
            risk_class="medium",
            fallback_if_failed=list(contract.allowed_branches or []),
            supporting_agent_opinions=[f"workflow_contract:{contract.contract_id}:{contract.version}"],
        )
        state["execution"]["current_action"] = selected.model_dump(mode="json")
        state["execution"]["action_status"] = "selected"
        state["workflow"]["next_action"] = selected.action_family
        state["services"]["selected_action_requires_execution"] = True
        state = self._sync_execution_checkpoint(
            state,
            contract=contract,
            current_capability=selected.target_capability,
            next_capability=selected.target_capability,
            needs_deliberation=False,
            deliberation_reason=None,
        )
        state = self._append_decision_ledger(
            state,
            entry_type="contract_action_activated",
            reason="follow_active_contract",
            round_id=int((state.get("deliberation", {}) or {}).get("round_index", 0) or 0),
            contract=contract,
            agent_names=list(contract.approved_by_agents or []),
            selected_action=selected.model_dump(mode="json"),
            summary={"next_capability": selected.target_capability},
        )
        return self._persist_state(
            state,
            event_type="contract_action_activated",
            extra={
                "contract_id": contract.contract_id,
                "contract_version": contract.version,
                "selected_capability": selected.target_capability,
                "selected_action_family": selected.action_family,
            },
        )

    def _execute_escalation_action(self, state: dict[str, Any]) -> dict[str, Any]:
        selected_action = dict((state.get("execution", {}) or {}).get("current_action", {}) or {})
        round_id = int((state.get("deliberation", {}) or {}).get("round_index", 0) or 0)
        target_capability = str(selected_action.get("target_capability") or "") or None
        risk_class = str(selected_action.get("risk_class") or "medium").strip().lower() or "medium"
        timeout_attempts = self._count_timeout_auto_attempts(state, stage=target_capability)
        default_timeout_action = self._timeout_default_action_for_risk(
            risk_class=risk_class,
            prior_attempts=timeout_attempts,
        )
        should_notify = _should_notify_human_for_escalation(state, target_capability=target_capability)
        state["workflow"]["run_status"] = "needs_human"
        payload = build_human_escalation_payload(
            state,
            recommended_options=list(
                selected_action.get("parameters", {}).get("recommended_options", [])
                or ["manual_fix_resume", "retry_current_stage", "rerun_previous_stage", "skip_material", "abort_task"]
            ),
        )
        payload["risk_class"] = risk_class
        payload["human_window_seconds"] = int(self.runtime.agent_runtime.human_review_timeout_seconds or 300)
        payload["default_timeout_action"] = default_timeout_action
        payload["auto_recovery_attempt"] = timeout_attempts + 1
        previous_timeout_patch = None
        for item in reversed(list((state.get("diagnostics", {}) or {}).get("recovery_history", []) or [])):
            if isinstance(item, dict) and str(item.get("origin") or "") == "timeout_auto":
                previous_timeout_patch = dict(item)
                break
        if previous_timeout_patch is not None:
            payload["previous_timeout_patch"] = previous_timeout_patch
        workdir = state["execution"]["workdir"]
        paths = write_escalation_payload(
            workdir=workdir,
            payload=payload,
            checkpoint_subdir=self.runtime.checkpoint_subdir,
        )
        if should_notify:
            notify_result = notify_escalation(payload)
        else:
            notify_result = {
                "sent": False,
                "reason": "non_failure_escalation_suppressed",
            }
        state["diagnostics"]["consultation_trace"] = dedupe_keep_order(
            list(state["diagnostics"].get("consultation_trace", []) or []) + [{"payload": payload, "paths": paths, "notify_result": notify_result}]
        )
        state["services"]["pending_human_payload"] = payload
        decision = resolve_human_decision(payload=payload, runtime=self.runtime)
        effective_action = str(decision.action)
        decision_payload = decision.model_dump(mode="json")
        if decision.source == "timeout_default":
            timeout_record = {
                "origin": "timeout_auto",
                "action": effective_action,
                "target_stage": target_capability,
                "risk_class": risk_class,
                "attempt_index": timeout_attempts + 1,
                "default_timeout_action": default_timeout_action,
                "reason": str(decision.reason or "timeout_default"),
            }
            state["diagnostics"]["recovery_history"] = dedupe_keep_order(
                list(state["diagnostics"].get("recovery_history", []) or []) + [timeout_record]
            )
            if effective_action not in {"skip_material", "abort_task"}:
                effective_action = "skip_material"
                decision_payload["warnings"] = dedupe_keep_order(
                    list(decision_payload.get("warnings", []) or []) + ["timeout_forced_skip_material"]
                )
                decision_payload["effective_action"] = effective_action
        state["services"]["latest_human_decision"] = decision_payload
        state["services"]["pending_human_payload"] = {}
        state["workflow"]["escalated_to_human"] = True
        state["workflow"]["wait_reason"] = None
        if effective_action == "complete_material":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "completed"
            state["workflow"]["termination_reason"] = "timeout_auto_completed_after_mobility_results"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family="escalate_human",
                target_capability=target_capability,
                status="success",
                result_summary={"human_action": "complete_material"},
            )
        elif effective_action == "skip_material":
            state["services"]["termination_requested"] = True
            if _has_completed_mobility_without_runtime_errors(state):
                state["workflow"]["run_status"] = "skipped"
                state["workflow"]["termination_reason"] = "skip_material_with_computed_results"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family="escalate_human",
                    target_capability=target_capability,
                    status="skipped",
                    result_summary={"human_action": "skip_material", "computed_results_preserved": True},
                )
            else:
                state["workflow"]["run_status"] = "skipped"
                state["workflow"]["termination_reason"] = "skip_material"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family="escalate_human",
                    target_capability=target_capability,
                    status="skipped",
                    result_summary={"human_action": "skip_material"},
                )
        elif effective_action == "abort_task":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "failed"
            state["workflow"]["termination_reason"] = "abort_task"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family="escalate_human",
                target_capability=target_capability,
                status="failed",
                error_summary="abort_task",
                result_summary={"human_action": "abort_task"},
            )
        else:
            resume_stage = target_capability or str(state["workflow"].get("current_stage") or "")
            if effective_action == "manual_fix_resume" and decision.instruction is not None:
                instruction = ManualFixInstruction.model_validate(decision.instruction.model_dump(mode="json"))
                cleanup = preview_cleanup(
                    workdir=state["execution"]["workdir"],
                    resume_stage=instruction.resume_stage,
                    cleanup_policy=instruction.cleanup_policy,
                )
                apply_cleanup(workdir=state["execution"]["workdir"], preview=cleanup)
                resume_stage = instruction.resume_stage
                state["diagnostics"]["recovery_history"] = dedupe_keep_order(
                    list(state["diagnostics"].get("recovery_history", []) or [])
                    + [{"action": "manual_fix_resume", "target_stage": resume_stage, "preview": cleanup.model_dump(mode="json")}]
                )
            elif effective_action == "rerun_previous_stage":
                resume_stage = find_previous_stage(target_capability or "") or (target_capability or "")
            state = _apply_resume_strategy(state, resume_stage, reason=effective_action)
            state["workflow"]["run_status"] = "running"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family="escalate_human",
                target_capability=resume_stage,
                status="success",
                result_summary={"human_action": effective_action, "resume_stage": resume_stage},
            )
        state["execution"]["latest_execution_observation"] = observation.model_dump(mode="json")
        state["execution"]["action_status"] = observation.status
        _append_decision_trace(state, {"node": "agentic_controller", "message_type": "execution_observation", **observation.model_dump(mode="json")})
        return self._persist_state(
            state,
            event_type="human_escalation_resolved",
            extra={
                "human_action": effective_action,
                "selected_capability": target_capability,
                "run_status": state["workflow"].get("run_status"),
            },
        )

    def _update_checkpoint_after_execution(self, state: dict[str, Any]) -> dict[str, Any]:
        contract = self._workflow_contract(state)
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        current_capability = str(latest.get("target_capability") or "") or None
        next_capability = None
        if contract is not None:
            next_capability = next_capability_after(
                list((state.get("workflow", {}) or {}).get("completed_stages", []) or []),
                list(contract.planned_capabilities or []),
            )
        reason = self._deliberation_reason(state, contract)
        state = self._sync_execution_checkpoint(
            state,
            contract=contract,
            current_capability=current_capability,
            next_capability=next_capability,
            needs_deliberation=bool(reason),
            deliberation_reason=reason,
        )
        services = dict(state.get("services", {}) or {})
        checkpoint_payload = dict((state.get("execution", {}) or {}).get("execution_checkpoint", {}) or {})
        checkpoint_entry = {
            "timestamp": utc_now_iso(),
            "capability": current_capability,
            "next_capability": next_capability,
            "run_status": str((state.get("workflow", {}) or {}).get("run_status") or ""),
            "execution_status": str(latest.get("status") or ""),
            "deliberation_reason": reason,
            "checkpoint": checkpoint_payload,
            "resume_options": {
                "from_last_success": True,
                "from_specified_step": True,
                "from_human_confirmation": True,
            },
        }
        services["step_checkpoints"] = dedupe_keep_order(
            list(services.get("step_checkpoints", []) or []) + [checkpoint_entry]
        )
        state["services"] = services
        return state

    def _execute_current_action(self, state: dict[str, Any]) -> dict[str, Any]:
        selected_action = dict((state.get("execution", {}) or {}).get("current_action", {}) or {})
        if not selected_action:
            return state
        before_retry_counts = _retry_counts_snapshot(state)
        if str(selected_action.get("action_family") or "") == "escalate_human":
            state = self._execute_escalation_action(state)
        else:
            state = apply_state_patch(state, self.execute_node(state))
            state = _ensure_selected_retry_accounted_after_execution(
                state,
                selected_action=selected_action,
                before_retry_counts=before_retry_counts,
            )
            state = self._persist_state(
                state,
                event_type="selected_action_executed",
                extra={
                    "selected_action_family": selected_action.get("action_family"),
                    "selected_capability": selected_action.get("target_capability"),
                    "execution_status": dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {}).get("status"),
                },
            )
        state = apply_state_patch(state, self.reflect_node(state))
        state = self._update_checkpoint_after_execution(state)
        return self._persist_state(
            state,
            event_type="execution_reflected",
            extra={
                "selected_capability": dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {}).get("target_capability"),
                "execution_status": dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {}).get("status"),
                "run_status": state.get("workflow", {}).get("run_status"),
            },
        )

    def _finalize_if_needed(self, state: dict[str, Any]) -> dict[str, Any]:
        final_report_status = str(((state.get("workflow", {}) or {}).get("stage_status", {}) or {}).get("final_report") or "")
        if final_report_status == "success":
            return state
        if not _terminal_or_finalizable(state):
            return state
        contract = self._workflow_contract(state)
        if contract is not None:
            compute_status = derive_compute_status_from_state_payload(state)
            if compute_status == "failed":
                contract.plan_status = "aborted"
            else:
                contract.plan_status = "completed"
            state = self._update_contract_state(state, contract)
        state = apply_state_patch(state, self.final_report_node(state))
        contract = self._workflow_contract(state)
        if contract is not None:
            contract.plan_status = "completed" if derive_compute_status_from_state_payload(state) == "completed" else "aborted"
            state = self._update_contract_state(state, contract)
        state = self._sync_execution_checkpoint(
            state,
            contract=contract,
            current_capability=None,
            next_capability=None,
            needs_deliberation=False,
            deliberation_reason=None,
        )
        return self._persist_state(
            state,
            event_type="final_report_written",
            extra={
                "run_status": state.get("workflow", {}).get("run_status"),
                "material_id": state.get("material", {}).get("material_id"),
            },
        )

    def _normalize_loaded_state(
        self,
        state: dict[str, Any],
        *,
        material_id: str,
        root_path: str,
        workdir: str,
        poscar_path: str,
        potcar_path: str,
        user_goal: str,
        parent_batch_id: str | None,
    ) -> dict[str, Any]:
        normalized = _state_dict(state)
        normalized["task"]["task_type"] = "single_material"
        normalized["task"]["root_path"] = os.path.abspath(root_path)
        normalized["task"]["user_goal"] = user_goal
        normalized["task"]["parent_batch_id"] = parent_batch_id
        normalized["task"]["dry_run"] = bool(self.runtime.dry_run)
        normalized["material"]["material_id"] = material_id
        normalized["material"]["poscar_path"] = os.path.abspath(poscar_path)
        normalized["material"]["potcar_path"] = os.path.abspath(potcar_path)
        normalized["execution"]["workdir"] = os.path.abspath(workdir)
        normalized["execution"]["compatibility_checkpoint_path"] = os.path.join(os.path.abspath(workdir), "checkpoint.pkl")
        normalized["agent"]["decision_engine"] = self.runtime.agent_runtime.decision_engine.value
        normalized["agent"]["llm_required"] = True
        normalized["agent"]["llm_provider"] = self.runtime.agent_runtime.llm_provider
        normalized["workflow"]["max_refinement_rounds"] = int(self.runtime.agent_runtime.max_refinement_rounds)
        mission = dict(normalized.get("mission", {}) or {})
        runtime_constraints = dict(mission.get("runtime_constraints", {}) or {})
        runtime_constraints["dry_run"] = bool(self.runtime.dry_run)
        runtime_constraints["max_refinement_rounds"] = int(self.runtime.agent_runtime.max_refinement_rounds)
        runtime_constraints["full_autonomy"] = bool(self.runtime.full_autonomy)
        runtime_constraints["allow_external_wait"] = bool(self.runtime.allow_external_wait)
        mission["runtime_constraints"] = runtime_constraints
        normalized["mission"] = mission
        normalized["services"]["capability_registry"] = capability_registry_payload()
        normalized["services"]["runtime_strategy"] = {
            "council_policy_mode": str(self.runtime.council_policy_mode or "balanced"),
            "deliberation_mode": "segment_agentic",
        }
        return _state_dict(normalized)

    def _initial_state(
        self,
        *,
        material_id: str,
        root_path: str,
        workdir: str,
        poscar_path: str,
        potcar_path: str,
        user_goal: str,
        parent_batch_id: str | None,
        task_id: str | None,
        thread_id: str,
    ) -> dict[str, Any]:
        state = make_initial_material_state(
            task_id=task_id,
            material_id=material_id,
            root_path=root_path,
            workdir=workdir,
            poscar_path=poscar_path,
            potcar_path=potcar_path,
            user_goal=user_goal,
            decision_engine=self.runtime.agent_runtime.decision_engine.value,
            llm_required=True,
            llm_provider=self.runtime.agent_runtime.llm_provider,
            max_refinement_rounds=self.runtime.agent_runtime.max_refinement_rounds,
            dry_run=self.runtime.dry_run,
            thread_id=thread_id,
        ).to_dict()
        state["task"]["parent_batch_id"] = parent_batch_id
        state["task_board"] = build_initial_task_board()
        state["mission"]["runtime_constraints"] = {
            **dict((state.get("mission", {}) or {}).get("runtime_constraints", {}) or {}),
            "dry_run": bool(self.runtime.dry_run),
            "max_refinement_rounds": int(self.runtime.agent_runtime.max_refinement_rounds),
            "full_autonomy": bool(self.runtime.full_autonomy),
            "allow_external_wait": bool(self.runtime.allow_external_wait),
        }
        state["services"]["capability_registry"] = capability_registry_payload()
        state["services"]["runtime_strategy"] = {
            "council_policy_mode": str(self.runtime.council_policy_mode or "balanced"),
            "deliberation_mode": "segment_agentic",
        }
        return _state_dict(state)

    def _inject_external_event(self, state: dict[str, Any], event: ExternalEventRecord | dict[str, Any]) -> dict[str, Any]:
        normalized_event = normalize_external_event(
            event,
            default_thread_id=str((state.get("execution", {}) or {}).get("thread_id") or ""),
            default_run_id=str((state.get("task", {}) or {}).get("task_id") or ""),
        )
        state["execution"]["pending_events"] = list((state.get("execution", {}) or {}).get("pending_events", []) or []) + [
            normalized_event.model_dump(mode="json")
        ]
        state["execution"]["latest_event"] = {}
        return self._persist_state(
            state,
            event_type="external_event_submitted",
            extra={
                "external_event_id": normalized_event.event_id,
                "external_event_type": normalized_event.event_type,
                "external_job_id": normalized_event.job_id,
            },
        )

    def _recover_unexpected_wait_boundary(
        self,
        state: dict[str, Any],
        *,
        run_status: str,
        event_type: str,
        error_code: str,
    ) -> dict[str, Any]:
        execution = dict(state.get("execution", {}) or {})
        current_action = dict(execution.get("current_action", {}) or {})
        resume_markers = dict(execution.get("resume_markers", {}) or {})
        blocked_job_id = str(resume_markers.get("job_id") or current_action.get("parameters", {}).get("job_id") or "")
        blocked_capability = str(
            resume_markers.get("awaiting_capability") or current_action.get("target_capability") or ""
        )
        removed_jobs: list[str] = []
        cleaned_jobs: list[dict[str, Any]] = []
        for item in list(execution.get("external_jobs", []) or []):
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("job_id") or "")
            capability = str(item.get("target_capability") or "")
            if (blocked_job_id and job_id == blocked_job_id) or (blocked_capability and capability == blocked_capability):
                removed_jobs.append(job_id or capability)
                continue
            cleaned_jobs.append(item)
        execution["external_jobs"] = cleaned_jobs
        execution["resume_markers"] = {}
        if current_action:
            execution["current_action"] = {
                **current_action,
                "submit_external_job": False,
                "wait_for_event_after_submission": False,
                "unexpected_wait_boundary_recovered": True,
            }
        latest_observation = dict(execution.get("latest_execution_observation", {}) or {})
        if latest_observation and str(latest_observation.get("status") or "") == "running":
            latest_raw_evidence = dict(latest_observation.get("raw_evidence", {}) or {})
            if removed_jobs:
                latest_raw_evidence["recovered_external_jobs"] = removed_jobs
            latest_observation["status"] = "failed"
            latest_observation["error_summary"] = f"{error_code}:{run_status}"
            latest_observation["raw_evidence"] = latest_raw_evidence
            execution["latest_execution_observation"] = latest_observation
            execution["action_status"] = "failed"
            state["blackboard"]["latest_execution_observation"] = latest_observation
        state["execution"] = execution
        state = _append_framework_diagnostic(
            state,
            code=error_code,
            detail={
                "run_status": run_status,
                "removed_external_jobs": removed_jobs,
                "recovered_target_capability": blocked_capability or None,
            },
        )
        state["workflow"]["run_status"] = "needs_recovery"
        state["workflow"]["wait_reason"] = None
        state["diagnostics"]["last_error"] = f"{error_code}:{run_status}"
        return self._persist_state(
            state,
            event_type=event_type,
            extra={
                "previous_run_status": run_status,
                "removed_external_jobs": removed_jobs,
                "recovered_target_capability": blocked_capability or None,
            },
        )

    def _handle_external_event_resume_guard(
        self,
        state: dict[str, Any],
        event: ExternalEventRecord | dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        if event is None:
            return state, False
        workflow = dict(state.get("workflow", {}) or {})
        normalized_event = normalize_external_event(
            event,
            default_thread_id=str((state.get("execution", {}) or {}).get("thread_id") or ""),
            default_run_id=str((state.get("task", {}) or {}).get("task_id") or ""),
        )
        if str(workflow.get("run_status") or "") in TERMINAL_RUN_STATUSES:
            state = _append_framework_diagnostic(
                state,
                code="external_event_resume_refused_terminal_state",
                detail={
                    "event_id": normalized_event.event_id,
                    "event_type": normalized_event.event_type,
                    "run_status": workflow.get("run_status"),
                },
            )
            return self._persist_state(
                state,
                event_type="external_event_rejected",
                extra={
                    "event_id": normalized_event.event_id,
                    "event_type": normalized_event.event_type,
                    "reason": "terminal_state",
                },
            ), True
        consumed_ids = {
            str(item or "").strip()
            for item in list((state.get("execution", {}) or {}).get("consumed_event_ids", []) or [])
            if str(item or "").strip()
        }
        if normalized_event.event_id and normalized_event.event_id in consumed_ids:
            state = _append_framework_diagnostic(
                state,
                code="duplicate_external_event_ignored",
                detail={
                    "event_id": normalized_event.event_id,
                    "event_type": normalized_event.event_type,
                },
            )
            return self._persist_state(
                state,
                event_type="external_event_ignored",
                extra={
                    "event_id": normalized_event.event_id,
                    "event_type": normalized_event.event_type,
                    "reason": "duplicate",
                },
            ), True
        return state, False

    def _sync_external_event_progress(self, state: dict[str, Any]) -> dict[str, Any]:
        latest_event = dict((state.get("execution", {}) or {}).get("latest_event", {}) or {})
        latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        workflow = dict(state.get("workflow", {}) or {})
        stage_status = dict(workflow.get("stage_status", {}) or {})
        completed = list(workflow.get("completed_stages", []) or [])
        target_capability = str(
            latest_event.get("target_capability")
            or latest_observation.get("target_capability")
            or ""
        ).strip()
        event_type = str(latest_event.get("event_type") or "").strip()
        if target_capability and event_type == "job_completed":
            workflow["completed_stages"] = dedupe_keep_order(completed + [target_capability])
            stage_status[target_capability] = "success"
        elif target_capability and event_type in {"job_failed", "job_timeout", "artifact_missing"}:
            stage_status[target_capability] = "failed"
        workflow["stage_status"] = stage_status
        state["workflow"] = workflow
        return state

    def drive(
        self,
        *,
        material_id: str,
        root_path: str,
        workdir: str,
        poscar_path: str,
        potcar_path: str,
        user_goal: str,
        parent_batch_id: str | None = None,
        fresh: bool = False,
        thread_id: str | None = None,
        external_event: ExternalEventRecord | dict[str, Any] | None = None,
    ) -> MaterialRunOutcome:
        root_abs = os.path.abspath(root_path)
        workdir_abs = os.path.abspath(workdir)
        poscar_abs = os.path.abspath(poscar_path)
        potcar_abs = os.path.abspath(potcar_path)
        os.makedirs(workdir_abs, exist_ok=True)
        saved_state = None if fresh else load_state_snapshot(workdir=workdir_abs, checkpoint_subdir=self.runtime.checkpoint_subdir)
        resolved_thread_id = thread_id or str((saved_state or {}).get("execution", {}).get("thread_id") or "") or build_material_thread_id(
            task_id=str((saved_state or {}).get("task", {}).get("task_id") or material_id),
            material_id=material_id,
        )
        if fresh:
            remove_checkpoints(
                workdir=workdir_abs,
                checkpoint_subdir=self.runtime.checkpoint_subdir,
                database_uri=self.runtime.resolved_db_uri,
                thread_id=resolved_thread_id,
            )
        has_durable_checkpoint = False if fresh else langgraph_checkpoint_exists(
            database_uri=self.runtime.resolved_db_uri,
            thread_id=resolved_thread_id,
        )
        if saved_state and not fresh and not has_durable_checkpoint:
            raise CheckpointRestoreError(
                f"langgraph_checkpoint_restore_failed:thread_id={resolved_thread_id}:workdir={workdir_abs}"
            )
        try:
            self._open_compatibility_app()
            durable_state = None if fresh or not has_durable_checkpoint else self._load_durable_state(resolved_thread_id)
            resume_state = durable_state or saved_state
            if resume_state and not fresh:
                state = self._normalize_loaded_state(
                    resume_state,
                    material_id=material_id,
                    root_path=root_abs,
                    workdir=workdir_abs,
                    poscar_path=poscar_abs,
                    potcar_path=potcar_abs,
                    user_goal=user_goal,
                    parent_batch_id=parent_batch_id,
                )
                emit_progress(
                    "resuming agent-first material runtime",
                    workdir=workdir_abs,
                    details={
                        "material_id": material_id,
                        "thread_id": resolved_thread_id,
                        "run_status": dict(state.get("workflow", {}) or {}).get("run_status"),
                        "current_stage": dict(state.get("workflow", {}) or {}).get("current_stage"),
                    },
                )
            else:
                state = self._initial_state(
                    material_id=material_id,
                    root_path=root_abs,
                    workdir=workdir_abs,
                    poscar_path=poscar_abs,
                    potcar_path=potcar_abs,
                    user_goal=user_goal,
                    parent_batch_id=parent_batch_id,
                    task_id=str((saved_state or {}).get("task", {}).get("task_id") or "") or None,
                    thread_id=resolved_thread_id,
                )
                emit_progress(
                    "starting agent-first material runtime",
                    workdir=workdir_abs,
                    details={
                        "material_id": material_id,
                        "thread_id": resolved_thread_id,
                        "fresh": fresh,
                        "dry_run": self.runtime.dry_run,
                    },
                )
            state["execution"]["thread_id"] = resolved_thread_id
            save_thread_id(workdir=workdir_abs, thread_id=resolved_thread_id, checkpoint_subdir=self.runtime.checkpoint_subdir)
            save_checkpoint_metadata(
                workdir=workdir_abs,
                thread_id=resolved_thread_id,
                database_uri=self.runtime.resolved_db_uri,
                checkpoint_subdir=self.runtime.checkpoint_subdir,
            )
            state = self._persist_state(
                state,
                event_type="agentic_runtime_started",
                extra={"thread_id": resolved_thread_id, "material_id": material_id},
            )
            state, should_return = self._handle_external_event_resume_guard(state, external_event)
            if should_return:
                return build_material_outcome(state)
            if external_event is not None:
                state = self._inject_external_event(state, external_event)

            while True:
                state = _state_dict(state)
                workflow = dict(state.get("workflow", {}) or {})
                execution = dict(state.get("execution", {}) or {})
                if list(execution.get("pending_events", []) or []):
                    state = _consume_pending_external_event(state)
                    state = self._sync_external_event_progress(state)
                    latest_event = dict((state.get("execution", {}) or {}).get("latest_event", {}) or {})
                    state = self._persist_state(
                        state,
                        event_type="external_event_consumed",
                        extra={
                            "event_id": latest_event.get("event_id"),
                            "event_type": latest_event.get("event_type"),
                            "job_id": latest_event.get("job_id"),
                            "run_status": dict((state.get("workflow", {}) or {})).get("run_status"),
                        },
                    )
                    workflow = dict(state.get("workflow", {}) or {})
                    execution = dict(state.get("execution", {}) or {})
                run_status = str(workflow.get("run_status") or "pending")
                if run_status in TERMINAL_RUN_STATUSES:
                    break
                if run_status in WAITING_RUN_STATUSES and not list(execution.get("pending_events", []) or []):
                    if self.runtime.full_autonomy:
                        state = self._recover_unexpected_wait_boundary(
                            state,
                            run_status=run_status,
                            error_code="unexpected_wait_boundary_in_full_autonomy",
                            event_type="full_autonomy_wait_boundary_recovered",
                        )
                        continue
                    break
                contract = self._workflow_contract(state)
                deliberation_reason = self._deliberation_reason(state, contract)
                if deliberation_reason:
                    state = self._deliberate(state, reason=deliberation_reason)
                else:
                    state = self._activate_contract_action(state)
                selected_action = dict((state.get("execution", {}) or {}).get("current_action", {}) or {})
                if not selected_action:
                    run_status = str((state.get("workflow", {}) or {}).get("run_status") or "")
                    action_status = str((state.get("execution", {}) or {}).get("action_status") or "")
                    if run_status in TERMINAL_RUN_STATUSES or run_status == "ready_to_finalize":
                        break
                    if run_status in WAITING_RUN_STATUSES:
                        if self.runtime.full_autonomy:
                            state = self._recover_unexpected_wait_boundary(
                                state,
                                run_status=run_status,
                                error_code="unexpected_wait_boundary_in_full_autonomy",
                                event_type="full_autonomy_wait_boundary_recovered",
                            )
                            continue
                        break
                    checkpoint_reason = "contract_completed" if action_status == "waiting_council" else "missing_selected_action"
                    state = self._sync_execution_checkpoint(
                        state,
                        contract=self._workflow_contract(state),
                        current_capability=None,
                        next_capability=None,
                        needs_deliberation=True,
                        deliberation_reason=checkpoint_reason,
                    )
                    if action_status == "waiting_council":
                        continue
                    state["services"]["termination_requested"] = True
                    state["workflow"]["run_status"] = "failed"
                    state["workflow"]["termination_reason"] = "missing_selected_action"
                    state["diagnostics"]["last_error"] = "missing_selected_action_after_deliberation"
                    break
                state = self._execute_current_action(state)
                if _terminal_or_finalizable(state):
                    break
                execution = dict(state.get("execution", {}) or {})
                if str((state.get("workflow", {}) or {}).get("run_status") or "") in WAITING_RUN_STATUSES and not list(execution.get("pending_events", []) or []):
                    if self.runtime.full_autonomy:
                        run_status = str((state.get("workflow", {}) or {}).get("run_status") or "")
                        state = self._recover_unexpected_wait_boundary(
                            state,
                            run_status=run_status,
                            error_code="post_execution_wait_boundary_in_full_autonomy",
                            event_type="full_autonomy_post_execution_wait_recovered",
                        )
                        continue
                    break

            state = self._finalize_if_needed(state)
            return build_material_outcome(state)
        finally:
            self._close_compatibility_app()


def run_agentic_material(
    *,
    runtime: RuntimeContext,
    material_id: str,
    root_path: str,
    workdir: str,
    poscar_path: str,
    potcar_path: str,
    user_goal: str,
    parent_batch_id: str | None = None,
    fresh: bool = False,
    thread_id: str | None = None,
) -> MaterialRunOutcome:
    controller = AgenticMaterialController(runtime)
    return controller.drive(
        material_id=material_id,
        root_path=root_path,
        workdir=workdir,
        poscar_path=poscar_path,
        potcar_path=potcar_path,
        user_goal=user_goal,
        parent_batch_id=parent_batch_id,
        fresh=fresh,
        thread_id=thread_id,
    )


def run_agentic_material_external_event(
    *,
    runtime: RuntimeContext,
    workdir: str,
    event: ExternalEventRecord | dict[str, Any],
    thread_id: str | None = None,
) -> MaterialRunOutcome:
    saved_state = load_state_snapshot(workdir=workdir, checkpoint_subdir=runtime.checkpoint_subdir)
    if not saved_state:
        raise RuntimeError(f"missing_state_snapshot_for_external_event:workdir={os.path.abspath(workdir)}")
    controller = AgenticMaterialController(runtime)
    state = _state_dict(saved_state)
    return controller.drive(
        material_id=str((state.get("material", {}) or {}).get("material_id") or "2D_Material"),
        root_path=str((state.get("task", {}) or {}).get("root_path") or os.path.dirname(os.path.abspath(workdir))),
        workdir=os.path.abspath(workdir),
        poscar_path=str((state.get("material", {}) or {}).get("poscar_path") or os.path.join(os.path.abspath(workdir), "..", "POSCAR")),
        potcar_path=str((state.get("material", {}) or {}).get("potcar_path") or os.path.join(os.path.abspath(workdir), "..", "POTCAR")),
        user_goal=str((state.get("task", {}) or {}).get("user_goal") or "calculate_2d_mobility"),
        parent_batch_id=str((state.get("task", {}) or {}).get("parent_batch_id") or "") or None,
        fresh=False,
        thread_id=thread_id or str((state.get("execution", {}) or {}).get("thread_id") or "") or None,
        external_event=event,
    )
