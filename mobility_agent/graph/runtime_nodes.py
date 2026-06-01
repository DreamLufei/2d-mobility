from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from typing import Any, Callable

from langgraph.graph import END
from langgraph.types import Command, interrupt

from ..agents import ManualFixInstruction
from ..agents.context_engineering import build_llm_context_summary
from ..agents.cost_guardian import CostGuardianAgent
from ..agents.critic import CriticAgent
from ..agents.executor import ExecutorAgent
from ..agents.orchestrator import OrchestratorAgent
from ..agents.physics_judge import PhysicsJudgeAgent
from ..agents.planner import PlannerAgent
from ..agents.refinement import RefinementAgent
from ..agents.recovery import RecoveryAgent
from ..agents.reporter import ReporterAgent
from ..agents.schemas import (
    ArbitrationRecord,
    Critique,
    ExecutionObservation,
    Preference,
    Proposal,
    ReflectionRecord,
    SelectedAction,
)
from ..hitl.cleanup import apply_cleanup, preview_cleanup
from ..hitl.escalation import notify_escalation, write_escalation_payload
from ..hitl.resume import normalize_hitl_decision
from ..policy.engine import has_relax_failure_signature
from ..runtime.store import open_memory_store, record_skill_metadata
from ..runtime.action_registry import capability_sequence
from ..runtime.agent_tools import AgentToolGateway
from ..runtime.checkpointing import save_state_snapshot
from ..runtime.channel_utils import derive_direction_acceptance, subchannel_tokens_from_targets
from ..runtime.context import RuntimeContext
from ..runtime.deliberation_loop import (
    all_tasks_resolved,
    build_round_snapshot,
    build_initial_task_board,
    has_blocked_or_abandoned_tasks,
    mark_task_abandoned,
    mark_task_blocked,
    mark_task_completed,
    mark_task_started,
    next_pending_task,
    reset_from_capability,
)
from ..runtime.refinement_policy import (
    DEFAULT_FIT_R2_THRESHOLD,
    DEFAULT_REFINEMENT_TARGET_POINTS,
    resolve_refinement_sampling,
)
from ..runtime.validation_policy import build_validation_report
from ..policy.engine import AgenticPolicyEngine
from ..skills import discover_skills
from ..tools.anomaly_detector import detect_basic_anomalies
from ..tools.band_tool import BandTool, BandToolInput
from ..tools.errors import StageDependencyError
from ..tools.mass_tool import MassTool, MassToolInput
from ..tools.mobility_tool import MobilityTool, MobilityToolInput
from ..tools.physics_validator import validate_physics_window
from ..tools.relax_tool import RelaxTool, RelaxToolInput
from ..tools.schemas import ToolExecutionResult, ToolFailureEvidence
from ..tools.scf_tool import ScfTool, ScfToolInput
from ..tools.strain_tool import StrainTool, StrainToolInput
from ..tools.vasp_common import classify_vasp_failure_text
from ..utils import dedupe_keep_order, summarize_poscar
from ..runtime.telemetry import emit_progress
from .human_gate import build_human_escalation_payload
from .stage_contracts import find_previous_stage, get_stage_contract
from .state import (
    EXTERNAL_EVENT_TYPES,
    STABLE_CHECKPOINT_STAGES,
    TERMINAL_RUN_STATUSES,
    MaterialTaskState,
    apply_state_updates,
    build_state_patch,
    build_material_outcome,
    clear_resolved_error_state,
    export_compatibility_checkpoint,
    has_completed_compute_state_payload,
    normalize_external_event,
    record_stage_status,
    register_tool_result,
    state_payload_to_dict,
    utc_now_iso,
)


def _state(payload: MaterialTaskState | dict[str, Any]) -> MaterialTaskState:
    if isinstance(payload, MaterialTaskState):
        return payload
    return MaterialTaskState.from_dict(payload)


def _state_dict(payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
    return state_payload_to_dict(payload)


def _finalize_node_output(
    before_payload: MaterialTaskState | dict[str, Any],
    after_payload: dict[str, Any],
    *,
    sections: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, Any]:
    return build_state_patch(before_payload, MaterialTaskState.from_dict(after_payload).to_dict(), sections=sections)


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for token in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(token)
    return current


def _stage_workdir(root: str, stage: str) -> str:
    mapping = {
        "relax": os.path.join(root, "01_relax"),
        "scf": os.path.join(root, "02_scf"),
        "band": os.path.join(root, "03_band"),
        "effective_mass": root,
        "strain_loop": os.path.join(root, "05_strain"),
        "mobility": root,
        "validation": root,
    }
    return mapping.get(stage, root)


def _log_paths_for_stage(root: str, stage: str) -> list[str]:
    stage_dir = _stage_workdir(root, stage)
    candidates = [
        os.path.join(stage_dir, "sout"),
        os.path.join(root, "vasp_relax_retry.log"),
    ]
    return [path for path in candidates if os.path.exists(path)]


def _memory_store_scope(langgraph_runtime: Any, database_uri: str):
    store = getattr(langgraph_runtime, "store", None)
    if store is not None:
        return nullcontext(store)
    return open_memory_store(database_uri)


def _validate_stage_dependencies(state_payload: dict[str, Any], stage: str) -> None:
    missing = []
    for field_path in get_stage_contract(stage).required_inputs:
        value = _nested_get(state_payload, field_path)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(field_path)
    if missing:
        raise StageDependencyError(f"{stage}_missing_inputs:{','.join(missing)}")


def _classify_failure(error_summary: str) -> str:
    value = str(error_summary or "").lower()
    vasp_error_type, _trigger = classify_vasp_failure_text(value)
    if vasp_error_type in {"runner_environment_failure", "chgcar_compatibility_failure"}:
        return vasp_error_type
    if "runner/environment failure" in value or "runner_environment_failure" in value:
        return "runner_environment_failure"
    if "chgcar compatibility failure" in value or "chgcar_compatibility_failure" in value:
        return "chgcar_compatibility_failure"
    if "zbrent" in value:
        return "zbrent_fatal"
    if "contcar" in value and "missing" in value:
        return "missing_output"
    if "nonconverged" in value or "not converged" in value:
        return "nonconverged"
    if "returncode" in value:
        return "nonzero_exit"
    if "missing_input" in value:
        return "missing_input"
    return "unknown_failure"


def _normalize_tool_output(stage: str, output: Any, workdir: str) -> ToolExecutionResult:
    artifact_paths = {str(k): str(v) for k, v in dict(output.artifact_paths or {}).items() if v}
    raw_payload = dict(output.state_updates or {})
    parser_payload = dict(output.key_summary or {})
    stdout_path = os.path.join(_stage_workdir(workdir, stage), "sout")
    evidence = ToolFailureEvidence(
        returncode=raw_payload.get("returncode") or (raw_payload.get("recovery_summary", {}) or {}).get("returncode"),
        stdout_path=(stdout_path if os.path.exists(stdout_path) else None),
        stderr_path=None,
        log_paths=dedupe_keep_order(list(artifact_paths.values()) + _log_paths_for_stage(workdir, stage)),
        parser_payload=parser_payload,
        raw_payload=raw_payload,
    )
    return ToolExecutionResult(
        stage=stage,
        status="success" if bool(output.success) else "failed",
        error_summary=output.error_summary,
        warnings=list(output.warnings or []),
        artifact_paths=artifact_paths,
        key_summary=parser_payload,
        state_updates=raw_payload,
        raw_evidence=evidence,
        invocation_source=str(raw_payload.get("_tool_source") or "native_tool"),
        duration_s=float(output.duration_s or 0.0),
    )


def _dry_run_stage(stage: str, state_payload: dict[str, Any], runtime: RuntimeContext) -> ToolExecutionResult:
    should_fail = stage in set(runtime.dry_run_fail_stages or ())
    workdir = str(state_payload.get("execution", {}).get("workdir") or "")
    if should_fail:
        return ToolExecutionResult(
            stage=stage,
            status="failed",
            error_summary=f"dry_run_injected_failure:{stage}",
            warnings=["dry_run_mode"],
            raw_evidence=ToolFailureEvidence(
                log_paths=_log_paths_for_stage(workdir, stage),
                raw_payload={"dry_run": True, "injected_failure": stage},
            ),
        )
    if stage == "relax":
        state_updates = {
            "relaxed_poscar": state_payload.get("material", {}).get("poscar_path"),
            "reciprocal_lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]],
            "recovery_summary": {},
        }
    elif stage == "scf":
        state_updates = {"fermi_energy": 0.5}
    elif stage == "band":
        state_updates = {
            "vbm_energy": -5.0,
            "cbm_energy": -3.5,
            "vbm_kpoint": [0.0, 0.0, 0.0],
            "cbm_kpoint": [0.5, 0.0, 0.0],
            "vbm_band_index": 5,
            "cbm_band_index": 6,
            "vbm_spin": 0,
            "cbm_spin": 0,
        }
    elif stage == "effective_mass":
        state_updates = {
            "electron_mass_x": 0.35,
            "electron_mass_y": 0.42,
            "electron_mass_dos": 0.38,
            "hole_mass_x": 0.55,
            "hole_mass_y": 0.61,
            "hole_mass_dos": 0.58,
        }
    elif stage == "strain_loop":
        strain_data = []
        for direction in ["x", "y"]:
            for strain in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                strain_data.append(
                    {
                        "direction": direction,
                        "strain": strain,
                        "total_energy": -20.0 + strain * strain,
                        "e_vbm": -5.0 + 0.2 * strain,
                        "e_cbm": -3.5 + 0.15 * strain,
                        "e_vacuum": 0.0,
                        "folder": os.path.join(workdir, "05_strain", direction, f"strain_{strain:+.4f}"),
                        "completed": True,
                    }
                )
        state_updates = {
            "strain_data": strain_data,
            "strain_summary": {"completed_points": len(strain_data), "failed_points": 0, "dry_run": True},
        }
    elif stage == "mobility":
        results_by_direction = {
            "x": {
                "electron": {"mobility_cm2_Vs": 1200.0, "E1_eV": 1.1, "C2D_J_m2": 90.0, "E1_fit_R2": 0.97, "C2D_fit_R2": 0.98},
                "hole": {"mobility_cm2_Vs": 800.0, "E1_eV": 1.3, "C2D_J_m2": 90.0, "E1_fit_R2": 0.96, "C2D_fit_R2": 0.98},
                "elastic_modulus_C2D_J_m2": 90.0,
                "n_points": 5,
            },
            "y": {
                "electron": {"mobility_cm2_Vs": 900.0, "E1_eV": 1.2, "C2D_J_m2": 85.0, "E1_fit_R2": 0.97, "C2D_fit_R2": 0.98},
                "hole": {"mobility_cm2_Vs": 700.0, "E1_eV": 1.4, "C2D_J_m2": 85.0, "E1_fit_R2": 0.96, "C2D_fit_R2": 0.98},
                "elastic_modulus_C2D_J_m2": 85.0,
                "n_points": 5,
            },
        }
        state_updates = {
            "results": {"results_by_direction": results_by_direction},
            "mobility_summary": {"status_label": "completed", "direction_count": 2, "dry_run": True},
            "fit_diagnostics": {
                "fit_r2_min": 0.96,
                "energy_fit_r2_min": 0.98,
                "edge_fit_r2_min": 0.96,
                "effective_fit_quality": 0.96,
                "e1_sigma_max": 0.1,
                "c2d_sigma_max": 1.0,
            },
        }
    else:
        state_updates = {}
    return ToolExecutionResult(
        stage=stage,
        status="success",
        warnings=["dry_run_mode"],
        artifact_paths={},
        key_summary={"dry_run": True},
        state_updates=state_updates,
        raw_evidence=ToolFailureEvidence(raw_payload={"dry_run": True}),
        invocation_source="dry_run",
    )


def _instantiate_tools(runtime: RuntimeContext) -> dict[str, Any]:
    policy_engine = AgenticPolicyEngine(runtime)
    return {
        "relax": RelaxTool(vasp_cmd=runtime.vasp_cmd, consider_spin=runtime.consider_spin, policy_engine=policy_engine),
        "scf": ScfTool(vasp_cmd=runtime.vasp_cmd, consider_spin=runtime.consider_spin, policy_engine=policy_engine),
        "band": BandTool(vasp_cmd=runtime.vasp_cmd, consider_spin=runtime.consider_spin, policy_engine=policy_engine),
        "effective_mass": MassTool(vasp_cmd=runtime.vasp_cmd, consider_spin=runtime.consider_spin),
        "strain_loop": StrainTool(
            vasp_cmd=runtime.vasp_cmd,
            consider_spin=runtime.consider_spin,
            vacuum_direction=runtime.vacuum_direction,
            policy_engine=policy_engine,
        ),
        "mobility": MobilityTool(temperature=runtime.temperature, c2d_prefac=runtime.c2d_prefac),
    }


def _build_tool_inputs(stage: str, state_payload: dict[str, Any]) -> Any:
    state = _state(state_payload)
    if stage == "relax":
        return RelaxToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            poscar_path=str(state.material.poscar_path),
            potcar_path=str(state.material.potcar_path),
            recovery_param_updates=dict(state.execution.pending_parameter_updates or {}),
        )
    if stage == "scf":
        return ScfToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            poscar_path=str(state.physics_results.relaxed_structure_path or state.material.poscar_path),
            potcar_path=str(state.material.potcar_path),
            material_name=state.material.material_id,
        )
    if stage == "band":
        chgcar = (state.execution.artifact_paths or {}).get("CHGCAR") or os.path.join(state.execution.workdir, "02_scf", "CHGCAR")
        return BandToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            poscar_path=str(state.physics_results.relaxed_structure_path or state.material.poscar_path),
            potcar_path=str(state.material.potcar_path),
            chgcar_path=chgcar if chgcar else None,
            fermi_energy=state.physics_results.fermi_energy,
        )
    if stage == "effective_mass":
        masses = state.physics_results
        return MassToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            poscar_path=str(state.physics_results.relaxed_structure_path or state.material.poscar_path),
            potcar_path=str(state.material.potcar_path),
            reciprocal_lattice=list(masses.reciprocal_lattice),
            vbm_kpoint=list(masses.vbm_kpoint),
            cbm_kpoint=list(masses.cbm_kpoint),
            vbm_band_index=int(masses.vbm_band_index or 0),
            cbm_band_index=int(masses.cbm_band_index or 0),
            vbm_spin=masses.vbm_spin,
            cbm_spin=masses.cbm_spin,
            fermi_energy=masses.fermi_energy,
        )
    if stage == "strain_loop":
        return StrainToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            relaxed_poscar=str(state.physics_results.relaxed_structure_path or state.material.poscar_path),
            potcar_path=str(state.material.potcar_path),
            strain_plan_by_direction={k: list(v) for k, v in dict(state.physics_results.strain_plan_by_direction).items()},
            relax_retry_backups=list((state.diagnostics.recovery_summary or {}).get("relax_retry_backups", []) or []),
        )
    if stage == "mobility":
        masses = state.physics_results.masses
        return MobilityToolInput(
            material_id=state.material.material_id,
            base_dir=state.execution.workdir,
            state_payload=state_payload,
            strain_data=list(state.physics_results.strain_data),
            electron_mass_x=masses.get("electron_mass_x"),
            electron_mass_y=masses.get("electron_mass_y"),
            electron_mass_dos=masses.get("electron_mass_dos"),
            hole_mass_x=masses.get("hole_mass_x"),
            hole_mass_y=masses.get("hole_mass_y"),
            hole_mass_dos=masses.get("hole_mass_dos"),
            mass_diagnostics=dict((state.physics_results.effective_mass_summary or {}).get("mass_diagnostics", {}) or {}),
        )
    raise KeyError(f"unknown_stage:{stage}")


def _translate_tool_updates(stage: str, result: ToolExecutionResult) -> dict[str, Any]:
    raw = dict(result.state_updates or {})
    service_updates: dict[str, Any] = {}
    raw_services = dict(raw.get("services", {}) or {})
    if raw_services.get("parameter_plans"):
        service_updates["parameter_plans"] = dict(raw_services.get("parameter_plans", {}) or {})
    if raw_services.get("retrieval_trace"):
        service_updates["retrieval_trace"] = list(raw_services.get("retrieval_trace", []) or [])
    if stage == "relax":
        updates = {
            "physics_results": {
                "relaxed_structure_path": raw.get("relaxed_poscar"),
                "reciprocal_lattice": list(raw.get("reciprocal_lattice", []) or []),
                "relax_summary": dict(result.key_summary),
            },
            "diagnostics": {"recovery_summary": dict(raw.get("recovery_summary", {}) or {})},
            "execution": {"pending_parameter_updates": {}},
        }
        if service_updates:
            updates["services"] = service_updates
        return updates
    if stage == "scf":
        updates = {"physics_results": {"fermi_energy": raw.get("fermi_energy"), "scf_summary": dict(result.key_summary)}}
        if service_updates:
            updates["services"] = service_updates
        return updates
    if stage == "band":
        updates = {
            "physics_results": {
                "band_summary": dict(result.key_summary),
                "vbm_energy": raw.get("vbm_energy"),
                "cbm_energy": raw.get("cbm_energy"),
                "vbm_kpoint": list(raw.get("vbm_kpoint", []) or []),
                "cbm_kpoint": list(raw.get("cbm_kpoint", []) or []),
                "vbm_band_index": raw.get("vbm_band_index"),
                "cbm_band_index": raw.get("cbm_band_index"),
                "vbm_spin": raw.get("vbm_spin"),
                "cbm_spin": raw.get("cbm_spin"),
            }
        }
        if service_updates:
            updates["services"] = service_updates
        return updates
    if stage == "effective_mass":
        return {
            "physics_results": {
                "effective_mass_summary": dict(result.key_summary),
                "masses": {
                    "electron_mass_x": raw.get("electron_mass_x"),
                    "electron_mass_y": raw.get("electron_mass_y"),
                    "electron_mass_dos": raw.get("electron_mass_dos"),
                    "hole_mass_x": raw.get("hole_mass_x"),
                    "hole_mass_y": raw.get("hole_mass_y"),
                    "hole_mass_dos": raw.get("hole_mass_dos"),
                },
            }
        }
    if stage == "strain_loop":
        strain_summary = dict((raw.get("strain_summary") or result.key_summary) or {})
        updates = {
            "physics_results": {
                "strain_data": list(raw.get("strain_data", []) or []),
                "strain_data_summary": strain_summary,
            },
            "diagnostics": {"strain_summary": strain_summary},
        }
        if service_updates:
            updates["services"] = service_updates
        return updates
    if stage == "mobility":
        results = dict(raw.get("results", {}) or {})
        results_by_direction = dict(results.get("results_by_direction", {}) or {})
        e1 = {}
        c2d = {}
        mobility = {}
        for direction, direction_data in results_by_direction.items():
            c2d[direction] = direction_data.get("elastic_modulus_C2D_J_m2")
            mobility[direction] = {}
            e1[direction] = {}
            for carrier in ["electron", "hole"]:
                carrier_data = dict(direction_data.get(carrier, {}) or {})
                if carrier_data:
                    mobility[direction][carrier] = carrier_data.get("mobility_cm2_Vs")
                    e1[direction][carrier] = carrier_data.get("E1_eV")
        updates = {
            "physics_results": {
                "results": results,
                "mobility": mobility,
                "E1": e1,
                "C2D": c2d,
                "mobility_summary": dict(raw.get("mobility_summary", {}) or {}),
            },
            "diagnostics": {"fit_diagnostics": dict(raw.get("fit_diagnostics", {}) or {})},
        }
        if service_updates:
            updates["services"] = service_updates
        return updates
    return {}


def _run_stage_tool(stage: str, state_payload: dict[str, Any], runtime: RuntimeContext, tools: dict[str, Any]) -> ToolExecutionResult:
    if runtime.dry_run:
        return _dry_run_stage(stage, state_payload, runtime)
    tool = tools[stage]
    inputs = _build_tool_inputs(stage, state_payload)
    output = tool.run(inputs)
    return _normalize_tool_output(stage, output, str(state_payload.get("execution", {}).get("workdir") or ""))


def _prepare_capability(state_payload: dict[str, Any]) -> ToolExecutionResult:
    state = _state(state_payload).to_dict()
    workdir = state["execution"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    poscar_path = state["material"].get("poscar_path")
    potcar_path = state["material"].get("potcar_path")
    summary = summarize_poscar(str(poscar_path or ""))
    warnings = []
    error = None
    if not poscar_path or not os.path.exists(poscar_path):
        error = "prepare_missing_poscar"
    elif not potcar_path or not os.path.exists(potcar_path):
        error = "prepare_missing_potcar"
    if summary.get("warning"):
        warnings.append(summary["warning"])
    return ToolExecutionResult(
        stage="prepare",
        status="failed" if error else "success",
        error_summary=error,
        warnings=warnings,
        key_summary={"structure_summary": summary},
        state_updates={
            "material": {
                "structure_summary": summary,
                "atom_count": int(summary.get("atom_count", 0) or 0),
                "preflight_summary": summary,
            },
            "physics_results": {"prepare_summary": summary},
            "execution": {"workdir_inputs_ready": error is None},
        },
        raw_evidence=ToolFailureEvidence(
            parser_payload={"structure_summary": summary},
            raw_payload={"poscar_path": poscar_path, "potcar_path": potcar_path},
        ),
        invocation_source="native_tool",
    )


def _append_decision_trace(state: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    state["agent"]["decision_trace"] = dedupe_keep_order(list(state["agent"].get("decision_trace", []) or []) + [record])
    state["agent"]["agent_decisions"] = dedupe_keep_order(list(state["agent"].get("agent_decisions", []) or []) + [record])
    return state


def _append_framework_diagnostic(state: dict[str, Any], *, code: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {"code": code, "detail": dict(detail or {}), "timestamp": utc_now_iso()}
    state["services"]["framework_diagnostics"] = dedupe_keep_order(
        list(state["services"].get("framework_diagnostics", []) or []) + [entry]
    )
    return state


def _record_agent_tool_call(
    state: dict[str, Any],
    *,
    tool_name: str,
    phase: str,
    payload: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "tool_name": tool_name,
        "phase": phase,
        "payload": payload,
        "result": result,
        "source": "agent_tool",
    }
    state["execution"]["tool_invocations"] = dedupe_keep_order(
        list(state["execution"].get("tool_invocations", []) or []) + [entry]
    )
    state["execution"]["tool_trace"] = dedupe_keep_order(list(state["execution"].get("tool_trace", []) or []) + [entry])
    return state


def _anomaly_flags(state: dict[str, Any]) -> list[str]:
    mobility_metrics = validate_physics_window(dict(state.get("physics_results", {}).get("results", {}) or {}))
    raw_flags = list(mobility_metrics.get("anomaly_flags", []) or [])
    raw_flags += detect_basic_anomalies(
        {
            "errors": state.get("diagnostics", {}).get("errors"),
            "run_status": state.get("workflow", {}).get("run_status"),
            "confidence_score": state.get("diagnostics", {}).get("confidence_score"),
        }
    )
    return [str(item) for item in dedupe_keep_order(raw_flags)]


def _consume_pending_external_event(state: dict[str, Any]) -> dict[str, Any]:
    pending_events = list(state["execution"].get("pending_events", []) or [])
    if not pending_events:
        return state
    raw_event = dict(pending_events.pop(0) or {})
    event = normalize_external_event(
        raw_event,
        default_thread_id=str((state.get("execution", {}) or {}).get("thread_id") or ""),
        default_run_id=str((state.get("task", {}) or {}).get("task_id") or ""),
    ).model_dump(mode="json")
    state["execution"]["pending_events"] = pending_events
    state["execution"]["latest_event"] = event
    state["execution"]["event_history"] = dedupe_keep_order(
        list(state["execution"].get("event_history", []) or []) + [event]
    )
    event_id = str(event.get("event_id") or "")
    if event_id:
        state["execution"]["consumed_event_ids"] = dedupe_keep_order(
            list(state["execution"].get("consumed_event_ids", []) or []) + [event_id]
        )
    job_id = str(event.get("job_id") or "")
    target_capability = str(
        event.get("target_capability")
        or state["execution"].get("resume_markers", {}).get("awaiting_capability")
        or ""
    ) or None
    event_type = str(event.get("event_type") or "resume_requested")
    event_status = str(event.get("status") or event_type or "completed")
    normalized_status = "success" if event_status in {"completed", "success", "job_completed", "resume_requested", "manual_override"} else "failed"
    updated_jobs: list[dict[str, Any]] = []
    matched_job = False
    for current in list(state["execution"].get("external_jobs", []) or []):
        if isinstance(current, dict) and job_id and str(current.get("job_id") or "") == job_id:
            matched_job = True
            updated_jobs.append(
                {
                    **current,
                    "status": event_status,
                    "last_event_id": event.get("event_id"),
                    "last_event": event,
                    "updated_at": utc_now_iso(),
                }
            )
        else:
            updated_jobs.append(current)
    state["execution"]["external_jobs"] = updated_jobs
    if job_id and event_type in {"job_completed", "job_failed", "job_timeout", "artifact_missing"} and not matched_job:
        state["workflow"]["run_status"] = "waiting_external"
        state = _append_framework_diagnostic(
            state,
            code="stale_external_event_ignored",
            detail={"event_id": event.get("event_id"), "job_id": job_id, "event_type": event_type},
        )
        return state

    state["workflow"]["wait_reason"] = None
    state["execution"]["resume_markers"] = {}
    observation = _build_execution_observation(
        state=state,
        round_id=int(state["deliberation"].get("round_index", 0) or 0),
        command_action_family=str(event.get("action_family") or "run_capability"),
        target_capability=target_capability,
        status=normalized_status,
        result_summary=dict(event.get("result_summary", {}) or {}),
        raw_evidence={"external_event": event},
        error_summary=(str(event.get("error_summary") or "") or None),
        artifact_paths={str(k): str(v) for k, v in dict(event.get("artifact_paths", {}) or {}).items() if v},
        confidence=0.88,
    )
    state["execution"]["latest_execution_observation"] = observation.model_dump(mode="json")
    if event_type == "manual_override":
        payload = dict(event.get("payload", {}) or {})
        manual_action = str(payload.get("action") or payload.get("decision") or "continue")
        state["services"]["latest_human_decision"] = payload
        if manual_action == "skip_material":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "skipped"
            state["workflow"]["termination_reason"] = "manual_override:skip_material"
        elif manual_action == "abort_task":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "aborted"
            state["workflow"]["termination_reason"] = "manual_override:abort_task"
        else:
            resume_stage = str(payload.get("resume_stage") or target_capability or "")
            if resume_stage and resume_stage in capability_sequence():
                state = _apply_resume_strategy(state, resume_stage, reason="manual_override")
            state["workflow"]["run_status"] = "running"
    elif event_type == "resume_requested":
        state["workflow"]["run_status"] = "running"
    elif normalized_status == "failed":
        if target_capability and target_capability in capability_sequence():
            state["task_board"] = mark_task_blocked(
                dict(state.get("task_board", {}) or {}),
                target_capability,
                reason=str(event.get("error_summary") or event_status),
            )
        state["workflow"]["run_status"] = "needs_recovery"
        state["diagnostics"]["last_error"] = str(event.get("error_summary") or event_status)
    else:
        if target_capability and target_capability in capability_sequence():
            state["task_board"] = mark_task_completed(dict(state.get("task_board", {}) or {}), target_capability)
        state["workflow"]["run_status"] = "running"
        state = clear_resolved_error_state(state, resolved_stage=target_capability or None)
    state = _append_framework_diagnostic(
        state,
        code="external_event_consumed",
        detail={
            "event_id": event.get("event_id"),
            "event_type": event_type,
            "job_id": job_id,
            "status": event_status,
            "target_capability": target_capability,
        },
    )
    return state


def _build_execution_observation(
    *,
    state: dict[str, Any],
    round_id: int,
    command_action_family: str,
    target_capability: str | None,
    result: ToolExecutionResult | None = None,
    status: str | None = None,
    result_summary: dict[str, Any] | None = None,
    raw_evidence: dict[str, Any] | None = None,
    error_summary: str | None = None,
    artifact_paths: dict[str, str] | None = None,
    confidence: float = 0.9,
) -> ExecutionObservation:
    return ExecutionObservation(
        agent_name="executor",
        round_id=round_id,
        target_task_id=str(state.get("task", {}).get("task_id") or ""),
        action_family=command_action_family,  # type: ignore[arg-type]
        target_capability=target_capability,
        status=(status or (result.status if result is not None else "success")),  # type: ignore[arg-type]
        error_summary=error_summary or (result.error_summary if result is not None else None),
        artifact_paths=artifact_paths or (dict(result.artifact_paths) if result is not None else {}),
        result_summary=result_summary or (dict(result.key_summary) if result is not None else {}),
        raw_evidence=raw_evidence or (result.raw_evidence.model_dump(mode="json") if result is not None else {}),
        content={"current_stage": state.get("workflow", {}).get("current_stage")},
        confidence=confidence,
    )


def _selected_target_channels(selected_action: dict[str, Any]) -> list[str]:
    params = dict(selected_action.get("parameters", {}) or {})
    raw = params.get("target_channels")
    if raw is None:
        for alias in ("channels_to_invalidate", "channels_to_skip", "channels", "target_directions", "directions"):
            if params.get(alias) is not None:
                raw = params.get(alias)
                break
    if raw is None:
        single = params.get("channel") or params.get("target_channel")
        raw = [single] if single else []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip() for item in list(raw or []) if str(item).strip()]


def _sync_retry_counts(state: dict[str, Any], capability: str) -> dict[str, Any]:
    retries = dict(state["execution"].get("retry_counts", {}) or {})
    retries[capability] = int(retries.get(capability, 0) or 0) + 1
    state["execution"]["retry_counts"] = retries
    state["workflow"]["retry_counts"] = retries
    return state


def _derive_run_status(state: dict[str, Any]) -> tuple[str, str | None]:
    workflow = dict(state.get("workflow", {}) or {})
    services = dict(state.get("services", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    latest_observation = dict(execution.get("latest_execution_observation", {}) or {})
    current_status = str(workflow.get("run_status") or "running")
    latest_status = str(latest_observation.get("status") or "")
    if services.get("pending_human_payload") and not services.get("latest_human_decision"):
        return "needs_human", "awaiting_human_decision"
    if current_status in {"completed", "failed", "aborted", "skipped"}:
        return current_status, str(workflow.get("wait_reason") or "") or None
    if current_status == "waiting_external":
        if list(execution.get("pending_events", []) or []):
            return "running", None
        return "waiting_external", str(workflow.get("wait_reason") or "external_job_pending") or "external_job_pending"
    if latest_status == "failed" or has_blocked_or_abandoned_tasks(state):
        return "needs_recovery", None
    if all_tasks_resolved(state):
        return "ready_to_finalize", None
    return "running", None


def _apply_resume_strategy(state: dict[str, Any], capability: str, *, reason: str) -> dict[str, Any]:
    board = reset_from_capability(dict(state.get("task_board", {}) or {}), capability)
    state["task_board"] = board
    state["workflow"]["current_stage"] = capability
    state["workflow"]["next_action"] = capability
    state["diagnostics"]["recovery_history"] = dedupe_keep_order(
        list(state["diagnostics"].get("recovery_history", []) or []) + [{"action": "resume", "target_stage": capability, "reason": reason}]
    )
    return state


def _validate_result_capability(state: dict[str, Any]) -> dict[str, Any]:
    anomaly_flags = _anomaly_flags(state)
    warnings = list(state.get("material", {}).get("warnings", []) or [])
    report = build_validation_report(
        state,
        fit_threshold=DEFAULT_FIT_R2_THRESHOLD,
        max_points_per_direction=DEFAULT_REFINEMENT_TARGET_POINTS,
        anomaly_flags=anomaly_flags,
        warnings=warnings,
    )
    confidence = 0.92
    if report.get("decision") == "fail":
        confidence = 0.99
    elif report.get("recommended_action") == "refine_sampling":
        confidence = 0.74
    elif report.get("decision") == "pass_with_warning":
        confidence = 0.78
    report["confidence_score"] = confidence
    return {"validation_report": report, "confidence_score": confidence, "anomaly_flags": anomaly_flags}


def _quality_grade_from_state(state: dict[str, Any]) -> str:
    workflow = dict(state.get("workflow", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    validation = dict(diagnostics.get("validation_report", {}) or {})
    run_status = str(workflow.get("run_status") or "")
    decision = str(validation.get("decision") or "")
    confidence = diagnostics.get("confidence_score")
    confidence_value = float(confidence) if confidence is not None else 0.0
    if run_status in {"failed", "aborted", "skipped"} and not has_completed_compute_state_payload(state):
        return "failed"
    if decision in {"fail", "rejected"}:
        # Mark low confidence rather than execution failure when calculations finished but validation rejects quality.
        return "low_confidence"
    if decision == "pass" and confidence_value >= 0.9:
        return "high_confidence"
    if decision in {"pass", "pass_with_warning"} and confidence_value >= 0.75:
        return "warning_usable"
    return "low_confidence"


def make_observe_state_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    runtime_ctx = runtime
    gateway = AgentToolGateway()

    def _node(state_payload: MaterialTaskState | dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        emit_progress(
            "enter observe_state",
            workdir=state["execution"].get("workdir"),
            channel="graph",
            details={"material_id": state["material"].get("material_id")},
        )
        if state["workflow"].get("run_status") in {"", "pending", None}:
            state["workflow"]["run_status"] = "running"
        if state["workflow"].get("run_status") == "waiting_external" and list(state["execution"].get("pending_events", []) or []):
            state = _consume_pending_external_event(state)
        state["workflow"]["current_stage"] = "observe_state"
        state["execution"]["environment_summary"] = {
            "cwd": os.getcwd(),
            "dry_run": runtime_ctx.dry_run,
            "hitl_policy": runtime_ctx.hitl_policy,
            "db_uri": runtime_ctx.db_uri_preview,
            "deprecation_warnings": list(runtime_ctx.deprecation_warnings),
        }
        if not any(state["task_board"].values()):
            state["task_board"] = build_initial_task_board()
        registry = discover_skills(skills_root)
        state["services"]["available_agent_tools"] = gateway.metadata()
        state["deliberation"]["round_index"] = int(state["deliberation"].get("round_index", 0) or 0) + 1
        round_id = int(state["deliberation"]["round_index"])
        state["deliberation"]["rounds"] = dedupe_keep_order(
            list(state["deliberation"].get("rounds", []) or []) + [build_round_snapshot(state) | {"round_id": round_id}]
        )
        latest_obs = dict(state["execution"].get("latest_execution_observation", {}) or {})
        if latest_obs:
            state["blackboard"]["latest_execution_observation"] = latest_obs
        workspace_inspection = gateway.call(
            "inspect_workspace",
            {
                "workdir": state["execution"]["workdir"],
                "poscar_path": state["material"].get("poscar_path"),
                "potcar_path": state["material"].get("potcar_path"),
                "checkpoint_path": runtime_ctx.checkpoint_path_for(state["execution"]["workdir"]),
                "artifact_registry": dict(state["execution"].get("artifact_registry", {}) or {}),
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="inspect_workspace",
            phase="observe_state",
            payload={"workdir": state["execution"]["workdir"]},
            result=workspace_inspection,
        )
        artifact_inspection = gateway.call(
            "inspect_artifacts",
            {
                "workdir": state["execution"]["workdir"],
                "target_capability": str(latest_obs.get("target_capability") or state["workflow"].get("current_stage") or ""),
                "artifact_registry": dict(state["execution"].get("artifact_registry", {}) or {}),
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="inspect_artifacts",
            phase="observe_state",
            payload={"target_capability": artifact_inspection.get("target_capability")},
            result=artifact_inspection,
        )
        status_summary = gateway.call("query_execution_status", {"state": state})
        state = _record_agent_tool_call(
            state,
            tool_name="query_execution_status",
            phase="observe_state",
            payload={"current_stage": state["workflow"].get("current_stage")},
            result=status_summary,
        )
        observation_summary = gateway.call("synthesize_observation", {"state": state})
        state = _record_agent_tool_call(
            state,
            tool_name="synthesize_observation",
            phase="observe_state",
            payload={"round_id": round_id},
            result=observation_summary,
        )
        hitl_status = gateway.call(
            "inspect_hitl_state",
            {
                "workdir": state["execution"]["workdir"],
                "timeout_seconds": runtime_ctx.agent_runtime.human_review_timeout_seconds,
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="inspect_hitl_state",
            phase="observe_state",
            payload={"workdir": state["execution"]["workdir"]},
            result=hitl_status,
        )
        state["blackboard"]["validated_facts"] = {
            **dict(state["blackboard"].get("validated_facts", {}) or {}),
            "workspace": workspace_inspection,
            "execution_status": status_summary,
            "hitl_status": hitl_status,
        }
        state["blackboard"]["parsed_artifacts"] = {
            **dict(state["blackboard"].get("parsed_artifacts", {}) or {}),
            "artifact_inspection": artifact_inspection,
        }
        state["blackboard"]["anomaly_flags"] = list(observation_summary.get("anomaly_flags", []) or [])
        state["blackboard"]["risk_flags"] = list(observation_summary.get("risk_flags", []) or [])
        skill_resolution = gateway.call(
            "resolve_skills",
            {
                "state": state,
                "role": "runtime",
                "limit": runtime_ctx.skill_auto_resolve_limit,
                "skills_root": skills_root,
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="resolve_skills",
            phase="observe_state",
            payload={"role": "runtime", "round_id": round_id},
            result=skill_resolution,
        )
        selected_skills = list(skill_resolution.get("selected_skills", []) or [])
        state["agent"]["loaded_skills"] = dedupe_keep_order(list(state["agent"].get("loaded_skills", []) or []) + selected_skills)
        state["services"]["loaded_skills"] = list(selected_skills)
        state["services"]["skill_resolution"] = dict(skill_resolution)
        state["execution"]["skill_trace"] = dedupe_keep_order(
            list(state["execution"].get("skill_trace", []) or [])
            + [{
                "phase": "observe_state",
                "round_id": round_id,
                "role": "runtime",
                "selected_skills": selected_skills,
                "candidate_count": len(list(skill_resolution.get("candidates", []) or [])),
                "run_status": state["workflow"].get("run_status"),
                "current_stage": state["workflow"].get("current_stage"),
            }]
        )
        state["services"]["llm_context_summary"] = build_llm_context_summary(
            state,
            execution_status=status_summary,
            observation_summary=observation_summary,
        )
        with _memory_store_scope(runtime, runtime_ctx.resolved_db_uri) as store:
            for skill_name, payload in registry.items():
                manifest = dict(payload.get("manifest", {}) or {})
                record_skill_metadata(
                    store,
                    skill_name=skill_name,
                    payload={
                        "skill_name": skill_name,
                        "path": payload.get("path"),
                        "description": payload.get("description"),
                        "summary": payload.get("summary"),
                        "manifest": manifest,
                        "resource_count": len(list(payload.get("resources", []) or [])),
                    },
                )
            memory_hits = gateway.call("query_memory_hits", {"state": state, "limit": 5}, store=store)
            state = _record_agent_tool_call(
                state,
                tool_name="query_memory_hits",
                phase="observe_state",
                payload={"round_id": round_id},
                result=memory_hits,
            )
            state["memory"]["recovered_case_patterns"] = list(memory_hits.get("recovered_case_patterns", []) or [])
            state["memory"]["validation_case_patterns"] = list(memory_hits.get("validation_case_patterns", []) or [])
            state["memory"]["reusable_heuristics"] = list(memory_hits.get("skill_registry", []) or [])
        return _finalize_node_output(before, state)

    return _node


def make_proposal_phase_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    planner = PlannerAgent(runtime, skills_root)
    recovery = RecoveryAgent(runtime, skills_root)
    refinement = RefinementAgent(runtime, skills_root)
    executor = ExecutorAgent(runtime, skills_root)

    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        round_id = int(state["deliberation"].get("round_index", 0) or 0)
        state["workflow"]["current_stage"] = "proposal_phase"
        proposals: list[Proposal] = []

        def _collect(agent_label: str, fn) -> None:
            nonlocal proposals
            try:
                proposals.extend(fn())
            except Exception as exc:
                emit_progress(
                    "proposal agent failed; continuing with remaining agents",
                    workdir=state["execution"].get("workdir"),
                    channel="graph",
                    details={
                        "round_id": round_id,
                        "agent": agent_label,
                        "error": f"{type(exc).__name__}:{exc}",
                    },
                )

        emit_progress(
            "proposal_phase started",
            workdir=state["execution"].get("workdir"),
            channel="graph",
            details={"round_id": round_id},
        )
        emit_progress("planner propose", workdir=state["execution"].get("workdir"), channel="graph", details={"round_id": round_id})
        _collect("planner", lambda: planner.propose(state=state, round_id=round_id))
        emit_progress("recovery propose", workdir=state["execution"].get("workdir"), channel="graph", details={"round_id": round_id})
        try:
            proposals.extend(recovery.propose(state=state, round_id=round_id))
            if recovery.last_failure_diagnosis:
                state["diagnostics"]["recovery_diagnosis"] = dict(recovery.last_failure_diagnosis)
                state["services"]["retrieval_trace"] = dedupe_keep_order(
                    list(state["services"].get("retrieval_trace", []) or [])
                    + [
                        {
                            "kind": "failure_diagnosis",
                            "stage": recovery.last_failure_diagnosis.get("stage"),
                            "source": recovery.last_failure_diagnosis.get("source"),
                            "confidence": recovery.last_failure_diagnosis.get("confidence"),
                            "recommended_action": recovery.last_failure_diagnosis.get("recommended_action"),
                            "evidence": list(recovery.last_failure_diagnosis.get("evidence_items", []) or []),
                        }
                    ]
                )
        except Exception as exc:
            emit_progress(
                "proposal agent failed; continuing with remaining agents",
                workdir=state["execution"].get("workdir"),
                channel="graph",
                details={
                    "round_id": round_id,
                    "agent": "recovery",
                    "error": f"{type(exc).__name__}:{exc}",
                },
            )
        emit_progress("refinement propose", workdir=state["execution"].get("workdir"), channel="graph", details={"round_id": round_id})
        _collect("refinement", lambda: refinement.propose(state=state, round_id=round_id))
        emit_progress("executor propose", workdir=state["execution"].get("workdir"), channel="graph", details={"round_id": round_id})
        _collect("executor", lambda: executor.propose(state=state, round_id=round_id))
        state["deliberation"]["proposals"] = dedupe_keep_order(
            list(state["deliberation"].get("proposals", []) or []) + [item.model_dump(mode="json") for item in proposals]
        )
        for item in proposals:
            _append_decision_trace(state, {"node": "proposal_phase", "message_type": "proposal", **item.model_dump(mode="json")})
        return _finalize_node_output(before, state)

    return _node


def make_critique_phase_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    critic = CriticAgent(runtime, skills_root)
    judge = PhysicsJudgeAgent(runtime, skills_root)
    cost_guardian = CostGuardianAgent(runtime, skills_root)

    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        round_id = int(state["deliberation"].get("round_index", 0) or 0)
        emit_progress(
            "critique_phase started",
            workdir=state["execution"].get("workdir"),
            channel="graph",
            details={"round_id": round_id},
        )
        proposals = [
            Proposal.model_validate(item)
            for item in list(state["deliberation"].get("proposals", []) or [])
            if isinstance(item, dict) and int(item.get("round_id", -1)) == round_id
        ]
        critiques: list[Critique] = []
        preferences: list[Preference] = []
        for agent in (critic, judge, cost_guardian):
            emit_progress(
                "reviewing proposals",
                workdir=state["execution"].get("workdir"),
                channel="graph",
                details={"round_id": round_id, "agent": agent.agent_name},
            )
            try:
                agent_critiques, agent_preferences = agent.review(state=state, proposals=proposals, round_id=round_id)
            except Exception as exc:
                emit_progress(
                    "review agent failed; continuing without this opinion",
                    workdir=state["execution"].get("workdir"),
                    channel="graph",
                    details={"round_id": round_id, "agent": agent.agent_name, "error": type(exc).__name__},
                )
                state = _append_framework_diagnostic(
                    state,
                    code="review_agent_failed",
                    detail={"agent": agent.agent_name, "round_id": round_id, "error_type": type(exc).__name__, "error_text": str(exc)},
                )
                continue
            critiques.extend(agent_critiques)
            preferences.extend(agent_preferences)
        state["workflow"]["current_stage"] = "critique_phase"
        state["deliberation"]["critiques"] = dedupe_keep_order(
            list(state["deliberation"].get("critiques", []) or []) + [item.model_dump(mode="json") for item in critiques]
        )
        state["deliberation"]["preferences"] = dedupe_keep_order(
            list(state["deliberation"].get("preferences", []) or []) + [item.model_dump(mode="json") for item in preferences]
        )
        for item in critiques + preferences:
            _append_decision_trace(state, {"node": "critique_phase", "message_type": item.message_type, **item.model_dump(mode="json")})
        return _finalize_node_output(before, state)

    return _node


def make_arbitration_phase_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
) -> Callable[[MaterialTaskState | dict[str, Any]], Command]:
    orchestrator = OrchestratorAgent(runtime, skills_root)

    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> Command:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        round_id = int(state["deliberation"].get("round_index", 0) or 0)
        emit_progress(
            "arbitration_phase started",
            workdir=state["execution"].get("workdir"),
            channel="graph",
            details={"round_id": round_id},
        )
        proposals = [
            Proposal.model_validate(item)
            for item in list(state["deliberation"].get("proposals", []) or [])
            if isinstance(item, dict) and int(item.get("round_id", -1)) == round_id
        ]
        critiques = [
            Critique.model_validate(item)
            for item in list(state["deliberation"].get("critiques", []) or [])
            if isinstance(item, dict) and int(item.get("round_id", -1)) == round_id
        ]
        preferences = [
            Preference.model_validate(item)
            for item in list(state["deliberation"].get("preferences", []) or [])
            if isinstance(item, dict) and int(item.get("round_id", -1)) == round_id
        ]
        arbitration = orchestrator.arbitrate(
            state=state,
            proposals=proposals,
            critiques=critiques,
            preferences=preferences,
            round_id=round_id,
        )
        latest_failure = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
        if arbitration.selected_action is None and str(latest_failure.get("status") or "") == "failed" and has_relax_failure_signature(
            stage=str(latest_failure.get("target_capability") or latest_failure.get("stage") or ""),
            latest_failure=latest_failure,
            state_payload=state,
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
                    "Relaxation failure evidence was found after arbitration no-op; "
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
        state["workflow"]["current_stage"] = "arbitration_phase"
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
        state["execution"]["current_action"] = arbitration.selected_action.model_dump(mode="json") if arbitration.selected_action is not None else {}
        state["execution"]["action_status"] = "selected" if arbitration.selected_action is not None else "noop"
        state["workflow"]["next_action"] = arbitration.selected_action.action_family if arbitration.selected_action is not None else None
        state["services"]["selected_action_requires_execution"] = bool(arbitration.selected_action is not None and not arbitration.whether_noop)
        if arbitration.whether_waiting_external:
            state["workflow"]["run_status"] = "waiting_external"
        elif arbitration.whether_ready_to_finalize:
            state["workflow"]["run_status"] = "ready_to_finalize"
        elif arbitration.whether_noop and (
            str((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}).get("status") or "") == "failed"
            or has_blocked_or_abandoned_tasks(state)
        ):
            state["workflow"]["run_status"] = "needs_recovery"
        elif arbitration.whether_noop:
            state["workflow"]["run_status"] = "running"
        state["services"]["termination_requested"] = False
        _append_decision_trace(state, {"node": "arbitration_phase", "message_type": "arbitration", **arbitration.model_dump(mode="json")})
        goto = "execute_selected_action" if bool(state["services"].get("selected_action_requires_execution")) else "check_termination"
        return Command(update=_finalize_node_output(before, state), goto=goto)

    return _node


def _execute_validation_action(state: dict[str, Any], round_id: int, selected_action: dict[str, Any]) -> tuple[dict[str, Any], ExecutionObservation]:
    validation = _validate_result_capability(state)
    state["diagnostics"]["validation_report"] = dict(validation["validation_report"])
    state["diagnostics"]["confidence_score"] = float(validation["confidence_score"])
    state["blackboard"]["anomaly_flags"] = list(validation["anomaly_flags"])
    retained_subchannels = list(validation["validation_report"].get("retained_subchannels", []) or [])
    rejected_subchannels = list(validation["validation_report"].get("rejected_subchannels", []) or [])
    accepted_directions, rejected_directions = derive_direction_acceptance(retained_subchannels, rejected_subchannels)
    state["physics_results"]["accepted_channels"] = accepted_directions
    state["physics_results"]["rejected_channels"] = rejected_directions
    state["diagnostics"]["validation_report"]["accepted_channels"] = accepted_directions
    state["diagnostics"]["validation_report"]["rejected_channels"] = rejected_directions
    if validation["validation_report"]["decision"] == "fail":
        state["material"]["warnings"] = dedupe_keep_order(list(state["material"].get("warnings", []) or []) + list(validation["anomaly_flags"]))
    observation = _build_execution_observation(
        state=state,
        round_id=round_id,
        command_action_family=str(selected_action.get("action_family") or "revalidate_result"),
        target_capability="validation",
        status="success",
        result_summary=validation["validation_report"],
        raw_evidence={"validation": validation},
    )
    state["workflow"]["stage_status"]["validation"] = "success"
    state["workflow"]["completed_stages"] = dedupe_keep_order(list(state["workflow"].get("completed_stages", []) or []) + ["validation"])
    return state, observation


def make_execute_selected_action_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
    tools: dict[str, Any],
) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    runtime_ctx = runtime
    executor = ExecutorAgent(runtime, skills_root)
    gateway = AgentToolGateway()

    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        round_id = int(state["deliberation"].get("round_index", 0) or 0)
        selected_action = dict(state["execution"].get("current_action", {}) or {})
        emit_progress(
            "execute_selected_action entered",
            workdir=state["execution"].get("workdir"),
            channel="graph",
            details={
                "round_id": round_id,
                "action_family": selected_action.get("action_family"),
                "target_capability": selected_action.get("target_capability"),
            },
        )
        if not selected_action:
            state["execution"]["action_status"] = "noop"
            return _finalize_node_output(before, state)
        action_family = str(selected_action.get("action_family") or "")
        target_capability = str(selected_action.get("target_capability") or "") or None
        legality = gateway.call(
            "check_action_legality",
            {
                "state": state,
                "action_family": action_family,
                "target_capability": target_capability,
                "parameters": dict(selected_action.get("parameters", {}) or {}),
                "submit_external_job": bool(selected_action.get("submit_external_job")),
                "wait_for_event_after_submission": bool(selected_action.get("wait_for_event_after_submission")),
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="check_action_legality",
            phase="execute_selected_action",
            payload={"action_family": action_family, "target_capability": target_capability},
            result=legality,
        )
        if not legality.get("allowed", False):
            refusal_reasons = [str(item) for item in list(legality.get("refusal_reasons", []) or []) if str(item).strip()]
            state = _append_framework_diagnostic(
                state,
                code="illegal_selected_action_strict_failure",
                detail={
                    "action_family": action_family,
                    "target_capability": target_capability,
                    "refusal_reasons": refusal_reasons,
                },
            )
            state["workflow"]["run_status"] = "failed"
            state["workflow"]["termination_reason"] = "illegal_selected_action"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family=action_family,
                target_capability=target_capability,
                status="failed",
                error_summary=("illegal_selected_action:" + ";".join(refusal_reasons))
                if refusal_reasons
                else "illegal_selected_action",
            )
            observation_signals = gateway.call("synthesize_observation", {"state": state})
            state = _record_agent_tool_call(
                state,
                tool_name="synthesize_observation",
                phase="execute_selected_action",
                payload={"action_family": action_family, "target_capability": target_capability},
                result=observation_signals,
            )
            observation.content = {**dict(observation.content or {}), "observation_signals": observation_signals}
            state["blackboard"]["anomaly_flags"] = list(observation_signals.get("anomaly_flags", []) or [])
            state["blackboard"]["risk_flags"] = list(observation_signals.get("risk_flags", []) or [])
            state["execution"]["latest_execution_observation"] = observation.model_dump(mode="json")
            state["execution"]["action_status"] = observation.status
            _append_decision_trace(state, {"node": "execute_selected_action", "message_type": "execution_observation", **observation.model_dump(mode="json")})
            return _finalize_node_output(before, state)
        command = executor.compile_selected_action(
            state=state,
            selected_action=SelectedAction.model_validate(selected_action),
            round_id=round_id,
        )
        state["workflow"]["current_stage"] = "execute_selected_action"
        _append_decision_trace(state, {"node": "execute_selected_action", "message_type": "execution_command", **command.model_dump(mode="json")})
        if target_capability and action_family in {"run_capability", "retry_capability", "rerun_from_capability", "refine_sampling", "revalidate_result"}:
            state["task_board"] = mark_task_started(dict(state.get("task_board", {}) or {}), target_capability)
        observation: ExecutionObservation

        if action_family == "refine_sampling":
            params = dict(selected_action.get("parameters", {}) or {})
            refinement_plan = resolve_refinement_sampling(
                state,
                params,
                max_points_per_direction=int(runtime.agent_runtime.strain_target_points or DEFAULT_REFINEMENT_TARGET_POINTS),
                fit_threshold=float(runtime.agent_runtime.fit_r2_threshold or DEFAULT_FIT_R2_THRESHOLD),
            )
            suggested = dict(refinement_plan.get("suggested_points", {}) or {})
            applied_points = dict(refinement_plan.get("applied_points", {}) or {})
            channels = [str(item) for item in list(refinement_plan.get("target_channels", []) or []) if str(item).strip()]
            if not channels:
                channels = list(applied_points)
            if not applied_points:
                state["workflow"]["run_status"] = "needs_recovery"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability or "strain_loop",
                    status="failed",
                    error_summary="no_fresh_refinement_points",
                    result_summary={
                        "requested_parameters": params,
                        "normalized_refinement_plan": refinement_plan,
                    },
                    raw_evidence={"selected_action": dict(selected_action)},
                )
                observation_signals = gateway.call("synthesize_observation", {"state": state})
                state = _record_agent_tool_call(
                    state,
                    tool_name="synthesize_observation",
                    phase="execute_selected_action",
                    payload={"action_family": action_family, "target_capability": target_capability or "strain_loop"},
                    result=observation_signals,
                )
                observation.content = {**dict(observation.content or {}), "observation_signals": observation_signals}
                state["blackboard"]["anomaly_flags"] = list(observation_signals.get("anomaly_flags", []) or [])
                state["blackboard"]["risk_flags"] = list(observation_signals.get("risk_flags", []) or [])
                state["execution"]["latest_execution_observation"] = observation.model_dump(mode="json")
                state["execution"]["action_status"] = observation.status
                _append_decision_trace(state, {"node": "execute_selected_action", "message_type": "execution_observation", **observation.model_dump(mode="json")})
                return _finalize_node_output(before, state)
            plan = dict(state.get("physics_results", {}).get("strain_plan_by_direction", {}) or {})
            for channel in channels:
                existing = [float(v) for v in list(plan.get(channel, []) or [])]
                extras = [float(v) for v in list(applied_points.get(channel, []) or [])]
                merged = sorted({round(float(v), 6) for v in existing + extras})
                if merged:
                    plan[channel] = [float(v) for v in merged]
            state["physics_results"]["strain_plan_by_direction"] = plan
            state["workflow"]["refinement_rounds"] = int(state["workflow"].get("refinement_rounds", 0) or 0) + 1
            if not target_capability:
                target_capability = "strain_loop"
            state["diagnostics"]["recovery_history"] = dedupe_keep_order(
                list(state["diagnostics"].get("recovery_history", []) or [])
                + [{
                    "action": "refine_sampling",
                    "target_stage": target_capability,
                    "target_channels": channels,
                    "suggested_points": suggested,
                    "applied_points": applied_points,
                    "max_points_per_direction": refinement_plan.get("max_points_per_direction"),
                    "refinement_strategy": refinement_plan.get("refinement_strategy"),
                    "refinement_round": int(state["workflow"].get("refinement_rounds", 0) or 0),
                }]
            )
            emit_progress(
                "refine_sampling plan updated",
                workdir=state["execution"].get("workdir"),
                channel="stage",
                details={
                    "target_capability": target_capability,
                    "target_channels": channels,
                    "applied_points": applied_points,
                    "refinement_rounds": int(state["workflow"].get("refinement_rounds", 0) or 0),
                },
            )

        if action_family == "finalize_material":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "ready_to_finalize"
            state["workflow"]["wait_reason"] = None
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family=action_family,
                target_capability=target_capability,
                status="success",
                result_summary={"reason": "finalize_material_selected"},
            )
        elif action_family == "abort_material":
            state["services"]["termination_requested"] = True
            state["workflow"]["run_status"] = "aborted"
            state["workflow"]["termination_reason"] = str(selected_action.get("parameters", {}).get("reason") or "abort_material")
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family=action_family,
                target_capability=target_capability,
                status="failed",
                error_summary=state["workflow"]["termination_reason"],
            )
        elif action_family == "escalate_human":
            state["workflow"]["run_status"] = "needs_human"
            payload = build_human_escalation_payload(
                state,
                recommended_options=list(
                    selected_action.get("parameters", {}).get("recommended_options", [])
                    or ["manual_fix_resume", "retry_current_stage", "rerun_previous_stage", "skip_material", "abort_task"]
                ),
            )
            workdir = state["execution"]["workdir"]
            paths = write_escalation_payload(
                workdir=workdir,
                payload=payload,
                checkpoint_subdir=runtime.checkpoint_subdir,
            )
            notify_result = notify_escalation(payload)
            state["diagnostics"]["consultation_trace"] = dedupe_keep_order(
                list(state["diagnostics"].get("consultation_trace", []) or []) + [{"payload": payload, "paths": paths, "notify_result": notify_result}]
            )
            state["services"]["pending_human_payload"] = payload
            if runtime.compatibility_export_enabled:
                checkpoint_path = export_compatibility_checkpoint(state, reason="deliberation_escalation")
                state["execution"]["compatibility_checkpoint_path"] = checkpoint_path
                state["execution"]["compatibility_checkpoint_history"] = dedupe_keep_order(
                    list(state["execution"].get("compatibility_checkpoint_history", []) or []) + [checkpoint_path]
                )
            save_state_snapshot(workdir=workdir, state=MaterialTaskState.from_dict(state).to_dict(), checkpoint_subdir=runtime.checkpoint_subdir)
            response = interrupt(payload)
            decision = normalize_hitl_decision(dict(response or {}), source="precomputed")
            state["services"]["latest_human_decision"] = decision.model_dump(mode="json")
            state["workflow"]["escalated_to_human"] = True
            state["workflow"]["wait_reason"] = None
            if decision.action == "skip_material":
                state["services"]["termination_requested"] = True
                state["workflow"]["run_status"] = "skipped"
                state["workflow"]["termination_reason"] = "skip_material"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    status="skipped",
                    result_summary={"human_action": "skip_material"},
                )
            elif decision.action == "abort_task":
                state["services"]["termination_requested"] = True
                state["workflow"]["run_status"] = "failed"
                state["workflow"]["termination_reason"] = "abort_task"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    status="failed",
                    error_summary="abort_task",
                    result_summary={"human_action": "abort_task"},
                )
            else:
                resume_stage = target_capability or str(state["workflow"].get("current_stage") or "")
                if decision.action == "manual_fix_resume" and decision.instruction is not None:
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
                elif decision.action == "rerun_previous_stage":
                    resume_stage = find_previous_stage(target_capability or "") or (target_capability or "")
                state = _apply_resume_strategy(state, resume_stage, reason=decision.action)
                state["workflow"]["run_status"] = "running"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=resume_stage,
                    status="success",
                    result_summary={"human_action": decision.action, "resume_stage": resume_stage},
                )
        elif action_family == "revalidate_result" or target_capability == "validation":
            state, observation = _execute_validation_action(state, round_id, selected_action)
            state["workflow"]["run_status"] = "running"
        elif action_family in {"invalidate_channel", "skip_channel"}:
            target_channels = _selected_target_channels(selected_action)
            if not target_channels:
                state = _append_framework_diagnostic(
                    state,
                    code="channel_action_missing_target_channels",
                    detail={
                        "action_family": action_family,
                        "target_capability": target_capability,
                        "parameter_keys": sorted(list(dict(selected_action.get("parameters", {}) or {}).keys())),
                    },
                )
                state["workflow"]["run_status"] = "needs_recovery"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    status="failed",
                    error_summary="channel_action_missing_target_channels",
                    result_summary={
                        "action": action_family,
                        "target_channels": target_channels,
                    },
                    raw_evidence={"selected_action": dict(selected_action)},
                )
            else:
                validation_report = dict(state["diagnostics"].get("validation_report", {}) or {})
                retained_subchannels = list(validation_report.get("retained_subchannels", []) or [])
                rejected_subchannels = list(validation_report.get("rejected_subchannels", []) or [])
                if validation_report:
                    targeted_subchannels = subchannel_tokens_from_targets(target_channels)
                    retained_subchannels = [item for item in retained_subchannels if item not in set(targeted_subchannels)]
                    rejected_subchannels = dedupe_keep_order(rejected_subchannels + targeted_subchannels)
                    validation_report["retained_subchannels"] = retained_subchannels
                    validation_report["rejected_subchannels"] = rejected_subchannels
                    accepted_directions, rejected_directions = derive_direction_acceptance(retained_subchannels, rejected_subchannels)
                else:
                    accepted = [
                        str(item)
                        for item in list((state.get("physics_results", {}) or {}).get("accepted_channels", []) or [])
                        if str(item).strip()
                    ]
                    rejected = [
                        str(item)
                        for item in list((state.get("physics_results", {}) or {}).get("rejected_channels", []) or [])
                        if str(item).strip()
                    ]
                    for channel in target_channels:
                        accepted = [item for item in accepted if item != channel]
                        rejected = dedupe_keep_order(rejected + [channel])
                    accepted_directions, rejected_directions = dedupe_keep_order(accepted), rejected
                state["physics_results"]["accepted_channels"] = dedupe_keep_order(accepted_directions)
                state["physics_results"]["rejected_channels"] = dedupe_keep_order(rejected_directions)
                if validation_report:
                    validation_report["accepted_channels"] = list(state["physics_results"]["accepted_channels"])
                    validation_report["rejected_channels"] = list(state["physics_results"]["rejected_channels"])
                    state["diagnostics"]["validation_report"] = validation_report
                state["workflow"]["run_status"] = "running"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    status="success",
                    result_summary={
                        "action": action_family,
                        "target_channels": target_channels,
                        "accepted_channels": list(state["physics_results"]["accepted_channels"]),
                        "rejected_channels": list(state["physics_results"]["rejected_channels"]),
                    },
                    raw_evidence={"selected_action": dict(selected_action)},
                )
        elif target_capability == "prepare":
            emit_progress(
                "waiting for deterministic stage completion",
                workdir=state["execution"].get("workdir"),
                channel="stage",
                details={
                    "stage": "prepare",
                    "action_family": action_family,
                    "wait_kind": "deterministic_stage",
                    "llm_tokens": 0,
                    "stage_dir": _stage_workdir(state["execution"]["workdir"], "prepare"),
                },
            )
            result = _prepare_capability(state)
            emit_progress(
                "stage execution finished",
                workdir=state["execution"].get("workdir"),
                channel="stage",
                details={
                    "stage": "prepare",
                    "status": result.status,
                    "duration_s": f"{float(result.duration_s or 0.0):.2f}",
                    "error": result.error_summary,
                },
            )
            state = register_tool_result(state, result)
            state = record_stage_status(_state(state), "prepare", "success" if result.success else "failed").to_dict()
            state["workflow"]["run_status"] = "running" if result.success else "needs_recovery"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family=action_family,
                target_capability="prepare",
                result=result,
            )
        elif target_capability:
            if action_family in {"rerun_from_capability", "repair_execution_context", "refine_sampling"}:
                state["task_board"] = reset_from_capability(dict(state.get("task_board", {}) or {}), target_capability)
            if action_family in {"retry_capability", "rerun_from_capability"}:
                state = _sync_retry_counts(state, target_capability)
            if bool(command.submit_external_job or command.wait_for_event_after_submission):
                job_id = (
                    str(selected_action.get("parameters", {}).get("job_id") or "")
                    or f"{state['task']['task_id']}::{round_id}::{target_capability or action_family}"
                )
                external_job = {
                    "job_id": job_id,
                    "action_family": action_family,
                    "target_capability": target_capability,
                    "status": "submitted",
                    "submitted_at": utc_now_iso(),
                    "expected_artifacts": list(command.expected_artifacts or []),
                    "wait_for_event_after_submission": bool(command.wait_for_event_after_submission),
                }
                existing_jobs = [
                    item
                    for item in list(state["execution"].get("external_jobs", []) or [])
                    if not (
                        isinstance(item, dict)
                        and str(item.get("job_id") or "") == job_id
                    )
                ]
                state["execution"]["external_jobs"] = existing_jobs + [external_job]
                state["execution"]["resume_markers"] = {
                    "job_id": job_id,
                    "awaiting_capability": target_capability,
                    "submitted_round_id": round_id,
                }
                state["workflow"]["run_status"] = "waiting_external"
                state["workflow"]["wait_reason"] = f"awaiting_external_event:{target_capability or action_family}"
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    status="running",
                    result_summary={"job_id": job_id, "wait_reason": state["workflow"]["wait_reason"]},
                    raw_evidence={"external_job": external_job},
                    confidence=0.9,
                )
            else:
                emit_progress(
                    "waiting for deterministic stage completion",
                    workdir=state["execution"].get("workdir"),
                    channel="stage",
                    details={
                        "stage": target_capability,
                        "action_family": action_family,
                        "wait_kind": "deterministic_stage",
                        "llm_tokens": 0,
                        "stage_dir": _stage_workdir(state["execution"]["workdir"], target_capability),
                        "stdout_path": os.path.join(_stage_workdir(state["execution"]["workdir"], target_capability), "sout"),
                    },
                )
                try:
                    _validate_stage_dependencies(state, target_capability)
                    result = _run_stage_tool(target_capability, state, runtime_ctx, tools)
                except Exception as exc:
                    result = ToolExecutionResult(
                        stage=target_capability,
                        status="failed",
                        error_summary=str(exc),
                        raw_evidence=ToolFailureEvidence(
                            exception_type=type(exc).__name__,
                            exception_message=str(exc),
                            log_paths=_log_paths_for_stage(state["execution"]["workdir"], target_capability),
                            raw_payload={"stage": target_capability},
                        ),
                    )
                emit_progress(
                    "stage execution finished",
                    workdir=state["execution"].get("workdir"),
                    channel="stage",
                    details={
                        "stage": target_capability,
                        "status": result.status,
                        "duration_s": f"{float(result.duration_s or 0.0):.2f}",
                        "error": result.error_summary,
                        "stdout_path": result.raw_evidence.stdout_path,
                    },
                )
                state = register_tool_result(state, result)
                state = apply_state_updates(state, _translate_tool_updates(target_capability, result))
                state = record_stage_status(_state(state), target_capability, "success" if result.success else "failed").to_dict()
                state["workflow"]["run_status"] = "running" if result.success else "needs_recovery"
                if result.success and target_capability in STABLE_CHECKPOINT_STAGES and runtime.compatibility_export_enabled:
                    checkpoint_path = export_compatibility_checkpoint(state, reason=f"stable_stage:{target_capability}")
                    state["execution"]["compatibility_checkpoint_path"] = checkpoint_path
                    state["execution"]["compatibility_checkpoint_history"] = dedupe_keep_order(
                        list(state["execution"].get("compatibility_checkpoint_history", []) or []) + [checkpoint_path]
                    )
                if not result.success:
                    if action_family in {"run_capability", "repair_execution_context", "refine_sampling", "revalidate_result"}:
                        state = _sync_retry_counts(state, target_capability)
                    state["diagnostics"]["recovery_summary"] = {
                        "stage": target_capability,
                        "current_stage": target_capability,
                        "error_type": _classify_failure(str(result.error_summary or "")),
                        "error_summary": result.error_summary,
                        "trigger_pattern": result.error_summary,
                        "retries_used": int((state["execution"].get("retry_counts", {}) or {}).get(target_capability, 0) or 0),
                        "max_retries": int(state["workflow"].get("retry_budget", 2) or 2),
                    }
                observation = _build_execution_observation(
                    state=state,
                    round_id=round_id,
                    command_action_family=action_family,
                    target_capability=target_capability,
                    result=result,
                )
        else:
            state["workflow"]["run_status"] = "needs_recovery"
            observation = _build_execution_observation(
                state=state,
                round_id=round_id,
                command_action_family=action_family or "abort_material",
                target_capability=target_capability,
                status="failed",
                error_summary="missing_target_capability",
            )

        observation_signals = gateway.call("synthesize_observation", {"state": state})
        state = _record_agent_tool_call(
            state,
            tool_name="synthesize_observation",
            phase="execute_selected_action",
            payload={"action_family": action_family, "target_capability": target_capability},
            result=observation_signals,
        )
        observation.content = {**dict(observation.content or {}), "observation_signals": observation_signals}
        state["blackboard"]["anomaly_flags"] = list(observation_signals.get("anomaly_flags", []) or [])
        state["blackboard"]["risk_flags"] = list(observation_signals.get("risk_flags", []) or [])
        state["execution"]["latest_execution_observation"] = observation.model_dump(mode="json")
        state["execution"]["action_status"] = observation.status
        _append_decision_trace(state, {"node": "execute_selected_action", "message_type": "execution_observation", **observation.model_dump(mode="json")})
        return _finalize_node_output(before, state)

    return _node


def make_reflect_round_node(runtime: RuntimeContext) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    runtime_ctx = runtime
    gateway = AgentToolGateway()

    def _node(state_payload: MaterialTaskState | dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        round_id = int(state["deliberation"].get("round_index", 0) or 0)
        state["workflow"]["current_stage"] = "reflect_round"
        observation = dict(state["execution"].get("latest_execution_observation", {}) or {})
        selected_action = dict(state["execution"].get("current_action", {}) or {})
        target_capability = str(observation.get("target_capability") or selected_action.get("target_capability") or "")
        if observation:
            state["blackboard"]["latest_execution_observation"] = observation
            state["blackboard"]["observations"] = dedupe_keep_order(list(state["blackboard"].get("observations", []) or []) + [observation])
        if target_capability:
            if observation.get("status") in {"success", "completed"} and target_capability in capability_sequence():
                state["task_board"] = mark_task_completed(dict(state.get("task_board", {}) or {}), target_capability)
            elif observation.get("status") == "failed":
                state["task_board"] = mark_task_blocked(
                    dict(state.get("task_board", {}) or {}),
                    target_capability,
                    reason=str(observation.get("error_summary") or "execution_failed"),
                )
            elif observation.get("status") == "skipped":
                state["task_board"] = mark_task_abandoned(
                    dict(state.get("task_board", {}) or {}),
                    target_capability,
                    reason=str(observation.get("error_summary") or "execution_skipped"),
                )
        reflection = ReflectionRecord(
            agent_name="orchestrator",
            round_id=round_id,
            target_task_id=str(state.get("task", {}).get("task_id") or ""),
            selected_action=selected_action,
            tradeoff_summary=list((state.get("deliberation", {}).get("arbitrations", []) or [{}])[-1].get("disagreement_summary", []) or [])
            if state.get("deliberation", {}).get("arbitrations")
            else [],
            failure_pattern=(str(observation.get("error_summary") or "") if observation.get("status") == "failed" else None),
            continue_deliberation=not bool(state.get("services", {}).get("termination_requested"))
            and str((state.get("workflow", {}) or {}).get("run_status") or "") not in {"waiting_external", "needs_human"},
            content={"observation_status": observation.get("status")},
            confidence=0.9,
        )
        state["deliberation"]["reflections"] = dedupe_keep_order(
            list(state["deliberation"].get("reflections", []) or []) + [reflection.model_dump(mode="json")]
        )
        if reflection.failure_pattern:
            state["deliberation"]["disagreement_records"] = dedupe_keep_order(
                list(state["deliberation"].get("disagreement_records", []) or [])
                + [{"round_id": round_id, "pattern": reflection.failure_pattern, "selected_action": selected_action}]
            )
            state["memory"]["historical_failures"] = dedupe_keep_order(
                list(state["memory"].get("historical_failures", []) or [])
                + [{"stage": target_capability, "error_summary": reflection.failure_pattern}]
            )
        with _memory_store_scope(runtime, runtime_ctx.resolved_db_uri) as store:
            memory_write = gateway.call(
                "write_memory_reflection",
                {"state": state, "round_id": round_id},
                store=store,
            )
            state = _record_agent_tool_call(
                state,
                tool_name="write_memory_reflection",
                phase="reflect_round",
                payload={"round_id": round_id},
                result=memory_write,
            )
        state["services"]["llm_context_summary"] = build_llm_context_summary(state)
        _append_decision_trace(state, {"node": "reflect_round", "message_type": "reflection", **reflection.model_dump(mode="json")})
        return _finalize_node_output(before, state)

    return _node


def make_check_termination_node(runtime: RuntimeContext) -> Callable[[MaterialTaskState | dict[str, Any]], Command]:
    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> Command:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        state["workflow"]["current_stage"] = "check_termination"
        derived_status, wait_reason = _derive_run_status(state)
        state["workflow"]["run_status"] = derived_status
        state["workflow"]["wait_reason"] = wait_reason
        if derived_status == "needs_recovery" and not state["diagnostics"].get("recovery_summary"):
            latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
            state = _append_framework_diagnostic(
                state,
                code="needs_recovery_without_structured_recovery_summary",
                detail={"latest_execution_observation": latest},
            )
        if derived_status == "ready_to_finalize" and not state["services"].get("termination_requested"):
            state = _append_framework_diagnostic(
                state,
                code="ready_to_finalize_waiting_for_deliberate_finalization",
                detail={"round_id": int((state.get("deliberation", {}) or {}).get("round_index", 0) or 0)},
            )
        if bool(state["services"].get("termination_requested")) or derived_status in TERMINAL_RUN_STATUSES:
            goto: str | object = "final_report"
        elif derived_status in {"waiting_external", "needs_human"}:
            goto = END
        else:
            goto = "observe_state"
        return Command(update=_finalize_node_output(before, state), goto=goto)

    return _node


def make_final_report_node(
    runtime: RuntimeContext,
    *,
    skills_root: str,
) -> Callable[[MaterialTaskState | dict[str, Any]], dict[str, Any]]:
    reporter = ReporterAgent(runtime, skills_root)
    gateway = AgentToolGateway()

    def _node(state_payload: MaterialTaskState | dict[str, Any]) -> dict[str, Any]:
        before = _state_dict(state_payload)
        state = copy.deepcopy(before)
        workdir = state["execution"]["workdir"]
        os.makedirs(workdir, exist_ok=True)
        run_status = str(state["workflow"].get("run_status") or "")
        if run_status in {"ready_to_finalize", "running", "pending", ""}:
            # Final completion reflects execution success/failure. Validation quality is tracked in report fields.
            state["workflow"]["run_status"] = "completed"
        quality_grade = _quality_grade_from_state(state)
        state["diagnostics"]["quality_grade"] = quality_grade
        validation_report = dict(state["diagnostics"].get("validation_report", {}) or {})
        validation_report["quality_grade"] = quality_grade
        state["diagnostics"]["validation_report"] = validation_report
        final_summary = reporter.summarize_material(state=state)
        final_summary["quality_grade"] = quality_grade
        state["services"]["final_report"] = final_summary
        predeclared_paths = {
            "mobility_results_path": os.path.join(workdir, "mobility_results.json"),
            "fit_diagnostics_path": os.path.join(workdir, "fit_diagnostics.json"),
            "decision_trace_path": os.path.join(workdir, "decision_trace.json"),
            "tool_trace_path": os.path.join(workdir, "tool_trace.json"),
            "retrieval_trace_path": os.path.join(workdir, "retrieval_trace.json"),
            "parameter_plan_path": os.path.join(workdir, "parameter_plan.json"),
            "skill_trace_path": os.path.join(workdir, "skill_trace.json"),
            "recovery_trace_path": os.path.join(workdir, "recovery_trace.json"),
            "recovery_diagnosis_path": os.path.join(workdir, "recovery_diagnosis.json"),
            "validation_report_path": os.path.join(workdir, "validation_report.json"),
            "deliberation_trace_path": os.path.join(workdir, "deliberation_trace.json"),
            "workflow_contract_path": os.path.join(workdir, "workflow_contract.json"),
            "workflow_contract_history_path": os.path.join(workdir, "workflow_contract_history.json"),
            "decision_ledger_path": os.path.join(workdir, "decision_ledger.json"),
            "execution_checkpoint_path": os.path.join(workdir, "execution_checkpoint.json"),
            "final_summary_path": os.path.join(workdir, "final_summary.json"),
            "material_outcome_path": os.path.join(workdir, "material_outcome.json"),
        }
        state["execution"]["artifact_paths"] = {**dict(state["execution"].get("artifact_paths", {}) or {}), **predeclared_paths}
        state["execution"]["artifact_registry"] = dict(state["execution"]["artifact_paths"])
        state["workflow"]["current_stage"] = "final_report"
        state["workflow"]["stage_status"]["final_report"] = "success"
        outcome = build_material_outcome(state).model_dump(mode="json")
        written = gateway.call(
            "write_runtime_artifacts",
            {
                "workdir": workdir,
                "state": state,
                "final_summary": final_summary,
                "material_outcome": outcome,
            },
        )
        state = _record_agent_tool_call(
            state,
            tool_name="write_runtime_artifacts",
            phase="final_report",
            payload={"workdir": workdir},
            result=written,
        )
        paths = dict(written.get("artifact_paths", {}) or {})
        state["execution"]["artifact_paths"] = {**dict(state["execution"].get("artifact_paths", {}) or {}), **paths}
        state["execution"]["artifact_registry"] = dict(state["execution"]["artifact_paths"])
        if runtime.compatibility_export_enabled:
            checkpoint_path = export_compatibility_checkpoint(state, reason="finalization")
            state["execution"]["compatibility_checkpoint_path"] = checkpoint_path
            state["execution"]["compatibility_checkpoint_history"] = dedupe_keep_order(
                list(state["execution"].get("compatibility_checkpoint_history", []) or []) + [checkpoint_path]
            )
        final_state = MaterialTaskState.from_dict(state).to_dict()
        save_state_snapshot(workdir=workdir, state=final_state, checkpoint_subdir=runtime.checkpoint_subdir)
        return _finalize_node_output(before, final_state)

    return _node
