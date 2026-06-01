from __future__ import annotations

import copy
import json
import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, model_validator

from ..config_runtime import normalize_llm_provider
from ..tools.schemas import ToolExecutionResult
from ..utils import dedupe_keep_order


_UTC = timezone.utc

CAPABILITY_SEQUENCE = [
    "prepare",
    "relax",
    "scf",
    "band",
    "effective_mass",
    "strain_loop",
    "mobility",
    "validation",
]

CAPABILITY_DEPENDENCIES: dict[str, list[str]] = {
    "prepare": [],
    "relax": ["prepare"],
    "scf": ["relax"],
    "band": ["scf"],
    "effective_mass": ["band"],
    "strain_loop": ["effective_mass"],
    "mobility": ["strain_loop"],
    "validation": ["mobility"],
}

COMPUTE_CAPABILITIES = tuple(
    capability for capability in CAPABILITY_SEQUENCE if capability != "validation"
)

APPEND_FIELDS = {
    "material.warnings",
    "workflow.completed_stages",
    "diagnostics.errors",
    "diagnostics.recovery_history",
    "diagnostics.consultation_trace",
    "execution.tool_trace",
    "execution.tool_invocations",
    "execution.failure_history",
    "execution.skill_trace",
    "execution.external_jobs",
    "execution.pending_events",
    "execution.event_history",
    "execution.consumed_event_ids",
    "execution.compatibility_checkpoint_history",
    "agent.agent_decisions",
    "agent.decision_trace",
    "blackboard.observations",
    "blackboard.risk_flags",
    "blackboard.anomaly_flags",
    "deliberation.rounds",
    "deliberation.proposals",
    "deliberation.critiques",
    "deliberation.preferences",
    "deliberation.arbitrations",
    "deliberation.selected_actions",
    "deliberation.reflections",
    "deliberation.disagreement_records",
    "deliberation.rationale_history",
    "memory.recovered_case_patterns",
    "memory.validation_case_patterns",
    "memory.historical_failures",
    "memory.reusable_heuristics",
    "services.framework_diagnostics",
    "services.retrieval_trace",
    "services.workflow_contract_history",
    "services.decision_ledger",
    "services.step_checkpoints",
    "batch.queue",
    "batch.running_items",
    "batch.completed_items",
    "batch.failed_items",
    "batch.skipped_items",
}

STABLE_CHECKPOINT_STAGES = {
    "prepare",
    "relax",
    "scf",
    "band",
    "effective_mass",
    "strain_loop",
    "mobility",
    "validation",
}

TERMINAL_RUN_STATUSES = {"completed", "failed", "aborted", "skipped"}
WAITING_RUN_STATUSES = {"waiting_external", "needs_human"}
EXTERNAL_EVENT_TYPES = {
    "job_completed",
    "job_failed",
    "job_timeout",
    "artifact_missing",
    "manual_override",
    "resume_requested",
}


def utc_now() -> datetime:
    return datetime.now(_UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def has_mobility_results_payload(results_payload: Mapping[str, Any] | None) -> bool:
    results = dict(results_payload or {})
    return bool(dict(results.get("results_by_direction", {}) or {}))


def has_failed_compute_stage(stage_status_payload: Mapping[str, Any] | None) -> bool:
    stage_status = dict(stage_status_payload or {})
    return any(str(stage_status.get(stage) or "") == "failed" for stage in COMPUTE_CAPABILITIES)


def has_completed_compute_payload(
    *,
    results: Mapping[str, Any] | None,
    stage_status: Mapping[str, Any] | None,
) -> bool:
    return has_mobility_results_payload(results) and not has_failed_compute_stage(stage_status)


def derive_compute_status(
    *,
    run_status: str | None,
    results: Mapping[str, Any] | None,
    stage_status: Mapping[str, Any] | None,
) -> str:
    raw_status = str(run_status or "").strip() or "pending"
    if raw_status == "ready_to_finalize":
        return "running"
    if raw_status in WAITING_RUN_STATUSES or raw_status in {"pending", "running"}:
        return raw_status

    compute_failed = has_failed_compute_stage(stage_status)
    if has_completed_compute_payload(results=results, stage_status=stage_status):
        return "completed"
    if compute_failed or raw_status in {"failed", "aborted"}:
        return "failed"
    if raw_status in {"completed", "skipped"}:
        return "skipped"
    return raw_status


def derive_compute_status_from_state_payload(state_payload: MaterialTaskState | dict[str, Any]) -> str:
    state = state_payload if isinstance(state_payload, MaterialTaskState) else MaterialTaskState.from_dict(state_payload)
    return derive_compute_status(
        run_status=state.workflow.run_status,
        results=state.physics_results.results,
        stage_status=state.workflow.stage_status,
    )


def has_completed_compute_state_payload(state_payload: MaterialTaskState | dict[str, Any]) -> bool:
    state = state_payload if isinstance(state_payload, MaterialTaskState) else MaterialTaskState.from_dict(state_payload)
    return has_completed_compute_payload(results=state.physics_results.results, stage_status=state.workflow.stage_status)


def _payload_has_substance(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def has_structured_validation_report_payload(validation_report: Mapping[str, Any] | None) -> bool:
    validation = dict(validation_report or {})
    if not validation:
        return False
    meaningful_keys = (
        "decision",
        "recommended_action",
        "fit_metrics",
        "channel_reviews",
        "accepted_channels",
        "rejected_channels",
        "retained_subchannels",
        "rejected_subchannels",
        "all_subchannels",
        "warnings",
        "failed_checks",
        "anomaly_flags",
        "confidence_score",
    )
    return any(_payload_has_substance(validation.get(key)) for key in meaningful_keys)


def validation_report_supports_finalize(
    *,
    validation_report: Mapping[str, Any] | None,
    validation_stage_status: str | None,
    latest_observation: Mapping[str, Any] | None = None,
) -> bool:
    validation = dict(validation_report or {})
    if not validation:
        return False
    if str(validation_stage_status or "").strip() == "success":
        return True
    latest = dict(latest_observation or {})
    if (
        str(latest.get("target_capability") or "").strip() == "validation"
        and str(latest.get("status") or "").strip() in {"success", "completed"}
    ):
        return True
    return has_structured_validation_report_payload(validation)


def derive_compute_status_from_outcome_payload(outcome_payload: Mapping[str, Any] | None) -> str:
    outcome = dict(outcome_payload or {})
    explicit_status = str(outcome.get("status") or "").strip()
    explicit_final_status = str(outcome.get("final_status") or "").strip()
    if explicit_status and not explicit_final_status:
        if explicit_status == "ready_to_finalize":
            return "running"
        if explicit_status in WAITING_RUN_STATUSES or explicit_status in TERMINAL_RUN_STATUSES or explicit_status in {"pending", "running"}:
            return explicit_status
    return derive_compute_status(
        run_status=str(explicit_final_status or explicit_status or "pending"),
        results=dict(outcome.get("results", {}) or {}),
        stage_status=dict(outcome.get("stage_status", {}) or {}),
    )


def resolve_outcome_scientific_decision(outcome_payload: Mapping[str, Any] | None) -> str | None:
    outcome = dict(outcome_payload or {})
    validation = dict(outcome.get("validation_report", {}) or {})
    decision = str(outcome.get("final_acceptance") or validation.get("decision") or "").strip()
    return decision or None


def scientific_decision_bucket(decision: str | None) -> str:
    normalized = str(decision or "").strip().lower()
    if normalized in {"pass", "accepted"}:
        return "passed"
    if normalized in {"pass_with_warning", "accepted_with_warning"}:
        return "warning"
    if normalized in {"fail", "rejected"}:
        return "failed"
    return "unknown"


class TaskSection(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_type: Literal["single_material", "batch_database"] = "single_material"
    user_goal: str = ""
    root_path: str = ""
    collection_name: str | None = None
    parent_batch_id: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    dry_run: bool = False


class MaterialSection(BaseModel):
    material_id: str = "2D_Material"
    composition: str | None = None
    structure_summary: dict[str, Any] = Field(default_factory=dict)
    structure_metadata: dict[str, Any] = Field(default_factory=dict)
    atom_count: int = 0
    preflight_summary: dict[str, Any] = Field(default_factory=dict)
    preflight_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    poscar_path: str | None = None
    potcar_path: str | None = None


class WorkflowSection(BaseModel):
    current_stage: str = "observe_state"
    completed_stages: list[str] = Field(default_factory=list)
    stage_status: dict[str, str] = Field(default_factory=dict)
    run_status: str = "pending"
    next_action: str | None = None
    retry_budget: int = 2
    retry_counts: dict[str, int] = Field(default_factory=dict)
    refinement_rounds: int = 0
    max_refinement_rounds: int = 1
    termination_reason: str | None = None
    wait_reason: str | None = None
    escalated_to_human: bool = False
    pending_human_action: dict[str, Any] = Field(default_factory=dict)
    pending_action_payload: dict[str, Any] = Field(default_factory=dict)


class MissionSection(BaseModel):
    user_goal: str = ""
    material_id: str = "2D_Material"
    required_outputs: list[str] = Field(default_factory=lambda: ["mobility_results", "validation_report", "final_summary"])
    reliability_target: str = "high"
    cost_budget: dict[str, Any] = Field(default_factory=lambda: {"retry_budget": 2, "refinement_budget": 1})
    runtime_constraints: dict[str, Any] = Field(default_factory=dict)


class BlackboardSection(BaseModel):
    validated_facts: dict[str, Any] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    parsed_artifacts: dict[str, Any] = Field(default_factory=dict)
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[str] = Field(default_factory=list)
    anomaly_flags: list[str] = Field(default_factory=list)
    latest_execution_observation: dict[str, Any] = Field(default_factory=dict)


class TaskBoardSection(BaseModel):
    pending_tasks: list[dict[str, Any]] = Field(default_factory=list)
    active_tasks: list[dict[str, Any]] = Field(default_factory=list)
    completed_tasks: list[dict[str, Any]] = Field(default_factory=list)
    blocked_tasks: list[dict[str, Any]] = Field(default_factory=list)
    abandoned_tasks: list[dict[str, Any]] = Field(default_factory=list)


class DeliberationSection(BaseModel):
    round_index: int = 0
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    proposals: list[dict[str, Any]] = Field(default_factory=list)
    critiques: list[dict[str, Any]] = Field(default_factory=list)
    preferences: list[dict[str, Any]] = Field(default_factory=list)
    arbitrations: list[dict[str, Any]] = Field(default_factory=list)
    selected_actions: list[dict[str, Any]] = Field(default_factory=list)
    reflections: list[dict[str, Any]] = Field(default_factory=list)
    disagreement_records: list[dict[str, Any]] = Field(default_factory=list)
    rationale_history: list[str] = Field(default_factory=list)


class ExecutionSection(BaseModel):
    workdir: str = ""
    thread_id: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_registry: dict[str, str] = Field(default_factory=dict)
    latest_tool_name: str | None = None
    latest_tool_result: dict[str, Any] = Field(default_factory=dict)
    latest_execution_observation: dict[str, Any] = Field(default_factory=dict)
    current_action: dict[str, Any] = Field(default_factory=dict)
    action_status: str = "pending"
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    tool_invocations: list[dict[str, Any]] = Field(default_factory=list)
    skill_trace: list[dict[str, Any]] = Field(default_factory=list)
    failure_history: list[dict[str, Any]] = Field(default_factory=list)
    job_ids: dict[str, Any] = Field(default_factory=dict)
    external_jobs: list[dict[str, Any]] = Field(default_factory=list)
    pending_events: list[dict[str, Any]] = Field(default_factory=list)
    event_history: list[dict[str, Any]] = Field(default_factory=list)
    consumed_event_ids: list[str] = Field(default_factory=list)
    latest_event: dict[str, Any] = Field(default_factory=dict)
    resume_markers: dict[str, Any] = Field(default_factory=dict)
    environment_summary: dict[str, Any] = Field(default_factory=dict)
    compatibility_checkpoint_path: str | None = None
    compatibility_checkpoint_history: list[str] = Field(default_factory=list)
    workdir_inputs_ready: bool = False
    pending_parameter_updates: dict[str, Any] = Field(default_factory=dict)
    execution_checkpoint: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsSection(BaseModel):
    errors: list[str] = Field(default_factory=list)
    last_error: str | None = None
    recovery_history: list[dict[str, Any]] = Field(default_factory=list)
    recovery_summary: dict[str, Any] = Field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = Field(default_factory=dict)
    strain_summary: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float | None = None
    low_confidence_reason: str | None = None
    validation_report: dict[str, Any] = Field(default_factory=dict)
    quality_grade: str | None = None
    consultation_trace: list[dict[str, Any]] = Field(default_factory=list)
    raw_evidence: dict[str, Any] = Field(default_factory=dict)
    recovery_diagnosis: dict[str, Any] = Field(default_factory=dict)


class PhysicsResultsSection(BaseModel):
    prepare_summary: dict[str, Any] = Field(default_factory=dict)
    relax_summary: dict[str, Any] = Field(default_factory=dict)
    scf_summary: dict[str, Any] = Field(default_factory=dict)
    band_summary: dict[str, Any] = Field(default_factory=dict)
    effective_mass_summary: dict[str, Any] = Field(default_factory=dict)
    strain_data_summary: dict[str, Any] = Field(default_factory=dict)
    deformation_potential_summary: dict[str, Any] = Field(default_factory=dict)
    elasticity_summary: dict[str, Any] = Field(default_factory=dict)
    mobility_summary: dict[str, Any] = Field(default_factory=dict)
    masses: dict[str, Any] = Field(default_factory=dict)
    E1: dict[str, Any] = Field(default_factory=dict)
    C2D: dict[str, Any] = Field(default_factory=dict)
    mobility: dict[str, Any] = Field(default_factory=dict)
    accepted_channels: list[str] = Field(default_factory=lambda: ["x", "y"])
    rejected_channels: list[str] = Field(default_factory=list)
    strain_plan_by_direction: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "x": [-0.02, -0.01, 0.0, 0.01, 0.02],
            "y": [-0.02, -0.01, 0.0, 0.01, 0.02],
        }
    )
    strain_data: list[dict[str, Any]] = Field(default_factory=list)
    relaxed_structure_path: str | None = None
    reciprocal_lattice: list[list[float]] = Field(default_factory=list)
    fermi_energy: float | None = None
    vbm_energy: float | None = None
    cbm_energy: float | None = None
    vbm_kpoint: list[float] = Field(default_factory=list)
    cbm_kpoint: list[float] = Field(default_factory=list)
    vbm_band_index: int | None = None
    cbm_band_index: int | None = None
    vbm_spin: int | None = None
    cbm_spin: int | None = None
    results: dict[str, Any] = Field(default_factory=dict)


class AgentSection(BaseModel):
    decision_engine: str = "llm_required"
    llm_required: bool = True
    llm_provider: str = "openai"
    agent_decisions: list[dict[str, Any]] = Field(default_factory=list)
    decision_trace: list[dict[str, Any]] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)


class AgentWorkspacesSection(BaseModel):
    orchestrator_workspace: dict[str, Any] = Field(default_factory=dict)
    planner_workspace: dict[str, Any] = Field(default_factory=dict)
    recovery_workspace: dict[str, Any] = Field(default_factory=dict)
    critic_workspace: dict[str, Any] = Field(default_factory=dict)
    judge_workspace: dict[str, Any] = Field(default_factory=dict)
    cost_guardian_workspace: dict[str, Any] = Field(default_factory=dict)
    executor_workspace: dict[str, Any] = Field(default_factory=dict)
    reporter_workspace: dict[str, Any] = Field(default_factory=dict)


class MemorySection(BaseModel):
    recovered_case_patterns: list[dict[str, Any]] = Field(default_factory=list)
    validation_case_patterns: list[dict[str, Any]] = Field(default_factory=list)
    historical_failures: list[dict[str, Any]] = Field(default_factory=list)
    reusable_heuristics: list[dict[str, Any]] = Field(default_factory=list)


class ServicesSection(BaseModel):
    loaded_skills: list[str] = Field(default_factory=list)
    skill_resolution: dict[str, Any] = Field(default_factory=dict)
    available_agent_tools: list[dict[str, Any]] = Field(default_factory=list)
    llm_context_summary: dict[str, Any] = Field(default_factory=dict)
    pending_human_payload: dict[str, Any] = Field(default_factory=dict)
    latest_human_decision: dict[str, Any] = Field(default_factory=dict)
    termination_requested: bool = False
    final_report: dict[str, Any] = Field(default_factory=dict)
    selected_action_requires_execution: bool = False
    deliberation_mode: str = "hierarchical_multi_agent"
    framework_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    parameter_plans: dict[str, Any] = Field(default_factory=dict)
    retrieval_trace: list[dict[str, Any]] = Field(default_factory=list)
    workflow_contract: dict[str, Any] = Field(default_factory=dict)
    workflow_contract_history: list[dict[str, Any]] = Field(default_factory=list)
    capability_registry: list[dict[str, Any]] = Field(default_factory=list)
    decision_ledger: list[dict[str, Any]] = Field(default_factory=list)
    step_checkpoints: list[dict[str, Any]] = Field(default_factory=list)
    material_job: dict[str, Any] = Field(default_factory=dict)
    council_output_cache: dict[str, Any] = Field(default_factory=dict)
    council_round_metrics: list[dict[str, Any]] = Field(default_factory=list)
    runtime_strategy: dict[str, Any] = Field(default_factory=dict)


class BatchSection(BaseModel):
    queue: list[dict[str, Any]] = Field(default_factory=list)
    running_items: list[dict[str, Any]] = Field(default_factory=list)
    completed_items: list[dict[str, Any]] = Field(default_factory=list)
    failed_items: list[dict[str, Any]] = Field(default_factory=list)
    skipped_items: list[dict[str, Any]] = Field(default_factory=list)
    global_statistics: dict[str, Any] = Field(default_factory=dict)


class MaterialTaskState(BaseModel):
    task: TaskSection = Field(default_factory=TaskSection)
    material: MaterialSection = Field(default_factory=MaterialSection)
    workflow: WorkflowSection = Field(default_factory=WorkflowSection)
    mission: MissionSection = Field(default_factory=MissionSection)
    blackboard: BlackboardSection = Field(default_factory=BlackboardSection)
    task_board: TaskBoardSection = Field(default_factory=TaskBoardSection)
    deliberation: DeliberationSection = Field(default_factory=DeliberationSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    diagnostics: DiagnosticsSection = Field(default_factory=DiagnosticsSection)
    physics_results: PhysicsResultsSection = Field(default_factory=PhysicsResultsSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    agent_workspaces: AgentWorkspacesSection = Field(default_factory=AgentWorkspacesSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    services: ServicesSection = Field(default_factory=ServicesSection)
    batch: BatchSection = Field(default_factory=BatchSection)

    @model_validator(mode="after")
    def _normalize(self):
        self.task.updated_at = utc_now_iso()
        if not self.mission.material_id:
            self.mission.material_id = self.material.material_id
        if not self.mission.user_goal:
            self.mission.user_goal = self.task.user_goal
        if not self.material.material_id:
            self.material.material_id = self.mission.material_id
        if not self.task.user_goal:
            self.task.user_goal = self.mission.user_goal
        self.material.preflight_tags = [str(tag) for tag in dedupe_keep_order(self.material.preflight_tags or [])]
        self.material.warnings = [str(item) for item in dedupe_keep_order(self.material.warnings or [])]
        self.workflow.completed_stages = [str(item) for item in dedupe_keep_order(self.workflow.completed_stages or [])]
        self.physics_results.accepted_channels = [str(item) for item in dedupe_keep_order(self.physics_results.accepted_channels or [])]
        self.physics_results.rejected_channels = [str(item) for item in dedupe_keep_order(self.physics_results.rejected_channels or [])]
        self.agent.loaded_skills = [str(item) for item in dedupe_keep_order(self.agent.loaded_skills or [])]
        self.services.loaded_skills = [str(item) for item in dedupe_keep_order(self.services.loaded_skills or [])]
        self.blackboard.risk_flags = [str(item) for item in dedupe_keep_order(self.blackboard.risk_flags or [])]
        self.blackboard.anomaly_flags = [str(item) for item in dedupe_keep_order(self.blackboard.anomaly_flags or [])]
        self.agent.llm_provider = normalize_llm_provider(self.agent.llm_provider)
        self.execution.artifact_registry = {
            **dict(self.execution.artifact_paths or {}),
            **dict(self.execution.artifact_registry or {}),
        }
        if not self.task_board.pending_tasks and not self.task_board.completed_tasks and not self.task_board.active_tasks:
            self.task_board.pending_tasks = _initial_pending_tasks()
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MaterialTaskState":
        if not payload:
            return cls()
        if "task" in payload:
            return cls.model_validate(payload)
        return legacy_state_to_shared_state(payload)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


STATE_TOP_LEVEL_SECTIONS = tuple(MaterialTaskState.model_fields.keys())


class ExternalEventRecord(BaseModel):
    event_id: str = ""
    event_type: str
    thread_id: str | None = None
    run_id: str | None = None
    job_id: str | None = None
    target_capability: str | None = None
    action_family: str | None = None
    timestamp: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    error_summary: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    provenance: str = "external"
    status: str | None = None
    retryable: bool | None = None
    terminal: bool | None = None

    @model_validator(mode="after")
    def _normalize(self):
        self.event_type = str(self.event_type or "").strip() or "resume_requested"
        self.thread_id = str(self.thread_id or "").strip() or None
        self.run_id = str(self.run_id or "").strip() or None
        self.job_id = str(self.job_id or "").strip() or None
        self.target_capability = str(self.target_capability or "").strip() or None
        self.action_family = str(self.action_family or "").strip() or None
        self.provenance = str(self.provenance or "").strip() or "external"
        self.payload = dict(self.payload or {})
        self.result_summary = dict(self.result_summary or {})
        self.artifact_paths = {str(k): str(v) for k, v in dict(self.artifact_paths or {}).items() if v}
        if self.status is not None:
            self.status = str(self.status).strip() or None
        if self.error_summary is not None:
            self.error_summary = str(self.error_summary)
        if self.event_type not in EXTERNAL_EVENT_TYPES:
            self.event_type = self.event_type or "resume_requested"
        if not self.event_id:
            self.event_id = "::".join(
                [
                    self.thread_id or "unknown-thread",
                    self.event_type or "unknown-event",
                    self.job_id or self.target_capability or "no-target",
                    self.timestamp or "no-timestamp",
                ]
            )
        return self


class BatchTaskState(BaseModel):
    task: TaskSection = Field(default_factory=lambda: TaskSection(task_type="batch_database"))
    mission: MissionSection = Field(default_factory=MissionSection)
    blackboard: BlackboardSection = Field(default_factory=BlackboardSection)
    task_board: TaskBoardSection = Field(default_factory=TaskBoardSection)
    deliberation: DeliberationSection = Field(default_factory=DeliberationSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    diagnostics: DiagnosticsSection = Field(default_factory=DiagnosticsSection)
    agent: AgentSection = Field(default_factory=AgentSection)
    agent_workspaces: AgentWorkspacesSection = Field(default_factory=AgentWorkspacesSection)
    memory: MemorySection = Field(default_factory=MemorySection)
    services: ServicesSection = Field(default_factory=ServicesSection)
    batch: BatchSection = Field(default_factory=BatchSection)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class MaterialRunOutcome(BaseModel):
    task_id: str
    material_id: str
    final_status: str
    status: str = ""
    final_acceptance: str | None = None
    termination_reason: str | None = None
    workdir: str
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    validation_report: dict[str, Any] = Field(default_factory=dict)
    final_summary: dict[str, Any] = Field(default_factory=dict)
    accepted_channels: list[str] = Field(default_factory=list)
    rejected_channels: list[str] = Field(default_factory=list)
    stage_status: dict[str, str] = Field(default_factory=dict)
    escalation_count: int = 0
    manual_fix_used: bool = False

    @model_validator(mode="after")
    def _sync_status(self):
        if not self.final_status and self.status:
            self.final_status = self.status
        if not self.status and self.final_status:
            self.status = self.final_status
        return self


def _initial_pending_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"capability::{capability}",
            "task_type": "capability",
            "capability": capability,
            "depends_on": list(CAPABILITY_DEPENDENCIES.get(capability, [])),
            "status": "pending",
        }
        for capability in CAPABILITY_SEQUENCE
    ]


def _append_path(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def state_payload_to_dict(payload: MaterialTaskState | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, MaterialTaskState):
        return payload.to_dict()
    return MaterialTaskState.from_dict(payload or {}).to_dict()


def deep_merge(base: Any, updates: Any, *, append_fields: set[str] | None = None, path: str = "") -> Any:
    append_fields = append_fields or APPEND_FIELDS
    if isinstance(base, dict) and isinstance(updates, dict):
        merged = {**base}
        for key, value in updates.items():
            child_path = f"{path}.{key}" if path else key
            if key in merged:
                merged[key] = deep_merge(merged[key], value, append_fields=append_fields, path=child_path)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(updates, list):
        if path in append_fields:
            return copy.deepcopy(dedupe_keep_order(list(base) + list(updates)))
        return copy.deepcopy(updates)
    return copy.deepcopy(updates)


def build_state_patch(
    before_payload: MaterialTaskState | dict[str, Any] | None,
    after_payload: MaterialTaskState | dict[str, Any] | None,
    *,
    sections: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, Any]:
    before = state_payload_to_dict(before_payload)
    after = state_payload_to_dict(after_payload)
    changed_sections = sections or STATE_TOP_LEVEL_SECTIONS
    patch: dict[str, Any] = {}
    for section in changed_sections:
        before_section = copy.deepcopy(before.get(section))
        after_section = copy.deepcopy(after.get(section))
        if section == "task" and isinstance(before_section, dict) and isinstance(after_section, dict):
            before_section["updated_at"] = ""
            after_section["updated_at"] = ""
        if before_section != after_section:
            patch[section] = copy.deepcopy(after.get(section))
    return patch


def apply_state_patch(
    state_payload: MaterialTaskState | dict[str, Any] | None,
    patch: dict[str, Any] | None,
) -> dict[str, Any]:
    base = state_payload_to_dict(state_payload)
    updated = dict(base)
    for section, value in dict(patch or {}).items():
        updated[str(section)] = copy.deepcopy(value)
    return MaterialTaskState.from_dict(updated).to_dict()


def normalize_external_event(
    payload: ExternalEventRecord | dict[str, Any],
    *,
    default_thread_id: str | None = None,
    default_run_id: str | None = None,
) -> ExternalEventRecord:
    if isinstance(payload, ExternalEventRecord):
        event = payload
    else:
        event = ExternalEventRecord.model_validate(dict(payload or {}))
    if default_thread_id and not event.thread_id:
        event.thread_id = str(default_thread_id)
    if default_run_id and not event.run_id:
        event.run_id = str(default_run_id)
    if not event.event_id:
        event = ExternalEventRecord.model_validate(event.model_dump(mode="json"))
    return event


def legacy_state_to_shared_state(payload: dict[str, Any]) -> MaterialTaskState:
    base_dir = str(payload.get("base_dir") or "")
    material_id = str(payload.get("material_id") or "2D_Material")
    user_goal = str(payload.get("user_goal") or "calculate_2d_mobility")
    state = MaterialTaskState(
        task=TaskSection(
            task_type="single_material",
            user_goal=user_goal,
            root_path=os.path.abspath(base_dir or os.getcwd()),
            dry_run=False,
        ),
        material=MaterialSection(
            material_id=material_id,
            structure_summary=dict(payload.get("structure_summary", {}) or {}),
            structure_metadata=dict(payload.get("material_metadata", {}) or {}),
            atom_count=int((payload.get("structure_summary", {}) or {}).get("atom_count", 0) or 0),
            preflight_summary=dict(payload.get("preflight_summary", {}) or {}),
            preflight_tags=list(payload.get("preflight_tags", []) or []),
            warnings=list(payload.get("warnings", []) or []),
            poscar_path=payload.get("poscar_path"),
            potcar_path=payload.get("potcar_path"),
        ),
        workflow=WorkflowSection(
            current_stage=str(payload.get("current_stage") or "observe_state"),
            completed_stages=[
                stage
                for stage, status in {
                    "relax": payload.get("relax_completed"),
                    "scf": payload.get("scf_completed"),
                    "band": payload.get("band_completed"),
                    "effective_mass": payload.get("effmass_completed"),
                    "strain_loop": payload.get("strain_completed"),
                }.items()
                if status
            ],
            stage_status={
                "relax": "success" if payload.get("relax_completed") else "pending",
                "scf": "success" if payload.get("scf_completed") else "pending",
                "band": "success" if payload.get("band_completed") else "pending",
                "effective_mass": "success" if payload.get("effmass_completed") else "pending",
                "strain_loop": "success" if payload.get("strain_completed") else "pending",
            },
            run_status=str(payload.get("run_status") or "pending"),
            retry_budget=3,
            retry_counts={"relax": int(len(payload.get("relax_retry_backups", []) or []))},
            refinement_rounds=int(payload.get("refinement_rounds", 0) or 0),
            max_refinement_rounds=int(payload.get("max_refinement_rounds", 1) or 1),
            termination_reason=payload.get("low_confidence_reason"),
            escalated_to_human=bool(payload.get("human_review_summary")),
        ),
        mission=MissionSection(
            user_goal=user_goal,
            material_id=material_id,
            runtime_constraints={"legacy_imported": True},
        ),
        blackboard=BlackboardSection(
            validated_facts={"legacy_imported": True},
            parsed_artifacts=dict(payload.get("artifact_paths", {}) or {}),
            intermediate_results=dict(payload.get("results", {}) or {}),
        ),
        execution=ExecutionSection(
            workdir=os.path.abspath(base_dir) if base_dir else "",
            thread_id=payload.get("thread_id"),
            artifact_paths=dict(payload.get("artifact_paths", {}) or {}),
            artifact_registry=dict(payload.get("artifact_paths", {}) or {}),
            latest_tool_name=payload.get("latest_tool_name"),
            latest_tool_result=dict(payload.get("latest_tool_result", {}) or {}),
            tool_trace=list(payload.get("tool_trace", []) or []),
            tool_invocations=list(payload.get("tool_trace", []) or []),
            environment_summary={},
            compatibility_checkpoint_path=os.path.join(base_dir, "checkpoint.pkl") if base_dir else None,
        ),
        diagnostics=DiagnosticsSection(
            errors=list(payload.get("errors", []) or []),
            last_error=(list(payload.get("errors", []) or [])[-1] if payload.get("errors") else None),
            recovery_history=list(payload.get("recovery_trace", []) or []),
            recovery_summary=dict(payload.get("recovery_summary", {}) or {}),
            fit_diagnostics=dict(payload.get("fit_diagnostics", {}) or {}),
            strain_summary=dict(payload.get("strain_summary", {}) or {}),
            confidence_score=payload.get("confidence_score"),
            low_confidence_reason=payload.get("low_confidence_reason"),
            validation_report=dict(payload.get("validation_report", {}) or {}),
            consultation_trace=list(payload.get("decision_trace", []) or []),
        ),
        physics_results=PhysicsResultsSection(
            band_summary=dict(payload.get("band_summary", {}) or {}),
            effective_mass_summary=dict(payload.get("mass_summary", {}) or {}),
            strain_data_summary=dict(payload.get("strain_summary", {}) or {}),
            mobility_summary=dict(payload.get("mobility_summary", {}) or {}),
            masses={
                "electron_mass_x": payload.get("electron_mass_x"),
                "electron_mass_y": payload.get("electron_mass_y"),
                "electron_mass_dos": payload.get("electron_mass_dos"),
                "hole_mass_x": payload.get("hole_mass_x"),
                "hole_mass_y": payload.get("hole_mass_y"),
                "hole_mass_dos": payload.get("hole_mass_dos"),
            },
            accepted_channels=["x", "y"],
            rejected_channels=[],
            strain_data=list(payload.get("strain_data", []) or []),
            relaxed_structure_path=payload.get("relaxed_poscar"),
            reciprocal_lattice=list(payload.get("reciprocal_lattice", []) or []),
            fermi_energy=payload.get("fermi_energy"),
            vbm_energy=payload.get("vbm_energy"),
            cbm_energy=payload.get("cbm_energy"),
            vbm_kpoint=list(payload.get("vbm_kpoint", []) or []),
            cbm_kpoint=list(payload.get("cbm_kpoint", []) or []),
            vbm_band_index=payload.get("vbm_band_index"),
            cbm_band_index=payload.get("cbm_band_index"),
            vbm_spin=payload.get("vbm_spin"),
            cbm_spin=payload.get("cbm_spin"),
            results=dict(payload.get("results", {}) or {}),
        ),
        agent=AgentSection(
            decision_engine=str(payload.get("decision_engine") or payload.get("planner_mode") or "llm_required"),
            llm_required=bool(payload.get("llm_required", payload.get("llm_enabled", True))),
            llm_provider=normalize_llm_provider(str(payload.get("llm_provider") or "openai")),
            agent_decisions=list(payload.get("decision_trace", []) or []),
            decision_trace=list(payload.get("decision_trace", []) or []),
        ),
        services=ServicesSection(loaded_skills=[]),
    )
    return state


def make_initial_material_state(
    *,
    material_id: str,
    root_path: str,
    workdir: str,
    poscar_path: str,
    potcar_path: str,
    user_goal: str,
    decision_engine: str,
    llm_required: bool,
    llm_provider: str,
    max_refinement_rounds: int,
    dry_run: bool,
    task_id: str | None = None,
    thread_id: str | None = None,
) -> MaterialTaskState:
    root_abs = os.path.abspath(root_path)
    workdir_abs = os.path.abspath(workdir)
    runtime_constraints = {
        "dry_run": bool(dry_run),
        "max_refinement_rounds": int(max_refinement_rounds),
    }
    state = MaterialTaskState(
        task=TaskSection(
            task_id=task_id or uuid.uuid4().hex,
            task_type="single_material",
            user_goal=user_goal,
            root_path=root_abs,
            dry_run=bool(dry_run),
        ),
        material=MaterialSection(
            material_id=material_id,
            poscar_path=os.path.abspath(poscar_path),
            potcar_path=os.path.abspath(potcar_path),
        ),
        workflow=WorkflowSection(
            current_stage="observe_state",
            stage_status={},
            retry_budget=3,
            retry_counts={},
            max_refinement_rounds=int(max_refinement_rounds),
        ),
        mission=MissionSection(
            user_goal=user_goal,
            material_id=material_id,
            cost_budget={"retry_budget": 2, "refinement_budget": int(max_refinement_rounds)},
            runtime_constraints=runtime_constraints,
        ),
        blackboard=BlackboardSection(
            validated_facts={"material_id": material_id},
            parsed_artifacts={
                "poscar_path": os.path.abspath(poscar_path),
                "potcar_path": os.path.abspath(potcar_path),
            },
        ),
        task_board=TaskBoardSection(pending_tasks=_initial_pending_tasks()),
        execution=ExecutionSection(
            workdir=workdir_abs,
            thread_id=thread_id,
            environment_summary={"cwd": root_abs},
            compatibility_checkpoint_path=os.path.join(workdir_abs, "checkpoint.pkl"),
            artifact_registry={},
        ),
        agent=AgentSection(
            decision_engine=decision_engine,
            llm_required=llm_required,
            llm_provider=normalize_llm_provider(llm_provider),
        ),
        services=ServicesSection(),
    )
    return state


def record_stage_status(state: MaterialTaskState, stage: str, status: str) -> MaterialTaskState:
    payload = state.to_dict()
    payload["workflow"]["current_stage"] = stage
    payload["workflow"]["stage_status"][stage] = status
    if status == "success":
        payload["workflow"]["completed_stages"] = dedupe_keep_order(
            list(payload["workflow"].get("completed_stages", []) or []) + [stage]
        )
        payload["blackboard"]["validated_facts"][f"{stage}_completed"] = True
    return MaterialTaskState.from_dict(payload)


def apply_state_updates(state_payload: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    base = MaterialTaskState.from_dict(state_payload).to_dict()
    merged = deep_merge(base, updates)
    return MaterialTaskState.from_dict(merged).to_dict()


def clear_resolved_error_state(state_payload: dict[str, Any], *, resolved_stage: str | None = None) -> dict[str, Any]:
    payload = MaterialTaskState.from_dict(state_payload).to_dict()
    diagnostics = dict(payload.get("diagnostics", {}) or {})
    latest_observation = dict((payload.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    recovery_summary = dict(diagnostics.get("recovery_summary", {}) or {})
    recovery_stage = str(recovery_summary.get("stage") or recovery_summary.get("current_stage") or "")
    previous_failed_stage = str(latest_observation.get("target_capability") or latest_observation.get("stage") or "")
    previous_failed = str(latest_observation.get("status") or "") == "failed"
    has_errors = bool(diagnostics.get("last_error")) or bool(list(diagnostics.get("errors", []) or []))
    should_clear = has_errors and (
        not resolved_stage
        or recovery_stage == resolved_stage
        or (previous_failed and previous_failed_stage == resolved_stage)
    )
    if should_clear:
        payload["diagnostics"]["last_error"] = None
        payload["diagnostics"]["errors"] = []
        if not resolved_stage or recovery_stage == resolved_stage:
            payload["diagnostics"]["recovery_summary"] = {}
    return MaterialTaskState.from_dict(payload).to_dict()


def register_tool_result(state_payload: dict[str, Any], result: ToolExecutionResult) -> dict[str, Any]:
    state = MaterialTaskState.from_dict(state_payload)
    payload = state.to_dict()
    payload["execution"]["latest_tool_name"] = result.stage
    payload["execution"]["latest_tool_result"] = result.model_dump(mode="json")
    payload["execution"]["artifact_paths"] = deep_merge(payload["execution"]["artifact_paths"], result.artifact_paths)
    payload["execution"]["artifact_registry"] = deep_merge(payload["execution"]["artifact_registry"], result.artifact_paths)
    trace_item = {
        "stage": result.stage,
        "status": result.status,
        "error_summary": result.error_summary,
        "warnings": list(result.warnings),
        "artifact_paths": dict(result.artifact_paths),
        "key_summary": dict(result.key_summary),
        "invocation_source": result.invocation_source,
        "duration_s": float(result.duration_s),
    }
    payload["execution"]["tool_trace"] = dedupe_keep_order(
        list(payload["execution"].get("tool_trace", []) or []) + [trace_item]
    )
    payload["execution"]["tool_invocations"] = dedupe_keep_order(
        list(payload["execution"].get("tool_invocations", []) or []) + [trace_item]
    )
    payload["diagnostics"]["raw_evidence"] = {
        **dict(payload["diagnostics"].get("raw_evidence", {}) or {}),
        result.stage: result.raw_evidence.model_dump(mode="json"),
    }
    if result.error_summary:
        payload["diagnostics"]["last_error"] = result.error_summary
        payload["diagnostics"]["errors"] = dedupe_keep_order(
            list(payload["diagnostics"].get("errors", []) or []) + [result.error_summary]
        )
        payload["execution"]["failure_history"] = dedupe_keep_order(
            list(payload["execution"].get("failure_history", []) or [])
            + [{"stage": result.stage, "error_summary": result.error_summary}]
        )
    if result.warnings:
        payload["material"]["warnings"] = dedupe_keep_order(
            list(payload["material"].get("warnings", []) or []) + list(result.warnings)
        )
        payload["blackboard"]["risk_flags"] = dedupe_keep_order(
            list(payload["blackboard"].get("risk_flags", []) or []) + list(result.warnings)
        )
    payload["blackboard"]["parsed_artifacts"] = deep_merge(
        dict(payload["blackboard"].get("parsed_artifacts", {}) or {}),
        result.artifact_paths,
    )
    payload = deep_merge(payload, result.state_updates)
    if result.success:
        payload = clear_resolved_error_state(payload, resolved_stage=result.stage)
    return MaterialTaskState.from_dict(payload).to_dict()


def build_material_outcome(state_payload: dict[str, Any]) -> MaterialRunOutcome:
    state = MaterialTaskState.from_dict(state_payload)
    final_summary = dict(state.services.final_report or {})
    if not final_summary:
        final_summary = dict(state.execution.latest_tool_result) if state.workflow.current_stage == "final_report" else {}
    if state.execution.artifact_paths.get("final_summary_path"):
        try:
            with open(state.execution.artifact_paths["final_summary_path"], "r", encoding="utf-8") as handle:
                final_summary = json.load(handle)
        except Exception:
            pass
    validation_report = {
        **dict(state.diagnostics.validation_report),
        **({"quality_grade": state.diagnostics.quality_grade} if state.diagnostics.quality_grade else {}),
    }
    accepted_channels = list(validation_report.get("accepted_channels", []) or [])
    rejected_channels = list(validation_report.get("rejected_channels", []) or [])
    if not accepted_channels and not rejected_channels and has_mobility_results_payload(state.physics_results.results):
        accepted_channels = list(state.physics_results.accepted_channels)
        rejected_channels = list(state.physics_results.rejected_channels)
    return MaterialRunOutcome(
        task_id=state.task.task_id,
        material_id=state.material.material_id,
        status=derive_compute_status_from_state_payload(state),
        final_status=state.workflow.run_status,
        final_acceptance=(state.diagnostics.validation_report or {}).get("decision")
        or final_summary.get("final_acceptance"),
        termination_reason=state.workflow.termination_reason,
        workdir=state.execution.workdir,
        artifact_paths=dict(state.execution.artifact_paths),
        results=dict(state.physics_results.results),
        warnings=list(state.material.warnings),
        errors=list(state.diagnostics.errors),
        confidence_score=state.diagnostics.confidence_score,
        validation_report=validation_report,
        final_summary=final_summary,
        accepted_channels=accepted_channels,
        rejected_channels=rejected_channels,
        stage_status=dict(state.workflow.stage_status),
        escalation_count=sum(1 for item in list(state.diagnostics.consultation_trace or []) if isinstance(item, dict)),
        manual_fix_used=any(
            str((item or {}).get("action") or "") == "manual_fix_resume"
            for item in list(state.diagnostics.recovery_history or [])
            if isinstance(item, dict)
        ),
    )


def build_legacy_state_snapshot(state_payload: dict[str, Any]) -> dict[str, Any]:
    state = MaterialTaskState.from_dict(state_payload)
    return {
        "material_id": state.material.material_id,
        "base_dir": state.execution.workdir,
        "poscar_path": state.material.poscar_path,
        "potcar_path": state.material.potcar_path,
        "current_stage": state.workflow.current_stage,
        "run_status": state.workflow.run_status,
        "warnings": list(state.material.warnings),
        "errors": list(state.diagnostics.errors),
        "artifact_paths": dict(state.execution.artifact_paths),
        "thread_id": state.execution.thread_id,
        "results": dict(state.physics_results.results),
        "validation_report": dict(state.diagnostics.validation_report),
        "decision_trace": list(state.agent.decision_trace),
        "tool_trace": list(state.execution.tool_trace),
        "recovery_trace": list(state.diagnostics.recovery_history),
        "fit_diagnostics": dict(state.diagnostics.fit_diagnostics),
        "strain_summary": dict(state.diagnostics.strain_summary),
        "relaxed_poscar": state.physics_results.relaxed_structure_path,
        "fermi_energy": state.physics_results.fermi_energy,
        "vbm_energy": state.physics_results.vbm_energy,
        "cbm_energy": state.physics_results.cbm_energy,
        "vbm_kpoint": list(state.physics_results.vbm_kpoint),
        "cbm_kpoint": list(state.physics_results.cbm_kpoint),
        "vbm_band_index": state.physics_results.vbm_band_index,
        "cbm_band_index": state.physics_results.cbm_band_index,
        "vbm_spin": state.physics_results.vbm_spin,
        "cbm_spin": state.physics_results.cbm_spin,
        "reciprocal_lattice": list(state.physics_results.reciprocal_lattice),
        "strain_data": list(state.physics_results.strain_data),
        "decision_engine": state.agent.decision_engine,
        "llm_required": state.agent.llm_required,
        "planner_mode": state.agent.decision_engine,
        "llm_enabled": state.agent.llm_required,
        "llm_provider": state.agent.llm_provider,
    }


def export_compatibility_checkpoint(state_payload: dict[str, Any], *, reason: str) -> str:
    state = MaterialTaskState.from_dict(state_payload)
    checkpoint_path = state.execution.compatibility_checkpoint_path or os.path.join(state.execution.workdir, "checkpoint.pkl")
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
    export_payload = {
        "reason": reason,
        "exported_at": utc_now_iso(),
        "shared_state": state.to_dict(),
        "legacy_view": build_legacy_state_snapshot(state.to_dict()),
    }
    with open(checkpoint_path, "wb") as handle:
        pickle.dump(export_payload, handle)
    history = list(state.execution.compatibility_checkpoint_history or [])
    _append_path(history, checkpoint_path)
    return checkpoint_path
