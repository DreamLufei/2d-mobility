from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
try:
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover - optional for low-dependency tests
    StructuredTool = None  # type: ignore

from ..graph.stage_contracts import STAGE_ORDER, get_stage_contract
from ..graph.state import (
    TERMINAL_RUN_STATUSES,
    derive_compute_status_from_outcome_payload,
    resolve_outcome_scientific_decision,
    scientific_decision_bucket,
    validation_report_supports_finalize,
)
from ..policy.probe import build_stage_probe_from_state
from ..policy.retrieval import PolicyKnowledgeBase, default_knowledge_base
from ..policy.schemas import RetrievedEvidence
from ..rag.wiki_sync import load_house_policy_documents
from ..runtime.checkpointing import runtime_state_snapshot_path
from ..runtime.refinement_policy import (
    DEFAULT_FIT_R2_THRESHOLD,
    DEFAULT_REFINEMENT_TARGET_POINTS,
    resolve_refinement_sampling,
    validation_requires_refinement,
)
from ..runtime.store import (
    find_recovery_cases,
    find_validation_heuristics,
    list_batch_statistics,
    list_skill_metadata,
    record_recovery_case,
    put_memory_item,
)
from ..skills import SkillResolutionRequest, discover_skills, load_skill, resolve_skills
from ..skills.models import SkillLoadResult as LoadedSkillResult
from ..skills.registry import canonical_skill_name
from ..tools.anomaly_detector import detect_basic_anomalies
from ..tools.physics_validator import validate_physics_window
from ..utils import dedupe_keep_order, summarize_poscar
from .action_registry import capability_dependencies, get_action_spec, list_action_families
from .deliberation_loop import all_tasks_resolved, next_pending_task


class WorkspaceInspectionInput(BaseModel):
    workdir: str
    poscar_path: str | None = None
    potcar_path: str | None = None
    checkpoint_path: str | None = None
    artifact_registry: dict[str, str] = Field(default_factory=dict)


class WorkspaceInspectionResult(BaseModel):
    available_inputs: dict[str, bool] = Field(default_factory=dict)
    existing_stage_dirs: list[str] = Field(default_factory=list)
    existing_artifacts: list[str] = Field(default_factory=list)
    compatibility_artifacts: dict[str, bool] = Field(default_factory=dict)
    checkpoint_available: bool = False
    warnings: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class ArtifactInspectionInput(BaseModel):
    workdir: str
    target_capability: str | None = None
    artifact_registry: dict[str, str] = Field(default_factory=dict)


class ArtifactInspectionResult(BaseModel):
    target_capability: str | None = None
    stage_dir_exists: bool = False
    expected_artifacts: list[str] = Field(default_factory=list)
    existing_artifacts: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    log_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class CapabilityMetadataQueryInput(BaseModel):
    action_family: str | None = None
    capability: str | None = None


class CapabilityMetadataQueryResult(BaseModel):
    action_family: str | None = None
    capability: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    legal_parameters: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    fallback_actions: list[str] = Field(default_factory=list)
    cost_class: str = "medium"
    risk_class: str = "medium"
    required_inputs: list[str] = Field(default_factory=list)
    canonical_outputs: list[str] = Field(default_factory=list)
    invalidates_downstream: list[str] = Field(default_factory=list)


class ExecutionStatusQueryInput(BaseModel):
    state: dict[str, Any]


class ExecutionStatusQueryResult(BaseModel):
    current_stage: str = ""
    run_status: str = ""
    latest_status: str = ""
    pending_capabilities: list[str] = Field(default_factory=list)
    active_capabilities: list[str] = Field(default_factory=list)
    completed_capabilities: list[str] = Field(default_factory=list)
    blocked_capabilities: list[str] = Field(default_factory=list)
    abandoned_capabilities: list[str] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)
    retry_budget_remaining: dict[str, int] = Field(default_factory=dict)
    refinement_rounds: int = 0
    max_refinement_rounds: int = 0
    next_pending_capability: str | None = None
    latest_error: str | None = None
    all_tasks_resolved: bool = False
    waiting_external: bool = False
    ready_to_finalize: bool = False
    needs_recovery: bool = False
    needs_human: bool = False
    wait_reason: str | None = None
    external_jobs_pending: list[str] = Field(default_factory=list)
    pending_event_count: int = 0


class ObservationSynthesisInput(BaseModel):
    state: dict[str, Any]


class ObservationSynthesisResult(BaseModel):
    anomaly_flags: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    fit_quality: float | None = None
    mobility_window_summary: dict[str, Any] = Field(default_factory=dict)
    validation_summary: dict[str, Any] = Field(default_factory=dict)
    accepted_channels: list[str] = Field(default_factory=list)
    rejected_channels: list[str] = Field(default_factory=list)
    confidence_score: float | None = None
    warnings: list[str] = Field(default_factory=list)


class LegalityCheckInput(BaseModel):
    state: dict[str, Any]
    action_family: str
    target_capability: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    submit_external_job: bool = False
    wait_for_event_after_submission: bool = False


class LegalityCheckResult(BaseModel):
    allowed: bool
    action_family: str
    target_capability: str | None = None
    refusal_reasons: list[str] = Field(default_factory=list)
    dependency_status: dict[str, bool] = Field(default_factory=dict)
    required_evidence_status: dict[str, bool] = Field(default_factory=dict)
    task_board_status: str = "unknown"
    fallback_action: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MemoryQueryInput(BaseModel):
    state: dict[str, Any]
    limit: int = 5


class MemoryQueryResult(BaseModel):
    recovered_case_patterns: list[dict[str, Any]] = Field(default_factory=list)
    validation_case_patterns: list[dict[str, Any]] = Field(default_factory=list)
    batch_statistics: list[dict[str, Any]] = Field(default_factory=list)
    skill_registry: list[dict[str, Any]] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemoryWriteInput(BaseModel):
    state: dict[str, Any]
    round_id: int


class MemoryWriteResult(BaseModel):
    recorded_categories: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HITLInspectionInput(BaseModel):
    workdir: str
    timeout_seconds: int = 300


class HITLInspectionResult(BaseModel):
    payload_path: str
    response_path: str
    log_path: str
    response_exists: bool
    pending_payload_exists: bool
    timeout_seconds: int
    warnings: list[str] = Field(default_factory=list)


class TraceWriteInput(BaseModel):
    workdir: str
    state: dict[str, Any]
    final_summary: dict[str, Any]
    material_outcome: dict[str, Any]


class TraceWriteResult(BaseModel):
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class BatchAggregationInput(BaseModel):
    outcomes: list[dict[str, Any]] = Field(default_factory=list)


class BatchAggregationResult(BaseModel):
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    scientifically_passed: int = 0
    scientifically_warning: int = 0
    scientifically_failed: int = 0
    scientifically_unknown: int = 0
    common_failure_stages: dict[str, int] = Field(default_factory=dict)


class SkillResolveInput(BaseModel):
    state: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None
    explicit_skills: list[str] = Field(default_factory=list)
    limit: int = 6
    skills_root: str | None = None


class SkillResolveResult(BaseModel):
    selected_skills: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    available_skills: list[dict[str, Any]] = Field(default_factory=list)


class SkillLoadInput(BaseModel):
    skill_name: str
    include_body: bool = True
    include_resources: bool = False
    resource_limit: int | None = 20
    skills_root: str | None = None


class SkillResourceListInput(BaseModel):
    skill_name: str
    skills_root: str | None = None


class SkillResourceListResult(BaseModel):
    skill_name: str
    resources: list[dict[str, Any]] = Field(default_factory=list)


class SkillResourceReadInput(BaseModel):
    skill_name: str
    resource_path: str
    skills_root: str | None = None


class SkillResourceReadResult(BaseModel):
    skill_name: str
    resource_path: str
    content: Any = Field(default_factory=dict)
    kind: str = "reference"


class PolicyEvidenceQueryInput(BaseModel):
    state: dict[str, Any]
    stage: str | None = None
    query: str = ""
    top_k: int = 5


class PolicyEvidenceQueryResult(BaseModel):
    stage: str = ""
    query: str = ""
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    source_corpora: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FailureEvidenceQueryInput(BaseModel):
    state: dict[str, Any]
    stage: str | None = None
    top_k: int = 5


class FailureEvidenceQueryResult(BaseModel):
    stage: str = ""
    query: str = ""
    evidence: list[RetrievedEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class AgentToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    callable_entrypoint: Callable[..., BaseModel]
    mcp_exposed: bool = True


def _existing_paths(paths: list[str]) -> list[str]:
    return [os.path.abspath(path) for path in paths if path and os.path.exists(path)]


def _escalation_paths(workdir: str) -> dict[str, str]:
    return {
        "payload_path": os.path.join(workdir, "human_escalation_payload.json"),
        "response_path": os.path.join(workdir, "human_escalation_response.json"),
        "log_path": os.path.join(workdir, "human_escalation_log.json"),
    }


def _default_skills_root() -> str:
    return os.path.abspath(
        os.environ.get("MOBILITY_SKILLS_ROOT")
        or os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "skills")
    )


def _artifact_paths_for_stage(workdir: str, capability: str | None) -> list[str]:
    if not capability or capability not in STAGE_ORDER:
        return []
    contract = get_stage_contract(capability)
    resolved: list[str] = []
    for pattern in contract.artifact_patterns:
        candidate = os.path.join(workdir, pattern)
        if "*" in pattern:
            resolved.extend(str(path.resolve()) for path in Path(workdir).glob(pattern) if path.exists())
        elif os.path.exists(candidate):
            resolved.append(os.path.abspath(candidate))
    return dedupe_keep_order(resolved)


def _policy_stage(payload_state: dict[str, Any], explicit_stage: str | None = None) -> str:
    if explicit_stage:
        return str(explicit_stage or "")
    latest = dict((payload_state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    workflow = dict(payload_state.get("workflow", {}) or {})
    return str(latest.get("target_capability") or latest.get("stage") or workflow.get("current_stage") or "")


def _build_policy_query(state: dict[str, Any], *, stage: str, query: str = "") -> str:
    if query.strip():
        return query.strip()
    probe = build_stage_probe_from_state(state, stage=stage)
    return json.dumps(probe.model_dump(mode="json"), ensure_ascii=False)


def _build_failure_query(state: dict[str, Any], *, stage: str) -> str:
    latest = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
    if latest.get("status") != "failed":
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    payload = {
        "stage": stage,
        "error_summary": latest.get("error_summary") or (state.get("diagnostics", {}) or {}).get("last_error"),
        "error_category": latest.get("error_category"),
        "artifact_paths": dict(latest.get("artifact_paths", {}) or {}),
        "result_summary": dict(latest.get("result_summary", {}) or {}),
    }
    return json.dumps(payload, ensure_ascii=False)


def _rag_strict_no_fallback_enabled() -> bool:
    raw = str(os.environ.get("RAG_STRICT_NO_FALLBACK", "") or "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _fallback_policy_knowledge_base() -> PolicyKnowledgeBase:
    house_policy_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "policy", "corpus", "house_policy.json")
    )
    fallback_documents = [item.model_dump(mode="json") for item in load_house_policy_documents(house_policy_path)]
    return PolicyKnowledgeBase(
        database_uri="memory://policy-kb",
        embedding_model="fallback-embedding",
        embedding_base_url="",
        embedding_api_key="",
        qa_model="",
        qa_base_url="",
        qa_api_key="",
        strict_rag=False,
        fallback_documents=fallback_documents,
    )


def inspect_workspace_tool(payload: WorkspaceInspectionInput, *, store: BaseStore | None = None) -> WorkspaceInspectionResult:
    del store
    workdir = os.path.abspath(payload.workdir)
    runtime_dir = (
        os.path.dirname(os.path.abspath(payload.checkpoint_path))
        if payload.checkpoint_path
        else os.path.dirname(runtime_state_snapshot_path(workdir))
    )
    available_inputs = {
        "poscar": bool(payload.poscar_path and os.path.exists(payload.poscar_path)),
        "potcar": bool(payload.potcar_path and os.path.exists(payload.potcar_path)),
        "workdir": os.path.isdir(workdir),
    }
    compatibility_artifacts = {
        "checkpoint_pkl": os.path.exists(os.path.join(workdir, "checkpoint.pkl")),
        "shared_state_json": os.path.exists(os.path.join(runtime_dir, "shared_state.json")),
        "material_outcome_json": os.path.exists(os.path.join(workdir, "material_outcome.json")),
        "final_summary_json": os.path.exists(os.path.join(workdir, "final_summary.json")),
    }
    existing_stage_dirs = [
        path.name
        for path in sorted(Path(workdir).glob("[0-9][0-9]_*"))
        if path.is_dir()
    ]
    registry_paths = _existing_paths(list(dict(payload.artifact_registry or {}).values()))
    warnings: list[str] = []
    if not available_inputs["poscar"]:
        warnings.append("missing_poscar")
    if not available_inputs["potcar"]:
        warnings.append("missing_potcar")
    return WorkspaceInspectionResult(
        available_inputs=available_inputs,
        existing_stage_dirs=existing_stage_dirs,
        existing_artifacts=registry_paths,
        compatibility_artifacts=compatibility_artifacts,
        checkpoint_available=bool(payload.checkpoint_path and os.path.exists(payload.checkpoint_path)),
        warnings=warnings,
        facts={
            "structure_summary": summarize_poscar(str(payload.poscar_path or "")),
            "workdir": workdir,
            "runtime_dir": runtime_dir,
        },
    )


def inspect_artifacts_tool(payload: ArtifactInspectionInput, *, store: BaseStore | None = None) -> ArtifactInspectionResult:
    del store
    workdir = os.path.abspath(payload.workdir)
    expected = _artifact_paths_for_stage(workdir, payload.target_capability)
    existing = _existing_paths(list(dict(payload.artifact_registry or {}).values()))
    stage_dir_exists = False
    if payload.target_capability:
        stage_dir_exists = any(
            Path(path).name.endswith(payload.target_capability) or payload.target_capability in Path(path).name
            for path in existing
        ) or bool(expected)
    log_paths = _existing_paths(
        [
            os.path.join(workdir, "sout"),
            os.path.join(workdir, "vasp_relax_retry.log"),
            os.path.join(workdir, "human_escalation_log.json"),
        ]
    )
    missing = [path for path in expected if path not in existing and not os.path.exists(path)]
    warnings: list[str] = []
    if payload.target_capability and not expected:
        warnings.append("no_expected_artifacts_declared")
    if missing:
        warnings.append("missing_expected_artifacts")
    return ArtifactInspectionResult(
        target_capability=payload.target_capability,
        stage_dir_exists=stage_dir_exists,
        expected_artifacts=expected,
        existing_artifacts=existing,
        missing_artifacts=missing,
        log_paths=log_paths,
        warnings=warnings,
        facts={"workdir": workdir},
    )


def retrieve_policy_evidence_tool(
    payload: PolicyEvidenceQueryInput,
    *,
    store: BaseStore | None = None,
) -> PolicyEvidenceQueryResult:
    del store
    state = dict(payload.state or {})
    stage = _policy_stage(state, payload.stage)
    query = _build_policy_query(state, stage=stage, query=str(payload.query or ""))
    warnings: list[str] = []
    evidence: list[RetrievedEvidence] = []
    try:
        kb = default_knowledge_base()
        evidence = kb.retrieve(query=query, stage=stage, top_k=max(1, int(payload.top_k or 5)), corpora=("house_policy", "vasp_wiki"))
    except Exception as exc:
        if _rag_strict_no_fallback_enabled():
            raise RuntimeError(f"policy_evidence_strict_rag_failed:{type(exc).__name__}:{exc}") from exc
        warnings.append(f"policy_evidence_unavailable:{type(exc).__name__}")
        try:
            fallback_kb = _fallback_policy_knowledge_base()
            evidence = fallback_kb.retrieve(query=query, stage=stage, top_k=max(1, int(payload.top_k or 5)), corpora=("house_policy",))
            if evidence:
                warnings.append("policy_evidence_fallback_house_policy")
        except Exception as inner_exc:
            warnings.append(f"policy_evidence_fallback_failed:{type(inner_exc).__name__}")
    if not evidence:
        warnings.append("no_policy_evidence_found")
    return PolicyEvidenceQueryResult(
        stage=stage,
        query=query,
        evidence=evidence,
        source_corpora=sorted({item.corpus for item in evidence}),
        warnings=warnings,
    )


def retrieve_failure_evidence_tool(
    payload: FailureEvidenceQueryInput,
    *,
    store: BaseStore | None = None,
) -> FailureEvidenceQueryResult:
    del store
    state = dict(payload.state or {})
    stage = _policy_stage(state, payload.stage)
    query = _build_failure_query(state, stage=stage)
    warnings: list[str] = []
    evidence: list[RetrievedEvidence] = []
    try:
        kb = default_knowledge_base()
        evidence = kb.retrieve(query=query, stage=stage, top_k=max(1, int(payload.top_k or 5)), corpora=("house_policy", "vasp_wiki"))
    except Exception as exc:
        if _rag_strict_no_fallback_enabled():
            raise RuntimeError(f"failure_evidence_strict_rag_failed:{type(exc).__name__}:{exc}") from exc
        warnings.append(f"failure_evidence_unavailable:{type(exc).__name__}")
        try:
            fallback_kb = _fallback_policy_knowledge_base()
            evidence = fallback_kb.retrieve(query=query, stage=stage, top_k=max(1, int(payload.top_k or 5)), corpora=("house_policy",))
            if evidence:
                warnings.append("failure_evidence_fallback_house_policy")
        except Exception as inner_exc:
            warnings.append(f"failure_evidence_fallback_failed:{type(inner_exc).__name__}")
    if not evidence:
        warnings.append("no_failure_evidence_found")
    return FailureEvidenceQueryResult(
        stage=stage,
        query=query,
        evidence=evidence,
        warnings=warnings,
    )


def query_capability_metadata_tool(
    payload: CapabilityMetadataQueryInput,
    *,
    store: BaseStore | None = None,
) -> CapabilityMetadataQueryResult:
    del store
    action_family = payload.action_family
    capability = payload.capability
    dependencies: list[str] = []
    legal_parameters: list[str] = []
    required_evidence: list[str] = []
    expected_artifacts: list[str] = []
    fallback_actions: list[str] = []
    cost_class = "medium"
    risk_class = "medium"
    if action_family:
        spec = get_action_spec(action_family)
        dependencies = list(spec.dependencies)
        legal_parameters = list(spec.legal_parameters)
        required_evidence = list(spec.required_evidence)
        expected_artifacts = list(spec.expected_artifacts)
        fallback_actions = list(spec.fallback_actions)
        cost_class = spec.cost_class
        risk_class = spec.risk_class
    stage_deps: list[str] = []
    required_inputs: list[str] = []
    canonical_outputs: list[str] = []
    invalidates_downstream: list[str] = []
    if capability and capability in STAGE_ORDER:
        contract = get_stage_contract(capability)
        stage_deps = capability_dependencies(capability)
        required_inputs = list(contract.required_inputs)
        canonical_outputs = list(contract.canonical_outputs)
        invalidates_downstream = list(contract.invalidates_downstream)
        if not expected_artifacts:
            expected_artifacts = list(contract.artifact_patterns)
    return CapabilityMetadataQueryResult(
        action_family=action_family,
        capability=capability,
        dependencies=stage_deps or dependencies,
        legal_parameters=legal_parameters,
        required_evidence=required_evidence,
        expected_artifacts=expected_artifacts,
        fallback_actions=fallback_actions,
        cost_class=cost_class,
        risk_class=risk_class,
        required_inputs=required_inputs,
        canonical_outputs=canonical_outputs,
        invalidates_downstream=invalidates_downstream,
    )


def query_execution_status_tool(payload: ExecutionStatusQueryInput, *, store: BaseStore | None = None) -> ExecutionStatusQueryResult:
    del store
    state = dict(payload.state or {})
    workflow = dict(state.get("workflow", {}) or {})
    task_board = dict(state.get("task_board", {}) or {})
    retry_counts = {str(k): int(v or 0) for k, v in dict(workflow.get("retry_counts", {}) or {}).items()}
    retry_budget = int(workflow.get("retry_budget", 0) or 0)

    def _caps(key: str) -> list[str]:
        values = list(task_board.get(key, []) or [])
        return [
            str(item.get("capability") or "")
            for item in values
            if isinstance(item, dict) and str(item.get("capability") or "").strip()
        ]

    pending = _caps("pending_tasks")
    active = _caps("active_tasks")
    completed = _caps("completed_tasks")
    blocked = _caps("blocked_tasks")
    abandoned = _caps("abandoned_tasks")
    run_status = str(workflow.get("run_status") or "")
    external_jobs_pending = [
        str(item.get("job_id") or "")
        for item in list((state.get("execution", {}) or {}).get("external_jobs", []) or [])
        if isinstance(item, dict) and str(item.get("status") or "") in {"submitted", "running", "waiting"}
    ]
    retry_budget_remaining = {
        capability: max(0, retry_budget - int(retry_counts.get(capability, 0) or 0))
        for capability in set([*pending, *active, *completed, *blocked, *abandoned, *retry_counts.keys()])
    }
    return ExecutionStatusQueryResult(
        current_stage=str(workflow.get("current_stage") or ""),
        run_status=run_status,
        latest_status=str((state.get("execution", {}) or {}).get("action_status") or ""),
        pending_capabilities=pending,
        active_capabilities=active,
        completed_capabilities=completed,
        blocked_capabilities=blocked,
        abandoned_capabilities=abandoned,
        retry_counts=retry_counts,
        retry_budget_remaining=retry_budget_remaining,
        refinement_rounds=int(workflow.get("refinement_rounds", 0) or 0),
        max_refinement_rounds=int(workflow.get("max_refinement_rounds", 0) or 0),
        next_pending_capability=(str((next_pending_task(state) or {}).get("capability") or "") or None),
        latest_error=str((state.get("diagnostics", {}) or {}).get("last_error") or "") or None,
        all_tasks_resolved=all_tasks_resolved(state),
        waiting_external=run_status == "waiting_external",
        ready_to_finalize=run_status == "ready_to_finalize",
        needs_recovery=run_status == "needs_recovery",
        needs_human=run_status == "needs_human",
        wait_reason=str(workflow.get("wait_reason") or "") or None,
        external_jobs_pending=[item for item in external_jobs_pending if item],
        pending_event_count=len(list((state.get("execution", {}) or {}).get("pending_events", []) or [])),
    )


def synthesize_observation_tool(payload: ObservationSynthesisInput, *, store: BaseStore | None = None) -> ObservationSynthesisResult:
    del store
    state = dict(payload.state or {})
    physics = dict(state.get("physics_results", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    raw_results = dict(physics.get("results", {}) or {})
    has_mobility_results = bool(dict(raw_results.get("results_by_direction", {}) or {}))
    mobility_metrics = (
        {"results_present": True, **validate_physics_window(raw_results)}
        if has_mobility_results
        else {
            "results_present": False,
            "fit_r2_min": None,
            "energy_fit_r2_min": None,
            "edge_fit_r2_min": None,
            "effective_fit_quality": None,
            "e1_sigma_max": None,
            "c2d_sigma_max": None,
            "warnings": [],
            "anomaly_flags": [],
            "per_direction": {},
        }
    )
    anomaly_flags = list(mobility_metrics.get("anomaly_flags", []) or [])
    anomaly_flags.extend(
        detect_basic_anomalies(
            {
                "errors": diagnostics.get("errors"),
                "run_status": workflow.get("run_status"),
                "confidence_score": diagnostics.get("confidence_score"),
            }
        )
    )
    anomaly_flags = [str(item) for item in dedupe_keep_order(anomaly_flags)]
    fit_metrics = dict(diagnostics.get("fit_diagnostics", {}) or {})
    fit_quality = fit_metrics.get("effective_fit_quality", fit_metrics.get("fit_r2_min"))
    if fit_quality is None:
        fit_quality = mobility_metrics.get("effective_fit_quality")
    warnings = list((state.get("material", {}) or {}).get("warnings", []) or [])
    risk_flags = list(anomaly_flags)
    if fit_quality is not None and float(fit_quality) < 0.90:
        risk_flags.append(f"fit_quality_below_threshold:{float(fit_quality):.3f}")
    return ObservationSynthesisResult(
        anomaly_flags=anomaly_flags,
        risk_flags=dedupe_keep_order(risk_flags),
        fit_quality=(float(fit_quality) if fit_quality is not None else None),
        mobility_window_summary=mobility_metrics,
        validation_summary=dict(diagnostics.get("validation_report", {}) or {}),
        accepted_channels=list(physics.get("accepted_channels", []) or []),
        rejected_channels=list(physics.get("rejected_channels", []) or []),
        confidence_score=diagnostics.get("confidence_score"),
        warnings=warnings,
    )


def _task_status_for_capability(state: dict[str, Any], capability: str | None) -> str:
    if not capability:
        return "none"
    task_board = dict(state.get("task_board", {}) or {})
    for status_name, field_name in (
        ("pending", "pending_tasks"),
        ("active", "active_tasks"),
        ("completed", "completed_tasks"),
        ("blocked", "blocked_tasks"),
        ("abandoned", "abandoned_tasks"),
    ):
        for item in list(task_board.get(field_name, []) or []):
            if isinstance(item, dict) and str(item.get("capability") or "") == capability:
                return status_name
    return "unknown"


def _required_evidence_status(state: dict[str, Any], required_evidence: list[str]) -> dict[str, bool]:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    blackboard = dict(state.get("blackboard", {}) or {})
    physics = dict(state.get("physics_results", {}) or {})
    workflow = dict(state.get("workflow", {}) or {})
    fit_metrics = dict(diagnostics.get("fit_diagnostics", {}) or {})
    anomaly_flags = list(blackboard.get("anomaly_flags", []) or [])
    status: dict[str, bool] = {}
    for evidence in required_evidence:
        if evidence == "latest_failure":
            status[evidence] = str((execution.get("latest_execution_observation", {}) or {}).get("status") or "") == "failed"
        elif evidence == "dependency_mismatch_or_failed_downstream":
            status[evidence] = bool(diagnostics.get("last_error") or blackboard.get("risk_flags"))
        elif evidence == "fit_quality_warning":
            value = fit_metrics.get("effective_fit_quality", fit_metrics.get("fit_r2_min", 1.0))
            status[evidence] = float(value or 1.0) < 0.90
        elif evidence == "results_present":
            status[evidence] = bool(physics.get("results") or diagnostics.get("validation_report"))
        elif evidence == "channel_specific_anomaly":
            status[evidence] = bool(anomaly_flags)
        elif evidence == "channel_specific_failure":
            status[evidence] = bool(physics.get("rejected_channels"))
        elif evidence == "insufficient_automation_confidence":
            confidence = diagnostics.get("confidence_score")
            status[evidence] = confidence is None or float(confidence or 0.0) < 0.75
        elif evidence == "result_or_terminal_status":
            status[evidence] = bool(physics.get("results")) or str(workflow.get("run_status") or "") in {
                "ready_to_finalize",
                "completed",
                "failed",
                "aborted",
                "skipped",
            }
        elif evidence == "terminal_failure":
            status[evidence] = str(workflow.get("run_status") or "") in {"failed", "aborted", "needs_recovery"} or bool(
                diagnostics.get("last_error")
            )
        elif evidence == "context_corruption_or_missing_artifacts":
            status[evidence] = bool(diagnostics.get("last_error"))
        else:
            status[evidence] = False
    return status


def _canonicalize_channel_action_parameters(action_family: str, parameters: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(parameters or {})
    family = str(action_family or "")
    if family not in {"invalidate_channel", "skip_channel"}:
        return params

    alias_keys = (
        ["channels_to_invalidate", "channels", "target_directions", "directions"]
        if family == "invalidate_channel"
        else ["channels_to_skip", "channels", "target_directions", "directions"]
    )
    raw = params.get("target_channels")
    if raw is None:
        for key in alias_keys:
            if params.get(key) is not None:
                raw = params.get(key)
                break
    if raw is None:
        single = params.get("channel") or params.get("target_channel")
        raw = [single] if single else []
    if isinstance(raw, str):
        raw = [raw]
    target_channels = [str(item).strip() for item in list(raw or []) if str(item).strip()]
    if target_channels:
        params["target_channels"] = target_channels
    return params


def check_legality_tool(payload: LegalityCheckInput, *, store: BaseStore | None = None) -> LegalityCheckResult:
    del store
    state = dict(payload.state or {})
    action_family = str(payload.action_family or "")
    target = payload.target_capability
    reasons: list[str] = []
    warnings: list[str] = []
    fallback_action = "abort_material"

    if action_family not in list_action_families():
        reasons.append(f"unknown_action_family:{action_family or 'unset'}")
        return LegalityCheckResult(
            allowed=False,
            action_family=action_family,
            target_capability=target,
            refusal_reasons=reasons,
            task_board_status=_task_status_for_capability(state, target),
            fallback_action=fallback_action,
        )

    spec = get_action_spec(action_family)
    if spec.fallback_actions:
        fallback_action = spec.fallback_actions[0]
    parameters = _canonicalize_channel_action_parameters(action_family, dict(payload.parameters or {}))

    task_status = _task_status_for_capability(state, target)
    workflow = dict(state.get("workflow", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    mission = dict(state.get("mission", {}) or {})
    runtime_constraints = dict(mission.get("runtime_constraints", {}) or {})
    run_status = str(workflow.get("run_status") or "")
    external_jobs = list(execution.get("external_jobs", []) or [])
    full_autonomy = bool(runtime_constraints.get("full_autonomy", False))
    allow_external_wait = bool(runtime_constraints.get("allow_external_wait", False))
    if state.get("services", {}).get("pending_human_payload") and not state.get("services", {}).get("latest_human_decision"):
        if action_family != "escalate_human":
            reasons.append("human_override_pending")

    if run_status in TERMINAL_RUN_STATUSES:
        reasons.append(f"terminal_run_status:{run_status}")

    if bool(payload.submit_external_job or payload.wait_for_event_after_submission) and full_autonomy and not allow_external_wait:
        reasons.append("external_wait_not_allowed_in_full_autonomy")

    if run_status == "waiting_external" and not list(execution.get("pending_events", []) or []):
        if action_family not in {"abort_material", "escalate_human"}:
            reasons.append("waiting_external_event")

    if action_family in {"run_capability", "retry_capability", "rerun_from_capability", "refine_sampling", "revalidate_result"} and not target:
        reasons.append("missing_target_capability")
    if action_family in {"invalidate_channel", "skip_channel"} and not list(parameters.get("target_channels", []) or []):
        reasons.append("missing_required_parameter:target_channels")
    if target and target not in STAGE_ORDER:
        reasons.append(f"unknown_target_capability:{target}")

    completed = {
        str(item.get("capability"))
        for item in list((state.get("task_board", {}) or {}).get("completed_tasks", []) or [])
        if isinstance(item, dict)
    }
    dependency_status = {dep: dep in completed for dep in capability_dependencies(target or "")}
    if action_family in {"run_capability", "retry_capability"} and target:
        missing = [dep for dep, ok in dependency_status.items() if not ok]
        if missing:
            reasons.append(f"missing_dependencies:{','.join(missing)}")
        if task_status == "completed" and action_family == "run_capability":
            reasons.append(f"target_task_already_completed:{target}")
        if task_status == "active":
            reasons.append(f"target_task_already_active:{target}")

    if target:
        pending_external = next(
            (
                item
                for item in external_jobs
                if isinstance(item, dict)
                and str(item.get("target_capability") or "") == target
                and str(item.get("status") or "") in {"submitted", "running", "waiting"}
            ),
            None,
        )
        if pending_external and action_family in {"run_capability", "retry_capability", "rerun_from_capability", "refine_sampling", "revalidate_result"}:
            reasons.append(f"external_job_already_pending:{target}")

    if action_family == "retry_capability" and target:
        retries = int(((state.get("workflow", {}) or {}).get("retry_counts", {}) or {}).get(target, 0) or 0)
        budget = int((state.get("workflow", {}) or {}).get("retry_budget", 0) or 0)
        if retries >= budget:
            reasons.append(f"retry_budget_exhausted:{target}:{retries}/{budget}")

    if action_family == "refine_sampling":
        refinement_rounds = int((state.get("workflow", {}) or {}).get("refinement_rounds", 0) or 0)
        max_rounds = int((state.get("workflow", {}) or {}).get("max_refinement_rounds", 0) or 0)
        if refinement_rounds >= max_rounds:
            reasons.append(f"refinement_budget_exhausted:{refinement_rounds}/{max_rounds}")
        refinement_plan = resolve_refinement_sampling(
            state,
            parameters,
            max_points_per_direction=DEFAULT_REFINEMENT_TARGET_POINTS,
            fit_threshold=DEFAULT_FIT_R2_THRESHOLD,
        )
        if not refinement_plan.get("applied_points"):
            reasons.append("no_fresh_refinement_points")

    if (
        target == "validation"
        and action_family in {"run_capability", "revalidate_result"}
        and validation_requires_refinement(
            state,
            max_points_per_direction=DEFAULT_REFINEMENT_TARGET_POINTS,
            fit_threshold=DEFAULT_FIT_R2_THRESHOLD,
        )
    ):
        warnings.append("validation_has_available_refinement_followup")

    validation_report = dict((state.get("diagnostics", {}) or {}).get("validation_report", {}) or {})
    validation_stage_status = str(((state.get("workflow", {}) or {}).get("stage_status", {}) or {}).get("validation") or "")
    latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    validation_ready_to_finalize = validation_report_supports_finalize(
        validation_report=validation_report,
        validation_stage_status=validation_stage_status,
        latest_observation=latest_observation,
    )
    if action_family == "finalize_material" and not validation_ready_to_finalize and not all_tasks_resolved(state):
        reasons.append("finalize_requires_all_tasks_resolved")
    if action_family == "finalize_material" and run_status == "waiting_external":
        reasons.append("cannot_finalize_while_waiting_external")

    if action_family == "abort_material":
        latest = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
        if run_status not in {"failed", "needs_recovery", "needs_human", "skipped", "aborted"} and latest.get("status") not in {"failed", "skipped"}:
            warnings.append("abort_without_terminal_failure")

    if action_family in {"run_capability", "retry_capability"} and task_status == "abandoned":
        reasons.append(f"target_task_abandoned:{target}")
    if action_family == "escalate_human" and task_status == "completed":
        reasons.append(f"completed_task_does_not_require_escalation:{target}")

    evidence_status = _required_evidence_status(state, list(spec.required_evidence))
    for evidence, ok in evidence_status.items():
        if not ok:
            reasons.append(f"missing_required_evidence:{evidence}")

    allowed = not reasons
    return LegalityCheckResult(
        allowed=allowed,
        action_family=action_family,
        target_capability=target,
        refusal_reasons=reasons,
        dependency_status=dependency_status,
        required_evidence_status=evidence_status,
        task_board_status=task_status,
        fallback_action=fallback_action,
        warnings=warnings,
    )


def query_memory_hits_tool(payload: MemoryQueryInput, *, store: BaseStore | None = None) -> MemoryQueryResult:
    if store is None:
        return MemoryQueryResult(warnings=["memory_store_unavailable"])  # type: ignore[call-arg]
    state = dict(payload.state or {})
    latest_observation = dict((state.get("blackboard", {}) or {}).get("latest_execution_observation", {}) or {})
    stage = str(latest_observation.get("target_capability") or state.get("workflow", {}).get("current_stage") or "")
    latest_error = str((state.get("diagnostics", {}) or {}).get("last_error") or latest_observation.get("error_summary") or "")
    anomaly_flags = list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or [])
    warnings = list((state.get("material", {}) or {}).get("warnings", []) or [])
    collection_name = str((state.get("task", {}) or {}).get("collection_name") or "")
    recovered = find_recovery_cases(store, stage=stage or "unknown", error_signature=latest_error or None, limit=payload.limit)
    validation = find_validation_heuristics(store, anomaly_flags=anomaly_flags, warnings=warnings, limit=payload.limit)
    batch_statistics = list_batch_statistics(store, collection_name=collection_name) if collection_name else []
    skill_registry = list_skill_metadata(store)
    refs = [
        f"recovery_cases:{len(recovered)}",
        f"validation_heuristics:{len(validation)}",
        f"skill_registry:{len(skill_registry)}",
    ]
    if batch_statistics:
        refs.append(f"batch_statistics:{len(batch_statistics)}")
    return MemoryQueryResult(
        recovered_case_patterns=recovered,
        validation_case_patterns=validation,
        batch_statistics=batch_statistics[: payload.limit],
        skill_registry=skill_registry[: payload.limit],
        evidence_refs=refs,
    )


def write_memory_reflection_tool(payload: MemoryWriteInput, *, store: BaseStore | None = None) -> MemoryWriteResult:
    if store is None:
        return MemoryWriteResult(warnings=["memory_store_unavailable"])
    state = dict(payload.state or {})
    recorded: list[str] = []
    latest_observation = dict((state.get("execution", {}) or {}).get("latest_execution_observation", {}) or {})
    if latest_observation.get("status") == "failed":
        task_id = str((state.get("task", {}) or {}).get("task_id") or "unknown_task")
        stage = str(latest_observation.get("target_capability") or latest_observation.get("stage") or "unknown")
        error_summary = str(latest_observation.get("error_summary") or "unknown_failure")
        selected_action = dict((state.get("execution", {}) or {}).get("current_action", {}) or {})
        record_recovery_case(
            store,
            task_id=task_id,
            payload={
                "task_id": task_id,
                "round_id": payload.round_id,
                "stage": stage,
                "error_signature": error_summary,
                "chosen_action": selected_action.get("action_family"),
                "target_capability": selected_action.get("target_capability"),
                "success_or_failure": latest_observation.get("status"),
            },
        )
        recorded.append("recovery_cases")
    put_memory_item(
        store,
        category="runtime_reflections",
        item_key=f"{state.get('task', {}).get('task_id', 'task')}:{payload.round_id}",
        payload={
            "task_id": state.get("task", {}).get("task_id"),
            "round_id": payload.round_id,
            "selected_action": dict((state.get("execution", {}) or {}).get("current_action", {}) or {}),
            "latest_execution_observation": latest_observation,
            "disagreement_records": list((state.get("deliberation", {}) or {}).get("disagreement_records", []) or []),
        },
    )
    recorded.append("runtime_reflections")
    return MemoryWriteResult(recorded_categories=recorded)


def inspect_hitl_state_tool(payload: HITLInspectionInput, *, store: BaseStore | None = None) -> HITLInspectionResult:
    del store
    paths = _escalation_paths(payload.workdir)
    warnings: list[str] = []
    if os.path.exists(paths["payload_path"]) and not os.path.exists(paths["response_path"]):
        warnings.append("pending_human_response")
    return HITLInspectionResult(
        payload_path=paths["payload_path"],
        response_path=paths["response_path"],
        log_path=paths["log_path"],
        response_exists=os.path.exists(paths["response_path"]),
        pending_payload_exists=os.path.exists(paths["payload_path"]),
        timeout_seconds=int(payload.timeout_seconds),
        warnings=warnings,
    )


def resolve_skills_tool(payload: SkillResolveInput, *, store: BaseStore | None = None) -> SkillResolveResult:
    del store
    state = dict(payload.state or {})
    workflow = dict(state.get("workflow", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    blackboard = dict(state.get("blackboard", {}) or {})
    task = dict(state.get("task", {}) or {})
    root = os.path.abspath(payload.skills_root or _default_skills_root())
    registry = discover_skills(root)
    selection = resolve_skills(
        registry,
        request=SkillResolutionRequest(
            role=payload.role,
            task_type=str(task.get("task_type") or ""),
            stage=str(workflow.get("current_stage") or ""),
            run_status=str(workflow.get("run_status") or ""),
            has_error=bool(diagnostics.get("last_error") or list(diagnostics.get("errors", []) or [])),
            latest_error=str(diagnostics.get("last_error") or ""),
            anomaly_flags=list(blackboard.get("anomaly_flags", []) or []),
            explicit_skills=list(payload.explicit_skills or []),
            limit=max(1, int(payload.limit or 1)),
        ),
    )
    available_skills = []
    for name, entry in sorted(registry.items()):
        manifest = dict(entry.get("manifest", {}) or {})
        available_skills.append(
            {
                "name": name,
                "description": entry.get("description"),
                "load_strategy": manifest.get("load_strategy"),
                "roles": list(manifest.get("roles", []) or []),
                "task_types": list(manifest.get("task_types", []) or []),
                "stages": list(manifest.get("stages", []) or []),
            }
        )
    return SkillResolveResult(
        selected_skills=list(selection.selected_skills),
        candidates=[item.model_dump(mode="json") for item in selection.candidates],
        available_skills=available_skills,
    )


def load_skill_tool(payload: SkillLoadInput, *, store: BaseStore | None = None) -> LoadedSkillResult:
    del store
    root = os.path.abspath(payload.skills_root or _default_skills_root())
    loaded = load_skill(
        root,
        payload.skill_name,
        include_body=bool(payload.include_body),
        include_resources=bool(payload.include_resources),
        resource_limit=payload.resource_limit,
    )
    return LoadedSkillResult.model_validate(loaded)


def list_skill_resources_tool(payload: SkillResourceListInput, *, store: BaseStore | None = None) -> SkillResourceListResult:
    del store
    root = os.path.abspath(payload.skills_root or _default_skills_root())
    registry = discover_skills(root)
    skill = registry.get(canonical_skill_name(payload.skill_name))
    if skill is None:
        raise FileNotFoundError(f"skill_not_found:{payload.skill_name}")
    return SkillResourceListResult(
        skill_name=str(skill.get("name") or payload.skill_name),
        resources=list(skill.get("resources", []) or []),
    )


def read_skill_resource_tool(payload: SkillResourceReadInput, *, store: BaseStore | None = None) -> SkillResourceReadResult:
    del store
    root = os.path.abspath(payload.skills_root or _default_skills_root())
    loaded = load_skill(root, payload.skill_name, include_body=False, include_resources=True)
    resource_payloads = dict(loaded.get("resource_payloads", {}) or {})
    resources = {str(item.get("path") or ""): dict(item) for item in list(loaded.get("resources", []) or [])}
    relative_path = str(payload.resource_path or "").strip()
    if relative_path not in resource_payloads:
        raise FileNotFoundError(f"skill_resource_not_found:{payload.skill_name}:{relative_path}")
    return SkillResourceReadResult(
        skill_name=str(loaded.get("name") or payload.skill_name),
        resource_path=relative_path,
        content=resource_payloads[relative_path],
        kind=str(resources.get(relative_path, {}).get("kind") or "reference"),
    )


def write_runtime_artifacts_tool(payload: TraceWriteInput, *, store: BaseStore | None = None) -> TraceWriteResult:
    del store
    workdir = os.path.abspath(payload.workdir)
    os.makedirs(workdir, exist_ok=True)
    state = dict(payload.state or {})
    deliberation_trace = {
        "rounds": list((state.get("deliberation", {}) or {}).get("rounds", []) or []),
        "proposals": list((state.get("deliberation", {}) or {}).get("proposals", []) or []),
        "critiques": list((state.get("deliberation", {}) or {}).get("critiques", []) or []),
        "preferences": list((state.get("deliberation", {}) or {}).get("preferences", []) or []),
        "arbitrations": list((state.get("deliberation", {}) or {}).get("arbitrations", []) or []),
        "selected_actions": list((state.get("deliberation", {}) or {}).get("selected_actions", []) or []),
        "reflections": list((state.get("deliberation", {}) or {}).get("reflections", []) or []),
        "disagreement_records": list((state.get("deliberation", {}) or {}).get("disagreement_records", []) or []),
    }
    files = {
        "mobility_results_path": ("mobility_results.json", dict((state.get("physics_results", {}) or {}).get("results", {}) or {})),
        "fit_diagnostics_path": ("fit_diagnostics.json", dict((state.get("diagnostics", {}) or {}).get("fit_diagnostics", {}) or {})),
        "decision_trace_path": ("decision_trace.json", list((state.get("agent", {}) or {}).get("decision_trace", []) or [])),
        "tool_trace_path": ("tool_trace.json", list((state.get("execution", {}) or {}).get("tool_trace", []) or [])),
        "retrieval_trace_path": ("retrieval_trace.json", list((state.get("services", {}) or {}).get("retrieval_trace", []) or [])),
        "parameter_plan_path": ("parameter_plan.json", dict((state.get("services", {}) or {}).get("parameter_plans", {}) or {})),
        "skill_trace_path": (
            "skill_trace.json",
            {
                "loaded_skills": list((state.get("services", {}) or {}).get("loaded_skills", []) or []),
                "skill_resolution": dict((state.get("services", {}) or {}).get("skill_resolution", {}) or {}),
                "skill_trace": list((state.get("execution", {}) or {}).get("skill_trace", []) or []),
                "available_agent_tools": list((state.get("services", {}) or {}).get("available_agent_tools", []) or []),
            },
        ),
        "recovery_trace_path": ("recovery_trace.json", list((state.get("diagnostics", {}) or {}).get("recovery_history", []) or [])),
        "recovery_diagnosis_path": ("recovery_diagnosis.json", dict((state.get("diagnostics", {}) or {}).get("recovery_diagnosis", {}) or {})),
        "validation_report_path": ("validation_report.json", dict((state.get("diagnostics", {}) or {}).get("validation_report", {}) or {})),
        "deliberation_trace_path": ("deliberation_trace.json", deliberation_trace),
        "workflow_contract_path": ("workflow_contract.json", dict((state.get("services", {}) or {}).get("workflow_contract", {}) or {})),
        "workflow_contract_history_path": (
            "workflow_contract_history.json",
            list((state.get("services", {}) or {}).get("workflow_contract_history", []) or []),
        ),
        "decision_ledger_path": ("decision_ledger.json", list((state.get("services", {}) or {}).get("decision_ledger", []) or [])),
        "execution_checkpoint_path": (
            "execution_checkpoint.json",
            dict((state.get("execution", {}) or {}).get("execution_checkpoint", {}) or {}),
        ),
        "final_summary_path": ("final_summary.json", dict(payload.final_summary or {})),
        "material_outcome_path": ("material_outcome.json", dict(payload.material_outcome or {})),
    }
    artifact_paths: dict[str, str] = {}
    for key, (filename, body) in files.items():
        path = os.path.join(workdir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(body, handle, ensure_ascii=False, indent=2)
        artifact_paths[key] = path
    return TraceWriteResult(artifact_paths=artifact_paths)


def summarize_batch_outcomes_tool(payload: BatchAggregationInput, *, store: BaseStore | None = None) -> BatchAggregationResult:
    del store
    processed = len(payload.outcomes)
    succeeded = 0
    failed = 0
    skipped = 0
    scientifically_passed = 0
    scientifically_warning = 0
    scientifically_failed = 0
    scientifically_unknown = 0
    common_failure_stages: dict[str, int] = {}
    for outcome in payload.outcomes:
        status = derive_compute_status_from_outcome_payload(outcome or {})
        if status == "completed":
            succeeded += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
        science_bucket = scientific_decision_bucket(resolve_outcome_scientific_decision(outcome or {}))
        if science_bucket == "passed":
            scientifically_passed += 1
        elif science_bucket == "warning":
            scientifically_warning += 1
        elif science_bucket == "failed":
            scientifically_failed += 1
        else:
            scientifically_unknown += 1
        if status != "completed":
            stage_status = dict((outcome or {}).get("stage_status", {}) or {})
            failure_stage = next((str(stage) for stage, value in stage_status.items() if value == "failed"), None)
            failure_stage = failure_stage or str((outcome or {}).get("termination_reason") or "unknown")
            common_failure_stages[failure_stage] = int(common_failure_stages.get(failure_stage, 0) or 0) + 1
    return BatchAggregationResult(
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        scientifically_passed=scientifically_passed,
        scientifically_warning=scientifically_warning,
        scientifically_failed=scientifically_failed,
        scientifically_unknown=scientifically_unknown,
        common_failure_stages=common_failure_stages,
    )


AGENT_TOOL_REGISTRY: dict[str, AgentToolDefinition] = {
    "inspect_workspace": AgentToolDefinition(
        name="inspect_workspace",
        description="Inspect a material workspace for inputs, checkpoints, compatibility artifacts, and existing stage directories.",
        input_model=WorkspaceInspectionInput,
        output_model=WorkspaceInspectionResult,
        callable_entrypoint=inspect_workspace_tool,
    ),
    "inspect_artifacts": AgentToolDefinition(
        name="inspect_artifacts",
        description="Inspect stage artifacts, expected outputs, and log paths for a capability.",
        input_model=ArtifactInspectionInput,
        output_model=ArtifactInspectionResult,
        callable_entrypoint=inspect_artifacts_tool,
    ),
    "retrieve_policy_evidence": AgentToolDefinition(
        name="retrieve_policy_evidence",
        description="Retrieve house-policy and optional VASP Wiki evidence relevant to a stage-level planning question.",
        input_model=PolicyEvidenceQueryInput,
        output_model=PolicyEvidenceQueryResult,
        callable_entrypoint=retrieve_policy_evidence_tool,
    ),
    "retrieve_failure_evidence": AgentToolDefinition(
        name="retrieve_failure_evidence",
        description="Retrieve house-policy and optional VASP Wiki evidence relevant to a current execution failure.",
        input_model=FailureEvidenceQueryInput,
        output_model=FailureEvidenceQueryResult,
        callable_entrypoint=retrieve_failure_evidence_tool,
    ),
    "query_capability_metadata": AgentToolDefinition(
        name="query_capability_metadata",
        description="Query action-registry and stage-contract metadata for a capability or action family.",
        input_model=CapabilityMetadataQueryInput,
        output_model=CapabilityMetadataQueryResult,
        callable_entrypoint=query_capability_metadata_tool,
    ),
    "query_execution_status": AgentToolDefinition(
        name="query_execution_status",
        description="Summarize task-board, retry-budget, refinement-budget, and run-status information.",
        input_model=ExecutionStatusQueryInput,
        output_model=ExecutionStatusQueryResult,
        callable_entrypoint=query_execution_status_tool,
    ),
    "synthesize_observation": AgentToolDefinition(
        name="synthesize_observation",
        description="Synthesize anomaly flags, fit quality, validation summary, and channel status from runtime state.",
        input_model=ObservationSynthesisInput,
        output_model=ObservationSynthesisResult,
        callable_entrypoint=synthesize_observation_tool,
    ),
    "check_action_legality": AgentToolDefinition(
        name="check_action_legality",
        description="Enforce runtime legality guardrails across action family, dependencies, evidence, task-board state, and budgets.",
        input_model=LegalityCheckInput,
        output_model=LegalityCheckResult,
        callable_entrypoint=check_legality_tool,
    ),
    "query_memory_hits": AgentToolDefinition(
        name="query_memory_hits",
        description="Query recovery, validation, skill, and batch memory hits relevant to the current runtime state.",
        input_model=MemoryQueryInput,
        output_model=MemoryQueryResult,
        callable_entrypoint=query_memory_hits_tool,
    ),
    "write_memory_reflection": AgentToolDefinition(
        name="write_memory_reflection",
        description="Write recovery/reflection memory records after a deliberation round.",
        input_model=MemoryWriteInput,
        output_model=MemoryWriteResult,
        callable_entrypoint=write_memory_reflection_tool,
    ),
    "inspect_hitl_state": AgentToolDefinition(
        name="inspect_hitl_state",
        description="Inspect HITL payload/response paths and timeout context for a workdir.",
        input_model=HITLInspectionInput,
        output_model=HITLInspectionResult,
        callable_entrypoint=inspect_hitl_state_tool,
    ),
    "resolve_skills": AgentToolDefinition(
        name="resolve_skills",
        description="Resolve the most relevant skill packages for the current role, state, and workflow context.",
        input_model=SkillResolveInput,
        output_model=SkillResolveResult,
        callable_entrypoint=resolve_skills_tool,
    ),
    "load_skill": AgentToolDefinition(
        name="load_skill",
        description="Load a skill package body and optional resource payloads on demand.",
        input_model=SkillLoadInput,
        output_model=LoadedSkillResult,
        callable_entrypoint=load_skill_tool,
    ),
    "list_skill_resources": AgentToolDefinition(
        name="list_skill_resources",
        description="List the indexed resources available inside a skill package.",
        input_model=SkillResourceListInput,
        output_model=SkillResourceListResult,
        callable_entrypoint=list_skill_resources_tool,
    ),
    "read_skill_resource": AgentToolDefinition(
        name="read_skill_resource",
        description="Read a specific resource from a skill package.",
        input_model=SkillResourceReadInput,
        output_model=SkillResourceReadResult,
        callable_entrypoint=read_skill_resource_tool,
    ),
    "write_runtime_artifacts": AgentToolDefinition(
        name="write_runtime_artifacts",
        description="Write deliberation trace, compatibility traces, final summary, and material outcome artifacts.",
        input_model=TraceWriteInput,
        output_model=TraceWriteResult,
        callable_entrypoint=write_runtime_artifacts_tool,
    ),
    "summarize_batch_outcomes": AgentToolDefinition(
        name="summarize_batch_outcomes",
        description="Aggregate processed, succeeded, failed, skipped counts and common failure stages for batch outcomes.",
        input_model=BatchAggregationInput,
        output_model=BatchAggregationResult,
        callable_entrypoint=summarize_batch_outcomes_tool,
    ),
}


def _tool_metadata_item(definition: AgentToolDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.input_model.model_json_schema(),
        "output_schema": definition.output_model.model_json_schema(),
        "callable_entrypoint": definition.callable_entrypoint.__name__,
        "mcp_exposed": definition.mcp_exposed,
    }


@lru_cache(maxsize=1)
def _cached_tool_metadata() -> tuple[dict[str, Any], ...]:
    return tuple(
        _tool_metadata_item(AGENT_TOOL_REGISTRY[item_name])
        for item_name in sorted(AGENT_TOOL_REGISTRY)
    )


class AgentToolGateway:
    def __init__(self) -> None:
        self._registry = dict(AGENT_TOOL_REGISTRY)

    def metadata(self, name: str | None = None) -> list[dict[str, Any]]:
        if name:
            return [_tool_metadata_item(self._registry[name])]
        return [dict(item) for item in _cached_tool_metadata()]

    def call(self, name: str, payload: dict[str, Any], *, store: BaseStore | None = None) -> dict[str, Any]:
        if name not in self._registry:
            raise KeyError(f"unknown_agent_tool:{name}")
        definition = self._registry[name]
        request = definition.input_model.model_validate(payload)
        result = definition.callable_entrypoint(request, store=store)
        if isinstance(result, BaseModel):
            return result.model_dump(mode="json")
        return dict(result)

    def as_langchain_tools(self, *, store: BaseStore | None = None, names: list[str] | None = None) -> list[Any]:
        if StructuredTool is None:
            return []
        selected = list(names or sorted(self._registry))
        tools: list[Any] = []
        for name in selected:
            if name not in self._registry:
                continue
            definition = self._registry[name]

            def _runner(store_ref=store, tool_name=name, **kwargs: Any) -> dict[str, Any]:
                return self.call(tool_name, kwargs, store=store_ref)

            tools.append(
                StructuredTool.from_function(
                    name=definition.name,
                    description=definition.description,
                    func=_runner,
                    args_schema=definition.input_model,
                    infer_schema=False,
                )
            )
        return tools


def list_agent_tool_metadata() -> list[dict[str, Any]]:
    return AgentToolGateway().metadata()
