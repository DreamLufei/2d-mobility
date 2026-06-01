from __future__ import annotations

import glob
import os
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from ..tools.errors import ManualFixValidationError


STAGE_ORDER = [
    "prepare",
    "relax",
    "scf",
    "band",
    "effective_mass",
    "strain_loop",
    "refinement",
    "mobility",
    "validation",
    "report",
]


CleanupPolicyName = Literal["retry_current_stage_only", "invalidate_downstream", "restart_from_stage"]
ManualFixModificationType = Literal["INCAR", "KPOINTS", "POSCAR", "multiple", "custom"]
ResumeStrategyName = Literal["default_rule", "retry_current_stage", "rerun_previous_stage", "rerun_from_relax", "custom_stage"]


class StageContract(BaseModel):
    stage: str
    required_inputs: list[str] = Field(default_factory=list)
    canonical_outputs: list[str] = Field(default_factory=list)
    failure_outputs: list[str] = Field(default_factory=list)
    retryable: bool = True
    invalidates_downstream: list[str] = Field(default_factory=list)
    artifact_patterns: list[str] = Field(default_factory=list)


class CleanupPreview(BaseModel):
    cleanup_policy: CleanupPolicyName
    resume_stage: str
    invalidated_stages: list[str] = Field(default_factory=list)
    invalidated_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResumeRuleContract(BaseModel):
    modified_files: list[str] = Field(default_factory=list)
    modification_type: ManualFixModificationType
    requested_resume_strategy: ResumeStrategyName
    resume_stage: str
    cleanup_policy: CleanupPolicyName
    invalidated_stages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


MANUAL_FIX_DEFAULTS: dict[ManualFixModificationType, dict[str, Any]] = {
    "INCAR": {
        "modified_files": ["INCAR"],
        "requested_resume_strategy": "default_rule",
        "resume_stage_ref": "current",
        "cleanup_policy": "retry_current_stage_only",
        "warnings": [],
    },
    "KPOINTS": {
        "modified_files": ["KPOINTS"],
        "requested_resume_strategy": "default_rule",
        "resume_stage_ref": "scf",
        "cleanup_policy": "invalidate_downstream",
        "warnings": [],
    },
    "POSCAR": {
        "modified_files": ["POSCAR"],
        "requested_resume_strategy": "default_rule",
        "resume_stage_ref": "relax",
        "cleanup_policy": "restart_from_stage",
        "warnings": [],
    },
    "multiple": {
        "modified_files": [],
        "requested_resume_strategy": "default_rule",
        "resume_stage_ref": "relax",
        "cleanup_policy": "restart_from_stage",
        "warnings": ["explicit_confirmation_required_for_multiple_files"],
    },
    "custom": {
        "modified_files": [],
        "requested_resume_strategy": "default_rule",
        "resume_stage_ref": "current",
        "cleanup_policy": "retry_current_stage_only",
        "warnings": [],
    },
}


def _downstream_from(stage: str) -> list[str]:
    if stage not in STAGE_ORDER:
        return []
    idx = STAGE_ORDER.index(stage)
    return list(STAGE_ORDER[idx + 1 :])


STAGE_CONTRACTS: dict[str, StageContract] = {
    "prepare": StageContract(
        stage="prepare",
        required_inputs=["task.root_path", "material.poscar_path", "material.potcar_path"],
        canonical_outputs=["execution.workdir", "material.structure_summary", "material.preflight_summary", "execution.environment_summary"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.prepare"],
        retryable=True,
        invalidates_downstream=_downstream_from("prepare"),
        artifact_patterns=[],
    ),
    "relax": StageContract(
        stage="relax",
        required_inputs=["execution.workdir", "material.poscar_path", "material.potcar_path"],
        canonical_outputs=["physics_results.relaxed_structure_path", "physics_results.reciprocal_lattice", "physics_results.relax_summary"],
        failure_outputs=["diagnostics.recovery_summary", "diagnostics.raw_evidence.relax"],
        retryable=True,
        invalidates_downstream=["scf", "band", "effective_mass", "strain_loop", "mobility", "validation", "report"],
        artifact_patterns=["01_relax"],
    ),
    "scf": StageContract(
        stage="scf",
        required_inputs=["physics_results.relaxed_structure_path", "material.potcar_path"],
        canonical_outputs=["physics_results.fermi_energy", "execution.artifact_paths.CHGCAR", "physics_results.scf_summary"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.scf"],
        retryable=True,
        invalidates_downstream=["band", "effective_mass", "strain_loop", "mobility", "validation", "report"],
        artifact_patterns=["02_scf"],
    ),
    "band": StageContract(
        stage="band",
        required_inputs=["physics_results.relaxed_structure_path", "material.potcar_path"],
        canonical_outputs=["physics_results.band_summary", "physics_results.vbm_energy", "physics_results.cbm_energy"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.band"],
        retryable=True,
        invalidates_downstream=["effective_mass", "mobility", "validation", "report"],
        artifact_patterns=["03_band"],
    ),
    "effective_mass": StageContract(
        stage="effective_mass",
        required_inputs=["physics_results.band_summary", "physics_results.reciprocal_lattice"],
        canonical_outputs=["physics_results.masses", "physics_results.effective_mass_summary"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.effective_mass"],
        retryable=True,
        invalidates_downstream=["mobility", "validation", "report"],
        artifact_patterns=["04_effmass_*"],
    ),
    "strain_loop": StageContract(
        stage="strain_loop",
        required_inputs=["physics_results.relaxed_structure_path", "material.potcar_path"],
        canonical_outputs=["physics_results.strain_data", "diagnostics.strain_summary"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.strain_loop"],
        retryable=True,
        invalidates_downstream=["mobility", "validation", "report"],
        artifact_patterns=["05_strain"],
    ),
    "refinement": StageContract(
        stage="refinement",
        required_inputs=["diagnostics.strain_summary", "workflow.refinement_rounds"],
        canonical_outputs=["workflow.refinement_rounds", "physics_results.accepted_channels", "physics_results.rejected_channels"],
        failure_outputs=["diagnostics.consultation_trace"],
        retryable=False,
        invalidates_downstream=["mobility", "validation", "report"],
        artifact_patterns=[],
    ),
    "mobility": StageContract(
        stage="mobility",
        required_inputs=["physics_results.strain_data", "physics_results.masses"],
        canonical_outputs=["physics_results.mobility", "physics_results.E1", "physics_results.C2D", "physics_results.mobility_summary", "diagnostics.fit_diagnostics"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.mobility"],
        retryable=True,
        invalidates_downstream=["validation", "report"],
        artifact_patterns=[
            "mobility_results.json",
            "fit_diagnostics.json",
            "decision_trace.json",
            "tool_trace.json",
            "recovery_trace.json",
            "validation_report.json",
            "final_summary.json",
            "strain_data.csv",
            "strain_status.csv",
            "material_outcome.json",
        ],
    ),
    "validation": StageContract(
        stage="validation",
        required_inputs=["physics_results.mobility", "diagnostics.fit_diagnostics"],
        canonical_outputs=["diagnostics.validation_report", "diagnostics.confidence_score"],
        failure_outputs=["diagnostics.validation_report"],
        retryable=False,
        invalidates_downstream=["report"],
        artifact_patterns=["validation_report.json", "final_summary.json", "material_outcome.json"],
    ),
    "report": StageContract(
        stage="report",
        required_inputs=["diagnostics.validation_report", "execution.artifact_paths"],
        canonical_outputs=["execution.artifact_paths.final_summary_path"],
        failure_outputs=["diagnostics.last_error", "diagnostics.raw_evidence.report"],
        retryable=False,
        invalidates_downstream=[],
        artifact_patterns=["final_summary.json", "material_outcome.json"],
    ),
}


def get_stage_contract(stage: str) -> StageContract:
    if stage not in STAGE_CONTRACTS:
        raise KeyError(f"unknown_stage_contract:{stage}")
    return STAGE_CONTRACTS[stage]


def stage_before(left: str, right: str) -> bool:
    return STAGE_ORDER.index(left) < STAGE_ORDER.index(right)


def downstream_stages(stage: str) -> list[str]:
    return list(get_stage_contract(stage).invalidates_downstream)


def find_previous_stage(stage: str) -> str | None:
    if stage not in STAGE_ORDER:
        return None
    idx = STAGE_ORDER.index(stage)
    if idx <= 0:
        return None
    return STAGE_ORDER[idx - 1]


def validate_stage_name(stage: str) -> str:
    if stage not in STAGE_CONTRACTS:
        raise ManualFixValidationError(f"invalid_resume_stage:{stage}")
    return stage


def default_cleanup_policy_for_resume_stage(*, current_stage: str, resume_stage: str) -> CleanupPolicyName:
    current = validate_stage_name(current_stage)
    target = validate_stage_name(resume_stage)
    if target == current:
        return "retry_current_stage_only"
    if target == "relax":
        return "restart_from_stage"
    return "invalidate_downstream"


def _resolve_stage_reference(*, current_stage: str, stage_ref: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if stage_ref == "current":
        return validate_stage_name(current_stage), warnings
    if stage_ref == "previous":
        previous = find_previous_stage(current_stage)
        if previous is None:
            warnings.append("no_previous_stage_available_using_current_stage")
            return validate_stage_name(current_stage), warnings
        return previous, warnings
    return validate_stage_name(stage_ref), warnings


def resolve_resume_contract(
    *,
    current_stage: str,
    modification_type: ManualFixModificationType,
    requested_resume_strategy: ResumeStrategyName | None = None,
    selected_resume_stage: str | None = None,
    selected_cleanup_policy: CleanupPolicyName | None = None,
    modified_files: list[str] | None = None,
) -> ResumeRuleContract:
    current = validate_stage_name(current_stage)
    defaults = dict(MANUAL_FIX_DEFAULTS.get(modification_type, MANUAL_FIX_DEFAULTS["custom"]))
    requested = (requested_resume_strategy or defaults.get("requested_resume_strategy") or "default_rule")
    warnings = [str(item) for item in list(defaults.get("warnings", []) or [])]

    if requested == "default_rule":
        stage_name, stage_warnings = _resolve_stage_reference(
            current_stage=current,
            stage_ref=str(defaults.get("resume_stage_ref") or "current"),
        )
        warnings.extend(stage_warnings)
        cleanup_policy = defaults.get("cleanup_policy") or default_cleanup_policy_for_resume_stage(
            current_stage=current,
            resume_stage=stage_name,
        )
    elif requested == "retry_current_stage":
        stage_name = current
        cleanup_policy = selected_cleanup_policy or "retry_current_stage_only"
    elif requested == "rerun_previous_stage":
        stage_name, stage_warnings = _resolve_stage_reference(current_stage=current, stage_ref="previous")
        warnings.extend(stage_warnings)
        cleanup_policy = selected_cleanup_policy or default_cleanup_policy_for_resume_stage(
            current_stage=current,
            resume_stage=stage_name,
        )
    elif requested == "rerun_from_relax":
        stage_name = "relax"
        cleanup_policy = selected_cleanup_policy or "restart_from_stage"
    elif requested == "custom_stage":
        if not str(selected_resume_stage or "").strip():
            raise ManualFixValidationError("custom_stage_requires_resume_stage")
        stage_name = validate_stage_name(selected_resume_stage)
        cleanup_policy = selected_cleanup_policy or default_cleanup_policy_for_resume_stage(
            current_stage=current,
            resume_stage=stage_name,
        )
    else:
        raise ManualFixValidationError(f"invalid_requested_resume_strategy:{requested}")

    validate_resume_request(current_stage=current, resume_stage=stage_name, cleanup_policy=cleanup_policy)
    return ResumeRuleContract(
        modified_files=list(modified_files or defaults.get("modified_files", []) or []),
        modification_type=modification_type,
        requested_resume_strategy=requested,
        resume_stage=stage_name,
        cleanup_policy=cleanup_policy,
        invalidated_stages=invalidated_stages_for(
            resume_stage=stage_name,
            cleanup_policy=cleanup_policy,
        ),
        warnings=list(dict.fromkeys(warnings)),
    )


def invalidated_stages_for(*, resume_stage: str, cleanup_policy: CleanupPolicyName) -> list[str]:
    stage_name = validate_stage_name(resume_stage)
    if cleanup_policy == "retry_current_stage_only":
        return [stage_name]
    if cleanup_policy == "invalidate_downstream":
        return list(downstream_stages(stage_name))
    return [stage_name] + list(downstream_stages(stage_name))


def validate_resume_request(*, current_stage: str, resume_stage: str, cleanup_policy: CleanupPolicyName) -> str:
    current = validate_stage_name(current_stage)
    target = validate_stage_name(resume_stage)
    if STAGE_ORDER.index(target) > STAGE_ORDER.index(current):
        raise ManualFixValidationError(f"incompatible_resume_stage:{target}:after_current_stage:{current}")
    if cleanup_policy == "retry_current_stage_only" and target != current:
        raise ManualFixValidationError(
            f"incompatible_cleanup_policy:{cleanup_policy}:requires_current_stage:{current}:got:{target}"
        )
    if cleanup_policy == "invalidate_downstream" and target == current:
        raise ManualFixValidationError(
            f"incompatible_cleanup_policy:{cleanup_policy}:requires_restart_before_current_stage:{current}"
        )
    return target


def _collect_stage_artifacts(workdir: str, stage: str) -> list[str]:
    artifacts: list[str] = []
    for pattern in get_stage_contract(stage).artifact_patterns:
        candidate = os.path.join(workdir, pattern)
        if "*" in pattern:
            artifacts.extend(sorted(glob.glob(candidate)))
        else:
            artifacts.append(candidate)
    return [os.path.abspath(path) for path in artifacts if os.path.exists(path)]


def build_cleanup_preview(*, workdir: str, resume_stage: str, cleanup_policy: CleanupPolicyName) -> CleanupPreview:
    invalidated_stages = invalidated_stages_for(resume_stage=resume_stage, cleanup_policy=cleanup_policy)
    invalidated_artifacts: list[str] = []
    for stage in invalidated_stages:
        invalidated_artifacts.extend(_collect_stage_artifacts(workdir, stage))
    warnings: list[str] = []
    if not invalidated_stages:
        warnings.append("no_stage_invalidation")
    if invalidated_stages and not invalidated_artifacts:
        warnings.append("no_matching_artifacts_found")
    return CleanupPreview(
        cleanup_policy=cleanup_policy,
        resume_stage=resume_stage,
        invalidated_stages=invalidated_stages,
        invalidated_artifacts=sorted({os.path.abspath(path) for path in invalidated_artifacts}),
        warnings=warnings,
    )
