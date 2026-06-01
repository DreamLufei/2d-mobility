from __future__ import annotations

from ..graph.stage_contracts import (
    CleanupPolicyName,
    ManualFixModificationType as ModificationType,
    ResumeRuleContract as ResumeRulePreview,
    ResumeStrategyName,
    resolve_resume_contract,
    validate_stage_name as _validate_stage_name,
)


def validate_stage_name(stage: str) -> str:
    return _validate_stage_name(stage)


def compute_default_resume_rule(*, current_stage: str, modification_type: ModificationType) -> ResumeRulePreview:
    return resolve_resume_contract(
        current_stage=current_stage,
        modification_type=modification_type,
        requested_resume_strategy="default_rule",
    )


def build_custom_resume_rule(
    *,
    resume_stage: str,
    cleanup_policy: CleanupPolicyName,
    modified_files: list[str],
    current_stage: str | None = None,
    modification_type: ModificationType = "custom",
    requested_resume_strategy: ResumeStrategyName = "custom_stage",
) -> ResumeRulePreview:
    current = current_stage or resume_stage
    return resolve_resume_contract(
        current_stage=current,
        modification_type=modification_type,
        requested_resume_strategy=requested_resume_strategy,
        selected_resume_stage=resume_stage,
        selected_cleanup_policy=cleanup_policy,
        modified_files=modified_files,
    )
