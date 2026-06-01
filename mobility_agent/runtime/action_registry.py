from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..graph.state import CAPABILITY_DEPENDENCIES, CAPABILITY_SEQUENCE


CostClass = Literal["low", "medium", "high"]
RiskClass = Literal["low", "medium", "high"]


class ActionSpec(BaseModel):
    action_family: str
    dependencies: list[str] = Field(default_factory=list)
    legal_parameters: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    fallback_actions: list[str] = Field(default_factory=list)
    cost_class: CostClass = "medium"
    risk_class: RiskClass = "medium"


ACTION_REGISTRY: dict[str, ActionSpec] = {
    "run_capability": ActionSpec(
        action_family="run_capability",
        dependencies=["target_capability"],
        legal_parameters=["target_capability"],
        required_evidence=[],
        expected_artifacts=["stage_outputs"],
        fallback_actions=["retry_capability", "escalate_human"],
        cost_class="medium",
        risk_class="medium",
    ),
    "retry_capability": ActionSpec(
        action_family="retry_capability",
        dependencies=["target_capability"],
        legal_parameters=["target_capability", "parameter_updates"],
        required_evidence=["latest_failure"],
        expected_artifacts=["retried_stage_outputs"],
        fallback_actions=["rerun_from_capability", "escalate_human"],
        cost_class="medium",
        risk_class="medium",
    ),
    "rerun_from_capability": ActionSpec(
        action_family="rerun_from_capability",
        dependencies=["target_capability"],
        legal_parameters=["target_capability"],
        required_evidence=["dependency_mismatch_or_failed_downstream"],
        expected_artifacts=["recomputed_downstream_outputs"],
        fallback_actions=["escalate_human", "abort_material"],
        cost_class="high",
        risk_class="medium",
    ),
    "repair_execution_context": ActionSpec(
        action_family="repair_execution_context",
        dependencies=[],
        legal_parameters=["repair_kind"],
        required_evidence=["context_corruption_or_missing_artifacts"],
        expected_artifacts=["repaired_execution_context"],
        fallback_actions=["escalate_human"],
        cost_class="low",
        risk_class="medium",
    ),
    "refine_sampling": ActionSpec(
        action_family="refine_sampling",
        dependencies=["target_capability"],
        legal_parameters=["target_capability", "target_channels", "target_directions", "suggested_points", "refinement_strategy", "target_fits"],
        required_evidence=["fit_quality_warning"],
        expected_artifacts=["additional_sampling_points"],
        fallback_actions=["revalidate_result", "abort_material"],
        cost_class="high",
        risk_class="medium",
    ),
    "revalidate_result": ActionSpec(
        action_family="revalidate_result",
        dependencies=["target_capability"],
        legal_parameters=["target_capability"],
        required_evidence=["results_present"],
        expected_artifacts=["validation_report"],
        fallback_actions=["escalate_human", "abort_material"],
        cost_class="low",
        risk_class="low",
    ),
    "invalidate_channel": ActionSpec(
        action_family="invalidate_channel",
        dependencies=[],
        legal_parameters=["target_channels"],
        required_evidence=["channel_specific_anomaly"],
        expected_artifacts=["updated_channel_registry"],
        fallback_actions=["skip_channel", "revalidate_result"],
        cost_class="low",
        risk_class="low",
    ),
    "skip_channel": ActionSpec(
        action_family="skip_channel",
        dependencies=[],
        legal_parameters=["target_channels"],
        required_evidence=["channel_specific_failure"],
        expected_artifacts=["updated_channel_registry"],
        fallback_actions=["abort_material"],
        cost_class="low",
        risk_class="medium",
    ),
    "escalate_human": ActionSpec(
        action_family="escalate_human",
        dependencies=[],
        legal_parameters=["recommended_options"],
        required_evidence=["insufficient_automation_confidence"],
        expected_artifacts=["human_escalation_payload"],
        fallback_actions=["abort_material"],
        cost_class="low",
        risk_class="low",
    ),
    "finalize_material": ActionSpec(
        action_family="finalize_material",
        dependencies=[],
        legal_parameters=[],
        required_evidence=["result_or_terminal_status"],
        expected_artifacts=["final_summary", "material_outcome"],
        fallback_actions=[],
        cost_class="low",
        risk_class="low",
    ),
    "abort_material": ActionSpec(
        action_family="abort_material",
        dependencies=[],
        legal_parameters=["reason"],
        required_evidence=["terminal_failure"],
        expected_artifacts=["final_summary", "material_outcome"],
        fallback_actions=[],
        cost_class="low",
        risk_class="high",
    ),
}


def get_action_spec(action_family: str) -> ActionSpec:
    return ACTION_REGISTRY[action_family]


def list_action_families() -> list[str]:
    return list(ACTION_REGISTRY.keys())


def capability_dependencies(capability: str) -> list[str]:
    return list(CAPABILITY_DEPENDENCIES.get(capability, []))


def capability_sequence() -> list[str]:
    return list(CAPABILITY_SEQUENCE)


def action_is_legal(action_family: str, *, target_capability: str | None = None, parameters: dict[str, Any] | None = None) -> bool:
    if action_family not in ACTION_REGISTRY:
        return False
    if action_family in {"run_capability", "retry_capability", "rerun_from_capability", "refine_sampling", "revalidate_result"}:
        return bool(target_capability)
    if action_family in {"invalidate_channel", "skip_channel"}:
        return bool((parameters or {}).get("target_channels"))
    return True
