from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..graph.stage_contracts import get_stage_contract
from ..graph.state import CAPABILITY_DEPENDENCIES, CAPABILITY_SEQUENCE


class CapabilityDescriptor(BaseModel):
    capability: str
    dependencies: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    canonical_outputs: list[str] = Field(default_factory=list)
    failure_outputs: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    retryable: bool = True
    default_action_family: str = "run_capability"

    @model_validator(mode="after")
    def _normalize(self) -> "CapabilityDescriptor":
        self.capability = str(self.capability or "").strip()
        self.dependencies = [str(item) for item in self.dependencies if str(item or "").strip()]
        self.required_inputs = [str(item) for item in self.required_inputs if str(item or "").strip()]
        self.canonical_outputs = [str(item) for item in self.canonical_outputs if str(item or "").strip()]
        self.failure_outputs = [str(item) for item in self.failure_outputs if str(item or "").strip()]
        self.expected_artifacts = [str(item) for item in self.expected_artifacts if str(item or "").strip()]
        self.default_action_family = str(self.default_action_family or "run_capability").strip() or "run_capability"
        return self


def capability_descriptors() -> list[CapabilityDescriptor]:
    descriptors: list[CapabilityDescriptor] = []
    for capability in CAPABILITY_SEQUENCE:
        contract = get_stage_contract(capability)
        descriptors.append(
            CapabilityDescriptor(
                capability=capability,
                dependencies=list(CAPABILITY_DEPENDENCIES.get(capability, [])),
                required_inputs=list(contract.required_inputs),
                canonical_outputs=list(contract.canonical_outputs),
                failure_outputs=list(contract.failure_outputs),
                expected_artifacts=list(contract.artifact_patterns),
                retryable=bool(contract.retryable),
            )
        )
    return descriptors


def capability_registry_payload() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in capability_descriptors()]


def plan_from_capability(start_capability: str | None) -> list[str]:
    capability = str(start_capability or "").strip()
    if not capability or capability not in CAPABILITY_SEQUENCE:
        return list(CAPABILITY_SEQUENCE)
    start_index = CAPABILITY_SEQUENCE.index(capability)
    return list(CAPABILITY_SEQUENCE[start_index:])


def next_capability_after(completed_capabilities: list[str], planned_capabilities: list[str]) -> str | None:
    completed = {str(item or "").strip() for item in list(completed_capabilities or []) if str(item or "").strip()}
    for capability in list(planned_capabilities or []):
        normalized = str(capability or "").strip()
        if normalized and normalized not in completed:
            return normalized
    return None
