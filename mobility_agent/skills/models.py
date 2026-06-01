from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..utils import dedupe_keep_order


LoadStrategyName = Literal["summary_only", "summary_and_body", "manual"]


def _string_list(values: list[Any] | tuple[Any, ...] | None) -> list[str]:
    normalized = [str(item or "").strip() for item in list(values or [])]
    return [item for item in dedupe_keep_order(normalized) if item]


class SkillResource(BaseModel):
    path: str
    kind: str = "reference"
    description: str = ""
    size_bytes: int | None = None

    @property
    def basename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class SkillManifest(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    purpose: str = ""
    when_to_use: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    relevant_state_fields: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    decision_rules: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    expected_output_schema: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    stages: list[str] = Field(default_factory=list)
    run_statuses: list[str] = Field(default_factory=list)
    error_patterns: list[str] = Field(default_factory=list)
    anomaly_patterns: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    load_strategy: LoadStrategyName = "summary_only"
    resource_roots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self):
        self.name = str(self.name or "").strip()
        self.version = str(self.version or "1").strip() or "1"
        self.description = str(self.description or "").strip()
        self.purpose = str(self.purpose or "").strip()
        self.when_to_use = _string_list(self.when_to_use)
        self.required_inputs = _string_list(self.required_inputs)
        self.relevant_state_fields = _string_list(self.relevant_state_fields)
        self.allowed_tools = _string_list(self.allowed_tools)
        self.decision_rules = _string_list(self.decision_rules)
        self.stop_conditions = _string_list(self.stop_conditions)
        self.expected_output_schema = _string_list(self.expected_output_schema)
        self.caveats = _string_list(self.caveats)
        self.roles = _string_list(self.roles)
        self.task_types = _string_list(self.task_types)
        self.stages = _string_list(self.stages)
        self.run_statuses = _string_list(self.run_statuses)
        self.error_patterns = _string_list(self.error_patterns)
        self.anomaly_patterns = _string_list(self.anomaly_patterns)
        self.tags = _string_list(self.tags)
        self.resource_roots = _string_list(self.resource_roots)
        return self


class SkillRegistryEntry(BaseModel):
    name: str
    path: str
    skill_md: str
    description: str = ""
    manifest: SkillManifest
    summary: str = ""
    resources: list[SkillResource] = Field(default_factory=list)


class SkillLoadResult(BaseModel):
    name: str
    path: str
    skill_md: str
    description: str = ""
    manifest: SkillManifest
    summary: str = ""
    text: str = ""
    resource_payloads: dict[str, Any] = Field(default_factory=dict)
    resources: list[SkillResource] = Field(default_factory=list)


class SkillResolutionRequest(BaseModel):
    role: str | None = None
    task_type: str | None = None
    stage: str | None = None
    run_status: str | None = None
    has_error: bool = False
    latest_error: str | None = None
    anomaly_flags: list[str] = Field(default_factory=list)
    explicit_skills: list[str] = Field(default_factory=list)
    limit: int = 6

    @model_validator(mode="after")
    def _normalize(self):
        self.role = str(self.role or "").strip().lower() or None
        self.task_type = str(self.task_type or "").strip() or None
        self.stage = str(self.stage or "").strip() or None
        self.run_status = str(self.run_status or "").strip() or None
        self.latest_error = str(self.latest_error or "").strip() or None
        self.anomaly_flags = _string_list(self.anomaly_flags)
        self.explicit_skills = _string_list(self.explicit_skills)
        self.limit = max(1, int(self.limit or 1))
        return self


class SkillCandidate(BaseModel):
    name: str
    score: float = 0.0
    selected: bool = False
    reasons: list[str] = Field(default_factory=list)
    manifest: SkillManifest

    @model_validator(mode="after")
    def _normalize(self):
        self.reasons = _string_list(self.reasons)
        return self


class SkillSelectionRecord(BaseModel):
    role: str | None = None
    task_type: str | None = None
    stage: str | None = None
    run_status: str | None = None
    selected_skills: list[str] = Field(default_factory=list)
    candidates: list[SkillCandidate] = Field(default_factory=list)
    resolution_mode: str = "resolver"

    @model_validator(mode="after")
    def _normalize(self):
        self.selected_skills = _string_list(self.selected_skills)
        self.resolution_mode = str(self.resolution_mode or "resolver").strip() or "resolver"
        return self
