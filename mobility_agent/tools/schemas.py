from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..utils import dedupe_keep_order


class ToolFailureEvidence(BaseModel):
    returncode: int | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_paths: list[str] = Field(default_factory=list)
    parser_payload: dict[str, Any] = Field(default_factory=dict)
    exception_type: str | None = None
    exception_message: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self):
        self.log_paths = [str(path) for path in dedupe_keep_order(self.log_paths or []) if str(path)]
        self.parser_payload = dict(self.parser_payload or {})
        self.raw_payload = dict(self.raw_payload or {})
        if self.exception_type is not None:
            self.exception_type = str(self.exception_type)
        if self.exception_message is not None:
            self.exception_message = str(self.exception_message)
        return self


class ToolExecutionResult(BaseModel):
    stage: str
    status: Literal["success", "failed", "skipped"] = "success"
    error_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    key_summary: dict[str, Any] = Field(default_factory=dict)
    state_updates: dict[str, Any] = Field(default_factory=dict)
    raw_evidence: ToolFailureEvidence = Field(default_factory=ToolFailureEvidence)
    invocation_source: str = "native_tool"
    duration_s: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == "success"

    @model_validator(mode="after")
    def _normalize(self):
        self.stage = str(self.stage)
        self.warnings = [str(item) for item in dedupe_keep_order(self.warnings or [])]
        self.artifact_paths = {str(k): str(v) for k, v in dict(self.artifact_paths or {}).items() if v}
        self.key_summary = dict(self.key_summary or {})
        self.state_updates = dict(self.state_updates or {})
        self.invocation_source = str(self.invocation_source or "native_tool")
        return self

