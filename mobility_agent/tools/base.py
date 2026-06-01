from __future__ import annotations

import time
from typing import Any, Callable, Type

from pydantic import BaseModel, Field


class ToolInputBase(BaseModel):
    material_id: str
    base_dir: str
    state_payload: dict[str, Any] = Field(default_factory=dict)


class ToolOutputBase(BaseModel):
    success: bool
    warnings: list[str] = Field(default_factory=list)
    key_summary: dict[str, Any] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    error_summary: str | None = None
    duration_s: float = 0.0
    state_updates: dict[str, Any] = Field(default_factory=dict)


class DeterministicTool:
    name: str = "tool"
    description: str = "deterministic tool"
    input_model: Type[ToolInputBase] = ToolInputBase
    output_model: Type[ToolOutputBase] = ToolOutputBase

    def __init__(self, executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        self.executor = executor

    def _execute(self, inputs: ToolInputBase) -> dict[str, Any]:
        if self.executor is None:
            raise NotImplementedError(f"{self.name} executor is not configured")
        raw = dict(self.executor(dict(inputs.state_payload)) or {})
        raw.setdefault("_tool_source", "legacy_bridge")
        return raw

    def _build_output(self, inputs: ToolInputBase, raw: dict[str, Any], duration_s: float) -> ToolOutputBase:
        success = not bool(raw.get("errors"))
        warnings = list(raw.get("warnings", []) or [])
        error_summary = "; ".join(list(raw.get("errors", []) or [])) if raw.get("errors") else None
        key_summary = {k: v for k, v in raw.items() if k not in {"strain_data", "results", "errors", "warnings"}}
        return self.output_model(
            success=success,
            warnings=warnings,
            key_summary=key_summary,
            artifact_paths={},
            error_summary=error_summary,
            duration_s=duration_s,
            state_updates=raw,
        )

    def run(self, inputs: ToolInputBase) -> ToolOutputBase:
        started = time.time()
        try:
            raw = self._execute(inputs)
            return self._build_output(inputs, raw, time.time() - started)
        except Exception as e:
            return self.output_model(
                success=False,
                warnings=[],
                key_summary={},
                artifact_paths={},
                error_summary=str(e),
                duration_s=time.time() - started,
                state_updates={"errors": [str(e)]},
            )


def build_artifact_map(*paths: str) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in paths:
        if path:
            artifacts[path.split("/")[-1]] = path
    return artifacts