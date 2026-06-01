from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from ..utils import dedupe_keep_order


class RetrievedEvidence(BaseModel):
    corpus: str
    source_id: str
    title: str
    chunk_id: str = ""
    revision_id: str = ""
    url_or_path: str = ""
    heading: str = ""
    stage: str = ""
    snippet: str
    score: float = 0.0
    tags: list[str] = Field(default_factory=list)

    @property
    def reference(self) -> str:
        return f"{self.corpus}:{self.source_id}"

    @model_validator(mode="after")
    def _normalize(self):
        self.corpus = str(self.corpus or "unknown")
        self.source_id = str(self.source_id or "unknown")
        self.title = str(self.title or self.source_id)
        self.chunk_id = str(self.chunk_id or "")
        self.revision_id = str(self.revision_id or "")
        self.url_or_path = str(self.url_or_path or "")
        self.heading = str(self.heading or "")
        self.stage = str(self.stage or "")
        self.snippet = str(self.snippet or "")
        self.score = float(self.score or 0.0)
        self.tags = [str(item) for item in dedupe_keep_order(self.tags or []) if str(item)]
        return self


class StageProbe(BaseModel):
    stage: str
    material_id: str
    atom_count: int = 0
    composition: str | None = None
    structure_summary: dict[str, Any] = Field(default_factory=dict)
    resource_summary: dict[str, Any] = Field(default_factory=dict)
    kpoint_summary: dict[str, Any] = Field(default_factory=dict)
    prior_execution_summary: dict[str, Any] = Field(default_factory=dict)
    extra_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self):
        self.stage = str(self.stage or "")
        self.material_id = str(self.material_id or "")
        self.atom_count = int(self.atom_count or 0)
        self.composition = str(self.composition) if self.composition is not None else None
        self.structure_summary = dict(self.structure_summary or {})
        self.resource_summary = dict(self.resource_summary or {})
        self.kpoint_summary = dict(self.kpoint_summary or {})
        self.prior_execution_summary = dict(self.prior_execution_summary or {})
        self.extra_context = dict(self.extra_context or {})
        return self


class ParameterPlan(BaseModel):
    stage: str
    source: str = "fallback"
    incar_overrides: dict[str, Any] = Field(default_factory=dict)
    kpoints_policy: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_items: list[RetrievedEvidence] = Field(default_factory=list)
    house_rule_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""

    @model_validator(mode="after")
    def _normalize(self):
        self.stage = str(self.stage or "")
        self.source = str(self.source or "fallback")
        self.incar_overrides = {str(k): v for k, v in dict(self.incar_overrides or {}).items()}
        self.kpoints_policy = {str(k): v for k, v in dict(self.kpoints_policy or {}).items()}
        self.runtime_policy = {str(k): v for k, v in dict(self.runtime_policy or {}).items()}
        self.evidence_refs = [str(item) for item in dedupe_keep_order(self.evidence_refs or []) if str(item)]
        self.evidence_items = list(self.evidence_items or [])
        self.house_rule_refs = [str(item) for item in dedupe_keep_order(self.house_rule_refs or []) if str(item)]
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.rationale = str(self.rationale or "")
        return self


class FailureDiagnosis(BaseModel):
    stage: str
    source: str = "fallback"
    hypotheses: list[str] = Field(default_factory=list)
    recommended_action: str = "retry_capability"
    parameter_patch: dict[str, Any] = Field(default_factory=dict)
    needs_human: bool = False
    evidence_refs: list[str] = Field(default_factory=list)
    evidence_items: list[RetrievedEvidence] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""

    @model_validator(mode="after")
    def _normalize(self):
        self.stage = str(self.stage or "")
        self.source = str(self.source or "fallback")
        self.hypotheses = [str(item) for item in dedupe_keep_order(self.hypotheses or []) if str(item)]
        self.recommended_action = str(self.recommended_action or "retry_capability")
        self.parameter_patch = {str(k): v for k, v in dict(self.parameter_patch or {}).items()}
        self.needs_human = bool(self.needs_human)
        self.evidence_refs = [str(item) for item in dedupe_keep_order(self.evidence_refs or []) if str(item)]
        self.evidence_items = list(self.evidence_items or [])
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.rationale = str(self.rationale or "")
        return self
