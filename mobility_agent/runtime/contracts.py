from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..graph.state import utc_now_iso
from ..utils import dedupe_keep_order


ContractStatus = Literal["active", "superseded", "completed", "aborted"]


class CapabilityDecision(BaseModel):
    capability: str
    action_family: str = "run_capability"
    source_agents: list[str] = Field(default_factory=list)
    supporting_agents: list[str] = Field(default_factory=list)
    opposing_agents: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    fallback_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "CapabilityDecision":
        self.capability = str(self.capability or "").strip()
        self.action_family = str(self.action_family or "run_capability").strip() or "run_capability"
        self.source_agents = [str(item) for item in dedupe_keep_order(self.source_agents or []) if str(item or "").strip()]
        self.supporting_agents = [str(item) for item in dedupe_keep_order(self.supporting_agents or []) if str(item or "").strip()]
        self.opposing_agents = [str(item) for item in dedupe_keep_order(self.opposing_agents or []) if str(item or "").strip()]
        self.rationale = str(self.rationale or "").strip()
        self.fallback_actions = [str(item) for item in dedupe_keep_order(self.fallback_actions or []) if str(item or "").strip()]
        return self


class WorkflowContract(BaseModel):
    contract_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    version: int = 1
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    plan_status: ContractStatus = "active"
    deliberation_reason: str = "initial_plan"
    council_mode: str = "segment_council"
    approved_by_agents: list[str] = Field(default_factory=list)
    current_focus: str | None = None
    planned_capabilities: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)
    input_hash: str | None = None
    revisit_triggers: dict[str, Any] = Field(default_factory=dict)
    allowed_branches: list[str] = Field(default_factory=list)
    decision_rationale: str = ""
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    capability_decisions: list[CapabilityDecision] = Field(default_factory=list)
    reuse_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "WorkflowContract":
        self.contract_id = str(self.contract_id or uuid.uuid4().hex).strip()
        self.version = max(1, int(self.version or 1))
        self.updated_at = str(self.updated_at or self.created_at or utc_now_iso())
        self.deliberation_reason = str(self.deliberation_reason or "initial_plan").strip() or "initial_plan"
        self.council_mode = str(self.council_mode or "segment_council").strip() or "segment_council"
        self.approved_by_agents = [str(item) for item in dedupe_keep_order(self.approved_by_agents or []) if str(item or "").strip()]
        self.current_focus = str(self.current_focus).strip() if self.current_focus is not None and str(self.current_focus or "").strip() else None
        self.planned_capabilities = [str(item) for item in dedupe_keep_order(self.planned_capabilities or []) if str(item or "").strip()]
        self.milestones = [str(item) for item in dedupe_keep_order(self.milestones or []) if str(item or "").strip()]
        self.input_hash = str(self.input_hash).strip() if self.input_hash is not None and str(self.input_hash or "").strip() else None
        self.allowed_branches = [str(item) for item in dedupe_keep_order(self.allowed_branches or []) if str(item or "").strip()]
        self.decision_rationale = str(self.decision_rationale or "").strip()
        self.evidence_summary = dict(self.evidence_summary or {})
        self.reuse_metadata = dict(self.reuse_metadata or {})
        decisions = []
        for item in list(self.capability_decisions or []):
            if isinstance(item, CapabilityDecision):
                decisions.append(item)
            else:
                decisions.append(CapabilityDecision.model_validate(item))
        self.capability_decisions = decisions
        if not self.current_focus and self.planned_capabilities:
            self.current_focus = self.planned_capabilities[0]
        return self


class ExecutionCheckpoint(BaseModel):
    contract_id: str | None = None
    contract_version: int = 0
    current_capability: str | None = None
    next_capability: str | None = None
    completed_capabilities: list[str] = Field(default_factory=list)
    last_observation: dict[str, Any] = Field(default_factory=dict)
    needs_deliberation: bool = True
    deliberation_reason: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)

    @model_validator(mode="after")
    def _normalize(self) -> "ExecutionCheckpoint":
        self.contract_id = str(self.contract_id).strip() if self.contract_id is not None and str(self.contract_id or "").strip() else None
        self.contract_version = max(0, int(self.contract_version or 0))
        self.current_capability = str(self.current_capability).strip() if self.current_capability is not None and str(self.current_capability or "").strip() else None
        self.next_capability = str(self.next_capability).strip() if self.next_capability is not None and str(self.next_capability or "").strip() else None
        self.completed_capabilities = [str(item) for item in dedupe_keep_order(self.completed_capabilities or []) if str(item or "").strip()]
        self.last_observation = dict(self.last_observation or {})
        self.deliberation_reason = (
            str(self.deliberation_reason).strip()
            if self.deliberation_reason is not None and str(self.deliberation_reason or "").strip()
            else None
        )
        self.updated_at = str(self.updated_at or utc_now_iso())
        return self


class DecisionLedgerEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: str = Field(default_factory=utc_now_iso)
    entry_type: str
    reason: str = ""
    round_id: int = 0
    contract_id: str | None = None
    contract_version: int | None = None
    agent_names: list[str] = Field(default_factory=list)
    selected_action: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "DecisionLedgerEntry":
        self.entry_id = str(self.entry_id or uuid.uuid4().hex).strip()
        self.timestamp = str(self.timestamp or utc_now_iso())
        self.entry_type = str(self.entry_type or "").strip() or "runtime_event"
        self.reason = str(self.reason or "").strip()
        self.round_id = max(0, int(self.round_id or 0))
        self.contract_id = str(self.contract_id).strip() if self.contract_id is not None and str(self.contract_id or "").strip() else None
        self.contract_version = int(self.contract_version or 0) if self.contract_version is not None else None
        self.agent_names = [str(item) for item in dedupe_keep_order(self.agent_names or []) if str(item or "").strip()]
        self.selected_action = dict(self.selected_action or {})
        self.summary = dict(self.summary or {})
        return self
