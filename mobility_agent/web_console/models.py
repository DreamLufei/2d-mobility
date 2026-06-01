from __future__ import annotations

import os
from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


RUNTIME_RUN_STATUSES = {
    "pending",
    "running",
    "waiting_external",
    "needs_human",
    "completed",
    "failed",
    "aborted",
    "skipped",
}

CONTROL_PLANE_STATUSES = {
    "queued",
    "starting",
    "live",
    "disconnected",
    "cancelled",
    "archived",
}

TERMINAL_RUNTIME_STATUSES = {"completed", "failed", "aborted", "skipped"}

ARTIFACT_FILENAME_WHITELIST = {
    "mobility_results.json": "mobility_results_path",
    "fit_diagnostics.json": "fit_diagnostics_path",
    "decision_trace.json": "decision_trace_path",
    "tool_trace.json": "tool_trace_path",
    "recovery_trace.json": "recovery_trace_path",
    "validation_report.json": "validation_report_path",
    "final_summary.json": "final_summary_path",
    "material_outcome.json": "material_outcome_path",
    "checkpoint.pkl": "compatibility_checkpoint_path",
    "human_escalation_payload.json": "human_escalation_payload_path",
    "human_escalation_response.json": "human_escalation_response_path",
    "human_escalation_log.json": "human_escalation_log_path",
}


def imported_job_id(prefix: str, anchor: str) -> str:
    digest = sha1(anchor.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}::{digest}"


class RuntimeOverrides(BaseModel):
    dry_run: bool | None = None
    dry_run_fail_stages: list[str] = Field(default_factory=list)
    hitl_policy: str | None = None
    compatibility_export_enabled: bool | None = None
    compatibility_export_pickle: bool | None = None
    checkpoint_subdir: str | None = None
    db_uri: str | None = None
    skills_root: str | None = None
    skill_auto_resolve_limit: int | None = None
    skill_inline_body_limit: int | None = None

    @field_validator("dry_run_fail_stages", mode="before")
    @classmethod
    def _normalize_fail_stages(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in list(value or []) if str(item).strip()]


class RuntimeSettingsView(BaseModel):
    service_preset: str = "custom"
    mobility_db_uri: str = ""
    llm_provider: str = "openai"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key_present: bool = False
    llm_api_key_preview: str | None = None
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key_present: bool = False
    embedding_api_key_preview: str | None = None
    wiki_qa_model: str = ""
    agentic_policy_enabled: bool = True
    policy_allowlist_mode: str = "restricted"
    policy_retrieval_top_k: int = 5
    policy_trace_enabled: bool = True
    rag_top_k: int = 6
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 180
    rag_reindex_batch_size: int = 64
    hitl_policy: str = "interactive"
    human_review_timeout_seconds: int = 300
    human_review_default_action: str = "skip_material"
    enable_email_notifications: bool = False
    email_notify_to: str = ""
    email_dry_run: bool = True
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: str = ""
    smtp_from: str = ""
    smtp_password_present: bool = False
    smtp_password_preview: str | None = None


class RuntimeSettingsUpdateRequest(BaseModel):
    mobility_db_uri: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    clear_llm_api_key: bool = False
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    clear_embedding_api_key: bool = False
    wiki_qa_model: str | None = None
    agentic_policy_enabled: bool | None = None
    policy_allowlist_mode: str | None = None
    policy_retrieval_top_k: int | None = None
    policy_trace_enabled: bool | None = None
    rag_top_k: int | None = None
    rag_chunk_size: int | None = None
    rag_chunk_overlap: int | None = None
    rag_reindex_batch_size: int | None = None
    hitl_policy: str | None = None
    human_review_timeout_seconds: int | None = None
    human_review_default_action: str | None = None
    enable_email_notifications: bool | None = None
    email_notify_to: str | None = None
    email_dry_run: bool | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_use_tls: bool | None = None
    smtp_username: str | None = None
    smtp_from: str | None = None
    smtp_password: str | None = None
    clear_smtp_password: bool = False


class SingleJobRequest(BaseModel):
    display_name: str | None = None
    root_path: str
    workdir: str | None = None
    material_id: str | None = None
    poscar_path: str | None = None
    potcar_path: str | None = None
    user_goal: str = "calculate_2d_mobility"
    fresh: bool = False
    runtime: RuntimeOverrides = Field(default_factory=RuntimeOverrides)

    @field_validator("root_path", "workdir", "poscar_path", "potcar_path", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return os.path.abspath(text)


class BatchConfigPayload(BaseModel):
    mongo_uri: str
    mongo_db: str
    mongo_collection: str
    batch_tag: str
    runs_root: str
    potcar_method: str = "vaspkit"
    vaspkit_cmd: str = "vaspkit"
    vaspkit_task: int = 103
    potcar_root: str | None = None
    potcar_map_path: str | None = None
    retry_failed: bool = False
    running_stale_s: int = 12 * 3600

    @field_validator("runs_root", "potcar_root", "potcar_map_path", mode="before")
    @classmethod
    def _normalize_batch_paths(cls, value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return os.path.abspath(text)


class BatchConfigOverrides(BaseModel):
    mongo_uri: str | None = None
    mongo_db: str | None = None
    mongo_collection: str | None = None
    batch_tag: str | None = None
    runs_root: str | None = None
    potcar_method: str | None = None
    vaspkit_cmd: str | None = None
    vaspkit_task: int | None = None
    potcar_root: str | None = None
    potcar_map_path: str | None = None
    retry_failed: bool | None = None
    running_stale_s: int | None = None

    @field_validator("runs_root", "potcar_root", "potcar_map_path", mode="before")
    @classmethod
    def _normalize_override_paths(cls, value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return os.path.abspath(text)


class BatchJobRequest(BaseModel):
    display_name: str | None = None
    fresh_materials: bool = False
    thread_id: str | None = None
    config: BatchConfigPayload | None = None
    config_overrides: BatchConfigOverrides = Field(default_factory=BatchConfigOverrides)
    runtime: RuntimeOverrides = Field(default_factory=RuntimeOverrides)


class WikiQueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    stage: str | None = None
    corpora: list[str] = Field(default_factory=lambda: ["vasp_wiki"])


class WikiReindexRequest(BaseModel):
    mode: Literal["full", "incremental"] = "incremental"
    include_all_pages: bool = False
    max_pages: int | None = None
    delay_seconds: float = 0.2


class HitlResponseRequest(BaseModel):
    action: str
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    instruction: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", exclude_none=True)
        if payload.get("instruction") is None:
            payload.pop("instruction", None)
        return payload


class ExternalEventResumeRequest(BaseModel):
    thread_id: str | None = None
    event: dict[str, Any] = Field(default_factory=dict)


class WorkerJobSpec(BaseModel):
    job_id: str
    job_type: Literal["single_material", "batch", "wiki_reindex"]
    request: dict[str, Any] = Field(default_factory=dict)


class JobSnapshot(BaseModel):
    job_id: str
    job_type: str
    job_role: str
    display_name: str
    material_id: str | None = None
    batch_tag: str | None = None
    root_path: str
    workdir: str | None = None
    thread_id: str | None = None
    pid: int | None = None
    pgid: int | None = None
    runtime_run_status: str = "pending"
    control_plane_status: str = "queued"
    final_acceptance: str | None = None
    quality_grade: str | None = None
    current_stage: str | None = None
    hitl_pending: bool = False
    wait_reason: str | None = None
    error_summary: str | None = None
    last_progress_line: str | None = None
    last_state_updated_at: str | None = None
    last_heartbeat_at: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    parent_job_id: str | None = None
    child_job_ids: list[str] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
