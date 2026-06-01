from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config_runtime import AgentRuntimeConfig
from ..env import ensure_project_env_loaded
from .checkpointing import runtime_checkpoint_metadata_path, runtime_state_snapshot_path
from .database import is_postgres_uri, normalize_database_uri, redact_database_uri


HITL_POLICY_ALIASES = {
    "interactive": "interactive",
    "non_interactive_wait": "non_interactive_skip_on_timeout",
    "non_interactive_skip": "non_interactive_skip_on_timeout",
    "non_interactive_skip_on_timeout": "non_interactive_skip_on_timeout",
    "non_interactive_abort_on_timeout": "non_interactive_abort_on_timeout",
}

RUNTIME_PROFILE_ALIASES = {
    "": "default",
    "default": "default",
    "full": "full_autonomy",
    "full_power": "full_autonomy",
    "full-autonomy": "full_autonomy",
    "full_autonomy": "full_autonomy",
    "autonomy": "full_autonomy",
    "max": "full_autonomy",
    "strict": "full_autonomy",
}

FULL_AUTONOMY_PROFILE_DEFAULTS = {
    "FULL_AUTONOMY": "true",
    "ALLOW_EXTERNAL_WAIT": "false",
    "RAG_REQUIRED": "true",
    "AGENTIC_POLICY_ENABLED": "true",
    "ENABLE_HUMAN_REVIEW": "true",
    "HUMAN_REVIEW_TIMEOUT_SECONDS": "300",
    "HITL_POLICY": "interactive",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() or default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _dedupe_warnings(items: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _apply_runtime_profile_from_env() -> tuple[str, tuple[str, ...]]:
    raw_profile = _env_str("MOBILITY_PROFILE", _env_str("MOBILITY_RUNTIME_PROFILE", "default"))
    normalized_profile = RUNTIME_PROFILE_ALIASES.get(str(raw_profile or "").strip().lower(), "default")
    warnings: list[str] = []
    if str(raw_profile or "").strip() and normalized_profile == "default" and str(raw_profile).strip().lower() not in RUNTIME_PROFILE_ALIASES:
        warnings.append(f"unknown_runtime_profile:{raw_profile}->default")
    if normalized_profile == "full_autonomy":
        for key, value in FULL_AUTONOMY_PROFILE_DEFAULTS.items():
            os.environ.setdefault(key, value)
    return normalized_profile, _dedupe_warnings(warnings)


def normalize_hitl_policy(value: str | None) -> str:
    normalized, _ = normalize_hitl_policy_with_warnings(value)
    return normalized


def normalize_hitl_policy_with_warnings(value: str | None, *, source: str = "HITL_POLICY") -> tuple[str, tuple[str, ...]]:
    raw = str(value or "").strip().lower()
    if not raw:
        return "interactive", ()
    normalized = HITL_POLICY_ALIASES.get(raw, "interactive")
    warnings: list[str] = []
    if raw in {"non_interactive_wait", "non_interactive_skip"}:
        warnings.append(f"deprecated_{source}:{raw}->{normalized}")
    elif raw not in HITL_POLICY_ALIASES:
        warnings.append(f"unknown_{source}:{raw}->interactive")
    return normalized, _dedupe_warnings(warnings)


@dataclass(frozen=True)
class RuntimeContext:
    agent_runtime: AgentRuntimeConfig
    vasp_cmd: str = "mpirun -np 4 vasp_std > sout 2>&1"
    temperature: float = 300.0
    vacuum_direction: int = 2
    c2d_prefac: float = 1.0
    consider_spin: bool = False
    hitl_policy: str = "interactive"
    dry_run: bool = False
    dry_run_fail_stages: tuple[str, ...] = ()
    db_uri: str = ""
    store_path: str = ""
    notification_backend: str = "stdout"
    checkpoint_subdir: str = ".runtime"
    compatibility_export_enabled: bool = True
    compatibility_export_pickle: bool = True
    full_autonomy: bool = True
    allow_external_wait: bool = False
    rag_required: bool = False
    python_executable: str = sys.executable
    skills_root: str = ""
    skill_auto_resolve_limit: int = 6
    skill_inline_body_limit: int = 2400
    agentic_policy_enabled: bool = True
    policy_allowlist_mode: str = "restricted"
    policy_retrieval_top_k: int = 5
    policy_trace_enabled: bool = True
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    wiki_qa_model: str = ""
    rag_top_k: int = 6
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 180
    rag_reindex_batch_size: int = 64
    runtime_profile: str = "default"
    council_policy_mode: str = "balanced"
    deprecation_warnings: tuple[str, ...] = ()

    def require_llm_ready(self) -> None:
        self.agent_runtime.validate_llm_required()

    @classmethod
    def from_env(cls) -> "RuntimeContext":
        ensure_project_env_loaded()
        runtime_profile, profile_warnings = _apply_runtime_profile_from_env()
        agent_runtime = AgentRuntimeConfig.from_env()
        db_uri = _env_str("MOBILITY_DB_URI", "")
        if not db_uri:
            raise RuntimeError("MOBILITY_DB_URI is required. Configure a Postgres DSN with pgvector enabled before starting the runtime.")
        fail_stages = tuple(
            stage.strip() for stage in _env_str("DRY_RUN_FAIL_STAGES", "").split(",") if stage.strip()
        )
        hitl_policy, hitl_warnings = normalize_hitl_policy_with_warnings(_env_str("HITL_POLICY", "interactive"))
        deprecation_warnings = list(hitl_warnings) + list(profile_warnings)
        for env_name in ("HUMAN_REVIEW_POLL_SECONDS", "HUMAN_REVIEW_POLL_S"):
            if os.environ.get(env_name) is not None:
                deprecation_warnings.append(f"deprecated_env_ignored:{env_name}")
        for env_name in ("REFLECTION_MODEL", "MAX_PLAN_CYCLES", "MAX_DEEPEN_PER_MATERIAL", "MAX_FAILED_ATTEMPTS", "MIN_CONFIDENCE_FOR_STOP"):
            if os.environ.get(env_name) is not None:
                deprecation_warnings.append(f"deprecated_env_ignored:{env_name}")
        runtime = cls(
            agent_runtime=agent_runtime,
            vasp_cmd=_env_str("VASP_CMD", "mpirun -np 4 vasp_std > sout 2>&1"),
            temperature=float(_env_str("MOBILITY_TEMPERATURE_K", "300.0")),
            vacuum_direction=_env_int("VACUUM_DIRECTION", 2),
            c2d_prefac=float(_env_str("C2D_PREFAC", "1.0")),
            consider_spin=_env_bool("CONSIDER_SPIN", False),
            hitl_policy=hitl_policy,
            dry_run=_env_bool("DRY_RUN", False),
            dry_run_fail_stages=fail_stages,
            db_uri=normalize_database_uri(db_uri, default_memory_name="runtime"),
            store_path=normalize_database_uri(db_uri, default_memory_name="runtime"),
            notification_backend=_env_str("NOTIFICATION_BACKEND", "stdout"),
            checkpoint_subdir=_env_str("CHECKPOINT_SUBDIR", ".runtime"),
            compatibility_export_enabled=_env_bool("COMPATIBILITY_CHECKPOINT_EXPORT", True),
            compatibility_export_pickle=_env_bool("COMPATIBILITY_CHECKPOINT_PICKLE", True),
            full_autonomy=_env_bool("FULL_AUTONOMY", True),
            allow_external_wait=_env_bool("ALLOW_EXTERNAL_WAIT", False),
            rag_required=_env_bool("RAG_REQUIRED", True),
            python_executable=os.environ.get("PYTHON_EXECUTABLE") or sys.executable,
            skills_root=os.path.abspath(
                _env_str(
                    "MOBILITY_SKILLS_ROOT",
                    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "skills"),
                )
            ),
            skill_auto_resolve_limit=max(1, _env_int("SKILL_AUTO_RESOLVE_LIMIT", 6)),
            skill_inline_body_limit=max(400, _env_int("SKILL_INLINE_BODY_LIMIT", 2400)),
            agentic_policy_enabled=_env_bool("AGENTIC_POLICY_ENABLED", True),
            policy_allowlist_mode=_env_str("POLICY_ALLOWLIST_MODE", "restricted"),
            policy_retrieval_top_k=max(1, _env_int("POLICY_RETRIEVAL_TOP_K", 5)),
            policy_trace_enabled=_env_bool("POLICY_TRACE_ENABLED", True),
            embedding_model=_env_str("EMBEDDING_MODEL", ""),
            embedding_base_url=_env_str("EMBEDDING_BASE_URL", _env_str("LLM_BASE_URL", "")),
            embedding_api_key=_env_str("EMBEDDING_API_KEY", _env_str("LLM_API_KEY", "")),
            wiki_qa_model=_env_str("WIKI_QA_MODEL", _env_str("LLM_MODEL", "")),
            rag_top_k=max(1, _env_int("RAG_TOP_K", 6)),
            rag_chunk_size=max(200, _env_int("RAG_CHUNK_SIZE", 1200)),
            rag_chunk_overlap=max(0, _env_int("RAG_CHUNK_OVERLAP", 180)),
            rag_reindex_batch_size=max(1, _env_int("RAG_REINDEX_BATCH_SIZE", 64)),
            runtime_profile=runtime_profile,
            council_policy_mode=_env_str("AGENTIC_COUNCIL_MODE", "balanced").strip().lower() or "balanced",
            deprecation_warnings=_dedupe_warnings(deprecation_warnings),
        )
        runtime.require_llm_ready()
        if runtime.full_autonomy and runtime.allow_external_wait:
            raise RuntimeError("ALLOW_EXTERNAL_WAIT=true is incompatible with FULL_AUTONOMY=true.")
        if runtime.full_autonomy and not runtime.agentic_policy_enabled:
            raise RuntimeError("AGENTIC_POLICY_ENABLED=false is incompatible with FULL_AUTONOMY=true.")
        if runtime.rag_required and not is_postgres_uri(runtime.resolved_db_uri):
            raise RuntimeError("RAG_REQUIRED=true needs a Postgres MOBILITY_DB_URI with pgvector enabled.")
        if not runtime.embedding_model:
            raise RuntimeError("EMBEDDING_MODEL is required. Configure an embedding model before starting the runtime.")
        return runtime

    def checkpoint_path_for(self, workdir: str) -> str:
        return runtime_checkpoint_metadata_path(workdir, checkpoint_subdir=self.checkpoint_subdir)

    def state_snapshot_path_for(self, workdir: str) -> str:
        return runtime_state_snapshot_path(workdir, checkpoint_subdir=self.checkpoint_subdir)

    @property
    def resolved_db_uri(self) -> str:
        return normalize_database_uri(self.db_uri or self.store_path, default_memory_name="runtime")

    @property
    def db_uri_preview(self) -> str:
        return redact_database_uri(self.resolved_db_uri)
