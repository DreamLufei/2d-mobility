from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import dotenv_values, set_key, unset_key

from ..config_runtime import normalize_llm_provider
from ..runtime.database import redact_database_uri
from .models import RuntimeSettingsUpdateRequest, RuntimeSettingsView


_ENV_FIELD_MAP = {
    "mobility_db_uri": "MOBILITY_DB_URI",
    "llm_provider": "LLM_PROVIDER",
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_base_url": "EMBEDDING_BASE_URL",
    "wiki_qa_model": "WIKI_QA_MODEL",
    "agentic_policy_enabled": "AGENTIC_POLICY_ENABLED",
    "policy_allowlist_mode": "POLICY_ALLOWLIST_MODE",
    "policy_retrieval_top_k": "POLICY_RETRIEVAL_TOP_K",
    "policy_trace_enabled": "POLICY_TRACE_ENABLED",
    "rag_top_k": "RAG_TOP_K",
    "rag_chunk_size": "RAG_CHUNK_SIZE",
    "rag_chunk_overlap": "RAG_CHUNK_OVERLAP",
    "rag_reindex_batch_size": "RAG_REINDEX_BATCH_SIZE",
    "hitl_policy": "HITL_POLICY",
    "human_review_timeout_seconds": "HUMAN_REVIEW_TIMEOUT_SECONDS",
    "human_review_default_action": "HUMAN_REVIEW_DEFAULT_ACTION",
    "enable_email_notifications": "ENABLE_EMAIL_NOTIFICATIONS",
    "email_notify_to": "EMAIL_NOTIFY_TO",
    "email_dry_run": "EMAIL_DRY_RUN",
    "smtp_host": "SMTP_HOST",
    "smtp_port": "SMTP_PORT",
    "smtp_use_tls": "SMTP_USE_TLS",
    "smtp_username": "SMTP_USERNAME",
    "smtp_from": "SMTP_FROM",
}

_SECRET_FIELD_MAP = {
    "llm_api_key": "LLM_API_KEY",
    "embedding_api_key": "EMBEDDING_API_KEY",
    "smtp_password": "SMTP_PASSWORD",
}


def _as_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _stringify_env(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _secret_preview(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 10:
        return "*" * len(text)
    return f"{text[:6]}...{text[-4:]}"


def _detect_service_preset(base_url: str, model: str) -> str:
    url = str(base_url or "").strip().lower()
    model_text = str(model or "").strip().lower()
    if "openrouter.ai" in url:
        return "openrouter"
    if ("dashscope" in url and "aliyuncs.com" in url) or model_text.startswith("qwen"):
        return "qwen"
    return "custom"


@dataclass
class RuntimeSettingsStore:
    repo_root: str

    @property
    def env_path(self) -> str:
        return os.path.join(self.repo_root, ".env")

    @property
    def env_local_path(self) -> str:
        return os.path.join(self.repo_root, ".env.local")

    def _read_env_file(self, path: str) -> dict[str, str]:
        if not path or not os.path.exists(path):
            return {}
        try:
            payload = dotenv_values(path)
        except Exception:
            return {}
        return {str(key): str(value) for key, value in payload.items() if key and value is not None}

    def effective_env_values(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        merged.update(self._read_env_file(self.env_path))
        merged.update(self._read_env_file(self.env_local_path))
        return merged

    def read_settings(self) -> RuntimeSettingsView:
        values = self.effective_env_values()
        mobility_db_uri = str(values.get("MOBILITY_DB_URI") or "")
        llm_base_url = str(values.get("LLM_BASE_URL") or "")
        llm_model = str(values.get("LLM_MODEL") or "")
        llm_api_key = str(values.get("LLM_API_KEY") or "")
        embedding_base_url = str(values.get("EMBEDDING_BASE_URL") or llm_base_url)
        embedding_api_key = str(values.get("EMBEDDING_API_KEY") or values.get("LLM_API_KEY") or "")
        smtp_password = str(values.get("SMTP_PASSWORD") or "")
        return RuntimeSettingsView(
            service_preset=_detect_service_preset(llm_base_url, llm_model),
            mobility_db_uri=redact_database_uri(mobility_db_uri),
            llm_provider=normalize_llm_provider(values.get("LLM_PROVIDER"), default="openai"),
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key_present=bool(llm_api_key),
            llm_api_key_preview=_secret_preview(llm_api_key),
            embedding_model=str(values.get("EMBEDDING_MODEL") or ""),
            embedding_base_url=embedding_base_url,
            embedding_api_key_present=bool(embedding_api_key),
            embedding_api_key_preview=_secret_preview(embedding_api_key),
            wiki_qa_model=str(values.get("WIKI_QA_MODEL") or llm_model),
            agentic_policy_enabled=_as_bool(values.get("AGENTIC_POLICY_ENABLED"), True),
            policy_allowlist_mode=str(values.get("POLICY_ALLOWLIST_MODE") or "restricted"),
            policy_retrieval_top_k=_as_int(values.get("POLICY_RETRIEVAL_TOP_K"), 5),
            policy_trace_enabled=_as_bool(values.get("POLICY_TRACE_ENABLED"), True),
            rag_top_k=_as_int(values.get("RAG_TOP_K"), 6),
            rag_chunk_size=_as_int(values.get("RAG_CHUNK_SIZE"), 1200),
            rag_chunk_overlap=_as_int(values.get("RAG_CHUNK_OVERLAP"), 180),
            rag_reindex_batch_size=_as_int(values.get("RAG_REINDEX_BATCH_SIZE"), 64),
            hitl_policy=str(values.get("HITL_POLICY") or "interactive"),
            human_review_timeout_seconds=_as_int(values.get("HUMAN_REVIEW_TIMEOUT_SECONDS"), 300),
            human_review_default_action=str(values.get("HUMAN_REVIEW_DEFAULT_ACTION") or "skip_material"),
            enable_email_notifications=_as_bool(values.get("ENABLE_EMAIL_NOTIFICATIONS"), False),
            email_notify_to=str(values.get("EMAIL_NOTIFY_TO") or ""),
            email_dry_run=_as_bool(values.get("EMAIL_DRY_RUN"), True),
            smtp_host=str(values.get("SMTP_HOST") or ""),
            smtp_port=_as_int(values.get("SMTP_PORT"), 587),
            smtp_use_tls=_as_bool(values.get("SMTP_USE_TLS"), True),
            smtp_username=str(values.get("SMTP_USERNAME") or ""),
            smtp_from=str(values.get("SMTP_FROM") or ""),
            smtp_password_present=bool(smtp_password),
            smtp_password_preview=_secret_preview(smtp_password),
        )

    def update_settings(self, request: RuntimeSettingsUpdateRequest) -> RuntimeSettingsView:
        os.makedirs(self.repo_root, exist_ok=True)
        if not os.path.exists(self.env_local_path):
            with open(self.env_local_path, "a", encoding="utf-8"):
                pass

        fields_set = set(request.model_fields_set)

        for field_name, env_name in _ENV_FIELD_MAP.items():
            if field_name not in fields_set:
                continue
            value = getattr(request, field_name)
            if value is None or str(value).strip() == "":
                try:
                    unset_key(self.env_local_path, env_name)
                except Exception:
                    pass
                continue
            if field_name == "llm_provider":
                value = normalize_llm_provider(str(value), default="openai")
            set_key(self.env_local_path, env_name, _stringify_env(value), quote_mode="auto")

        if request.clear_llm_api_key:
            try:
                unset_key(self.env_local_path, _SECRET_FIELD_MAP["llm_api_key"])
            except Exception:
                pass
        elif "llm_api_key" in fields_set and str(request.llm_api_key or "").strip():
            set_key(
                self.env_local_path,
                _SECRET_FIELD_MAP["llm_api_key"],
                str(request.llm_api_key or "").strip(),
                quote_mode="auto",
            )

        if request.clear_embedding_api_key:
            try:
                unset_key(self.env_local_path, _SECRET_FIELD_MAP["embedding_api_key"])
            except Exception:
                pass
        elif "embedding_api_key" in fields_set and str(request.embedding_api_key or "").strip():
            set_key(
                self.env_local_path,
                _SECRET_FIELD_MAP["embedding_api_key"],
                str(request.embedding_api_key or "").strip(),
                quote_mode="auto",
            )

        if request.clear_smtp_password:
            try:
                unset_key(self.env_local_path, _SECRET_FIELD_MAP["smtp_password"])
            except Exception:
                pass
        elif "smtp_password" in fields_set and str(request.smtp_password or "").strip():
            set_key(
                self.env_local_path,
                _SECRET_FIELD_MAP["smtp_password"],
                str(request.smtp_password or "").strip(),
                quote_mode="auto",
            )

        return self.read_settings()
