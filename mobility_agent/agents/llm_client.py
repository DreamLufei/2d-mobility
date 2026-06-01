from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Any, Iterable, Tuple
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config_runtime import AgentRuntimeConfig


REQUIRED_LLM_ROLES = (
    "specialist",
    "planner",
    "critic",
    "recovery",
    "physics_judge",
    "cost_guardian",
    "orchestrator",
    "reporter",
    "executor",
)


_SERIALIZED_OFFICIAL_PROVIDER_HOSTS = {
    "dashscope.aliyuncs.com",
    "dashscope-intl.aliyuncs.com",
    "dashscope-us.aliyuncs.com",
}
_SERIALIZED_OFFICIAL_PROVIDER_LLM_LOCK = RLock()


def is_serialized_official_provider_base_url(base_url: str | None) -> bool:
    raw = str(base_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    hostname = str(parsed.hostname or "").strip().lower()
    return hostname in _SERIALIZED_OFFICIAL_PROVIDER_HOSTS


def runtime_uses_serialized_official_provider(runtime: AgentRuntimeConfig, *, role: str | None = None) -> bool:
    resolved = runtime.resolve_role_config(role)
    base_url = str(resolved.get("base_url") or runtime.llm_base_url or "").strip()
    return is_serialized_official_provider_base_url(base_url)


def _is_dashscope_base_url(base_url: str | None) -> bool:
    return is_serialized_official_provider_base_url(base_url)


def _is_openrouter_base_url(base_url: str | None) -> bool:
    raw = str(base_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return False
    hostname = str(parsed.hostname or "").strip().lower()
    return hostname == "openrouter.ai"


def _build_openrouter_extra_body(provider_preferences: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = dict(provider_preferences or {})
    provider: dict[str, Any] = {}
    order = [str(item).strip() for item in payload.get("order", []) or [] if str(item).strip()]
    only = [str(item).strip() for item in payload.get("only", []) or [] if str(item).strip()]
    ignore = [str(item).strip() for item in payload.get("ignore", []) or [] if str(item).strip()]
    sort = str(payload.get("sort") or "").strip()
    if order:
        provider["order"] = order
    if only:
        provider["only"] = only
    if ignore:
        provider["ignore"] = ignore
    if sort:
        provider["sort"] = sort
    if payload.get("require_parameters"):
        provider["require_parameters"] = True
    if payload.get("allow_fallbacks") is False:
        provider["allow_fallbacks"] = False
    if not provider:
        return None
    return {"provider": provider}


def _merge_extra_body(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged or None


@contextmanager
def llm_request_guard(runtime: AgentRuntimeConfig, *, role: str | None = None):
    if not runtime_uses_serialized_official_provider(runtime, role=role):
        yield
        return
    with _SERIALIZED_OFFICIAL_PROVIDER_LLM_LOCK:
        yield


def build_llm_client(
    runtime: AgentRuntimeConfig,
    *,
    role: str | None = None,
    require_real: bool = False,
) -> Tuple[BaseChatModel | None, str | None]:
    del require_real
    resolved = runtime.resolve_role_config(role)
    provider = str(resolved.get("provider") or runtime.llm_provider or "").strip().lower()
    model = str(resolved.get("model") or "").strip()
    base_url = str(resolved.get("base_url") or runtime.llm_base_url or "").strip()
    api_key = str(resolved.get("api_key") or runtime.llm_api_key or "").strip()
    temperature = float(resolved.get("temperature") or runtime.llm_temperature)
    timeout_seconds = int(resolved.get("timeout_seconds") or runtime.llm_timeout_seconds)
    max_tokens = int(resolved.get("max_tokens") or runtime.llm_max_tokens)
    use_responses_api = bool(resolved.get("use_responses_api") or False)
    reasoning_effort = str(resolved.get("reasoning_effort") or "").strip()
    provider_preferences = resolved.get("provider_preferences") if isinstance(resolved, dict) else None
    if provider == "mock":
        return None, "mock_provider_disallowed"
    if provider in {"openai", "openai_compatible"}:
        if not base_url:
            return None, "missing_llm_base_url"
        if not api_key:
            return None, "missing_llm_api_key"
        if not model:
            return None, f"missing_llm_model:{role or 'default'}"
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout_seconds,
            "max_retries": runtime.llm_max_retries,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_responses_api": use_responses_api,
            "reasoning_effort": reasoning_effort or None,
        }
        openrouter_extra_body = None
        dashscope_extra_body = None
        if _is_openrouter_base_url(base_url):
            openrouter_extra_body = _build_openrouter_extra_body(
                provider_preferences if isinstance(provider_preferences, dict) else None
            )
        if _is_dashscope_base_url(base_url):
            dashscope_extra_body = {"enable_thinking": False}
        extra_body = _merge_extra_body(openrouter_extra_body, dashscope_extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        return (ChatOpenAI(**kwargs), None)
    return None, f"unsupported_llm_provider:{provider or 'unset'}"


def validate_runtime_llm_config(
    runtime: AgentRuntimeConfig,
    *,
    roles: Iterable[str] = REQUIRED_LLM_ROLES,
) -> None:
    issues: list[str] = []
    normalized_roles = [str(role or "").strip() for role in roles if str(role or "").strip()]
    if not normalized_roles:
        normalized_roles = list(REQUIRED_LLM_ROLES)
    for role in normalized_roles:
        _, reason = build_llm_client(runtime, role=role, require_real=True)
        if reason:
            issues.append(f"{role}:{reason}")
    if issues:
        raise RuntimeError(
            "llm_configuration_invalid: "
            + "; ".join(issues)
            + ". Configure LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, and role/global model names before starting the runtime."
        )
