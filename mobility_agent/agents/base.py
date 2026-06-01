from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from pydantic import BaseModel, ValidationError

from .skill_specs import DEFAULT_SCIENCE_MAINLINE
from ..runtime.agent_tools import AgentToolGateway, list_agent_tool_metadata
from ..runtime.context import RuntimeContext
from ..runtime.telemetry import dump_json_trace, emit_progress, tool_evidence_enabled
from ..skills import canonical_skill_name, choose_skills, discover_skills, load_skill
from ..utils import dedupe_keep_order
from .llm_client import build_llm_client, llm_request_guard, runtime_uses_serialized_official_provider

_MAX_LLM_STRING_CHARS = 4000
_MAX_LLM_DICT_ITEMS = 20
_MAX_LLM_LIST_ITEMS = 6
_MAX_LLM_DEPTH = 5
_RECENT_HISTORY_ITEMS = 4
_SOFT_PROMPT_TOKEN_BUDGET = {
    "planner": 12000,
    "recovery": 10000,
    "critic": 9000,
    "physics_judge": 9000,
    "cost_guardian": 8000,
    "orchestrator": 12000,
    "reporter": 8000,
    "executor": 6000,
    "validation": 8000,
}
_STRUCTURED_CALL_MAX_ATTEMPTS = 3
_STRUCTURED_CALL_METHOD = "json_mode"
_STRUCTURED_RATE_LIMIT_BACKOFF_MAX_SECONDS = 8.0
_SERIALIZED_PROVIDER_STRUCTURED_BACKOFF_SECONDS = {
    1: 5.0,
    2: 15.0,
    3: 30.0,
}
_STRUCTURED_CONNECTION_BACKOFF_SECONDS = {
    1: 3.0,
    2: 10.0,
    3: 20.0,
}
_STRUCTURED_RETRY_GUIDANCE = (
    "Your previous response could not be parsed into the required schema. "
    "Return exactly one valid JSON object matching the schema fields and types. "
    "Do not include Markdown headings, code fences, explanatory text, or any extra prose."
)


def _structured_output_method() -> str:
    return _STRUCTURED_CALL_METHOD


def _structured_output_kwargs(*, include_raw: bool = True) -> dict[str, Any]:
    return {
        "include_raw": include_raw,
        "method": _structured_output_method(),
    }


def _is_rate_limit_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    error_type = type(exc).__name__.lower()
    error_text = str(exc or "").lower()
    if error_type == "ratelimiterror":
        return True
    markers = (
        "rate limit",
        "rate_limit",
        "status code: 429",
        "status code 429",
        "status=429",
        "http 429",
        " code: 429",
        "error code: 429",
        "code 1302",
        "\"code\":\"1302\"",
        "'code': '1302'",
        "达到速率限制",
        "控制请求频率",
    )
    return any(marker in error_text for marker in markers)


def _is_connection_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    error_type = type(exc).__name__.lower()
    error_text = str(exc or "").lower()
    if error_type in {"apiconnectionerror", "connecterror", "connectionerror"}:
        return True
    markers = (
        "connection error",
        "connectionerror",
        "api connection error",
        "failed to establish a new connection",
        "connection refused",
        "connection reset",
        "remoteprotocolerror",
        "readtimeout",
        "read timeout",
        "timed out",
        "connect timeout",
        "server disconnected",
        "temporary failure in name resolution",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in error_text for marker in markers)


def _structured_rate_limit_backoff_seconds(*, attempt: int, serialized_provider: bool = False) -> float:
    if serialized_provider:
        return float(_SERIALIZED_PROVIDER_STRUCTURED_BACKOFF_SECONDS.get(int(attempt), 30.0))
    exponent = max(0, int(attempt) - 1)
    return min(_STRUCTURED_RATE_LIMIT_BACKOFF_MAX_SECONDS, float(2**exponent))


def _structured_connection_backoff_seconds(*, attempt: int) -> float:
    return float(_STRUCTURED_CONNECTION_BACKOFF_SECONDS.get(int(attempt), 20.0))


def _message_to_trace_payload(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", None),
        "content": getattr(message, "content", None),
    }
    name = getattr(message, "name", None)
    if name:
        payload["name"] = name
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def _structured_response_trace_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, BaseModel):
        return {"parsed": response.model_dump(mode="json")}
    if isinstance(response, dict):
        raw = response.get("raw")
        return {
            "parsed": (
                response["parsed"].model_dump(mode="json")
                if isinstance(response.get("parsed"), BaseModel)
                else response.get("parsed")
            ),
            "raw": (_message_to_trace_payload(raw) if raw is not None else None),
            "parsing_error": str(response.get("parsing_error") or "") or None,
        }
    return {"value": str(response)}


def _response_message_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_first_json_payload(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    decoder = json.JSONDecoder()
    start_positions = [index for index, char in enumerate(candidate) if char in "[{"]
    for start in start_positions:
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _coerce_structured_payload(*, response: Any, schema: type[BaseModel], agent_name: str) -> dict[str, Any]:
    candidates: list[Any] = []
    if isinstance(response, dict):
        parsed = response.get("parsed")
        if parsed is not None:
            candidates.append(parsed)
        raw = response.get("raw")
        raw_text = _response_message_text(raw)
        extracted = _extract_first_json_payload(raw_text)
        if extracted is not None:
            candidates.append(extracted)
        if raw_text.strip():
            candidates.append(raw_text)
        content = response.get("content")
        if content is not None:
            candidates.append(content)
    else:
        candidates.append(response)
        if isinstance(response, str):
            extracted = _extract_first_json_payload(response)
            if extracted is not None:
                candidates.append(extracted)
    errors: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if isinstance(candidate, BaseModel):
                return schema.model_validate(candidate.model_dump(mode="json")).model_dump(mode="json")
            return schema.model_validate(candidate).model_dump(mode="json")
        except ValidationError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
            continue
    if errors:
        raise RuntimeError(f"{agent_name}_structured_output_invalid:{errors[-1]}")
    raise RuntimeError(f"{agent_name}_structured_output_invalid")


def _estimate_prompt_tokens(messages: list[Any]) -> int | None:
    try:
        return int(count_tokens_approximately(messages))
    except Exception:
        return None


def _split_text_for_trim(text: str) -> list[str]:
    tokens = str(text or "").split(" ")
    if len(tokens) <= 1:
        return list(str(text or ""))
    return tokens


def _bounded_prompt_messages(
    *,
    prompt: ChatPromptTemplate,
    invoke_payload: dict[str, Any],
    max_tokens: int,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], int | None, int | None, bool]:
    rendered = prompt.invoke(invoke_payload)
    rendered_message_objects = list(getattr(rendered, "messages", []) or [])
    rendered_messages = [_message_to_trace_payload(message) for message in rendered_message_objects]
    estimated_prompt_tokens = _estimate_prompt_tokens(rendered_message_objects)

    invoked_message_objects = list(rendered_message_objects)
    prompt_trimmed = False
    if estimated_prompt_tokens is not None and estimated_prompt_tokens > max_tokens:
        try:
            trimmed = trim_messages(
                rendered_message_objects,
                max_tokens=max_tokens,
                token_counter="approximate",
                strategy="last",
                include_system=True,
                allow_partial=True,
                text_splitter=_split_text_for_trim,
            )
            if trimmed and any(str(getattr(message, "type", "")) == "human" for message in trimmed):
                invoked_message_objects = list(trimmed)
                prompt_trimmed = True
        except Exception:
            pass

    estimated_invoked_tokens = _estimate_prompt_tokens(invoked_message_objects)
    invoked_messages = [_message_to_trace_payload(message) for message in invoked_message_objects]
    return (
        invoked_message_objects,
        rendered_messages,
        invoked_messages,
        estimated_prompt_tokens,
        estimated_invoked_tokens,
        prompt_trimmed,
    )


def _truncate_text(value: str, *, limit: int = _MAX_LLM_STRING_CHARS) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    remaining = max(0, len(text) - limit)
    return f"{text[:limit]}\n...[truncated {remaining} chars]"


def _compact_list(value: list[Any], *, depth: int) -> Any:
    if len(value) <= _MAX_LLM_LIST_ITEMS:
        return [_compact_for_llm(item, depth=depth + 1) for item in value]
    keep_head = 2
    keep_tail = max(1, _MAX_LLM_LIST_ITEMS - keep_head)
    head_items = [_compact_for_llm(item, depth=depth + 1) for item in value[:keep_head]]
    tail_items = [_compact_for_llm(item, depth=depth + 1) for item in value[-keep_tail:]]
    return {
        "_summary": {
            "kind": "list",
            "total_items": len(value),
            "kept_head": keep_head,
            "kept_tail": keep_tail,
        },
        "head": head_items,
        "tail": tail_items,
    }


def _compact_dict(value: dict[str, Any], *, depth: int) -> Any:
    items = list(value.items())
    if len(items) <= _MAX_LLM_DICT_ITEMS:
        return {str(key): _compact_for_llm(item, depth=depth + 1) for key, item in items}
    kept_items = items[:_MAX_LLM_DICT_ITEMS]
    omitted_keys = [str(key) for key, _ in items[_MAX_LLM_DICT_ITEMS : _MAX_LLM_DICT_ITEMS + 8]]
    compacted = {str(key): _compact_for_llm(item, depth=depth + 1) for key, item in kept_items}
    compacted["_summary"] = {
        "kind": "dict",
        "total_keys": len(items),
        "kept_keys": [str(key) for key, _ in kept_items],
        "omitted_key_examples": omitted_keys,
    }
    return compacted


def _recent_items(value: Any, *, count: int = _RECENT_HISTORY_ITEMS) -> Any:
    if not isinstance(value, list):
        return _compact_for_llm(value, depth=0)
    if len(value) <= count:
        return [_compact_for_llm(item, depth=1) for item in value]
    return {
        "_summary": {"kind": "recent_list", "total_items": len(value), "kept_recent": count},
        "recent": [_compact_for_llm(item, depth=1) for item in value[-count:]],
    }


def _compact_deliberation_section(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_index": value.get("round_index"),
        "rounds": _recent_items(value.get("rounds", [])),
        "proposals": _recent_items(value.get("proposals", [])),
        "critiques": _recent_items(value.get("critiques", [])),
        "preferences": _recent_items(value.get("preferences", [])),
        "arbitrations": _recent_items(value.get("arbitrations", [])),
        "selected_actions": _recent_items(value.get("selected_actions", [])),
        "reflections": _recent_items(value.get("reflections", [])),
        "disagreement_records": _recent_items(value.get("disagreement_records", [])),
        "rationale_history": _recent_items(value.get("rationale_history", [])),
    }


def _compact_execution_section(value: dict[str, Any]) -> dict[str, Any]:
    keep_direct = [
        "workdir",
        "thread_id",
        "latest_tool_name",
        "latest_tool_result",
        "latest_execution_observation",
        "current_action",
        "action_status",
        "job_ids",
        "external_jobs",
        "pending_events",
        "latest_event",
        "resume_markers",
        "environment_summary",
        "compatibility_checkpoint_path",
        "compatibility_checkpoint_history",
        "workdir_inputs_ready",
        "pending_parameter_updates",
    ]
    compacted = {key: _compact_for_llm(value.get(key), depth=1) for key in keep_direct if key in value}
    compacted["artifact_paths"] = _compact_for_llm(value.get("artifact_paths", {}), depth=1)
    compacted["artifact_registry"] = _compact_for_llm(value.get("artifact_registry", {}), depth=1)
    compacted["tool_trace"] = _recent_items(value.get("tool_trace", []))
    compacted["tool_invocations"] = _recent_items(value.get("tool_invocations", []))
    compacted["skill_trace"] = _recent_items(value.get("skill_trace", []))
    compacted["failure_history"] = _recent_items(value.get("failure_history", []))
    compacted["event_history"] = _recent_items(value.get("event_history", []))
    compacted["consumed_event_ids"] = _recent_items(value.get("consumed_event_ids", []))
    return compacted


def _compact_agent_section(value: dict[str, Any]) -> dict[str, Any]:
    compacted = {
        "decision_engine": value.get("decision_engine"),
        "llm_required": value.get("llm_required"),
        "llm_provider": value.get("llm_provider"),
        "loaded_skills": _compact_for_llm(value.get("loaded_skills", []), depth=1),
    }
    compacted["agent_decisions"] = _recent_items(value.get("agent_decisions", []))
    compacted["decision_trace"] = _recent_items(value.get("decision_trace", []))
    return compacted


def _compact_blackboard_section(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "latest_execution_observation": _compact_for_llm(value.get("latest_execution_observation", {}), depth=1),
        "observations": _recent_items(value.get("observations", [])),
        "validated_facts": _recent_items(value.get("validated_facts", [])),
        "parsed_artifacts": _compact_for_llm(value.get("parsed_artifacts", {}), depth=1),
        "intermediate_results": _compact_for_llm(value.get("intermediate_results", {}), depth=1),
        "risk_flags": _compact_for_llm(value.get("risk_flags", []), depth=1),
        "anomaly_flags": _compact_for_llm(value.get("anomaly_flags", []), depth=1),
    }


def _compact_state_for_llm(state: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in state.items():
        if key == "deliberation" and isinstance(value, dict):
            compacted[key] = _compact_deliberation_section(value)
        elif key == "execution" and isinstance(value, dict):
            compacted[key] = _compact_execution_section(value)
        elif key == "agent" and isinstance(value, dict):
            compacted[key] = _compact_agent_section(value)
        elif key == "blackboard" and isinstance(value, dict):
            compacted[key] = _compact_blackboard_section(value)
        else:
            compacted[key] = _compact_for_llm(value, depth=0)
    return compacted


def _compact_for_llm(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_text(value)
    if depth >= _MAX_LLM_DEPTH:
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except Exception:
            serialized = str(value)
        return {
            "_summary": {
                "kind": type(value).__name__,
                "serialized_chars": len(serialized),
            },
            "preview": _truncate_text(serialized, limit=800),
        }
    if isinstance(value, list):
        return _compact_list(value, depth=depth)
    if isinstance(value, dict):
        return _compact_dict(value, depth=depth)
    try:
        return _truncate_text(json.dumps(value, ensure_ascii=False))
    except Exception:
        return _truncate_text(str(value))


def _compact_payload_for_llm(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "state" and isinstance(value, dict):
            compacted[key] = _compact_state_for_llm(value)
        elif key == "allowed_actions" and isinstance(value, list):
            compacted[key] = [str(item) for item in value]
        else:
            compacted[key] = _compact_for_llm(value, depth=0)
    return compacted


def _truncate_json_string(value: str, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    remaining = max(0, len(text) - limit)
    return f"{text[:limit]}\n...[truncated {remaining} chars]"


def _trim_recent_json_list(value: str, *, keep_recent: int) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    if isinstance(parsed, list):
        trimmed = parsed[-keep_recent:] if keep_recent > 0 else []
        return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
    return text


def _minimize_invoke_payload(invoke_payload: dict[str, Any], *, level: int) -> dict[str, Any]:
    payload = dict(invoke_payload or {})
    if level <= 0:
        return payload
    if level >= 1:
        payload["tool_context"] = _truncate_json_string(payload.get("tool_context", ""), limit=320)
        payload["tool_evidence_json"] = _truncate_json_string(payload.get("tool_evidence_json", ""), limit=1800)
        payload["payload"] = _truncate_json_string(
            _trim_recent_json_list(payload.get("payload", ""), keep_recent=2),
            limit=10000,
        )
    if level >= 2:
        payload["skill_context"] = _truncate_json_string(payload.get("skill_context", ""), limit=500)
        payload["tool_context"] = _truncate_json_string(payload.get("tool_context", ""), limit=180)
        payload["tool_evidence_json"] = _truncate_json_string(payload.get("tool_evidence_json", ""), limit=800)
        payload["payload"] = _truncate_json_string(payload.get("payload", ""), limit=5000)
    return payload


def _extract_usage_metadata(response: Any) -> dict[str, Any]:
    raw = None
    if isinstance(response, dict):
        raw = response.get("raw")
    metadata = {}
    usage = getattr(raw, "usage_metadata", None)
    if isinstance(usage, dict):
        metadata.update(usage)
    response_metadata = getattr(raw, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, dict):
                metadata.update(candidate)
    normalized: dict[str, Any] = {}
    prompt_tokens = metadata.get("input_tokens", metadata.get("prompt_tokens"))
    completion_tokens = metadata.get("output_tokens", metadata.get("completion_tokens"))
    reasoning_tokens = metadata.get("reasoning_tokens")
    if prompt_tokens is not None:
        normalized["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        normalized["completion_tokens"] = int(completion_tokens)
    if reasoning_tokens is not None:
        normalized["reasoning_tokens"] = int(reasoning_tokens)
    return normalized


class SkillAwareAgent:
    agent_name = "agent"
    llm_role = "specialist"

    def __init__(self, runtime: RuntimeContext, skills_root: str):
        self.runtime = runtime
        self.skills_root = skills_root
        self.llm, self.llm_reason = build_llm_client(runtime.agent_runtime, role=self.llm_role, require_real=True)
        if self.llm is None:
            raise RuntimeError(f"{self.agent_name}_llm_required:{self.llm_reason or 'unavailable'}")
        self.tool_gateway = AgentToolGateway()
        self._skill_registry_cache: dict[str, dict[str, Any]] | None = None
        self.last_llm_call_metadata: dict[str, Any] = {}

    def _uses_serialized_official_provider(self) -> bool:
        return runtime_uses_serialized_official_provider(self.runtime.agent_runtime, role=self.llm_role)

    def _emit_serialized_official_provider_mode(
        self,
        *,
        stage: str,
        schema: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self._uses_serialized_official_provider():
            return
        payload = {
            "agent": self.agent_name,
            "role": self.llm_role,
            "stage": stage,
        }
        if schema:
            payload["schema"] = schema
        if details:
            payload.update(details)
        emit_progress(
            "serialized official provider mode active",
            channel="agent",
            details=payload,
        )

    def _invoke_llm_with_guard(
        self,
        fn,
        *,
        stage: str,
        schema: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self._emit_serialized_official_provider_mode(stage=stage, schema=schema, details=details)
        with llm_request_guard(self.runtime.agent_runtime, role=self.llm_role):
            return fn()

    def _skill_registry(self) -> dict[str, dict[str, Any]]:
        if self._skill_registry_cache is None:
            self._skill_registry_cache = discover_skills(self.skills_root)
        return self._skill_registry_cache

    def _role_default_skills(self) -> list[str]:
        mapping = {
            "admission": ["admission"],
            "planner": ["planning"],
            "recovery": ["recovery"],
            "critic": ["critique"],
            "physics_judge": ["physics_validation"],
            "cost_guardian": ["cost_guardian"],
            "orchestrator": ["orchestration"],
            "reporter": ["reporting"],
            "report": ["reporting"],
            "executor": ["execution_feasibility"],
            "validation": ["validation"],
        }
        return dedupe_keep_order([canonical_skill_name(skill) for skill in list(mapping.get(self.llm_role, [])) if skill])

    def _default_explicit_skills(self, *, task_type: str, stage: str) -> list[str]:
        del stage
        skills = list(self._role_default_skills())
        if task_type == "single_material":
            skills.append("single_material_mobility")
        elif task_type == "batch_database":
            skills.append("batch_mobility_screening")
        return dedupe_keep_order([canonical_skill_name(skill) for skill in skills if skill])

    def _merge_explicit_skills(
        self,
        *,
        task_type: str,
        stage: str,
        explicit_skills: list[str] | None = None,
    ) -> list[str]:
        merged = self._default_explicit_skills(task_type=task_type, stage=stage) + list(explicit_skills or [])
        return dedupe_keep_order([canonical_skill_name(skill) for skill in merged if skill])

    def _skill_bundle(
        self,
        *,
        task_type: str,
        stage: str,
        has_error: bool = False,
        explicit_skills: list[str] | None = None,
    ) -> dict[str, Any]:
        registry = self._skill_registry()
        resolved_explicit = self._merge_explicit_skills(
            task_type=task_type,
            stage=stage,
            explicit_skills=explicit_skills,
        )
        skills = list(
            choose_skills(
                task_type=task_type,
                stage=stage,
                role=self.llm_role,
                has_error=has_error,
                explicit_skills=resolved_explicit,
                registry=registry,
                limit=self.runtime.skill_auto_resolve_limit,
            )
        )
        loaded = []
        for skill in skills:
            try:
                meta = dict(registry.get(skill, {}) or {})
                manifest = dict(meta.get("manifest", {}) or {})
                load_strategy = str(manifest.get("load_strategy") or "summary_only")
                loaded_skill = load_skill(
                    self.skills_root,
                    skill,
                    include_body=load_strategy == "summary_and_body",
                    include_resources=False,
                )
                if load_strategy == "summary_and_body" and len(str(loaded_skill.get("text") or "")) > self.runtime.skill_inline_body_limit:
                    loaded_skill["text"] = str(loaded_skill.get("text") or "")[: self.runtime.skill_inline_body_limit].rstrip() + "\n...[truncated]"
                loaded.append(loaded_skill)
            except FileNotFoundError:
                continue
        return {"selected": skills, "loaded": loaded, "registry": registry}

    def _tool_bundle(
        self,
        *,
        tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        registry = list_agent_tool_metadata()
        if tool_names:
            selected = [item for item in registry if item["name"] in tool_names]
        else:
            selected = registry
        return {"selected": selected, "registry": registry}

    def _langchain_tools(self, *, tool_names: list[str] | None = None) -> list[Any]:
        return self.tool_gateway.as_langchain_tools(names=tool_names)

    def _skill_prompt(self, bundle: dict[str, Any], *, summary_only: bool = True) -> str:
        sections: list[str] = []
        for skill in list(bundle.get("loaded", []) or []):
            name = str(skill.get("name") or "unknown_skill")
            manifest = dict(skill.get("manifest", {}) or {})
            summary = str(skill.get("summary") or skill.get("description") or manifest.get("description") or "").strip()
            body = str(skill.get("text") or "").strip()
            lines = [f"[{name}]"]
            if summary:
                lines.append(f"summary={summary}")
            purpose = str(manifest.get("purpose") or "").strip()
            if purpose:
                lines.append(f"purpose={purpose}")
            when_to_use = list(manifest.get("when_to_use", []) or [])
            if when_to_use:
                lines.append(f"when_to_use={'; '.join(when_to_use[:3])}")
            roles = list(manifest.get("roles", []) or [])
            stages = list(manifest.get("stages", []) or [])
            if roles:
                lines.append(f"roles={', '.join(roles)}")
            if stages:
                lines.append(f"stages={', '.join(stages[:6])}")
            resources = [str(item.get("path") or "") for item in list(skill.get("resources", []) or []) if item.get("path")]
            if resources:
                lines.append(f"resources={', '.join(resources[:6])}")
            if body and not summary_only:
                lines.append("body:")
                lines.append(body)
            elif body:
                lines.append("body=available_on_demand")
            sections.append("\n".join(lines))
        return "\n\n".join(sections) if sections else "No skill packages were loaded."

    def _tool_prompt(self, bundle: dict[str, Any]) -> str:
        selected = list(bundle.get("selected", []) or [])
        if not selected:
            return "No agent-callable tools were exposed."
        lines = []
        for item in selected:
            name = str(item.get("name") or "unknown_tool")
            description = str(item.get("description") or "").strip()
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)

    def _role_skill_prompt(self) -> str:
        registry = self._skill_registry()
        role_entries: list[dict[str, Any]] = []
        for name, entry in sorted(registry.items()):
            manifest = dict(entry.get("manifest", {}) or {})
            roles = [str(item or "").strip().lower() for item in list(manifest.get("roles", []) or [])]
            if self.llm_role not in roles:
                continue
            role_entries.append(
                {
                    "name": name,
                    "purpose": str(manifest.get("purpose") or entry.get("description") or "").strip(),
                    "when_to_use": list(manifest.get("when_to_use", []) or []),
                    "required_inputs": list(manifest.get("required_inputs", []) or []),
                    "expected_outputs": list(manifest.get("expected_output_schema", []) or []),
                    "decision_rules": list(manifest.get("decision_rules", []) or []),
                    "allowed_tools": list(manifest.get("allowed_tools", []) or []),
                    "tags": list(manifest.get("tags", []) or []),
                }
            )
        if role_entries:
            priority = {name: index for index, name in enumerate(self._role_default_skills())}
            role_entries = sorted(
                role_entries,
                key=lambda item: (
                    priority.get(str(item.get("name") or ""), len(priority) + 20),
                    str(item.get("name") or ""),
                ),
            )
            lines = [DEFAULT_SCIENCE_MAINLINE, "", "ROLE_SKILLS:"]
            for item in role_entries:
                purpose = _truncate_text(item["purpose"], limit=120)
                when_to_use = ", ".join(str(entry) for entry in list(item["when_to_use"] or [])[:2])
                lines.append(f"- {item['name']}: purpose={purpose}; when_to_use={when_to_use or 'as-needed'}")
            return "\n".join(lines)
        return DEFAULT_SCIENCE_MAINLINE

    def _tool_names_for_role(self) -> list[str]:
        mapping = {
            "planner": ["query_execution_status", "query_capability_metadata", "inspect_workspace", "inspect_artifacts", "synthesize_observation", "retrieve_policy_evidence"],
            "recovery": ["query_execution_status", "query_capability_metadata", "inspect_artifacts", "inspect_hitl_state", "synthesize_observation", "retrieve_failure_evidence", "retrieve_policy_evidence"],
            "critic": ["check_action_legality", "query_capability_metadata", "query_execution_status", "inspect_artifacts"],
            "physics_judge": ["synthesize_observation", "inspect_artifacts", "query_capability_metadata"],
            "cost_guardian": ["query_execution_status", "query_capability_metadata", "check_action_legality"],
            "orchestrator": ["check_action_legality", "query_execution_status", "synthesize_observation", "query_capability_metadata"],
            "reporter": ["query_execution_status", "synthesize_observation", "inspect_workspace"],
            "executor": ["check_action_legality", "query_capability_metadata", "inspect_artifacts"],
        }
        shared_skill_tools = ["resolve_skills", "load_skill", "list_skill_resources", "read_skill_resource"]
        return dedupe_keep_order(list(mapping.get(self.llm_role, [])) + shared_skill_tools)

    def _collect_tool_evidence(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        tool_names: list[str] | None = None,
        max_calls: int = 3,
    ) -> list[dict[str, Any]]:
        if not tool_evidence_enabled():
            emit_progress(
                "tool-evidence preflight skipped",
                channel="agent",
                details={"agent": self.agent_name, "role": self.llm_role},
            )
            return []
        compact_user_payload = _compact_payload_for_llm(user_payload)
        selected_names = tool_names or self._tool_names_for_role()
        tools = self._langchain_tools(tool_names=selected_names)
        if not tools:
            return []
        messages = [
            SystemMessage(
                content=(
                    f"{system_prompt}\n"
                    "Before making a decision, call tools only if they provide materially useful evidence. "
                    "Do not execute physics jobs. Use tools only for inspection, query, diagnosis, legality, memory, or reporting context."
                )
            ),
            HumanMessage(content=json.dumps(compact_user_payload, ensure_ascii=False, indent=2)),
        ]
        request_trace_path = dump_json_trace(
            "tool_evidence_prompt",
            {
                "agent_name": self.agent_name,
                "role": self.llm_role,
                "tool_names": selected_names,
                "messages": [_message_to_trace_payload(message) for message in messages],
                "user_payload": compact_user_payload,
            },
            role=self.llm_role,
        )
        emit_progress(
            "tool-evidence preflight started",
            channel="agent",
            details={"agent": self.agent_name, "role": self.llm_role, "trace": request_trace_path},
        )
        started = time.time()
        try:
            bound = self.llm.bind_tools(tools, tool_choice="auto")
            ai_message = self._invoke_llm_with_guard(
                lambda: bound.invoke(messages),
                stage="tool_evidence_preflight",
                details={"tool_calls_max": int(max_calls)},
            )
        except Exception as exc:
            error_trace_path = dump_json_trace(
                "tool_evidence_error",
                {
                    "agent_name": self.agent_name,
                    "role": self.llm_role,
                    "error_type": type(exc).__name__,
                    "error_text": str(exc),
                },
                role=self.llm_role,
            )
            emit_progress(
                "tool-evidence preflight failed",
                channel="agent",
                details={
                    "agent": self.agent_name,
                    "role": self.llm_role,
                    "duration_s": f"{time.time() - started:.2f}",
                    "error": type(exc).__name__,
                    "trace": error_trace_path,
                },
            )
            return []
        tool_records: list[dict[str, Any]] = []
        for call in list(getattr(ai_message, "tool_calls", []) or [])[: max(0, int(max_calls))]:
            name = str(call.get("name") or "")
            args = dict(call.get("args", {}) or {})
            try:
                result = self.tool_gateway.call(name, args)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}:{exc}"}
            tool_records.append({"tool_name": name, "arguments": args, "result": result})
        response_trace_path = dump_json_trace(
            "tool_evidence_result",
            {
                "agent_name": self.agent_name,
                "role": self.llm_role,
                "tool_calls": list(getattr(ai_message, "tool_calls", []) or []),
                "tool_records": tool_records,
            },
            role=self.llm_role,
        )
        emit_progress(
            "tool-evidence preflight finished",
            channel="agent",
            details={
                "agent": self.agent_name,
                "role": self.llm_role,
                "duration_s": f"{time.time() - started:.2f}",
                "tool_calls": len(tool_records),
                "trace": response_trace_path,
            },
        )
        return tool_records

    def _call_llm_structured_with_tools(
        self,
        *,
        schema: type[BaseModel],
        task_type: str,
        stage: str,
        payload: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        explicit_skills: list[str] | None = None,
        allowed_actions: list[str] | None = None,
        action_field: str = "action_type",
        tool_names: list[str] | None = None,
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError(f"{self.agent_name}_llm_required:{self.llm_reason or 'unavailable'}")
        self.last_llm_call_metadata = {}
        tool_records = self._collect_tool_evidence(
            system_prompt=system_prompt,
            user_payload=payload,
            tool_names=tool_names,
        )
        bundle = self._skill_bundle(
            task_type=task_type,
            stage=stage,
            has_error=False,
            explicit_skills=explicit_skills,
        )
        tool_bundle = self._tool_bundle(tool_names=tool_names or self._tool_names_for_role())
        compact_payload = _compact_payload_for_llm(payload)
        original_payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        compact_payload_json = json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))
        if len(compact_payload_json) < len(original_payload_json):
            emit_progress(
                "llm payload compacted",
                channel="agent",
                details={
                    "agent": self.agent_name,
                    "role": self.llm_role,
                    "original_chars": len(original_payload_json),
                    "compacted_chars": len(compact_payload_json),
                },
            )
        base_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                    + "\n"
                    + self._role_skill_prompt()
                    + "\nUse the gathered tool evidence and visible state to produce one structured response. "
                    + "Be concise: avoid restating the full state, keep rationales compact, and prefer short evidence lists."
                ),
                ("human", user_prompt),
            ]
        )
        base_invoke_payload = {
            "task_type": task_type,
            "stage": stage,
            "skill_context": self._skill_prompt(bundle, summary_only=True),
            "tool_context": self._tool_prompt(tool_bundle),
            "tool_evidence_json": json.dumps(tool_records, ensure_ascii=False, separators=(",", ":")),
            "payload": compact_payload_json,
            "allowed_actions": json.dumps(list(allowed_actions or []), ensure_ascii=False, separators=(",", ":")),
        }
        soft_budget = int(_SOFT_PROMPT_TOKEN_BUDGET.get(self.llm_role, 10000))
        started = time.time()
        llm_structured = self.llm.with_structured_output(schema, **_structured_output_kwargs(include_raw=True))
        max_attempts = max(1, int(_STRUCTURED_CALL_MAX_ATTEMPTS))
        response: Any | None = None
        result: dict[str, Any] | None = None
        last_error: Exception | None = None
        prompt_trace_path = None
        attempt_metadata: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            compaction_level = 0 if attempt == 1 else (1 if attempt == 2 else 2)
            invoke_payload = _minimize_invoke_payload(base_invoke_payload, level=compaction_level)
            rendered_messages: list[dict[str, Any]] = []
            invoked_messages: list[dict[str, Any]] = []
            estimated_prompt_tokens = None
            estimated_invoked_tokens = None
            prompt_trimmed = False
            try:
                (
                    attempt_messages,
                    rendered_messages,
                    invoked_messages,
                    estimated_prompt_tokens,
                    estimated_invoked_tokens,
                    prompt_trimmed,
                ) = _bounded_prompt_messages(
                    prompt=base_prompt,
                    invoke_payload=invoke_payload,
                    max_tokens=soft_budget,
                )
            except Exception:
                attempt_messages = []
            prompt_trace_path = dump_json_trace(
                "structured_prompt",
                {
                    "agent_name": self.agent_name,
                    "role": self.llm_role,
                    "task_type": task_type,
                    "stage": stage,
                    "schema": schema.__name__,
                    "attempt": attempt,
                    "compaction_level": compaction_level,
                    "invoke_payload": invoke_payload,
                    "rendered_messages": rendered_messages,
                    "invoked_messages": invoked_messages,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_invoked_tokens": estimated_invoked_tokens,
                    "prompt_trimmed": prompt_trimmed,
                    "soft_budget": soft_budget,
                },
                role=self.llm_role,
            )
            if prompt_trimmed:
                emit_progress(
                    "prompt trimmed to token budget via langchain trim_messages",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "attempt": attempt,
                        "compaction_level": compaction_level,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "estimated_invoked_tokens": estimated_invoked_tokens,
                        "soft_budget": soft_budget,
                        "trace": prompt_trace_path,
                    },
                )
            emit_progress(
                "structured LLM call started",
                channel="agent",
                details={
                    "agent": self.agent_name,
                    "role": self.llm_role,
                    "stage": stage,
                    "schema": schema.__name__,
                    "attempt": attempt,
                    "compaction_level": compaction_level,
                    "estimated_prompt_tokens": estimated_invoked_tokens,
                    "trace": prompt_trace_path,
                },
            )
            if attempt > 1 and attempt_messages:
                attempt_messages.append(
                    HumanMessage(
                        content=_STRUCTURED_RETRY_GUIDANCE
                    )
                )
            try:
                if attempt_messages:
                    response = self._invoke_llm_with_guard(
                        lambda: llm_structured.invoke(attempt_messages),
                        stage=stage,
                        schema=schema.__name__,
                        details={"attempt": attempt, "path": "structured_direct_messages"},
                    )
                else:
                    response = self._invoke_llm_with_guard(
                        lambda: (base_prompt | llm_structured).invoke(invoke_payload),
                        stage=stage,
                        schema=schema.__name__,
                        details={"attempt": attempt, "path": "structured_prompt_pipeline"},
                    )
                response_trace_path = dump_json_trace(
                    "structured_response",
                    {
                        "agent_name": self.agent_name,
                        "role": self.llm_role,
                        "task_type": task_type,
                        "stage": stage,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "response": _structured_response_trace_payload(response),
                    },
                    role=self.llm_role,
                )
                emit_progress(
                    "structured LLM call finished",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "stage": stage,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "compaction_level": compaction_level,
                        "max_attempts": max_attempts,
                        "duration_s": f"{time.time() - started:.2f}",
                        "trace": response_trace_path,
                    },
                )
                attempt_metadata = {
                    "attempts": attempt,
                    "compaction_level": compaction_level,
                    "estimated_prompt_tokens": estimated_prompt_tokens,
                    "estimated_invoked_tokens": estimated_invoked_tokens,
                    "prompt_trimmed": prompt_trimmed,
                    "duration_s": round(time.time() - started, 4),
                    "trace": response_trace_path,
                    **_extract_usage_metadata(response),
                }
                result = _coerce_structured_payload(response=response, schema=schema, agent_name=self.agent_name)
                break
            except Exception as exc:
                last_error = exc
                recovered_from_raw = False
                recovery_error: Exception | None = None
                parser_error_text = str(exc)
                if isinstance(exc, ValidationError) or "Invalid JSON" in parser_error_text or "json_invalid" in parser_error_text:
                    try:
                        raw_messages = list(attempt_messages)
                        if not raw_messages:
                            rendered = base_prompt.invoke(invoke_payload)
                            raw_messages = list(getattr(rendered, "messages", []) or [])
                        raw_response = self._invoke_llm_with_guard(
                            lambda: self.llm.invoke(raw_messages),
                            stage=stage,
                            details={"attempt": attempt, "path": "raw_recovery_invoke"},
                        )
                        response = {
                            "parsed": None,
                            "raw": raw_response,
                            "parsing_error": parser_error_text,
                        }
                        result = _coerce_structured_payload(response=response, schema=schema, agent_name=self.agent_name)
                        recovered_trace_path = dump_json_trace(
                            "structured_response_recovered",
                            {
                                "agent_name": self.agent_name,
                                "role": self.llm_role,
                                "task_type": task_type,
                                "stage": stage,
                                "schema": schema.__name__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                                "parser_error": parser_error_text,
                                "response": _structured_response_trace_payload(response),
                            },
                            role=self.llm_role,
                        )
                        emit_progress(
                            "structured parser failed; recovered from raw response",
                            channel="agent",
                            details={
                                "agent": self.agent_name,
                                "role": self.llm_role,
                                "stage": stage,
                                "schema": schema.__name__,
                                "attempt": attempt,
                                "compaction_level": compaction_level,
                                "max_attempts": max_attempts,
                                "trace": recovered_trace_path,
                            },
                        )
                        attempt_metadata = {
                            "attempts": attempt,
                            "compaction_level": compaction_level,
                            "estimated_prompt_tokens": estimated_prompt_tokens,
                            "estimated_invoked_tokens": estimated_invoked_tokens,
                            "prompt_trimmed": prompt_trimmed,
                            "duration_s": round(time.time() - started, 4),
                            "trace": recovered_trace_path,
                            **_extract_usage_metadata(response),
                        }
                        recovered_from_raw = True
                    except Exception as raw_exc:
                        recovery_error = raw_exc
                        last_error = raw_exc
                if recovered_from_raw:
                    break
                error_trace_path = dump_json_trace(
                    "structured_response_error",
                    {
                        "agent_name": self.agent_name,
                        "role": self.llm_role,
                        "task_type": task_type,
                        "stage": stage,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "error_text": str(exc),
                        "recovery_error": (f"{type(recovery_error).__name__}:{recovery_error}" if recovery_error is not None else None),
                        "response": _structured_response_trace_payload(response) if response is not None else None,
                    },
                    role=self.llm_role,
                )
                emit_progress(
                    "structured LLM call attempt failed",
                    channel="agent",
                    details={
                        "agent": self.agent_name,
                        "role": self.llm_role,
                        "stage": stage,
                        "schema": schema.__name__,
                        "attempt": attempt,
                        "compaction_level": compaction_level,
                        "max_attempts": max_attempts,
                        "error": f"{type(exc).__name__}:{exc}",
                        "trace": error_trace_path,
                    },
                )
                rate_limit_exc = recovery_error if _is_rate_limit_error(recovery_error) else exc
                connection_exc = recovery_error if _is_connection_error(recovery_error) else exc
                if attempt < max_attempts and _is_rate_limit_error(rate_limit_exc):
                    serialized_provider = self._uses_serialized_official_provider()
                    backoff_seconds = _structured_rate_limit_backoff_seconds(
                        attempt=attempt,
                        serialized_provider=serialized_provider,
                    )
                    emit_progress(
                        "structured LLM call rate limited; backing off before retry",
                        channel="agent",
                        details={
                            "agent": self.agent_name,
                            "role": self.llm_role,
                            "stage": stage,
                            "schema": schema.__name__,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "backoff_s": backoff_seconds,
                            "error": f"{type(rate_limit_exc).__name__}:{rate_limit_exc}",
                        },
                    )
                    if serialized_provider:
                        emit_progress(
                            "rate-limit cooldown applied",
                            channel="agent",
                            details={
                                "agent": self.agent_name,
                                "role": self.llm_role,
                                "stage": stage,
                                "schema": schema.__name__,
                                "attempt": attempt,
                                "cooldown_s": backoff_seconds,
                                "reason": "serialized_official_provider_structured_retry",
                            },
                        )
                    time.sleep(backoff_seconds)
                elif attempt < max_attempts and _is_connection_error(connection_exc):
                    backoff_seconds = _structured_connection_backoff_seconds(attempt=attempt)
                    emit_progress(
                        "structured LLM call hit transient connection issue; backing off before retry",
                        channel="agent",
                        details={
                            "agent": self.agent_name,
                            "role": self.llm_role,
                            "stage": stage,
                            "schema": schema.__name__,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "backoff_s": backoff_seconds,
                            "error": f"{type(connection_exc).__name__}:{connection_exc}",
                        },
                    )
                    time.sleep(backoff_seconds)
                if attempt >= max_attempts:
                    self.last_llm_call_metadata = {
                        "attempts": attempt,
                        "compaction_level": compaction_level,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "estimated_invoked_tokens": estimated_invoked_tokens,
                        "prompt_trimmed": prompt_trimmed,
                        "duration_s": round(time.time() - started, 4),
                        "trace": error_trace_path,
                        "error": f"{type(last_error).__name__}:{last_error}",
                    }
                    raise
        if result is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"{self.agent_name}_structured_output_invalid")
        if allowed_actions:
            chosen = str(result.get(action_field) or "")
            if chosen and chosen not in allowed_actions:
                raise RuntimeError(f"{self.agent_name}_unsupported_action:{chosen}")
        result["tool_evidence"] = tool_records
        if not attempt_metadata:
            attempt_metadata = {
                "attempts": 0,
                "compaction_level": 0,
                "duration_s": round(time.time() - started, 4),
                "trace": prompt_trace_path,
            }
        self.last_llm_call_metadata = {
            "task_type": task_type,
            "stage": stage,
            "schema": schema.__name__,
            "tool_calls": len(tool_records),
            **attempt_metadata,
        }
        return result

    def _maybe_call_llm(
        self,
        *,
        kind: str,
        schema: type[BaseModel],
        task_type: str,
        stage: str,
        summary: dict[str, Any],
        rule_payload: dict[str, Any],
        allowed_actions: list[str] | None = None,
        has_error: bool = False,
        explicit_skills: list[str] | None = None,
    ) -> dict[str, Any] | None:
        if self.llm is None:
            raise RuntimeError(f"{self.agent_name}_llm_required:{self.llm_reason or 'unavailable'}")
        self.last_llm_call_metadata = {}
        bundle = self._skill_bundle(
            task_type=task_type,
            stage=stage,
            has_error=has_error,
            explicit_skills=explicit_skills,
        )
        tool_bundle = self._tool_bundle()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a bounded scientific workflow decision agent.\n"
                    "Use the provided skill context and visible state only.\n"
                    "Agent-callable tools are available only for inspection, query, parsing, validation, and management.\n"
                    "Never invent unsupported actions.\n"
                    "Return a single structured response that matches the requested schema.",
                ),
                (
                    "human",
                    "AGENT_KIND: {kind}\n"
                    "TASK_TYPE: {task_type}\n"
                    "CURRENT_STAGE: {stage}\n"
                    "ALLOWED_ACTIONS: {allowed_actions}\n\n"
                    "SKILL_CONTEXT:\n{skill_context}\n\n"
                    "AVAILABLE_TOOLS:\n{tool_context}\n\n"
                    "RULE_PAYLOAD_JSON:\n{rule_payload}\n\n"
                    "VISIBLE_STATE_JSON:\n{summary}\n",
                ),
            ]
        )
        soft_budget = int(_SOFT_PROMPT_TOKEN_BUDGET.get(self.llm_role, 10000))
        try:
            invoke_payload = {
                "kind": kind,
                "task_type": task_type,
                "stage": stage,
                "allowed_actions": json.dumps(list(allowed_actions or []), ensure_ascii=False, separators=(",", ":")),
                "skill_context": self._skill_prompt(bundle, summary_only=True),
                "tool_context": self._tool_prompt(tool_bundle),
                "rule_payload": json.dumps(rule_payload, ensure_ascii=False, separators=(",", ":")),
                "summary": json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            }
            started = time.time()
            invoked_message_objects, _, _, _, _, _ = _bounded_prompt_messages(
                prompt=prompt,
                invoke_payload=invoke_payload,
                max_tokens=soft_budget,
            )
            llm_structured = self.llm.with_structured_output(schema, **_structured_output_kwargs(include_raw=True))
            if invoked_message_objects:
                response = self._invoke_llm_with_guard(
                    lambda: llm_structured.invoke(invoked_message_objects),
                    stage=stage,
                    schema=schema.__name__,
                    details={"path": "bounded_rule_direct_messages"},
                )
            else:
                response = self._invoke_llm_with_guard(
                    lambda: (prompt | llm_structured).invoke(invoke_payload),
                    stage=stage,
                    schema=schema.__name__,
                    details={"path": "bounded_rule_prompt_pipeline"},
                )
        except Exception:
            self.last_llm_call_metadata = {
                "task_type": task_type,
                "stage": stage,
                "schema": schema.__name__,
                "error": f"{self.agent_name}_bounded_decision_failed",
            }
            raise
        payload = _coerce_structured_payload(response=response, schema=schema, agent_name=self.agent_name)
        if allowed_actions and str(payload.get("decision") or "") not in allowed_actions:
            raise RuntimeError(f"{self.agent_name}_unsupported_decision:{payload.get('decision') or 'unset'}")
        payload["warnings"] = dedupe_keep_order(list(rule_payload.get("warnings", []) or []) + list(payload.get("warnings", []) or []))
        self.last_llm_call_metadata = {
            "task_type": task_type,
            "stage": stage,
            "schema": schema.__name__,
            "duration_s": round(time.time() - started, 4),
            **_extract_usage_metadata(response),
        }
        return payload

    def _call_llm_strict(
        self,
        *,
        schema: type[BaseModel],
        task_type: str,
        stage: str,
        payload: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        explicit_skills: list[str] | None = None,
        allowed_actions: list[str] | None = None,
        action_field: str = "action_type",
    ) -> dict[str, Any]:
        if self.llm is None:
            raise RuntimeError(f"{self.agent_name}_llm_required:{self.llm_reason or 'unavailable'}")
        bundle = self._skill_bundle(
            task_type=task_type,
            stage=stage,
            has_error=False,
            explicit_skills=explicit_skills,
        )
        tool_bundle = self._tool_bundle()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        invoke_payload = {
            "task_type": task_type,
            "stage": stage,
            "skill_context": self._skill_prompt(bundle),
            "tool_context": self._tool_prompt(tool_bundle),
            "payload": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "allowed_actions": json.dumps(list(allowed_actions or []), ensure_ascii=False, separators=(",", ":")),
        }
        soft_budget = int(_SOFT_PROMPT_TOKEN_BUDGET.get(self.llm_role, 10000))
        invoked_message_objects, _, _, _, _, _ = _bounded_prompt_messages(
            prompt=prompt,
            invoke_payload=invoke_payload,
            max_tokens=soft_budget,
        )
        llm_structured = self.llm.with_structured_output(schema, **_structured_output_kwargs(include_raw=True))
        if invoked_message_objects:
            response = self._invoke_llm_with_guard(
                lambda: llm_structured.invoke(invoked_message_objects),
                stage=stage,
                schema=schema.__name__,
                details={"path": "strict_direct_messages"},
            )
        else:
            response = self._invoke_llm_with_guard(
                lambda: (prompt | llm_structured).invoke(invoke_payload),
                stage=stage,
                schema=schema.__name__,
                details={"path": "strict_prompt_pipeline"},
            )
        result = _coerce_structured_payload(response=response, schema=schema, agent_name=self.agent_name)
        if allowed_actions:
            chosen = str(result.get(action_field) or "")
            if chosen not in allowed_actions:
                raise RuntimeError(f"{self.agent_name}_unsupported_action:{chosen or 'unset'}")
        return result
