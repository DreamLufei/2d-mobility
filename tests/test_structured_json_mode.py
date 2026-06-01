from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from mobility_agent.agents.base import SkillAwareAgent, _coerce_structured_payload, _structured_output_method


class _DecisionSchema(BaseModel):
    decision: str


class _StructuredRunnable:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.invocations: list[object] = []

    def invoke(self, payload: object) -> object:
        self.invocations.append(payload)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _SpyLLM:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.with_structured_output_calls: list[dict[str, object]] = []
        self.raw_invocations: list[object] = []

    def with_structured_output(self, schema: type[BaseModel], **kwargs: object) -> _StructuredRunnable:
        self.with_structured_output_calls.append({"schema": schema, "kwargs": dict(kwargs)})
        return _StructuredRunnable(self._responses)

    def invoke(self, payload: object) -> AIMessage:
        self.raw_invocations.append(payload)
        return AIMessage(content='{"decision":"prepare"}')


class _DummyAgent(SkillAwareAgent):
    agent_name = "dummy"
    llm_role = "planner"

    def __init__(self, llm: _SpyLLM) -> None:
        dummy_agent_runtime = SimpleNamespace(
            resolve_role_config=lambda role=None: {"base_url": ""},
            llm_base_url="",
        )
        self.runtime = SimpleNamespace(
            skill_auto_resolve_limit=0,
            skill_inline_body_limit=0,
            agent_runtime=dummy_agent_runtime,
        )
        self.skills_root = ""
        self.llm = llm
        self.llm_reason = None
        self.last_llm_call_metadata = {}

    def _collect_tool_evidence(self, **_: object) -> list[dict[str, object]]:
        return []

    def _skill_bundle(self, **_: object) -> dict[str, object]:
        return {"selected": [], "loaded": [], "registry": {}}

    def _tool_bundle(self, **_: object) -> dict[str, object]:
        return {"selected": [], "registry": []}

    def _skill_prompt(self, bundle: dict[str, object], *, summary_only: bool = True) -> str:
        del bundle, summary_only
        return ""

    def _tool_prompt(self, bundle: dict[str, object]) -> str:
        del bundle
        return ""

    def _role_skill_prompt(self) -> str:
        return ""


class StructuredJsonModeTests(unittest.TestCase):
    def test_strict_structured_calls_bind_json_mode(self) -> None:
        llm = _SpyLLM(
            [
                {
                    "parsed": {"decision": "prepare"},
                    "raw": AIMessage(content='{"decision":"prepare"}'),
                    "parsing_error": None,
                }
            ]
        )
        agent = _DummyAgent(llm)

        payload = agent._call_llm_strict(
            schema=_DecisionSchema,
            task_type="single_material",
            stage="prepare",
            payload={"material_id": "test"},
            system_prompt="Return exactly one JSON object.",
            user_prompt="VISIBLE_STATE_JSON:\n{payload}",
        )

        self.assertEqual(payload["decision"], "prepare")
        self.assertEqual(len(llm.with_structured_output_calls), 1)
        kwargs = dict(llm.with_structured_output_calls[0]["kwargs"])
        self.assertEqual(kwargs.get("method"), "json_mode")
        self.assertTrue(bool(kwargs.get("include_raw")))
        self.assertEqual(_structured_output_method(), "json_mode")

    def test_coerce_structured_payload_recovers_json_mode_text_variants(self) -> None:
        samples = [
            '```json\n{"decision":"prepare"}\n```',
            'Please use this object only: {"decision":"prepare"} Thanks.',
            '\n\n{"decision":"prepare"}\n',
        ]
        for raw_text in samples:
            with self.subTest(raw_text=raw_text):
                payload = _coerce_structured_payload(
                    response={
                        "parsed": None,
                        "raw": AIMessage(content=raw_text),
                        "parsing_error": "synthetic_invalid_json",
                    },
                    schema=_DecisionSchema,
                    agent_name="dummy",
                )
                self.assertEqual(payload["decision"], "prepare")

    def test_structured_call_retries_after_rate_limit_with_backoff(self) -> None:
        llm = _SpyLLM(
            [
                RuntimeError("Error code: 429 - {'error': {'code': '1302', 'message': '您的账户已达到速率限制，请您控制请求频率'}}"),
                {
                    "parsed": {"decision": "prepare"},
                    "raw": AIMessage(content='{"decision":"prepare"}'),
                    "parsing_error": None,
                },
            ]
        )
        agent = _DummyAgent(llm)

        with patch("mobility_agent.agents.base.time.sleep") as mocked_sleep:
            payload = agent._call_llm_structured_with_tools(
                schema=_DecisionSchema,
                task_type="single_material",
                stage="prepare",
                payload={"material_id": "test"},
                system_prompt="Return exactly one JSON object.",
                user_prompt="VISIBLE_STATE_JSON:\n{payload}",
            )

        self.assertEqual(payload["decision"], "prepare")
        mocked_sleep.assert_called_once_with(1.0)
        self.assertEqual(len(llm.with_structured_output_calls), 1)
        kwargs = dict(llm.with_structured_output_calls[0]["kwargs"])
        self.assertEqual(kwargs.get("method"), "json_mode")
