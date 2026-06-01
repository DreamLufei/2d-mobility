from __future__ import annotations

import unittest
from unittest.mock import patch

from mobility_agent.agents.base import _structured_rate_limit_backoff_seconds
from mobility_agent.agents.llm_client import (
    build_llm_client,
    is_serialized_official_provider_base_url,
    runtime_uses_serialized_official_provider,
)
from mobility_agent.config_runtime import AgentRuntimeConfig


class LLMClientTests(unittest.TestCase):
    def test_build_llm_client_passes_responses_api_and_reasoning_effort(self) -> None:
        runtime = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="http://127.0.0.1:9",
            llm_api_key="test-key",
            llm_model="gpt-5.4",
            llm_use_responses_api=True,
            llm_reasoning_effort="xhigh",
        )

        with patch("mobility_agent.agents.llm_client.ChatOpenAI") as mock_chat:
            build_llm_client(runtime, role="planner", require_real=True)

        _, kwargs = mock_chat.call_args
        self.assertTrue(kwargs["use_responses_api"])
        self.assertEqual(kwargs["reasoning_effort"], "xhigh")

    def test_build_llm_client_defaults_to_non_responses_openrouter_path(self) -> None:
        runtime = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="https://openrouter.ai/api/v1",
            llm_api_key="test-key",
            llm_model="minimax/minimax-m2.7",
        )

        with patch("mobility_agent.agents.llm_client.ChatOpenAI") as mock_chat:
            build_llm_client(runtime, role="planner", require_real=True)

        _, kwargs = mock_chat.call_args
        self.assertFalse(kwargs["use_responses_api"])
        self.assertIsNone(kwargs["reasoning_effort"])

    def test_build_llm_client_passes_openrouter_provider_preferences(self) -> None:
        runtime = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="https://openrouter.ai/api/v1",
            llm_api_key="test-key",
            llm_model="minimax/minimax-m2.7",
            llm_provider_order=("fireworks",),
            llm_provider_require_parameters=True,
        )

        with patch("mobility_agent.agents.llm_client.ChatOpenAI") as mock_chat:
            build_llm_client(runtime, role="planner", require_real=True)

        _, kwargs = mock_chat.call_args
        self.assertEqual(
            kwargs["extra_body"],
            {"provider": {"order": ["fireworks"], "require_parameters": True}},
        )

    def test_build_llm_client_disables_dashscope_thinking(self) -> None:
        runtime = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_api_key="test-key",
            llm_model="qwen3.6-plus",
        )

        with patch("mobility_agent.agents.llm_client.ChatOpenAI") as mock_chat:
            build_llm_client(runtime, role="planner", require_real=True)

        _, kwargs = mock_chat.call_args
        self.assertEqual(kwargs["extra_body"], {"enable_thinking": False})

    def test_serialized_provider_detection_matches_dashscope_hosts(self) -> None:
        self.assertTrue(is_serialized_official_provider_base_url("https://dashscope.aliyuncs.com/compatible-mode/v1"))
        self.assertTrue(is_serialized_official_provider_base_url("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"))
        self.assertTrue(is_serialized_official_provider_base_url("https://dashscope-us.aliyuncs.com/compatible-mode/v1"))
        self.assertFalse(is_serialized_official_provider_base_url("https://openrouter.ai/api/v1"))
        self.assertFalse(is_serialized_official_provider_base_url("https://api.example.com/v1"))

    def test_runtime_uses_serialized_provider_checks_resolved_role_base_url(self) -> None:
        dashscope = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            llm_api_key="test-key",
            llm_model="qwen3.6-plus",
        )
        openrouter = AgentRuntimeConfig(
            llm_provider="openai",
            llm_base_url="https://openrouter.ai/api/v1",
            llm_api_key="test-key",
            llm_model="minimax/minimax-m2.7",
        )
        self.assertTrue(runtime_uses_serialized_official_provider(dashscope, role="planner"))
        self.assertFalse(runtime_uses_serialized_official_provider(openrouter, role="planner"))

    def test_serialized_provider_structured_backoff_is_more_conservative(self) -> None:
        self.assertEqual(_structured_rate_limit_backoff_seconds(attempt=1, serialized_provider=True), 5.0)
        self.assertEqual(_structured_rate_limit_backoff_seconds(attempt=2, serialized_provider=True), 15.0)
        self.assertEqual(_structured_rate_limit_backoff_seconds(attempt=3, serialized_provider=True), 30.0)


if __name__ == "__main__":
    unittest.main()
