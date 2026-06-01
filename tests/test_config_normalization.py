from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.runtime.batch_config import load_config
from mobility_agent.runtime.context import RuntimeContext


class ConfigNormalizationTests(unittest.TestCase):
    _RUNTIME_ENV = {
        "MOBILITY_DB_URI": "memory://config-normalization",
        "EMBEDDING_MODEL": "test-embedding-model",
        "RAG_REQUIRED": "false",
    }

    def test_openai_compatible_alias_normalizes_to_openai(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "LLM_PROVIDER": "openai_compatible",
                "LLM_BASE_URL": "http://127.0.0.1:9/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            },
            clear=False,
        ):
            runtime = RuntimeContext.from_env()
        self.assertEqual(runtime.agent_runtime.llm_provider, "openai")

    def test_responses_api_and_reasoning_effort_are_loaded_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "http://127.0.0.1:9",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.4",
                "LLM_USE_RESPONSES_API": "true",
                "LLM_REASONING_EFFORT": "xhigh",
            },
            clear=False,
        ):
            runtime = RuntimeContext.from_env()
        self.assertTrue(runtime.agent_runtime.llm_use_responses_api)
        self.assertEqual(runtime.agent_runtime.llm_reasoning_effort, "xhigh")

    def test_hitl_alias_records_deprecation_warning(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "HITL_POLICY": "non_interactive_wait",
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "http://127.0.0.1:9/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            },
            clear=False,
        ):
            runtime = RuntimeContext.from_env()
        self.assertEqual(runtime.hitl_policy, "non_interactive_skip_on_timeout")
        self.assertTrue(any("deprecated_HITL_POLICY" in item for item in runtime.deprecation_warnings))

    def test_planner_mode_env_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "PLANNER_MODE": "llm",
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "http://127.0.0.1:9/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError) as exc:
                RuntimeContext.from_env()
        self.assertIn("PLANNER_MODE is no longer supported", str(exc.exception))

    def test_llm_enabled_env_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "LLM_ENABLED": "true",
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "http://127.0.0.1:9/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            },
            clear=False,
        ):
            with self.assertRaises(RuntimeError) as exc:
                RuntimeContext.from_env()
        self.assertIn("LLM_ENABLED is no longer supported", str(exc.exception))

    def test_batch_config_ignores_legacy_subprocess_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "MONGO_URI": "mongodb://example",
                "MONGO_DB": "db",
                "MONGO_COLLECTION": "collection",
                "RUNS_ROOT": tmpdir,
                "MOBALITY_SCRIPT": "/legacy/mobality.py",
                "VASP_TIMEOUT_S": "123",
            },
            clear=False,
        ):
            cfg = load_config()
        self.assertEqual(cfg.mongo_collection, "collection")
        self.assertIn("deprecated_env_ignored:MOBALITY_SCRIPT", cfg.deprecation_warnings)
        self.assertIn("deprecated_env_ignored:VASP_TIMEOUT_S", cfg.deprecation_warnings)

    def test_full_autonomy_profile_defaults_to_interactive_hitl(self) -> None:
        with patch.dict(
            os.environ,
            {
                **self._RUNTIME_ENV,
                "MOBILITY_PROFILE": "full_autonomy",
                "LLM_PROVIDER": "openai",
                "LLM_BASE_URL": "http://127.0.0.1:9/v1",
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "test-model",
            },
            clear=False,
        ):
            runtime = RuntimeContext.from_env()
        self.assertEqual(runtime.hitl_policy, "interactive")


if __name__ == "__main__":
    unittest.main()
