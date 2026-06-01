from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.agents.base import _is_connection_error, _structured_connection_backoff_seconds
from mobility_agent.runtime.agentic_controller import AgenticMaterialController, CouncilRoleFailure
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(*, store_path: str, llm_base_url: str) -> RuntimeContext:
    agent_runtime = build_test_agent_runtime(
        llm_base_url=llm_base_url,
        llm_model="qwen3.6-plus" if "dashscope" in llm_base_url else "minimax/minimax-m2.7",
        llm_api_key="test-key",
    )
    return RuntimeContext(
        agent_runtime=agent_runtime,
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
    )


def _working_state(tmpdir: str) -> dict[str, object]:
    return {
        "execution": {"workdir": os.path.join(tmpdir, "mobility_calculation")},
        "services": {"council_output_cache": {}},
    }


class SerializedProviderRuntimeTests(unittest.TestCase):
    def test_connection_error_detection_matches_api_connection_error(self) -> None:
        self.assertTrue(_is_connection_error(RuntimeError("APIConnectionError:Connection error.")))
        self.assertTrue(_is_connection_error(RuntimeError("Connection refused by upstream")))
        self.assertFalse(_is_connection_error(RuntimeError("plain failure")))

    def test_connection_error_backoff_schedule_is_progressive(self) -> None:
        self.assertEqual(_structured_connection_backoff_seconds(attempt=1), 3.0)
        self.assertEqual(_structured_connection_backoff_seconds(attempt=2), 10.0)
        self.assertEqual(_structured_connection_backoff_seconds(attempt=3), 20.0)
        self.assertEqual(_structured_connection_backoff_seconds(attempt=9), 20.0)

    def test_dashscope_reviewers_are_sequentialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_test_llm_clients():
                controller = AgenticMaterialController(
                    _runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                )
            reviewer_specs = [
                {"agent_name": "physics_judge", "critical": True, "invoke": lambda agent, state: (["physics"], ["pref"])},
                {"agent_name": "cost_guardian", "critical": False, "invoke": lambda agent, state: (["cost"], [])},
            ]
            with (
                patch.object(controller, "_run_role", side_effect=[(["physics"], ["pref"]), (["cost"], [])]) as mock_run_role,
                patch.object(controller, "_run_roles_parallel") as mock_parallel,
            ):
                results, failures = controller._run_reviewer_roles(
                    _working_state(tmpdir),
                    reviewer_specs=reviewer_specs,
                    round_id=3,
                    council_mode="validation_followup_council",
                    reason="post_mobility_quality_review",
                    proposals=[],
                    reused_roles=[],
                )
            self.assertEqual(len(results), 2)
            self.assertEqual(len(failures), 0)
            self.assertFalse(mock_parallel.called)
            self.assertEqual([call.kwargs["agent_name"] for call in mock_run_role.call_args_list], ["physics_judge", "cost_guardian"])

    def test_openrouter_reviewers_keep_parallel_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_test_llm_clients():
                controller = AgenticMaterialController(
                    _runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        llm_base_url="https://openrouter.ai/api/v1",
                    )
                )
            reviewer_specs = [
                {"agent_name": "physics_judge", "critical": True, "invoke": lambda agent, state: (["physics"], ["pref"])},
            ]
            sentinel_results = [{"index": 0, "agent_name": "physics_judge", "critical": True, "output": (["physics"], ["pref"]), "reused": False}]
            with (
                patch.object(controller, "_run_role") as mock_run_role,
                patch.object(controller, "_run_roles_parallel", return_value=(sentinel_results, [])) as mock_parallel,
            ):
                results, failures = controller._run_reviewer_roles(
                    _working_state(tmpdir),
                    reviewer_specs=reviewer_specs,
                    round_id=4,
                    council_mode="segment_council",
                    reason="resume",
                    proposals=[],
                    reused_roles=[],
                )
            self.assertEqual(results, sentinel_results)
            self.assertEqual(failures, [])
            self.assertTrue(mock_parallel.called)
            self.assertFalse(mock_run_role.called)

    def test_serialized_provider_reopen_cooldown_applies_only_for_rate_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_test_llm_clients():
                controller = AgenticMaterialController(
                    _runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                )
            exc = CouncilRoleFailure(
                agent_name="physics_judge",
                phase="review",
                critical=True,
                error=RuntimeError("Error code: 429 - {'error': {'code': '1302', 'message': 'rate limit'}}"),
            )
            with patch("mobility_agent.runtime.agentic_controller.time.sleep") as mock_sleep:
                controller._apply_council_reopen_cooldown_if_needed(
                    state=_working_state(tmpdir),
                    round_id=5,
                    exc=exc,
                )
            mock_sleep.assert_called_once_with(30.0)

    def test_serialized_provider_reopen_cooldown_is_skipped_for_non_rate_limit_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_test_llm_clients():
                controller = AgenticMaterialController(
                    _runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                )
            exc = CouncilRoleFailure(
                agent_name="physics_judge",
                phase="review",
                critical=True,
                error=RuntimeError("plain failure"),
            )
            with patch("mobility_agent.runtime.agentic_controller.time.sleep") as mock_sleep:
                controller._apply_council_reopen_cooldown_if_needed(
                    state=_working_state(tmpdir),
                    round_id=5,
                    exc=exc,
                )
            mock_sleep.assert_not_called()

    def test_connection_error_reopen_cooldown_applies_for_any_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_test_llm_clients():
                controller = AgenticMaterialController(
                    _runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        llm_base_url="https://openrouter.ai/api/v1",
                    )
                )
            exc = CouncilRoleFailure(
                agent_name="planner",
                phase="proposal",
                critical=True,
                error=RuntimeError("APIConnectionError:Connection error."),
            )
            with patch("mobility_agent.runtime.agentic_controller.time.sleep") as mock_sleep:
                controller._apply_council_reopen_cooldown_if_needed(
                    state=_working_state(tmpdir),
                    round_id=6,
                    exc=exc,
                )
            mock_sleep.assert_called_once_with(15.0)


if __name__ == "__main__":
    unittest.main()
