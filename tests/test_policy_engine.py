from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.policy.engine import AgenticPolicyEngine
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


class _FailingKnowledgeBase:
    def retrieve(self, *, query: str, stage: str, top_k: int, corpora: tuple[str, ...]):
        del query, stage, top_k, corpora
        raise ConnectionResetError("simulated_retrieval_disconnect")


def _runtime(*, store_path: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
        agentic_policy_enabled=False,
    )


class PolicyEngineTests(unittest.TestCase):
    def test_plan_stage_falls_back_when_retrieval_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = _runtime(store_path=os.path.join(tmpdir, "policy_store.sqlite"))
            state_payload = {
                "material": {
                    "material_id": "policy-fallback-test",
                    "structure_summary": {"atom_count": 2},
                    "atom_count": 2,
                },
                "execution": {},
                "diagnostics": {},
            }
            with patch_test_llm_clients():
                engine = AgenticPolicyEngine(runtime, knowledge_base=_FailingKnowledgeBase())
                plan = engine.plan_stage(
                    stage="relax",
                    state_payload=state_payload,
                    default_incar={"ENCUT": 520, "EDIFF": 1e-6},
                    default_kpoints_policy={"target_ka": 50.0, "gamma_centered": False},
                )
            self.assertEqual(plan.source, "fallback")
            self.assertEqual(plan.stage, "relax")
            self.assertTrue(str(plan.rationale).startswith("agentic_policy_disabled"))


if __name__ == "__main__":
    unittest.main()
