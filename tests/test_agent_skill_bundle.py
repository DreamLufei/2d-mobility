from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.agents.cost_guardian import CostGuardianAgent
from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.planner import PlannerAgent
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _skills_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "skills"))


def _runtime(tmpdir: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=os.path.join(tmpdir, "store.sqlite"),
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
        skills_root=_skills_root(),
    )


class AgentSkillBundleTests(unittest.TestCase):
    def test_planner_bundle_includes_role_and_workflow_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = PlannerAgent(_runtime(tmpdir), _skills_root())
            bundle = agent._skill_bundle(
                task_type="single_material",
                stage="proposal_phase",
                explicit_skills=["recovery"],
            )
        self.assertIn("planning", bundle["selected"])
        self.assertIn("single_material_mobility", bundle["selected"])
        self.assertIn("recovery", bundle["selected"])

    def test_cost_guardian_bundle_includes_cost_role_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = CostGuardianAgent(_runtime(tmpdir), _skills_root())
            bundle = agent._skill_bundle(
                task_type="single_material",
                stage="critique_phase",
                explicit_skills=["recovery"],
            )
        self.assertIn("cost_guardian", bundle["selected"])
        self.assertIn("single_material_mobility", bundle["selected"])
        self.assertIn("recovery", bundle["selected"])

    def test_orchestrator_bundle_normalizes_legacy_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = OrchestratorAgent(_runtime(tmpdir), _skills_root())
            bundle = agent._skill_bundle(
                task_type="single_material",
                stage="arbitration_phase",
                explicit_skills=["default_scientific_path_skill", "final_reporting_skill"],
            )
        self.assertIn("orchestration", bundle["selected"])
        self.assertIn("single_material_mobility", bundle["selected"])
        self.assertIn("reporting", bundle["selected"])
        self.assertNotIn("default_scientific_path_skill", bundle["selected"])
        self.assertNotIn("final_reporting_skill", bundle["selected"])


if __name__ == "__main__":
    unittest.main()
