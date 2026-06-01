from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.agents.executor import ExecutorAgent
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


class AgentSkillPromptTests(unittest.TestCase):
    def test_planner_role_prompt_prefers_disk_backed_skill_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = PlannerAgent(_runtime(tmpdir), _skills_root())
            prompt = agent._role_skill_prompt()
        self.assertIn("ROLE_SKILLS:", prompt)
        self.assertNotIn("compatibility fallback", prompt)
        self.assertIn("- planning:", prompt)
        self.assertIn("- single_material_mobility:", prompt)
        self.assertNotIn("planning_skill", prompt)
        self.assertNotIn("default_scientific_path_skill", prompt)

    def test_executor_role_prompt_uses_execution_feasibility_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = ExecutorAgent(_runtime(tmpdir), _skills_root())
            prompt = agent._role_skill_prompt()
        self.assertIn("ROLE_SKILLS:", prompt)
        self.assertIn("- execution_feasibility:", prompt)
        self.assertIn("- single_material_mobility:", prompt)
        self.assertNotIn("execution_feasibility_skill", prompt)


if __name__ == "__main__":
    unittest.main()
