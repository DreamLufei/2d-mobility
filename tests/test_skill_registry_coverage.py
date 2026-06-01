from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.agents.admission import AdmissionAgent
from mobility_agent.agents.cost_guardian import CostGuardianAgent
from mobility_agent.agents.critic import CriticAgent
from mobility_agent.agents.executor import ExecutorAgent
from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.physics_judge import PhysicsJudgeAgent
from mobility_agent.agents.planner import PlannerAgent
from mobility_agent.agents.recovery import RecoveryAgent
from mobility_agent.agents.reporter import ReporterAgent
from mobility_agent.agents.validation import ValidationAgent
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.skills import discover_skills
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


class SkillRegistryCoverageTests(unittest.TestCase):
    def test_all_in_tree_llm_roles_have_disk_backed_skill_packages(self) -> None:
        registry = discover_skills(_skills_root())
        covered_roles = {
            str(role)
            for entry in registry.values()
            for role in list((entry.get("manifest", {}) or {}).get("roles", []) or [])
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            runtime = _runtime(tmpdir)
            agents = [
                AdmissionAgent(runtime, _skills_root()),
                PlannerAgent(runtime, _skills_root()),
                RecoveryAgent(runtime, _skills_root()),
                CriticAgent(runtime, _skills_root()),
                PhysicsJudgeAgent(runtime, _skills_root()),
                CostGuardianAgent(runtime, _skills_root()),
                OrchestratorAgent(runtime, _skills_root()),
                ReporterAgent(runtime, _skills_root()),
                ExecutorAgent(runtime, _skills_root()),
                ValidationAgent(runtime, _skills_root()),
            ]
        missing = sorted({agent.llm_role for agent in agents if agent.llm_role not in covered_roles})
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
