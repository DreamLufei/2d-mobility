from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.agents.admission import AdmissionAgent
from mobility_agent.agents.reporter import ReporterAgent
from mobility_agent.agents.validation import ValidationAgent
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


class RuleAgentSkillPromptTests(unittest.TestCase):
    def test_admission_role_prompt_uses_disk_backed_admission_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = AdmissionAgent(_runtime(tmpdir), _skills_root())
            prompt = agent._role_skill_prompt()
        self.assertIn("ROLE_SKILLS:", prompt)
        self.assertIn("- admission:", prompt)
        self.assertNotIn("compatibility fallback", prompt)

    def test_validation_role_prompt_uses_disk_backed_validation_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = ValidationAgent(_runtime(tmpdir), _skills_root())
            prompt = agent._role_skill_prompt()
        self.assertIn("ROLE_SKILLS:", prompt)
        self.assertIn("- validation:", prompt)
        self.assertNotIn("compatibility fallback", prompt)

    def test_reporter_role_prompt_uses_reporting_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = ReporterAgent(_runtime(tmpdir), _skills_root())
            prompt = agent._role_skill_prompt()
        self.assertIn("ROLE_SKILLS:", prompt)
        self.assertIn("- reporting:", prompt)
        self.assertNotIn("compatibility fallback", prompt)


if __name__ == "__main__":
    unittest.main()
