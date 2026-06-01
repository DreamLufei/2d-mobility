from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mobility_agent.skills import SkillResolutionRequest, discover_skills, load_skill, resolve_skills


class SkillLoaderTests(unittest.TestCase):
    def test_registry_parses_frontmatter_and_nested_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "demo_skill"
            resource_dir = skill_dir / "references"
            resource_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(
                "\n".join(
                    [
                        "+++",
                        'name = "demo_skill"',
                        'version = "1"',
                        'description = "Demo skill"',
                        'load_strategy = "summary_only"',
                        'roles = ["planner"]',
                        'task_types = ["single_material"]',
                        'stages = ["proposal_phase"]',
                        'tags = ["demo", "mainline"]',
                        "+++",
                        "# demo_skill",
                        "",
                        "## purpose",
                        "Demo purpose",
                    ]
                ),
                encoding="utf-8",
            )
            (resource_dir / "rule.txt").write_text("demo rule\n", encoding="utf-8")
            registry = discover_skills(tmpdir)
            self.assertIn("demo_skill", registry)
            entry = registry["demo_skill"]
            self.assertEqual(entry["manifest"]["roles"], ["planner"])
            self.assertEqual(entry["resources"][0]["path"], "references/rule.txt")

    def test_loader_reads_resources_and_resolver_prefers_recovery_on_error(self) -> None:
        repo_skills = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "skills"))
        loaded = load_skill(repo_skills, "recovery", include_body=True, include_resources=True, resource_limit=5)
        self.assertEqual(loaded["name"], "recovery")
        self.assertIn("allowed_actions.json", loaded["resource_payloads"])

        registry = discover_skills(repo_skills)
        selection = resolve_skills(
            registry,
            request=SkillResolutionRequest(
                role="recovery",
                task_type="single_material",
                stage="proposal_phase",
                run_status="needs_recovery",
                has_error=True,
                latest_error="zbrent_fatal",
                limit=4,
            ),
        )
        self.assertIn("recovery", selection.selected_skills)

    def test_resolver_normalizes_legacy_explicit_skill_aliases(self) -> None:
        repo_skills = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "skills"))
        registry = discover_skills(repo_skills)
        selection = resolve_skills(
            registry,
            request=SkillResolutionRequest(
                role="planner",
                task_type="single_material",
                stage="proposal_phase",
                explicit_skills=["default_scientific_path_skill"],
                limit=4,
            ),
        )
        self.assertIn("single_material_mobility", selection.selected_skills)
        selected = next(item for item in selection.candidates if item.name == "single_material_mobility")
        self.assertIn("explicit_skill", selected.reasons)


if __name__ == "__main__":
    unittest.main()
