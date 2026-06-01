from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import mobality as single_cli
import run_mongo_batch as batch_cli
from tests.llm_test_utils import TEST_LLM_ENV, patch_test_llm_clients


class CliSkillOptionTests(unittest.TestCase):
    def _write_skill(self, root: str, name: str, *, roles: list[str] | None = None, task_types: list[str] | None = None) -> None:
        skill_dir = os.path.join(root, name)
        os.makedirs(skill_dir, exist_ok=True)
        with open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8") as handle:
            handle.write(
                "+++\n"
                f'name = "{name}"\n'
                f'description = "{name} description"\n'
                f"roles = {json.dumps(list(roles or []))}\n"
                f"task_types = {json.dumps(list(task_types or []))}\n"
                'load_strategy = "summary_only"\n'
                "+++\n\n"
                f"# {name}\n\n"
                "## Purpose\n"
                f"- {name} purpose\n"
            )
        with open(os.path.join(skill_dir, "notes.md"), "w", encoding="utf-8") as handle:
            handle.write(f"{name} resource\n")

    def test_single_cli_applies_skill_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
                )
            with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            custom_skills = os.path.join(tmpdir, "custom-skills")
            os.makedirs(custom_skills, exist_ok=True)
            stdout = io.StringIO()
            captured: dict[str, object] = {}
            env = {
                **TEST_LLM_ENV,
                "MOBILITY_STORE_PATH": os.path.join(tmpdir, "cli_store.sqlite"),
                "HUMAN_REVIEW_TIMEOUT_SECONDS": "0",
            }

            def fake_run_single_material(**kwargs):
                captured["runtime"] = kwargs["runtime"]
                return type(
                    "Outcome",
                    (),
                    {
                        "material_id": kwargs["material_id"],
                        "status": "completed",
                        "final_acceptance": None,
                        "termination_reason": None,
                        "workdir": kwargs["workdir"],
                        "warnings": [],
                        "errors": [],
                        "artifact_paths": {},
                        "model_dump": lambda self, mode="json": {"status": "completed"},
                    },
                )()

            argv = [
                "mobality.py",
                "--root-path",
                tmpdir,
                "--dry-run",
                "--skills-root",
                custom_skills,
                "--skill-auto-resolve-limit",
                "9",
                "--skill-inline-body-limit",
                "3000",
                "--json",
            ]
            with patch_test_llm_clients(), patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), patch(
                "mobality.run_single_material",
                side_effect=fake_run_single_material,
            ), contextlib.redirect_stdout(stdout):
                rc = single_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "completed")
            runtime = captured["runtime"]
            self.assertEqual(runtime.skills_root, os.path.abspath(custom_skills))
            self.assertEqual(runtime.skill_auto_resolve_limit, 9)
            self.assertEqual(runtime.skill_inline_body_limit, 3000)

    def test_batch_cli_applies_skill_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_skills = os.path.join(tmpdir, "custom-skills")
            os.makedirs(custom_skills, exist_ok=True)
            stdout = io.StringIO()
            captured: dict[str, object] = {}
            env = {
                **TEST_LLM_ENV,
                "MONGO_URI": "mongodb://example",
                "MONGO_DB": "db",
                "MONGO_COLLECTION": "collection",
                "BATCH_TAG": "cli-batch",
                "RUNS_ROOT": tmpdir,
                "POTCAR_METHOD": "concat",
                "POTCAR_ROOT": tmpdir,
                "MOBILITY_STORE_PATH": os.path.join(tmpdir, "cli_batch_store.sqlite"),
                "HUMAN_REVIEW_TIMEOUT_SECONDS": "0",
            }

            def fake_run_mongo_batch(**kwargs):
                captured["runtime"] = kwargs["runtime"]
                return {"batch": {"global_statistics": {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}}}

            argv = [
                "run_mongo_batch.py",
                "--dry-run",
                "--skills-root",
                custom_skills,
                "--skill-auto-resolve-limit",
                "7",
                "--skill-inline-body-limit",
                "2800",
                "--json",
            ]
            with patch_test_llm_clients(), patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), patch(
                "run_mongo_batch.run_mongo_batch",
                side_effect=fake_run_mongo_batch,
            ), contextlib.redirect_stdout(stdout):
                rc = batch_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["batch"]["global_statistics"]["processed"], 0)
            runtime = captured["runtime"]
            self.assertEqual(runtime.skills_root, os.path.abspath(custom_skills))
            self.assertEqual(runtime.skill_auto_resolve_limit, 7)
            self.assertEqual(runtime.skill_inline_body_limit, 2800)

    def test_single_cli_lists_skills_without_llm_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_skills = os.path.join(tmpdir, "custom-skills")
            os.makedirs(custom_skills, exist_ok=True)
            self._write_skill(custom_skills, "planning", roles=["planner"], task_types=["single_material_mobility"])
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = ["mobality.py", "--list-skills", "--skills-root", custom_skills, "--json"]
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "",
                    "LLM_BASE_URL": "",
                    "LLM_API_KEY": "",
                    "LLM_MODEL": "",
                },
                clear=False,
            ), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = single_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["skills_root"], os.path.abspath(custom_skills))
            self.assertEqual([item["name"] for item in payload["skills"]], ["planning"])
            self.assertEqual(payload["skills"][0]["roles"], ["planner"])
            self.assertEqual(payload["skills"][0]["resource_count"], 1)
            self.assertEqual(stderr.getvalue(), "")

    def test_batch_cli_lists_skills_without_llm_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_skills = os.path.join(tmpdir, "custom-skills")
            os.makedirs(custom_skills, exist_ok=True)
            self._write_skill(custom_skills, "reporting", roles=["reporter"], task_types=["batch_mobility_screening"])
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = ["run_mongo_batch.py", "--list-skills", "--skills-root", custom_skills, "--json"]
            with patch.dict(
                os.environ,
                {
                    "LLM_PROVIDER": "",
                    "LLM_BASE_URL": "",
                    "LLM_API_KEY": "",
                    "LLM_MODEL": "",
                },
                clear=False,
            ), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = batch_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["skills_root"], os.path.abspath(custom_skills))
            self.assertEqual([item["name"] for item in payload["skills"]], ["reporting"])
            self.assertEqual(payload["skills"][0]["roles"], ["reporter"])
            self.assertEqual(payload["skills"][0]["resource_count"], 1)
            self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
