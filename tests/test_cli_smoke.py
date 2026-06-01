from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

import mobality as single_cli
import run_mongo_batch as batch_cli
from mobility_agent.graph.state import MaterialRunOutcome
from tests.llm_test_utils import TEST_LLM_ENV, patch_test_llm_clients


class CliSmokeTests(unittest.TestCase):
    def test_single_material_cli_fails_without_llm_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
                )
            with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = ["mobality.py", "--root-path", tmpdir, "--dry-run", "--json"]
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
            self.assertEqual(rc, 2)
            self.assertIn("Configuration error:", stderr.getvalue())

    def test_batch_cli_fails_without_llm_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            stderr = io.StringIO()
            env = {
                "MONGO_URI": "mongodb://example",
                "MONGO_DB": "db",
                "MONGO_COLLECTION": "collection",
                "BATCH_TAG": "cli-batch",
                "RUNS_ROOT": tmpdir,
                "POTCAR_METHOD": "concat",
                "POTCAR_ROOT": tmpdir,
                "LLM_PROVIDER": "",
                "LLM_BASE_URL": "",
                "LLM_API_KEY": "",
                "LLM_MODEL": "",
            }
            argv = ["run_mongo_batch.py", "--dry-run", "--json"]
            with patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                rc = batch_cli.main()
            self.assertEqual(rc, 2)
            self.assertIn("Configuration error:", stderr.getvalue())

    def test_single_material_cli_dry_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
                )
            with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            stdout = io.StringIO()
            env = {
                **TEST_LLM_ENV,
                "ENABLE_HUMAN_REVIEW": "false",
                "MOBILITY_STORE_PATH": os.path.join(tmpdir, "cli_store.sqlite"),
                "HUMAN_REVIEW_TIMEOUT_SECONDS": "0",
            }
            argv = [
                "mobality.py",
                "--root-path",
                tmpdir,
                "--dry-run",
                "--hitl-policy",
                "non_interactive_skip_on_timeout",
                "--json",
            ]
            with patch_test_llm_clients(), patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                rc = single_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "completed")

    def test_single_material_cli_dry_run_fail_stage_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
                )
            with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            stdout = io.StringIO()
            env = {
                **TEST_LLM_ENV,
                "ENABLE_HUMAN_REVIEW": "false",
                "MOBILITY_STORE_PATH": os.path.join(tmpdir, "cli_store.sqlite"),
                "HUMAN_REVIEW_TIMEOUT_SECONDS": "0",
            }
            argv = [
                "mobality.py",
                "--root-path",
                tmpdir,
                "--dry-run",
                "--dry-run-fail-stages",
                "scf",
                "--hitl-policy",
                "non_interactive_skip_on_timeout",
                "--json",
            ]
            with patch_test_llm_clients(), patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                rc = single_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "skipped")

    def test_batch_cli_main_dry_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            structure = Structure(
                lattice=Lattice.hexagonal(3.0, 20.0),
                species=["Si", "Si"],
                coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
            )
            doc = {"_id": "doc-1", "material_id": "mat-1", "structure": structure.as_dict()}
            claims = [doc, None]

            def fake_claim(*args, **kwargs):
                return claims.pop(0)

            def fake_build_potcar(struct, *, potcar_root, dest_path, potcar_map_path=None):
                with open(dest_path, "w", encoding="utf-8") as handle:
                    handle.write("FAKE POTCAR\n")
                return ["Si"]

            def fake_run_single_material(**kwargs):
                workdir = kwargs["workdir"]
                return MaterialRunOutcome(
                    task_id="cli-task",
                    material_id=kwargs["material_id"],
                    final_status="completed",
                    workdir=workdir,
                    artifact_paths={"final_summary_path": os.path.join(workdir, "final_summary.json")},
                    results={},
                    warnings=[],
                    errors=[],
                    validation_report={},
                    final_summary={"material_id": kwargs["material_id"], "run_status": "completed"},
                )

            handles = SimpleNamespace(client=SimpleNamespace(close=lambda: None), collection=object())
            stdout = io.StringIO()
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
            argv = ["run_mongo_batch.py", "--dry-run", "--json", "--hitl-policy", "non_interactive_skip_on_timeout"]
            with patch_test_llm_clients(), patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), patch(
                "mobility_agent.runtime.batch_runner.connect",
                return_value=handles,
            ), patch(
                "mobility_agent.runtime.batch_runner.claim_next_material",
                side_effect=fake_claim,
            ), patch(
                "mobility_agent.runtime.batch_runner.build_potcar",
                side_effect=fake_build_potcar,
            ), patch(
                "mobility_agent.runtime.batch_runner.run_single_material",
                side_effect=fake_run_single_material,
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_completed",
                side_effect=lambda *args, **kwargs: None,
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_failed",
                side_effect=lambda *args, **kwargs: None,
            ), contextlib.redirect_stdout(stdout):
                rc = batch_cli.main()
            self.assertEqual(rc, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["batch"]["global_statistics"]["processed"], 1)


if __name__ == "__main__":
    unittest.main()
