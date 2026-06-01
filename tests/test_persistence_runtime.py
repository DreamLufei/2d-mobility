from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.runtime.checkpointing import load_thread_id, runtime_state_snapshot_path
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.database import delete_checkpoint_thread
from mobility_agent.runtime.runner import run_single_material
from mobility_agent.tools.errors import CheckpointRestoreError
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _prepare_material_root(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")


class PersistenceRuntimeTests(unittest.TestCase):
    def test_thread_id_persists_and_checkpoint_exports_at_stable_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                compatibility_export_enabled=True,
                compatibility_export_pickle=True,
            )
            with patch_test_llm_clients():
                outcome = run_single_material(runtime=runtime, material_id="persisted-mat", root_path=tmpdir, fresh=True)
            thread_id = load_thread_id(workdir=outcome.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            self.assertIsNotNone(thread_id)
            self.assertTrue(str(thread_id).startswith("material::"))
            self.assertIn("persisted-mat", str(thread_id))
            self.assertTrue(os.path.exists(os.path.join(outcome.workdir, "checkpoint.pkl")))

            with patch_test_llm_clients():
                second = run_single_material(runtime=runtime, material_id="persisted-mat", root_path=tmpdir, fresh=False)
            thread_id_second = load_thread_id(workdir=second.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            self.assertEqual(thread_id, thread_id_second)

    def test_debug_snapshots_do_not_drive_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                compatibility_export_enabled=True,
                compatibility_export_pickle=True,
            )
            with patch_test_llm_clients():
                outcome = run_single_material(runtime=runtime, material_id="restore-check", root_path=tmpdir, fresh=True)
            snapshot_path = runtime_state_snapshot_path(outcome.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            thread_id = load_thread_id(workdir=outcome.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            self.assertIsNotNone(thread_id)
            delete_checkpoint_thread(database_uri=runtime.resolved_db_uri, thread_id=thread_id)
            self.assertTrue(os.path.exists(snapshot_path))
            self.assertTrue(os.path.exists(os.path.join(outcome.workdir, "checkpoint.pkl")))
            with patch_test_llm_clients(), self.assertRaises(CheckpointRestoreError):
                run_single_material(runtime=runtime, material_id="restore-check", root_path=tmpdir, fresh=False)


if __name__ == "__main__":
    unittest.main()
