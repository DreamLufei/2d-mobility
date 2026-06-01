from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.runner import run_single_material
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _prepare_material_root(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")


class TimeoutPolicyTests(unittest.TestCase):
    def _runtime(self, policy: str, *, store_path: str) -> RuntimeContext:
        return RuntimeContext(
            agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
            hitl_policy=policy,
            dry_run=True,
            dry_run_fail_stages=("scf",),
            store_path=store_path,
            compatibility_export_enabled=False,
            compatibility_export_pickle=False,
        )

    def test_non_interactive_skip_on_timeout_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=self._runtime("non_interactive_skip_on_timeout", store_path=os.path.join(tmpdir, "timeout_store.sqlite")),
                    material_id="skip-on-timeout",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.final_status, "skipped")
            self.assertEqual(outcome.termination_reason, "skip_material")

    def test_non_interactive_abort_on_timeout_aborts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=self._runtime("non_interactive_abort_on_timeout", store_path=os.path.join(tmpdir, "timeout_store.sqlite")),
                    material_id="abort-on-timeout",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.final_status, "failed")
            self.assertEqual(outcome.termination_reason, "abort_task")


if __name__ == "__main__":
    unittest.main()
