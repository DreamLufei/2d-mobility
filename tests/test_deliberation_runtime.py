from __future__ import annotations

import json
import os
import tempfile
import unittest

from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.runner import run_single_material
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(*, store_path: str, fail_stages: tuple[str, ...] = ()) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        dry_run_fail_stages=fail_stages,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
    )


def _prepare_material_root(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")


class DeliberationRuntimeTests(unittest.TestCase):
    def test_deliberation_trace_is_auditable_for_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=_runtime(store_path=os.path.join(tmpdir, "store.sqlite")),
                    material_id="delib-happy",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.status, "completed")
            trace_path = outcome.artifact_paths["deliberation_trace_path"]
            self.assertTrue(os.path.exists(trace_path))
            with open(trace_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertGreaterEqual(len(payload["rounds"]), 1)
            self.assertGreaterEqual(len(payload["proposals"]), 1)
            self.assertGreaterEqual(len(payload["critiques"]), 1)
            self.assertGreaterEqual(len(payload["preferences"]), 1)
            self.assertGreaterEqual(len(payload["arbitrations"]), 1)
            self.assertGreaterEqual(len(payload["selected_actions"]), 1)
            self.assertGreaterEqual(len(payload["reflections"]), 1)
            first_action = payload["selected_actions"][0]
            self.assertEqual(first_action["action_family"], "run_capability")

    def test_failure_round_contains_multiple_recovery_proposals_and_objections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=_runtime(
                        store_path=os.path.join(tmpdir, "store.sqlite"),
                        fail_stages=("scf",),
                    ),
                    material_id="delib-fail",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.status, "skipped")
            with open(outcome.artifact_paths["deliberation_trace_path"], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            recovery_props = [item for item in payload["proposals"] if item["agent_name"] == "recovery"]
            self.assertGreaterEqual(len(recovery_props), 3)
            objection_like = [
                item
                for item in payload["critiques"] + payload["preferences"]
                if item["agent_name"] in {"critic", "physics_judge", "cost_guardian"}
            ]
            self.assertTrue(objection_like)
            self.assertGreaterEqual(len(payload["arbitrations"]), 1)
            self.assertGreaterEqual(len(payload["reflections"]), 1)


if __name__ == "__main__":
    unittest.main()
