from __future__ import annotations

import json
import os
import tempfile
import unittest

from mobility_agent.hitl.escalation import escalation_paths, write_human_response
from mobility_agent.runtime.checkpointing import runtime_ui_events_path, runtime_ui_state_path
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.runner import run_single_material
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(*, store_path: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
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


class WebConsoleRuntimeExportTests(unittest.TestCase):
    def test_ui_state_and_events_are_exported_during_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=runtime,
                    material_id="ui-export-ok",
                    root_path=tmpdir,
                    fresh=True,
                )
            ui_state_path = runtime_ui_state_path(outcome.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            ui_events_path = runtime_ui_events_path(outcome.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            self.assertTrue(os.path.exists(ui_state_path))
            self.assertTrue(os.path.exists(ui_events_path))
            with open(ui_state_path, "r", encoding="utf-8") as handle:
                ui_state = json.load(handle)
            self.assertEqual(ui_state["material_id"], "ui-export-ok")
            self.assertEqual(ui_state["runtime_run_status"], "completed")
            self.assertIn("updated_at", ui_state)
            self.assertGreaterEqual(int(ui_state["workflow_contract"]["version"]), 1)
            self.assertEqual(ui_state["workflow_contract"]["council_mode"], "validation_followup_council")
            self.assertEqual(ui_state["workflow_contract"]["planned_capabilities"], [])
            self.assertFalse(ui_state["execution_checkpoint"]["needs_deliberation"])
            with open(ui_events_path, "r", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle if line.strip()]
            self.assertGreaterEqual(len(events), 2)
            first = events[0]
            self.assertIn("timestamp", first)
            self.assertIn("event_type", first)
            self.assertIn("current_stage", first)
            self.assertIn("runtime_run_status", first)
            self.assertIn("latest_artifact_keys", first)
            event_types = {item["event_type"] for item in events}
            self.assertIn("state_initialized", event_types)
            self.assertIn("run_status_changed", event_types)
            artifact_paths = outcome.artifact_paths
            retrieval_trace_path = artifact_paths.get("retrieval_trace_path", "")
            parameter_plan_path = artifact_paths.get("parameter_plan_path", "")
            recovery_diagnosis_path = artifact_paths.get("recovery_diagnosis_path", "")
            workflow_contract_path = artifact_paths.get("workflow_contract_path", "")
            decision_ledger_path = artifact_paths.get("decision_ledger_path", "")
            self.assertTrue(os.path.exists(retrieval_trace_path))
            self.assertTrue(os.path.exists(parameter_plan_path))
            self.assertTrue(os.path.exists(recovery_diagnosis_path))
            self.assertTrue(os.path.exists(workflow_contract_path))
            self.assertTrue(os.path.exists(decision_ledger_path))
            with open(retrieval_trace_path, "r", encoding="utf-8") as handle:
                retrieval_trace = json.load(handle)
            with open(parameter_plan_path, "r", encoding="utf-8") as handle:
                parameter_plan = json.load(handle)
            with open(recovery_diagnosis_path, "r", encoding="utf-8") as handle:
                recovery_diagnosis = json.load(handle)
            with open(decision_ledger_path, "r", encoding="utf-8") as handle:
                decision_ledger = json.load(handle)
            self.assertIsInstance(retrieval_trace, list)
            self.assertIsInstance(parameter_plan, dict)
            self.assertIsInstance(recovery_diagnosis, dict)
            authored_segments = [
                list((item.get("summary") or {}).get("planned_capabilities", []) or [])
                for item in decision_ledger
                if str(item.get("entry_type") or "") == "workflow_contract_updated"
            ]
            self.assertIn(["prepare", "relax", "scf", "band", "effective_mass", "strain_loop"], authored_segments)
            self.assertIn(["mobility", "validation"], authored_segments)

    def test_human_response_writer_is_atomic_and_schema_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_human_response(
                workdir=tmpdir,
                response={"action": "retry_current_stage", "reason": "user_confirmed"},
            )
            response_path = escalation_paths(tmpdir)["response_path"]
            self.assertTrue(os.path.exists(response_path))
            with open(response_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["action"], "retry_current_stage")
            self.assertEqual(payload["reason"], "user_confirmed")
            self.assertEqual(result["decision"]["action"], "retry_current_stage")


if __name__ == "__main__":
    unittest.main()
