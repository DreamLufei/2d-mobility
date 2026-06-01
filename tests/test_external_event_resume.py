from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.schemas import ArbitrationRecord, SelectedAction
from mobility_agent.graph import build_material_graph
from mobility_agent.graph.state import MaterialTaskState, build_state_patch
from mobility_agent.runtime.checkpointing import open_sqlite_checkpointer
from mobility_agent.runtime.checkpointing import load_thread_id, runtime_state_snapshot_path
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.entrypoints import build_external_event_resume_entrypoint
from mobility_agent.runtime.runner import _build_nodes
from mobility_agent.runtime.runner import run_single_material, run_single_material_external_event
from mobility_agent.runtime.store import open_memory_store
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(*, store_path: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=store_path,
        full_autonomy=False,
        allow_external_wait=True,
        council_policy_mode="strict",
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


def _scripted_arbitrate(self, state, proposals, critiques, preferences, round_id):  # type: ignore[no-untyped-def]
    del proposals, critiques, preferences
    task_id = str((state.get("task", {}) or {}).get("task_id") or "")
    latest_event = dict((state.get("execution", {}) or {}).get("latest_event", {}) or {})
    completed = {
        str(item.get("capability"))
        for item in list((state.get("task_board", {}) or {}).get("completed_tasks", []) or [])
        if isinstance(item, dict)
    }
    if latest_event:
        event_type = str(latest_event.get("event_type") or "")
        if event_type == "resume_requested":
            selected = SelectedAction(
                action_family="run_capability",
                target_capability="relax",
                parameters={"job_id": "job-2"},
                rationale="requeue_after_resume_request",
                submit_external_job=True,
                wait_for_event_after_submission=True,
            )
            return ArbitrationRecord(
                agent_name="orchestrator",
                round_id=round_id,
                target_task_id=task_id,
                selected_proposal_id="scripted::resume_requested",
                selected_action=selected,
                rationale="resume_requested_requeues_external_job",
            )
        if event_type == "job_completed":
            selected = SelectedAction(
                action_family="abort_material",
                parameters={"reason": "post_event_completed_stop"},
                rationale="stop_after_consuming_completed_event",
            )
            return ArbitrationRecord(
                agent_name="orchestrator",
                round_id=round_id,
                target_task_id=task_id,
                selected_proposal_id="scripted::job_completed",
                selected_action=selected,
                rationale="job_completed_consumed",
            )
        if event_type in {"job_failed", "job_timeout", "artifact_missing"}:
            selected = SelectedAction(
                action_family="abort_material",
                parameters={"reason": f"post_{event_type}_stop"},
                rationale="stop_after_consuming_failed_event",
            )
            return ArbitrationRecord(
                agent_name="orchestrator",
                round_id=round_id,
                target_task_id=task_id,
                selected_proposal_id=f"scripted::{event_type}",
                selected_action=selected,
                rationale=f"{event_type}_consumed",
            )
    if str((state.get("workflow", {}) or {}).get("run_status") or "") == "waiting_external":
        return ArbitrationRecord(
            agent_name="orchestrator",
            round_id=round_id,
            target_task_id=task_id,
            rationale="waiting_for_external_event",
            whether_noop=True,
            whether_waiting_external=True,
        )
    if "prepare" not in completed:
        selected = SelectedAction(
            action_family="run_capability",
            target_capability="prepare",
            rationale="complete_prepare_on_default_mainline",
        )
        return ArbitrationRecord(
            agent_name="orchestrator",
            round_id=round_id,
            target_task_id=task_id,
            selected_proposal_id="scripted::prepare_first",
            selected_action=selected,
            rationale="prepare_precedes_external_relax",
        )
    selected = SelectedAction(
        action_family="run_capability",
        target_capability="relax",
        parameters={"job_id": "job-1"},
        rationale="submit_relax_as_external_job",
        submit_external_job=True,
        wait_for_event_after_submission=True,
    )
    return ArbitrationRecord(
        agent_name="orchestrator",
        round_id=round_id,
        target_task_id=task_id,
        selected_proposal_id="scripted::initial_external",
        selected_action=selected,
        rationale="initial_external_submission",
    )


class ExternalEventResumeTests(unittest.TestCase):
    def test_external_event_resume_entrypoint_consumes_job_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _scripted_arbitrate):
                waiting = run_single_material(runtime=runtime, material_id="evt-complete", root_path=tmpdir, fresh=True)
                self.assertEqual(waiting.status, "waiting_external")
                thread_id = load_thread_id(workdir=waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
                self.assertIsNotNone(thread_id)
                entrypoint = build_external_event_resume_entrypoint(
                    resume_material=lambda workdir, tid, event: run_single_material_external_event(
                        runtime=runtime,
                        workdir=workdir,
                        thread_id=tid,
                        event=event,
                    ).model_dump(mode="json")
                )
                result = entrypoint.invoke(
                    {
                        "workdir": waiting.workdir,
                        "thread_id": thread_id,
                        "event": {
                            "event_id": "evt-job-complete-1",
                            "event_type": "job_completed",
                            "thread_id": thread_id,
                            "job_id": "job-1",
                            "target_capability": "relax",
                            "action_family": "run_capability",
                            "artifact_paths": {"OUTCAR": os.path.join(waiting.workdir, "01_relax", "OUTCAR")},
                            "provenance": "test",
                        },
                    }
            )
            self.assertEqual(result["event"]["event_type"], "job_completed")
            self.assertIn(result["outcome"]["status"], {"aborted", "failed"})
            snapshot_path = runtime_state_snapshot_path(waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["execution"]["latest_event"]["event_id"], "evt-job-complete-1")
            self.assertIn("evt-job-complete-1", snapshot["execution"]["consumed_event_ids"])

    def test_job_failed_event_moves_run_out_of_waiting_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _scripted_arbitrate):
                waiting = run_single_material(runtime=runtime, material_id="evt-failed", root_path=tmpdir, fresh=True)
                resumed = run_single_material_external_event(
                    runtime=runtime,
                    workdir=waiting.workdir,
                    event={
                        "event_id": "evt-job-failed-1",
                        "event_type": "job_failed",
                        "job_id": "job-1",
                        "target_capability": "relax",
                        "action_family": "run_capability",
                        "error_summary": "scheduler_reported_failure",
                        "provenance": "test",
                    },
                )
            self.assertEqual(resumed.status, "aborted")
            snapshot_path = runtime_state_snapshot_path(waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["execution"]["latest_event"]["event_type"], "job_failed")
            self.assertEqual(snapshot["workflow"]["run_status"], "aborted")

    def test_duplicate_event_is_ignored_after_it_has_been_consumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _scripted_arbitrate):
                waiting = run_single_material(runtime=runtime, material_id="evt-dup", root_path=tmpdir, fresh=True)
                thread_id = load_thread_id(workdir=waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
                self.assertIsNotNone(thread_id)
                graph = build_material_graph(_build_nodes(runtime))
                config = {"configurable": {"thread_id": thread_id}}
                with open_sqlite_checkpointer(
                    workdir=waiting.workdir,
                    checkpoint_subdir=runtime.checkpoint_subdir,
                ) as checkpointer:
                    with open_memory_store(runtime.store_path) as store:
                        app = graph.compile(checkpointer=checkpointer, store=store)
                        persisted = MaterialTaskState.from_dict(app.get_state(config).values).to_dict()
                        updated = MaterialTaskState.from_dict(persisted).to_dict()
                        updated["execution"]["consumed_event_ids"] = list(updated["execution"].get("consumed_event_ids", []) or []) + [
                            "evt-resume-1"
                        ]
                        app.update_state(
                            config,
                            build_state_patch(persisted, updated, sections=("execution",)),
                            as_node="observe_state",
                        )
                second = run_single_material_external_event(
                    runtime=runtime,
                    workdir=waiting.workdir,
                    event={
                        "event_id": "evt-resume-1",
                        "event_type": "resume_requested",
                        "job_id": "job-1",
                        "target_capability": "relax",
                        "action_family": "run_capability",
                        "provenance": "test",
                    },
                )
            self.assertEqual(second.status, "waiting_external")
            snapshot_path = runtime_state_snapshot_path(waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            codes = [item["code"] for item in snapshot["services"]["framework_diagnostics"]]
            self.assertIn("duplicate_external_event_ignored", codes)

    def test_stale_job_event_is_ignored_while_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _scripted_arbitrate):
                waiting = run_single_material(runtime=runtime, material_id="evt-stale", root_path=tmpdir, fresh=True)
                stale = run_single_material_external_event(
                    runtime=runtime,
                    workdir=waiting.workdir,
                    event={
                        "event_id": "evt-stale-1",
                        "event_type": "job_completed",
                        "job_id": "missing-job",
                        "target_capability": "relax",
                        "action_family": "run_capability",
                        "provenance": "test",
                    },
                )
            self.assertEqual(stale.status, "waiting_external")
            snapshot_path = runtime_state_snapshot_path(waiting.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            codes = [item["code"] for item in snapshot["services"]["framework_diagnostics"]]
            self.assertIn("stale_external_event_ignored", codes)

    def test_completed_run_refuses_external_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            with patch_test_llm_clients():
                completed = run_single_material(runtime=runtime, material_id="evt-terminal", root_path=tmpdir, fresh=True)
                self.assertEqual(completed.status, "completed")
                refused = run_single_material_external_event(
                    runtime=runtime,
                    workdir=completed.workdir,
                    event={
                        "event_id": "evt-after-terminal",
                        "event_type": "resume_requested",
                        "provenance": "test",
                    },
                )
            self.assertEqual(refused.status, "completed")
            snapshot_path = runtime_state_snapshot_path(completed.workdir, checkpoint_subdir=runtime.checkpoint_subdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            codes = [item["code"] for item in snapshot["services"]["framework_diagnostics"]]
            self.assertIn("external_event_resume_refused_terminal_state", codes)


if __name__ == "__main__":
    unittest.main()
