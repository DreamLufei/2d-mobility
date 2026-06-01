from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from dotenv import dotenv_values
from fastapi.testclient import TestClient

from mobility_agent.runtime.checkpointing import write_json_atomic
from mobility_agent.web_console.api import create_app
from mobility_agent.web_console.config import WebConsoleSettings
from mobility_agent.web_console.registry import _MEMORY_REGISTRIES
from mobility_agent.web_console.service import WebConsoleService, _job_result_path, _process_group_alive, _utc_now_iso


def _write_text(path: str, payload: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def _fake_material_run(
    root: str,
    material_id: str,
    *,
    status: str = "completed",
    parent_batch_id: str | None = None,
    final_acceptance: str | None = None,
    quality_grade: str | None = None,
) -> str:
    material_root = os.path.join(root, material_id)
    workdir = os.path.join(material_root, "mobility_calculation")
    runtime_dir = os.path.join(workdir, ".runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    diagnostics_payload = {"last_error": "scf_failed" if status != "completed" else None}
    if final_acceptance or quality_grade:
        diagnostics_payload["validation_report"] = {
            "decision": str(final_acceptance or "pass_with_warning"),
            **({"quality_grade": str(quality_grade)} if quality_grade else {}),
        }
        if quality_grade:
            diagnostics_payload["quality_grade"] = str(quality_grade)

    shared_state = {
        "task": {
            "task_id": f"task::{material_id}",
            "task_type": "single_material",
            "parent_batch_id": parent_batch_id,
            "updated_at": _utc_now_iso(),
        },
        "material": {"material_id": material_id},
        "workflow": {
            "current_stage": "final_report" if status == "completed" else "scf",
            "run_status": status,
            "stage_status": {"prepare": "success", "relax": "success", "scf": "failed" if status != "completed" else "success"},
            "wait_reason": None,
        },
        "execution": {
            "thread_id": f"material::{material_id}::{material_id}::1234567890ab",
            "artifact_paths": {
                "final_summary_path": os.path.join(workdir, "final_summary.json"),
                "material_outcome_path": os.path.join(workdir, "material_outcome.json"),
            },
            "artifact_registry": {
                "final_summary_path": os.path.join(workdir, "final_summary.json"),
                "material_outcome_path": os.path.join(workdir, "material_outcome.json"),
            },
        },
        "diagnostics": diagnostics_payload,
    }
    ui_state = {
        "task_id": f"task::{material_id}",
        "material_id": material_id,
        "thread_id": shared_state["execution"]["thread_id"],
        "current_stage": shared_state["workflow"]["current_stage"],
        "runtime_run_status": status,
        "stage_status": dict(shared_state["workflow"]["stage_status"]),
        "selected_action": {"action_family": "run_capability", "target_capability": "scf"},
        "hitl_pending": False,
        "wait_reason": None,
        "latest_error": shared_state["diagnostics"]["last_error"],
        "artifact_paths": dict(shared_state["execution"]["artifact_paths"]),
        "updated_at": _utc_now_iso(),
    }
    write_json_atomic(os.path.join(runtime_dir, "shared_state.json"), shared_state)
    write_json_atomic(os.path.join(runtime_dir, "ui_state.json"), ui_state)
    _write_text(
        os.path.join(runtime_dir, "ui_events.jsonl"),
        json.dumps(
            {
                "timestamp": _utc_now_iso(),
                "event_type": "state_initialized",
                "current_stage": ui_state["current_stage"],
                "runtime_run_status": status,
                "selected_action_family": "run_capability",
                "selected_capability": "scf",
                "stage_status": "failed" if status != "completed" else "success",
                "hitl_pending": False,
                "wait_reason": None,
                "latest_error": ui_state["latest_error"],
                "latest_artifact_keys": ["final_summary_path", "material_outcome_path"],
            }
        )
        + "\n",
    )
    _write_text(os.path.join(runtime_dir, "thread_id.txt"), shared_state["execution"]["thread_id"] + "\n")
    write_json_atomic(
        os.path.join(workdir, "final_summary.json"),
        {
            "material_id": material_id,
            "status": status,
            **({"final_acceptance": str(final_acceptance)} if final_acceptance else {}),
            **({"quality_grade": str(quality_grade)} if quality_grade else {}),
        },
    )
    write_json_atomic(
        os.path.join(workdir, "material_outcome.json"),
        {
            "material_id": material_id,
            "status": status,
            "final_status": status,
            **({"final_acceptance": str(final_acceptance)} if final_acceptance else {}),
            "workdir": workdir,
            "artifact_paths": dict(shared_state["execution"]["artifact_paths"]),
            "errors": ["scf_failed"] if status != "completed" else [],
            "stage_status": dict(shared_state["workflow"]["stage_status"]),
            "validation_report": {
                **({"decision": str(final_acceptance)} if final_acceptance else {}),
                **({"quality_grade": str(quality_grade)} if quality_grade else {}),
            },
            "final_summary": {
                "material_id": material_id,
                "run_status": status,
                **({"final_acceptance": str(final_acceptance)} if final_acceptance else {}),
                **({"quality_grade": str(quality_grade)} if quality_grade else {}),
            },
        },
    )
    return workdir


class WebConsoleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        _MEMORY_REGISTRIES.clear()
        self._previous_mobility_db_uri = os.environ.pop("MOBILITY_DB_URI", None)

    def tearDown(self) -> None:
        if self._previous_mobility_db_uri is None:
            os.environ.pop("MOBILITY_DB_URI", None)
        else:
            os.environ["MOBILITY_DB_URI"] = self._previous_mobility_db_uri

    def test_worker_env_includes_repo_root_on_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir])
            service = WebConsoleService(settings)
            service._worker_shell_env = {}
            previous = os.environ.get("PYTHONPATH")
            try:
                os.environ["PYTHONPATH"] = "/tmp/existing-pythonpath"
                env = service._worker_env()
            finally:
                if previous is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous
            self.assertTrue(env["PYTHONPATH"].startswith(os.path.abspath(tmpdir)))
            self.assertIn("/tmp/existing-pythonpath", env["PYTHONPATH"])

    def test_worker_env_merges_shell_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir])
            service = WebConsoleService(settings)
            shell_payload = (
                b"PATH=/opt/hpc/bin:/usr/bin\0"
                b"LD_LIBRARY_PATH=/opt/hpc/lib\0"
                b"NVHPC_ROOT=/opt/nvhpc\0"
                b"UNRELATED=value\0"
            )
            completed = subprocess.CompletedProcess(
                args=["bash", "-lc", "env -0"],
                returncode=0,
                stdout=shell_payload,
                stderr=b"",
            )
            with patch("mobility_agent.web_console.service.subprocess.run", return_value=completed):
                env = service._worker_env()
            self.assertEqual(env.get("PATH"), "/opt/hpc/bin:/usr/bin")
            self.assertEqual(env.get("LD_LIBRARY_PATH"), "/opt/hpc/lib")
            self.assertEqual(env.get("NVHPC_ROOT"), "/opt/nvhpc")
            self.assertNotIn("UNRELATED", env)

    def test_runtime_settings_round_trip_updates_env_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_text(
                os.path.join(tmpdir, ".env.local"),
                "\n".join(
                    [
                        "MOBILITY_DB_URI=postgresql://mobility:old-db-secret@db.example:5432/mobility",
                        "LLM_PROVIDER=openai",
                        "LLM_BASE_URL=https://openrouter.ai/api/v1",
                        "LLM_MODEL=minimax/minimax-m2.5",
                        "LLM_API_KEY=old-secret",
                        "EMBEDDING_MODEL=text-embedding-3-small",
                        "EMBEDDING_API_KEY=old-embedding-secret",
                        "ENABLE_EMAIL_NOTIFICATIONS=true",
                        "SMTP_PASSWORD=old-smtp",
                    ]
                )
                + "\n",
            )
            app = create_app(WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir]))
            with TestClient(app) as client:
                settings = client.get("/api/settings/runtime")
                self.assertEqual(settings.status_code, 200)
                payload = settings.json()
                self.assertEqual(payload["service_preset"], "openrouter")
                self.assertEqual(payload["mobility_db_uri"], "postgresql://mobility:***@db.example:5432/mobility")
                self.assertTrue(payload["llm_api_key_present"])
                self.assertEqual(payload["embedding_model"], "text-embedding-3-small")
                self.assertTrue(payload["embedding_api_key_present"])
                self.assertTrue(payload["smtp_password_present"])

                updated = client.post(
                    "/api/settings/runtime",
                    json={
                        "mobility_db_uri": "postgresql://mobility:new-db-secret@db2.example:5432/mobility",
                        "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "llm_model": "qwen3.6-plus",
                        "llm_api_key": "new-secret",
                        "embedding_model": "text-embedding-v4",
                        "embedding_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "embedding_api_key": "new-embedding-secret",
                        "wiki_qa_model": "qwen3.6-plus",
                        "rag_top_k": 8,
                        "enable_email_notifications": False,
                        "smtp_password": "new-smtp",
                    },
                )
                self.assertEqual(updated.status_code, 200)
                response = updated.json()
                self.assertEqual(response["service_preset"], "qwen")
                self.assertEqual(response["mobility_db_uri"], "postgresql://mobility:***@db2.example:5432/mobility")
                self.assertEqual(response["llm_model"], "qwen3.6-plus")
                self.assertTrue(response["llm_api_key_present"])
                self.assertEqual(response["embedding_model"], "text-embedding-v4")
                self.assertEqual(response["embedding_base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
                self.assertTrue(response["embedding_api_key_present"])
                self.assertEqual(response["wiki_qa_model"], "qwen3.6-plus")
                self.assertEqual(response["rag_top_k"], 8)
                self.assertTrue(response["smtp_password_present"])

                env_values = dotenv_values(os.path.join(tmpdir, ".env.local"))
                self.assertEqual(env_values.get("MOBILITY_DB_URI"), "postgresql://mobility:new-db-secret@db2.example:5432/mobility")
                self.assertEqual(env_values.get("LLM_MODEL"), "qwen3.6-plus")
                self.assertEqual(env_values.get("LLM_API_KEY"), "new-secret")
                self.assertEqual(env_values.get("EMBEDDING_MODEL"), "text-embedding-v4")
                self.assertEqual(env_values.get("EMBEDDING_BASE_URL"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
                self.assertEqual(env_values.get("EMBEDDING_API_KEY"), "new-embedding-secret")
                self.assertEqual(env_values.get("WIKI_QA_MODEL"), "qwen3.6-plus")
                self.assertEqual(env_values.get("RAG_TOP_K"), "8")
                self.assertEqual(env_values.get("ENABLE_EMAIL_NOTIFICATIONS"), "false")
                self.assertEqual(env_values.get("SMTP_PASSWORD"), "new-smtp")

    def test_runtime_settings_detect_qwen_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_text(
                os.path.join(tmpdir, ".env.local"),
                "\n".join(
                    [
                        "MOBILITY_DB_URI=postgresql://mobility:db-secret@db.example:5432/mobility",
                        "LLM_PROVIDER=openai",
                        "LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "LLM_MODEL=qwen3.6-plus",
                        "LLM_API_KEY=test-secret",
                        "EMBEDDING_MODEL=text-embedding-v4",
                    ]
                )
                + "\n",
            )
            app = create_app(WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir]))
            with TestClient(app) as client:
                settings = client.get("/api/settings/runtime")
                self.assertEqual(settings.status_code, 200)
                payload = settings.json()
                self.assertEqual(payload["service_preset"], "qwen")
                self.assertEqual(payload["llm_base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
                self.assertEqual(payload["llm_model"], "qwen3.6-plus")
                self.assertTrue(payload["llm_api_key_present"])

    def test_wiki_endpoints_delegate_to_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir]))
            with TestClient(app) as client:
                service = client.app.state.console_service
                with (
                    patch.object(service, "wiki_health", return_value={"status": "ok", "document_count": 11}) as mock_health,
                    patch.object(
                        service,
                        "wiki_query",
                        return_value={
                            "query": "What does ENCUT control?",
                            "answer": "ENCUT controls the plane-wave cutoff energy.",
                            "citations": [
                                {
                                    "corpus": "vasp_wiki",
                                    "source_id": "ENCUT",
                                    "chunk_id": "chunk-1",
                                    "revision_id": "123",
                                    "title": "ENCUT",
                                    "heading": "ENCUT",
                                    "url": "https://www.vasp.at/wiki/ENCUT",
                                    "snippet": "ENCUT determines the kinetic energy cutoff.",
                                    "score": 0.93,
                                    "stage": "scf",
                                    "tags": ["scf"],
                                }
                            ],
                            "retrieval_metadata": {"top_k": 4},
                        },
                    ) as mock_query,
                    patch.object(
                        service,
                        "create_wiki_reindex_job",
                        return_value={"job_id": "wiki-job-1", "job_type": "wiki_reindex", "current_stage": "wiki_reindex"},
                    ) as mock_reindex,
                ):
                    health = client.get("/api/wiki/health")
                    self.assertEqual(health.status_code, 200)
                    self.assertEqual(health.json()["document_count"], 11)

                    query = client.post(
                        "/api/wiki/query",
                        json={"query": "What does ENCUT control?", "top_k": 4, "corpora": ["vasp_wiki"], "stage": "scf"},
                    )
                    self.assertEqual(query.status_code, 200)
                    self.assertIn("ENCUT controls", query.json()["answer"])

                    reindex = client.post("/api/wiki/reindex", json={"mode": "incremental", "max_pages": 25})
                    self.assertEqual(reindex.status_code, 200)
                    self.assertEqual(reindex.json()["job_type"], "wiki_reindex")

                mock_health.assert_called_once()
                query_args, _ = mock_query.call_args
                self.assertEqual(query_args[0].query, "What does ENCUT control?")
                self.assertEqual(query_args[0].top_k, 4)
                self.assertEqual(query_args[0].corpora, ["vasp_wiki"])
                self.assertEqual(query_args[0].stage, "scf")
                reindex_args, _ = mock_reindex.call_args
                self.assertEqual(reindex_args[0].mode, "incremental")
                self.assertEqual(reindex_args[0].max_pages, 25)

    def test_frontend_assets_are_served_from_root_asset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = os.path.join(tmpdir, "web_console", "frontend", "dist")
            assets_dir = os.path.join(dist_dir, "assets")
            os.makedirs(assets_dir, exist_ok=True)
            _write_text(
                os.path.join(dist_dir, "index.html"),
                """<!doctype html>
<html>
  <head>
    <script type="module" src="/assets/app.js"></script>
    <link rel="stylesheet" href="/assets/app.css">
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
""",
            )
            _write_text(os.path.join(assets_dir, "app.js"), "console.log('ok');\n")
            _write_text(os.path.join(assets_dir, "app.css"), "body { color: black; }\n")

            app = create_app(WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir]))
            with TestClient(app) as client:
                root = client.get("/")
                self.assertEqual(root.status_code, 200)
                self.assertIn('/assets/app.js', root.text)

                spa = client.get("/app/jobs")
                self.assertEqual(spa.status_code, 200)
                self.assertIn('/assets/app.js', spa.text)

                js = client.get("/assets/app.js")
                self.assertEqual(js.status_code, 200)
                self.assertIn("console.log('ok')", js.text)

    def test_reconciliation_imports_material_runs_and_batch_parent_child_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_root = os.path.join(tmpdir, "batch_runs")
            os.makedirs(batch_root, exist_ok=True)
            _fake_material_run(batch_root, "mat-a", status="failed", parent_batch_id="demo-batch")
            _fake_material_run(batch_root, "mat-b", status="completed", parent_batch_id="demo-batch")
            write_json_atomic(
                os.path.join(batch_root, "batch_summary_demo-batch.json"),
                {"batch_tag": "demo-batch", "processed": 2, "failed": 1, "succeeded": 1},
            )
            settings = WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir])
            service = WebConsoleService(settings)
            service.reconcile_once(True)
            jobs = service.list_job_snapshots()
            self.assertGreaterEqual(len(jobs), 3)
            parent = next(job for job in jobs if job["job_role"] == "batch_parent")
            self.assertEqual(parent["batch_tag"], "demo-batch")
            detail = service.get_job_detail(parent["job_id"])
            self.assertIsNotNone(detail)
            self.assertEqual(len(detail["children"]), 2)
            self.assertEqual(detail["failure_taxonomy"]["scf"], 1)

    def test_reconciliation_exposes_quality_signals_without_overwriting_completed_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _fake_material_run(
                tmpdir,
                "mat-quality-reject",
                status="completed",
                final_acceptance="fail",
                quality_grade="low_confidence",
            )
            service = WebConsoleService(WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir]))
            service.reconcile_once(True)
            jobs = service.list_job_snapshots()
            item = next(job for job in jobs if job.get("material_id") == "mat-quality-reject")
            self.assertEqual(item["runtime_run_status"], "completed")
            self.assertEqual(item.get("final_acceptance"), "fail")
            self.assertEqual(item.get("quality_grade"), "low_confidence")
            detail = service.get_job_detail(item["job_id"])
            self.assertIsNotNone(detail)
            self.assertEqual(detail.get("runtime_run_status"), "completed")
            self.assertEqual(detail.get("final_acceptance"), "fail")
            self.assertEqual(detail.get("quality_grade"), "low_confidence")

    def test_cancel_job_terminates_process_group(self) -> None:
        if os.name != "posix":
            self.skipTest("process-group cancellation is only supported on POSIX")
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "spawn_child.py")
            child_pid_path = os.path.join(tmpdir, "child.pid")
            _write_text(
                script_path,
                "\n".join(
                    [
                        "import os, subprocess, sys, time",
                        f"child_path = {child_pid_path!r}",
                        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
                        "with open(child_path, 'w', encoding='utf-8') as handle:",
                        "    handle.write(str(child.pid))",
                        "    handle.flush()",
                        "time.sleep(30)",
                    ]
                ),
            )
            proc = subprocess.Popen([sys.executable, script_path], start_new_session=True)
            pgid = os.getpgid(proc.pid)
            deadline = time.time() + 5
            while not os.path.exists(child_pid_path) and time.time() < deadline:
                time.sleep(0.05)
            self.assertTrue(os.path.exists(child_pid_path))
            with open(child_pid_path, "r", encoding="utf-8") as handle:
                child_pid = int(handle.read().strip())
            settings = WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir])
            service = WebConsoleService(settings)
            now = _utc_now_iso()
            service.registry.upsert_job(
                {
                    "job_id": "kill-me",
                    "job_type": "single_material",
                    "job_role": "standalone",
                    "parent_job_id": None,
                    "display_name": "kill-me",
                    "material_id": "kill-me",
                    "batch_tag": None,
                    "root_path": tmpdir,
                    "workdir": tmpdir,
                    "thread_id": None,
                    "pid": proc.pid,
                    "pgid": pgid,
                    "launch_cmd": "dummy",
                    "exit_code": None,
                    "signal": None,
                    "status_source": "control_plane",
                    "runtime_run_status": "running",
                    "control_plane_status": "live",
                    "current_stage": "scf",
                    "hitl_pending": False,
                    "wait_reason": None,
                    "error_summary": None,
                    "last_progress_line": None,
                    "last_state_updated_at": None,
                    "last_heartbeat_at": now,
                    "created_at": now,
                    "started_at": now,
                    "finished_at": None,
                    "job_request_json": {},
                    "updated_at": now,
                }
            )
            detail = service.cancel_job("kill-me")
            self.assertEqual(detail["control_plane_status"], "cancelled")
            self.assertFalse(_process_group_alive(pgid, proc.pid))
            proc.wait(timeout=5)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    def test_api_health_download_and_cancelled_vs_aborted_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = _fake_material_run(tmpdir, "mat-api", status="aborted")
            settings = WebConsoleSettings.from_repo(tmpdir, job_roots=[tmpdir])
            service = WebConsoleService(settings)
            now = _utc_now_iso()
            service.registry.upsert_job(
                {
                    "job_id": "aborted-job",
                    "job_type": "single_material",
                    "job_role": "standalone",
                    "parent_job_id": None,
                    "display_name": "aborted-job",
                    "material_id": "mat-api",
                    "batch_tag": None,
                    "root_path": os.path.dirname(workdir),
                    "workdir": workdir,
                    "thread_id": None,
                    "pid": None,
                    "pgid": None,
                    "launch_cmd": None,
                    "exit_code": None,
                    "signal": None,
                    "status_source": "imported_scan",
                    "runtime_run_status": "pending",
                    "control_plane_status": "archived",
                    "current_stage": None,
                    "hitl_pending": False,
                    "wait_reason": None,
                    "error_summary": None,
                    "last_progress_line": None,
                    "last_state_updated_at": None,
                    "last_heartbeat_at": None,
                    "created_at": now,
                    "started_at": None,
                    "finished_at": None,
                    "job_request_json": {},
                    "updated_at": now,
                }
            )
            cancelled_root = os.path.join(tmpdir, "cancelled")
            cancelled_workdir = _fake_material_run(cancelled_root, "mat-cancel", status="running")
            service.registry.upsert_job(
                {
                    "job_id": "cancelled-job",
                    "job_type": "single_material",
                    "job_role": "standalone",
                    "parent_job_id": None,
                    "display_name": "cancelled-job",
                    "material_id": "mat-cancel",
                    "batch_tag": None,
                    "root_path": os.path.dirname(cancelled_workdir),
                    "workdir": cancelled_workdir,
                    "thread_id": None,
                    "pid": None,
                    "pgid": None,
                    "launch_cmd": None,
                    "exit_code": -signal.SIGTERM,
                    "signal": "SIGTERM",
                    "status_source": "control_plane",
                    "runtime_run_status": "running",
                    "control_plane_status": "cancelled",
                    "current_stage": "scf",
                    "hitl_pending": False,
                    "wait_reason": None,
                    "error_summary": None,
                    "last_progress_line": None,
                    "last_state_updated_at": None,
                    "last_heartbeat_at": None,
                    "created_at": now,
                    "started_at": None,
                    "finished_at": now,
                    "job_request_json": {},
                    "updated_at": now,
                }
            )

            app = create_app(settings)
            with TestClient(app) as client:
                health = client.get("/api/health")
                self.assertEqual(health.status_code, 200)
                jobs = client.get("/api/jobs").json()
                aborted = next(job for job in jobs if job["job_id"] == "aborted-job")
                cancelled = next(job for job in jobs if job["job_id"] == "cancelled-job")
                self.assertEqual(aborted["runtime_run_status"], "aborted")
                self.assertEqual(aborted["control_plane_status"], "archived")
                self.assertEqual(cancelled["control_plane_status"], "cancelled")
                self.assertEqual(cancelled["runtime_run_status"], "running")
                detail = client.get("/api/jobs/aborted-job")
                self.assertEqual(detail.status_code, 200)
                state = client.get("/api/jobs/aborted-job/state")
                self.assertEqual(state.status_code, 200)
                timeline = client.get("/api/jobs/aborted-job/timeline")
                self.assertEqual(timeline.status_code, 200)
                artifacts = client.get("/api/jobs/aborted-job/artifacts").json()
                self.assertIn("final_summary_path", artifacts)
                download = client.get("/api/jobs/aborted-job/download/final_summary_path")
                self.assertEqual(download.status_code, 200)
                missing = client.get("/api/jobs/aborted-job/download/not_allowed")
                self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
