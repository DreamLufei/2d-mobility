from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from ..rag import VaspWikiRagService
from ..hitl.escalation import write_human_response
from ..runtime.batch_config import BatchConfig, load_config
from ..runtime.checkpointing import (
    append_ui_event,
    load_state_snapshot,
    load_thread_id,
    load_ui_state_snapshot,
    runtime_state_snapshot_path,
    runtime_ui_events_path,
)
from ..runtime.context import RuntimeContext
from ..runtime.entrypoints import build_external_event_resume_entrypoint
from ..runtime.runner import default_material_workdir, run_single_material_external_event
from .config import WebConsoleSettings
from .models import (
    ARTIFACT_FILENAME_WHITELIST,
    CONTROL_PLANE_STATUSES,
    RUNTIME_RUN_STATUSES,
    TERMINAL_RUNTIME_STATUSES,
    BatchConfigPayload,
    BatchJobRequest,
    ExternalEventResumeRequest,
    HitlResponseRequest,
    JobSnapshot,
    RuntimeSettingsUpdateRequest,
    SingleJobRequest,
    WikiQueryRequest,
    WikiReindexRequest,
    WorkerJobSpec,
    imported_job_id,
)
from .registry import ControlPlaneRegistry
from .runtime_settings import RuntimeSettingsStore

_WORKER_SHELL_ENV_KEYS = {
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "PKG_CONFIG_PATH",
    "MANPATH",
    "CMAKE_PREFIX_PATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CUDA_HOME",
    "CUDA_ROOT",
    "MKLROOT",
    "NVHPC_ROOT",
    "NV_VERSION",
    "ONEAPI_ROOT",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _safe_json_value(path: str) -> Any | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_jsonl(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if limit is not None:
        lines = lines[-max(0, int(limit)) :]
    payloads: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            payloads.append(item)
    return payloads


def _safe_text_lines(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle.readlines()]


def _load_worker_shell_env() -> dict[str, str]:
    try:
        proc = subprocess.run(
            # The user's HPC/MPI exports live behind an interactive-shell guard in ~/.bashrc.
            ["bash", "-ic", "env -0"],
            check=False,
            capture_output=True,
        )
    except Exception:
        return {}
    if proc.returncode != 0 or not proc.stdout:
        return {}
    payload: dict[str, str] = {}
    for raw_item in proc.stdout.split(b"\0"):
        if not raw_item or b"=" not in raw_item:
            continue
        raw_key, raw_value = raw_item.split(b"=", 1)
        try:
            key = raw_key.decode("utf-8")
            value = raw_value.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if key in _WORKER_SHELL_ENV_KEYS and value:
            payload[key] = value
    return payload


def _process_group_alive(pgid: int | None, pid: int | None = None) -> bool:
    def _has_live_process_in_group(group_id: int) -> bool | None:
        try:
            proc = subprocess.run(
                ["ps", "-o", "pgid=,stat=", "-ax"],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                parsed_pgid = int(parts[0])
            except Exception:
                continue
            if parsed_pgid != int(group_id):
                continue
            state = parts[1].strip()
            if state and not state.startswith("Z"):
                return True
        return False

    def _has_live_process(process_id: int) -> bool | None:
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(process_id)],
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        states = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return any(not state.startswith("Z") for state in states)

    if pgid:
        live_group = _has_live_process_in_group(int(pgid))
        if live_group is not None:
            return live_group
        try:
            os.killpg(int(pgid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    if pid:
        live_pid = _has_live_process(int(pid))
        if live_pid is not None:
            return live_pid
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    return False


def _job_result_path(settings: WebConsoleSettings, job_id: str) -> str:
    return os.path.join(settings.results_dir, f"{job_id}.json")


def _job_spec_path(settings: WebConsoleSettings, job_id: str) -> str:
    return os.path.join(settings.specs_dir, f"{job_id}.json")


def _normalize_runtime_status(value: str | None) -> str:
    text = str(value or "").strip()
    return text if text in RUNTIME_RUN_STATUSES else "pending"


def _normalize_control_plane_status(value: str | None) -> str:
    text = str(value or "").strip()
    return text if text in CONTROL_PLANE_STATUSES else "queued"


def _root_path_from_workdir(workdir: str) -> str:
    path = os.path.abspath(workdir)
    return os.path.dirname(path) if os.path.basename(path) == "mobility_calculation" else path


def _material_artifacts(
    *,
    workdir: str,
    ui_state: dict[str, Any] | None,
    shared_state: dict[str, Any] | None,
    outcome: dict[str, Any] | None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if isinstance(ui_state, dict):
        paths.update({str(k): str(v) for k, v in dict(ui_state.get("artifact_paths", {}) or {}).items() if v})
    execution = dict((shared_state or {}).get("execution", {}) or {})
    paths.update({str(k): str(v) for k, v in dict(execution.get("artifact_paths", {}) or {}).items() if v})
    paths.update({str(k): str(v) for k, v in dict(execution.get("artifact_registry", {}) or {}).items() if v})
    if isinstance(outcome, dict):
        paths.update({str(k): str(v) for k, v in dict(outcome.get("artifact_paths", {}) or {}).items() if v})
    for filename, artifact_key in ARTIFACT_FILENAME_WHITELIST.items():
        candidate = os.path.join(workdir, filename)
        if os.path.exists(candidate):
            paths.setdefault(artifact_key, candidate)
    return paths


def _quality_signals(
    *,
    outcome: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    shared_state: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    outcome_payload = dict(outcome or {})
    summary_payload = dict(summary or {})
    diagnostics = dict((shared_state or {}).get("diagnostics", {}) or {})
    validation = dict(diagnostics.get("validation_report", {}) or {})
    outcome_validation = dict(outcome_payload.get("validation_report", {}) or {})
    final_acceptance = (
        str(outcome_payload.get("final_acceptance") or "").strip()
        or str(summary_payload.get("final_acceptance") or "").strip()
        or str(outcome_validation.get("decision") or "").strip()
        or str(validation.get("decision") or "").strip()
        or None
    )
    quality_grade = (
        str(outcome_validation.get("quality_grade") or "").strip()
        or str(summary_payload.get("quality_grade") or "").strip()
        or str(validation.get("quality_grade") or "").strip()
        or str(diagnostics.get("quality_grade") or "").strip()
        or None
    )
    return final_acceptance, quality_grade


class WebConsoleService:
    def __init__(self, settings: WebConsoleSettings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.registry = ControlPlaneRegistry(settings.database_uri)
        self.runtime_settings = RuntimeSettingsStore(settings.repo_root)
        self._loop_task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._jobs_ws: set[WebSocket] = set()
        self._detail_ws: dict[str, set[WebSocket]] = {}
        self._last_poll_success_at: str | None = None
        self._scan_counter = 0
        self._worker_shell_env: dict[str, str] | None = None

    async def start(self) -> None:
        await asyncio.to_thread(self.reconcile_once, True)
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
        for ws in list(self._jobs_ws):
            with suppress(Exception):
                await ws.close()
        for ws_set in list(self._detail_ws.values()):
            for ws in list(ws_set):
                with suppress(Exception):
                    await ws.close()

    async def register_jobs_ws(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._jobs_ws.add(websocket)
        await websocket.send_json({"type": "jobs_snapshot", "jobs": self.list_job_snapshots(), "updated_at": _utc_now_iso()})

    async def register_detail_ws(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._detail_ws.setdefault(job_id, set()).add(websocket)
        detail = self.get_job_detail(job_id)
        await websocket.send_json({"type": "job_snapshot", "job_id": job_id, "detail": detail, "updated_at": _utc_now_iso()})

    def unregister_ws(self, websocket: WebSocket) -> None:
        self._jobs_ws.discard(websocket)
        for ws_set in self._detail_ws.values():
            ws_set.discard(websocket)

    async def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_counter += 1
                full_scan = self._scan_counter % 10 == 1
                await asyncio.to_thread(self.reconcile_once, full_scan)
                self._last_poll_success_at = _utc_now_iso()
                await self._broadcast_snapshots()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(self.settings.poll_interval_s)

    async def _broadcast_snapshots(self) -> None:
        jobs_payload = {"type": "jobs_snapshot", "jobs": self.list_job_snapshots(), "updated_at": _utc_now_iso()}
        for ws in list(self._jobs_ws):
            try:
                await ws.send_json(jobs_payload)
            except Exception:
                self.unregister_ws(ws)
        for job_id, sockets in list(self._detail_ws.items()):
            detail = self.get_job_detail(job_id)
            payload = {"type": "job_snapshot", "job_id": job_id, "detail": detail, "updated_at": _utc_now_iso()}
            for ws in list(sockets):
                try:
                    await ws.send_json(payload)
                except Exception:
                    self.unregister_ws(ws)

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self._worker_shell_env is None:
            self._worker_shell_env = _load_worker_shell_env()
        if self._worker_shell_env:
            env.update(self._worker_shell_env)
        env["MOBILITY_WEB_RESULTS_DIR"] = self.settings.results_dir
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        repo_root = os.path.abspath(self.settings.repo_root)
        env["PYTHONPATH"] = repo_root if not existing_pythonpath else os.pathsep.join([repo_root, existing_pythonpath])
        return env

    def _build_batch_config(self, request: BatchJobRequest) -> BatchConfig:
        if request.config is not None:
            payload = request.config.model_dump(mode="python", exclude_none=True)
            return BatchConfig(**payload)
        cfg = load_config()
        overrides = request.config_overrides.model_dump(mode="python", exclude_none=True)
        return BatchConfig(**{**cfg.__dict__, **overrides}) if overrides else cfg

    def _spawn_worker(self, job_id: str, spec: WorkerJobSpec, *, cwd: str) -> tuple[int, int, str]:
        spec_path = _job_spec_path(self.settings, job_id)
        with open(spec_path, "w", encoding="utf-8") as handle:
            handle.write(spec.model_dump_json(indent=2))
        cmd = [self.settings.python_executable, "-m", "mobility_agent.web_console.worker_main", "--job-spec", spec_path]
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=self._worker_env(),
            start_new_session=True,
        )
        pgid = proc.pid
        with suppress(Exception):
            pgid = os.getpgid(proc.pid)
        return proc.pid, int(pgid), shlex.join(cmd)

    def create_single_job(self, request: SingleJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        root_path = os.path.abspath(request.root_path)
        workdir = os.path.abspath(request.workdir or default_material_workdir(root_path))
        display_name = request.display_name or request.material_id or os.path.basename(root_path) or job_id
        spec = WorkerJobSpec(job_id=job_id, job_type="single_material", request=request.model_dump(mode="json"))
        pid, pgid, launch_cmd = self._spawn_worker(job_id, spec, cwd=root_path)
        now = _utc_now_iso()
        self.registry.upsert_job(
            {
                "job_id": job_id,
                "job_type": "single_material",
                "job_role": "standalone",
                "parent_job_id": None,
                "display_name": display_name,
                "material_id": request.material_id or os.path.basename(root_path) or None,
                "batch_tag": None,
                "root_path": root_path,
                "workdir": workdir,
                "thread_id": None,
                "pid": pid,
                "pgid": pgid,
                "launch_cmd": launch_cmd,
                "exit_code": None,
                "signal": None,
                "status_source": "control_plane",
                "runtime_run_status": "pending",
                "control_plane_status": "starting",
                "current_stage": None,
                "hitl_pending": False,
                "wait_reason": None,
                "error_summary": None,
                "last_progress_line": None,
                "last_state_updated_at": None,
                "last_heartbeat_at": now,
                "created_at": now,
                "started_at": now,
                "finished_at": None,
                "job_request_json": request.model_dump(mode="json"),
                "updated_at": now,
            }
        )
        self.reconcile_once(False)
        return self.get_job_detail(job_id) or {}

    def create_batch_job(self, request: BatchJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        cfg = self._build_batch_config(request)
        root_path = os.path.abspath(cfg.runs_root)
        spec = WorkerJobSpec(job_id=job_id, job_type="batch", request=request.model_dump(mode="json"))
        pid, pgid, launch_cmd = self._spawn_worker(job_id, spec, cwd=root_path)
        now = _utc_now_iso()
        display_name = request.display_name or cfg.batch_tag
        self.registry.upsert_job(
            {
                "job_id": job_id,
                "job_type": "batch",
                "job_role": "batch_parent",
                "parent_job_id": None,
                "display_name": display_name,
                "material_id": None,
                "batch_tag": cfg.batch_tag,
                "root_path": root_path,
                "workdir": root_path,
                "thread_id": request.thread_id,
                "pid": pid,
                "pgid": pgid,
                "launch_cmd": launch_cmd,
                "exit_code": None,
                "signal": None,
                "status_source": "control_plane",
                "runtime_run_status": "pending",
                "control_plane_status": "starting",
                "current_stage": None,
                "hitl_pending": False,
                "wait_reason": None,
                "error_summary": None,
                "last_progress_line": None,
                "last_state_updated_at": None,
                "last_heartbeat_at": now,
                "created_at": now,
                "started_at": now,
                "finished_at": None,
                "job_request_json": request.model_dump(mode="json"),
                "updated_at": now,
            }
        )
        self.reconcile_once(True)
        return self.get_job_detail(job_id) or {}

    def create_wiki_reindex_job(self, request: WikiReindexRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        root_path = os.path.abspath(self.settings.repo_root)
        workdir = os.path.join(root_path, ".wiki_rag")
        os.makedirs(workdir, exist_ok=True)
        spec = WorkerJobSpec(job_id=job_id, job_type="wiki_reindex", request=request.model_dump(mode="json"))
        pid, pgid, launch_cmd = self._spawn_worker(job_id, spec, cwd=root_path)
        now = _utc_now_iso()
        self.registry.upsert_job(
            {
                "job_id": job_id,
                "job_type": "wiki_reindex",
                "job_role": "wiki_reindex",
                "parent_job_id": None,
                "display_name": f"wiki-reindex:{request.mode}",
                "material_id": None,
                "batch_tag": None,
                "root_path": root_path,
                "workdir": workdir,
                "thread_id": None,
                "pid": pid,
                "pgid": pgid,
                "launch_cmd": launch_cmd,
                "exit_code": None,
                "signal": None,
                "status_source": "control_plane",
                "runtime_run_status": "running",
                "control_plane_status": "starting",
                "current_stage": "wiki_reindex",
                "hitl_pending": False,
                "wait_reason": None,
                "error_summary": None,
                "last_progress_line": None,
                "last_state_updated_at": None,
                "last_heartbeat_at": now,
                "created_at": now,
                "started_at": now,
                "finished_at": None,
                "job_request_json": request.model_dump(mode="json"),
                "updated_at": now,
            }
        )
        self.reconcile_once(False)
        return self.get_job_detail(job_id) or {}

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        row = self.registry.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        pgid = row.get("pgid")
        pid = row.get("pid")
        signal_name = "SIGTERM"
        try:
            if pgid:
                os.killpg(int(pgid), signal.SIGTERM)
            elif pid:
                os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 10.0
        while _process_group_alive(int(pgid) if pgid else None, int(pid) if pid else None):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if _process_group_alive(int(pgid) if pgid else None, int(pid) if pid else None):
            signal_name = "SIGKILL"
            try:
                if pgid:
                    os.killpg(int(pgid), signal.SIGKILL)
                elif pid:
                    os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        now = _utc_now_iso()
        exit_code = -signal.Signals[signal_name].value if signal_name in signal.Signals.__members__ else None
        self.registry.update_job(
            job_id,
            control_plane_status="cancelled",
            signal=signal_name,
            exit_code=exit_code,
            status_source="control_plane",
            finished_at=now,
            updated_at=now,
        )
        self.reconcile_once(False)
        return self.get_job_detail(job_id) or {}

    def submit_hitl_response(self, job_id: str, request: HitlResponseRequest) -> dict[str, Any]:
        row = self.registry.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        job_request = dict(row.get("job_request_json") or {})
        checkpoint_subdir = str(((job_request.get("runtime") or {}).get("checkpoint_subdir")) or ".runtime")
        result = write_human_response(
            workdir=str(row.get("workdir") or row.get("root_path") or ""),
            response=request.to_payload(),
            checkpoint_subdir=checkpoint_subdir,
        )
        self.reconcile_once(False)
        return result

    def resume_external_event(self, job_id: str, request: ExternalEventResumeRequest) -> dict[str, Any]:
        row = self.registry.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        job_request = dict(row.get("job_request_json") or {})
        runtime_overrides = dict((job_request.get("runtime") or {}))
        runtime = RuntimeContext.from_env()
        if runtime_overrides:
            from dataclasses import replace

            if runtime_overrides.get("checkpoint_subdir"):
                runtime = replace(runtime, checkpoint_subdir=str(runtime_overrides["checkpoint_subdir"]))
        entrypoint = build_external_event_resume_entrypoint(
            resume_material=lambda workdir, tid, event: run_single_material_external_event(
                runtime=runtime,
                workdir=workdir,
                thread_id=tid,
                event=event,
            ).model_dump(mode="json")
        )
        payload = entrypoint.invoke(
            {
                "workdir": str(row.get("workdir") or row.get("root_path") or ""),
                "thread_id": request.thread_id or row.get("thread_id"),
                "event": request.event,
            }
        )
        append_ui_event(
            workdir=str(row.get("workdir") or row.get("root_path") or ""),
            event_type="external_event_resume_requested",
            checkpoint_subdir=runtime.checkpoint_subdir,
            extra={"external_event_type": str((request.event or {}).get("event_type") or "") or None},
        )
        self.reconcile_once(False)
        return payload

    def health_snapshot(self) -> dict[str, Any]:
        active_jobs = [job for job in self.registry.list_jobs() if str(job.get("control_plane_status") or "") in {"starting", "live"}]
        wiki_health = self.wiki_health()
        return {
            "backend": "ok",
            "registry_ok": self.registry.healthcheck(),
            "aggregator_last_success_at": self._last_poll_success_at,
            "active_job_count": len(active_jobs),
            "websocket_client_count": len(self._jobs_ws) + sum(len(items) for items in self._detail_ws.values()),
            "host": self.settings.host,
            "port": self.settings.port,
            "wiki": wiki_health,
        }

    def get_runtime_settings(self) -> dict[str, Any]:
        return self.runtime_settings.read_settings().model_dump(mode="json")

    def update_runtime_settings(self, request: RuntimeSettingsUpdateRequest) -> dict[str, Any]:
        return self.runtime_settings.update_settings(request).model_dump(mode="json")

    def _wiki_service(self) -> VaspWikiRagService:
        runtime = RuntimeContext.from_env()
        return VaspWikiRagService.from_runtime(runtime)

    def wiki_health(self) -> dict[str, Any]:
        try:
            return self._wiki_service().health()
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}:{exc}"}

    def wiki_query(self, request: WikiQueryRequest) -> dict[str, Any]:
        response = self._wiki_service().query(
            query=request.query,
            top_k=request.top_k,
            corpora=request.corpora,
            stage=str(request.stage or ""),
        )
        return response.model_dump(mode="json")

    def list_job_snapshots(self) -> list[dict[str, Any]]:
        rows = self.registry.list_jobs()
        children_by_parent: dict[str, list[str]] = {}
        for row in rows:
            parent = str(row.get("parent_job_id") or "")
            if parent:
                children_by_parent.setdefault(parent, []).append(str(row.get("job_id") or ""))
        payloads: list[dict[str, Any]] = []
        for row in rows:
            summary = self._detail_from_row(row, include_logs=False, include_timeline=False, include_children=False)
            summary["child_job_ids"] = children_by_parent.get(str(row.get("job_id") or ""), [])
            payloads.append(summary)
        return payloads

    def get_job_detail(self, job_id: str) -> dict[str, Any] | None:
        row = self.registry.get_job(job_id)
        if row is None:
            return None
        return self._detail_from_row(row, include_logs=True, include_timeline=True, include_children=True)

    def get_job_state(self, job_id: str) -> dict[str, Any] | None:
        detail = self.get_job_detail(job_id)
        return None if detail is None else dict(detail.get("state") or {})

    def get_job_timeline(self, job_id: str) -> list[dict[str, Any]]:
        row = self.registry.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        workdir = str(row.get("workdir") or "")
        return _safe_jsonl(runtime_ui_events_path(workdir), limit=500)

    def get_job_logs(self, job_id: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        row = self.registry.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        path = os.path.join(str(row.get("workdir") or ""), ".runtime", "runtime_progress.log")
        lines = _safe_text_lines(path)
        total = len(lines)
        start = max(0, int(offset))
        end = min(total, start + max(1, int(limit)))
        return {"total": total, "offset": start, "limit": int(limit), "lines": lines[start:end]}

    def get_job_artifacts(self, job_id: str) -> dict[str, str]:
        detail = self.get_job_detail(job_id)
        if detail is None:
            raise KeyError(job_id)
        return dict(detail.get("artifacts") or {})

    def artifact_download_path(self, job_id: str, artifact_name: str) -> str:
        artifacts = self.get_job_artifacts(job_id)
        for key, path in artifacts.items():
            if key == artifact_name or os.path.basename(path) == artifact_name:
                return path
        raise KeyError(artifact_name)

    def artifact_json_preview(self, job_id: str, artifact_name: str) -> Any:
        path = self.artifact_download_path(job_id, artifact_name)
        payload = _safe_json_value(path)
        if payload is None:
            raise KeyError(artifact_name)
        return payload

    def reconcile_once(self, full_scan: bool) -> None:
        if full_scan:
            self._discover_jobs()
        for row in self.registry.list_jobs():
            refreshed = self._refresh_row(row)
            if refreshed:
                self.registry.upsert_job(refreshed)

    def _refresh_row(self, row: dict[str, Any]) -> dict[str, Any]:
        job = dict(row or {})
        workdir = str(job.get("workdir") or "")
        runtime_state = load_ui_state_snapshot(workdir=workdir) if workdir else None
        shared_state = load_state_snapshot(workdir=workdir) if workdir else None
        outcome = _safe_json(os.path.join(workdir, "material_outcome.json")) if workdir else None
        batch_summary = self._load_batch_summary(str(job.get("root_path") or ""))
        result_payload = _safe_json(_job_result_path(self.settings, str(job.get("job_id") or "")))
        logs = _safe_text_lines(os.path.join(workdir, ".runtime", "runtime_progress.log")) if workdir else []
        process_alive = _process_group_alive(job.get("pgid"), job.get("pid"))
        now = _utc_now_iso()

        status_source = "control_plane"
        runtime_run_status = _normalize_runtime_status(job.get("runtime_run_status"))
        final_acceptance = str(job.get("final_acceptance") or "").strip() or None
        quality_grade = str(job.get("quality_grade") or "").strip() or None
        current_stage = job.get("current_stage")
        hitl_pending = bool(job.get("hitl_pending"))
        wait_reason = job.get("wait_reason")
        error_summary = job.get("error_summary")
        thread_id = job.get("thread_id") or (runtime_state or {}).get("thread_id") or (shared_state or {}).get("execution", {}).get("thread_id")
        last_state_updated_at = job.get("last_state_updated_at")
        state_payload: dict[str, Any] = {}
        summary_payload: dict[str, Any] = {}

        if outcome:
            runtime_run_status = _normalize_runtime_status(str(outcome.get("status") or outcome.get("final_status") or "completed"))
            current_stage = current_stage or "final_report"
            error_summary = "; ".join(list(outcome.get("errors", []) or [])) or error_summary
            last_state_updated_at = datetime.fromtimestamp(os.path.getmtime(os.path.join(workdir, "material_outcome.json")), timezone.utc).isoformat().replace("+00:00", "Z")
            state_payload = dict(shared_state or runtime_state or {})
            summary_payload = dict(outcome.get("final_summary", {}) or {})
            status_source = "material_outcome"
        elif runtime_state:
            runtime_run_status = _normalize_runtime_status(runtime_state.get("runtime_run_status"))
            current_stage = runtime_state.get("current_stage")
            hitl_pending = bool(runtime_state.get("hitl_pending"))
            wait_reason = runtime_state.get("wait_reason")
            error_summary = runtime_state.get("latest_error") or error_summary
            last_state_updated_at = runtime_state.get("updated_at") or last_state_updated_at
            state_payload = dict(runtime_state)
            status_source = "ui_state"
        elif shared_state:
            workflow = dict(shared_state.get("workflow", {}) or {})
            runtime_run_status = _normalize_runtime_status(workflow.get("run_status"))
            current_stage = workflow.get("current_stage")
            hitl_pending = bool(workflow.get("run_status") == "needs_human")
            wait_reason = workflow.get("wait_reason")
            diagnostics = dict(shared_state.get("diagnostics", {}) or {})
            error_summary = str(diagnostics.get("last_error") or error_summary or "") or None
            last_state_updated_at = datetime.fromtimestamp(
                os.path.getmtime(runtime_state_snapshot_path(workdir)),
                timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            state_payload = dict(shared_state)
            status_source = "shared_state"
        elif batch_summary:
            runtime_run_status = "completed"
            current_stage = "batch_finalize"
            summary_payload = batch_summary
            last_state_updated_at = datetime.fromtimestamp(
                os.path.getmtime(batch_summary["_summary_path"]),
                timezone.utc,
            ).isoformat().replace("+00:00", "Z")
            status_source = "batch_summary"
        elif result_payload:
            runtime_run_status = "failed" if int(result_payload.get("exit_code", 1) or 1) else "completed"
            error_summary = str(result_payload.get("error") or error_summary or "") or None
            status_source = "worker_result"
        elif process_alive:
            runtime_run_status = "running" if runtime_run_status == "pending" else runtime_run_status

        detected_acceptance, detected_quality = _quality_signals(
            outcome=outcome,
            summary=summary_payload,
            shared_state=state_payload if state_payload else shared_state,
        )
        final_acceptance = detected_acceptance or final_acceptance
        quality_grade = detected_quality or quality_grade

        control_plane_status = _normalize_control_plane_status(job.get("control_plane_status"))
        if control_plane_status != "cancelled":
            if process_alive:
                control_plane_status = "live" if status_source != "control_plane" or current_stage else "starting"
            elif runtime_run_status in TERMINAL_RUNTIME_STATUSES or batch_summary:
                control_plane_status = "archived"
            elif job.get("pid") or job.get("pgid"):
                control_plane_status = "disconnected"
        last_heartbeat_at = now if process_alive else job.get("last_heartbeat_at")
        artifacts = _material_artifacts(workdir=workdir, ui_state=runtime_state, shared_state=shared_state, outcome=outcome) if workdir else {}
        child_ids = [item["job_id"] for item in self.registry.list_children(str(job.get("job_id") or ""))]

        if result_payload:
            job["exit_code"] = result_payload.get("exit_code")
            job["signal"] = result_payload.get("signal") or job.get("signal")

        job.update(
            {
                "thread_id": thread_id,
                "runtime_run_status": runtime_run_status,
                "control_plane_status": control_plane_status,
                "final_acceptance": final_acceptance,
                "quality_grade": quality_grade,
                "current_stage": current_stage,
                "hitl_pending": hitl_pending,
                "wait_reason": wait_reason,
                "error_summary": error_summary,
                "last_progress_line": logs[-1] if logs else job.get("last_progress_line"),
                "last_state_updated_at": last_state_updated_at,
                "last_heartbeat_at": last_heartbeat_at,
                "status_source": status_source,
                "finished_at": job.get("finished_at") or (now if control_plane_status in {"archived", "cancelled"} else None),
                "updated_at": now,
            }
        )
        job["_detail_state"] = state_payload
        job["_detail_summary"] = summary_payload
        job["_detail_artifacts"] = artifacts
        job["_detail_child_job_ids"] = child_ids
        return job

    def _detail_from_row(
        self,
        row: dict[str, Any],
        *,
        include_logs: bool,
        include_timeline: bool,
        include_children: bool,
    ) -> dict[str, Any]:
        refreshed = self._refresh_row(row)
        detail = JobSnapshot(
            job_id=str(refreshed.get("job_id") or ""),
            job_type=str(refreshed.get("job_type") or ""),
            job_role=str(refreshed.get("job_role") or ""),
            display_name=str(refreshed.get("display_name") or ""),
            material_id=refreshed.get("material_id"),
            batch_tag=refreshed.get("batch_tag"),
            root_path=str(refreshed.get("root_path") or ""),
            workdir=refreshed.get("workdir"),
            thread_id=refreshed.get("thread_id"),
            pid=refreshed.get("pid"),
            pgid=refreshed.get("pgid"),
            runtime_run_status=_normalize_runtime_status(refreshed.get("runtime_run_status")),
            control_plane_status=_normalize_control_plane_status(refreshed.get("control_plane_status")),
            final_acceptance=str(refreshed.get("final_acceptance") or "").strip() or None,
            quality_grade=str(refreshed.get("quality_grade") or "").strip() or None,
            current_stage=refreshed.get("current_stage"),
            hitl_pending=bool(refreshed.get("hitl_pending")),
            wait_reason=refreshed.get("wait_reason"),
            error_summary=refreshed.get("error_summary"),
            last_progress_line=refreshed.get("last_progress_line"),
            last_state_updated_at=refreshed.get("last_state_updated_at"),
            last_heartbeat_at=refreshed.get("last_heartbeat_at"),
            created_at=str(refreshed.get("created_at") or _utc_now_iso()),
            started_at=refreshed.get("started_at"),
            finished_at=refreshed.get("finished_at"),
            parent_job_id=refreshed.get("parent_job_id"),
            child_job_ids=list(refreshed.get("_detail_child_job_ids") or []),
            state=dict(refreshed.get("_detail_state") or {}),
            artifacts=dict(refreshed.get("_detail_artifacts") or {}),
            summary=dict(refreshed.get("_detail_summary") or {}),
        ).model_dump(mode="json")
        if include_logs and detail.get("workdir"):
            detail["logs"] = self.get_job_logs(detail["job_id"], limit=200, offset=0)
        if include_timeline and detail.get("workdir"):
            detail["timeline"] = self.get_job_timeline(detail["job_id"])
        if include_children:
            detail["children"] = [self._detail_from_row(child, include_logs=False, include_timeline=False, include_children=False) for child in self.registry.list_children(detail["job_id"])]
            if detail["job_role"] == "batch_parent":
                detail["failure_taxonomy"] = self._failure_taxonomy(detail["children"])
        return detail

    def _failure_taxonomy(self, children: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for child in children:
            if str(child.get("runtime_run_status") or "") not in {"failed", "aborted", "skipped"}:
                continue
            state = dict(child.get("state") or {})
            workflow = dict(state.get("workflow", {}) or {})
            stage_status = dict(workflow.get("stage_status", {}) or {})
            failed_stage = next((stage for stage, status in stage_status.items() if status == "failed"), None)
            label = failed_stage or str(child.get("error_summary") or "unknown")
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _load_batch_summary(self, root_path: str) -> dict[str, Any] | None:
        if not root_path or not os.path.isdir(root_path):
            return None
        candidates = sorted(Path(root_path).glob("batch_summary_*.json"))
        if not candidates:
            return None
        summary = _safe_json(str(candidates[-1])) or {}
        summary["_summary_path"] = str(candidates[-1])
        return summary

    def _discover_jobs(self) -> None:
        roots = list(self.settings.job_roots)
        for row in self.registry.list_jobs():
            for key in ("root_path", "workdir"):
                value = str(row.get(key) or "").strip()
                if value and value not in roots:
                    roots.append(value)

        for root in roots:
            if not os.path.exists(root):
                continue
            self._discover_batch_parents(root)
            self._discover_material_workdirs(root)

    def _discover_batch_parents(self, root: str) -> None:
        for summary_path in Path(root).glob("batch_summary_*.json"):
            summary = _safe_json(str(summary_path)) or {}
            batch_tag = str(summary.get("batch_tag") or summary_path.stem.replace("batch_summary_", ""))
            existing = self.registry.find_batch_parent(batch_tag)
            if existing is not None:
                continue
            job_id = imported_job_id("imported-batch", str(summary_path))
            now = _utc_now_iso()
            self.registry.upsert_job(
                {
                    "job_id": job_id,
                    "job_type": "batch",
                    "job_role": "batch_parent",
                    "parent_job_id": None,
                    "display_name": batch_tag,
                    "material_id": None,
                    "batch_tag": batch_tag,
                    "root_path": str(summary_path.parent),
                    "workdir": str(summary_path.parent),
                    "thread_id": None,
                    "pid": None,
                    "pgid": None,
                    "launch_cmd": None,
                    "exit_code": None,
                    "signal": None,
                    "status_source": "batch_summary",
                    "runtime_run_status": "completed",
                    "control_plane_status": "archived",
                    "current_stage": "batch_finalize",
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

    def _discover_material_workdirs(self, root: str) -> None:
        for dirpath, dirnames, filenames in os.walk(root):
            file_set = set(filenames)
            if ".web_runtime" in dirnames:
                dirnames.remove(".web_runtime")
            if ".runtime" not in dirnames and "material_outcome.json" not in file_set:
                continue
            workdir = os.path.abspath(dirpath)
            if self.registry.get_job_by_workdir(workdir) is not None:
                continue
            thread_id = load_thread_id(workdir=workdir) or ""
            if thread_id.startswith("batch::"):
                continue
            ui_state = load_ui_state_snapshot(workdir=workdir) or {}
            shared_state = load_state_snapshot(workdir=workdir) or {}
            outcome = _safe_json(os.path.join(workdir, "material_outcome.json")) or {}
            parent_batch_id = str(((shared_state.get("task") or {}).get("parent_batch_id")) or "")
            material_id = (
                str(ui_state.get("material_id") or "")
                or str((shared_state.get("material") or {}).get("material_id") or "")
                or str(outcome.get("material_id") or "")
                or os.path.basename(_root_path_from_workdir(workdir))
            )
            job_role = "batch_child" if parent_batch_id else "standalone"
            parent_job_id = None
            if parent_batch_id:
                parent = self.registry.find_batch_parent(parent_batch_id)
                if parent is None:
                    parent_root = os.path.dirname(_root_path_from_workdir(workdir))
                    imported_parent_id = imported_job_id("imported-batch", f"{parent_batch_id}:{parent_root}")
                    now = _utc_now_iso()
                    self.registry.upsert_job(
                        {
                            "job_id": imported_parent_id,
                            "job_type": "batch",
                            "job_role": "batch_parent",
                            "parent_job_id": None,
                            "display_name": parent_batch_id,
                            "material_id": None,
                            "batch_tag": parent_batch_id,
                            "root_path": parent_root,
                            "workdir": parent_root,
                            "thread_id": None,
                            "pid": None,
                            "pgid": None,
                            "launch_cmd": None,
                            "exit_code": None,
                            "signal": None,
                            "status_source": "imported_scan",
                            "runtime_run_status": "running",
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
                    parent_job_id = imported_parent_id
                else:
                    parent_job_id = str(parent.get("job_id") or "")
            job_id = imported_job_id("imported-job", workdir)
            now = _utc_now_iso()
            self.registry.upsert_job(
                {
                    "job_id": job_id,
                    "job_type": "single_material",
                    "job_role": job_role,
                    "parent_job_id": parent_job_id,
                    "display_name": material_id,
                    "material_id": material_id,
                    "batch_tag": parent_batch_id or None,
                    "root_path": _root_path_from_workdir(workdir),
                    "workdir": workdir,
                    "thread_id": thread_id or None,
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
