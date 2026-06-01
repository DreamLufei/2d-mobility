from __future__ import annotations

import copy
import json
from threading import Lock
from typing import Any

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..runtime.database import is_postgres_uri, normalize_database_uri


_SCHEMA = """
CREATE TABLE IF NOT EXISTS control_plane_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    job_role TEXT NOT NULL,
    parent_job_id TEXT,
    display_name TEXT NOT NULL,
    material_id TEXT,
    batch_tag TEXT,
    root_path TEXT NOT NULL,
    workdir TEXT,
    thread_id TEXT,
    pid BIGINT,
    pgid BIGINT,
    launch_cmd TEXT,
    exit_code BIGINT,
    signal TEXT,
    status_source TEXT,
    runtime_run_status TEXT NOT NULL,
    control_plane_status TEXT NOT NULL,
    current_stage TEXT,
    hitl_pending BOOLEAN NOT NULL DEFAULT FALSE,
    wait_reason TEXT,
    error_summary TEXT,
    last_progress_line TEXT,
    last_state_updated_at TEXT,
    last_heartbeat_at TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    job_request_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_control_plane_jobs_parent_job_id ON control_plane_jobs(parent_job_id);
CREATE INDEX IF NOT EXISTS idx_control_plane_jobs_batch_tag ON control_plane_jobs(batch_tag);
CREATE INDEX IF NOT EXISTS idx_control_plane_jobs_workdir ON control_plane_jobs(workdir);
CREATE INDEX IF NOT EXISTS idx_control_plane_jobs_control_plane_status ON control_plane_jobs(control_plane_status);
"""

_ALLOWED_COLUMNS = {
    "job_id",
    "job_type",
    "job_role",
    "parent_job_id",
    "display_name",
    "material_id",
    "batch_tag",
    "root_path",
    "workdir",
    "thread_id",
    "pid",
    "pgid",
    "launch_cmd",
    "exit_code",
    "signal",
    "status_source",
    "runtime_run_status",
    "control_plane_status",
    "current_stage",
    "hitl_pending",
    "wait_reason",
    "error_summary",
    "last_progress_line",
    "last_state_updated_at",
    "last_heartbeat_at",
    "created_at",
    "started_at",
    "finished_at",
    "job_request_json",
    "updated_at",
}

_MEMORY_REGISTRIES: dict[str, dict[str, dict[str, Any]]] = {}


class ControlPlaneRegistry:
    def __init__(self, database_uri: str) -> None:
        self.database_uri = normalize_database_uri(database_uri, default_memory_name="control-plane")
        self._lock = Lock()
        self._memory_rows = _MEMORY_REGISTRIES.setdefault(self.database_uri, {})
        self._initialize()

    @property
    def _is_memory_backend(self) -> bool:
        return not is_postgres_uri(self.database_uri)

    def _connect(self):
        return connect(self.database_uri, row_factory=dict_row, autocommit=True)

    def _initialize(self) -> None:
        if self._is_memory_backend:
            return
        with self._lock:
            with self._connect() as conn:
                conn.execute(_SCHEMA)

    def healthcheck(self) -> bool:
        if self._is_memory_backend:
            return True
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def upsert_job(self, payload: dict[str, Any]) -> None:
        job = {key: value for key, value in dict(payload or {}).items() if key in _ALLOWED_COLUMNS}
        if self._is_memory_backend:
            with self._lock:
                current = copy.deepcopy(self._memory_rows.get(str(job.get("job_id") or ""), {}))
                current.update(copy.deepcopy(job))
                self._memory_rows[str(job.get("job_id") or "")] = current
            return
        columns = sorted(job.keys())
        values = [self._serialize_value(column, job[column]) for column in columns]
        placeholders = ", ".join("%s" for _ in columns)
        assignments = ", ".join(f"{column}=EXCLUDED.{column}" for column in columns if column != "job_id")
        query = (
            f"INSERT INTO control_plane_jobs ({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(job_id) DO UPDATE SET {assignments}"
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(query, values)

    def update_job(self, job_id: str, **updates: Any) -> None:
        fields = {key: value for key, value in dict(updates or {}).items() if value is not None or key in updates}
        if not fields:
            return
        if self._is_memory_backend:
            with self._lock:
                current = copy.deepcopy(self._memory_rows.get(job_id, {}))
                current.update(copy.deepcopy(fields))
                self._memory_rows[job_id] = current
            return
        assignments = ", ".join(f"{key}=%s" for key in fields.keys())
        values = [self._serialize_value(key, value) for key, value in fields.items()] + [job_id]
        with self._lock:
            with self._connect() as conn:
                conn.execute(f"UPDATE control_plane_jobs SET {assignments} WHERE job_id=%s", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if self._is_memory_backend:
            return copy.deepcopy(self._memory_rows.get(job_id))
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM control_plane_jobs WHERE job_id=%s", [job_id]).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def get_job_by_workdir(self, workdir: str) -> dict[str, Any] | None:
        if self._is_memory_backend:
            matches = [row for row in self._memory_rows.values() if str(row.get("workdir") or "") == workdir]
            matches.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("job_id") or "")), reverse=True)
            return copy.deepcopy(matches[0]) if matches else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM control_plane_jobs WHERE workdir=%s ORDER BY created_at DESC LIMIT 1",
                [workdir],
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_jobs(self) -> list[dict[str, Any]]:
        if self._is_memory_backend:
            rows = list(self._memory_rows.values())
            rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("job_id") or "")), reverse=True)
            return [copy.deepcopy(row) for row in rows]
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM control_plane_jobs ORDER BY created_at DESC, job_id DESC").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_children(self, parent_job_id: str) -> list[dict[str, Any]]:
        if self._is_memory_backend:
            rows = [row for row in self._memory_rows.values() if str(row.get("parent_job_id") or "") == parent_job_id]
            rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("job_id") or "")))
            return [copy.deepcopy(row) for row in rows]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM control_plane_jobs WHERE parent_job_id=%s ORDER BY created_at ASC, job_id ASC",
                [parent_job_id],
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def find_batch_parent(self, batch_tag: str) -> dict[str, Any] | None:
        if self._is_memory_backend:
            rows = [
                row
                for row in self._memory_rows.values()
                if str(row.get("batch_tag") or "") == batch_tag and str(row.get("job_role") or "") == "batch_parent"
            ]
            rows.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("job_id") or "")), reverse=True)
            return copy.deepcopy(rows[0]) if rows else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM control_plane_jobs WHERE batch_tag=%s AND job_role='batch_parent' ORDER BY created_at DESC LIMIT 1",
                [batch_tag],
            ).fetchone()
        return self._row_to_dict(row) if row is not None else None

    @staticmethod
    def _serialize_value(column: str, value: Any) -> Any:
        if column == "job_request_json":
            return Jsonb(dict(value or {}))
        if isinstance(value, (dict, list)):
            return Jsonb(value)
        return value

    @staticmethod
    def _row_to_dict(row: dict[str, Any] | None) -> dict[str, Any]:
        if row is None:
            return {}
        payload = dict(row)
        payload["hitl_pending"] = bool(payload.get("hitl_pending"))
        raw_request = payload.get("job_request_json")
        if isinstance(raw_request, str):
            try:
                payload["job_request_json"] = json.loads(raw_request)
            except Exception:
                payload["job_request_json"] = {}
        elif raw_request is None:
            payload["job_request_json"] = {}
        return payload
