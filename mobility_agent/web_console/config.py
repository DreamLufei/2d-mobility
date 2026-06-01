from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass

from ..runtime.database import is_memory_uri, normalize_database_uri


@dataclass(frozen=True)
class WebConsoleSettings:
    repo_root: str
    control_dir: str
    database_uri: str
    specs_dir: str
    results_dir: str
    host: str = "127.0.0.1"
    port: int = 8765
    poll_interval_s: float = 1.0
    terminal_poll_interval_s: float = 5.0
    python_executable: str = sys.executable
    frontend_dist_dir: str | None = None
    job_roots: tuple[str, ...] = ()

    @classmethod
    def from_repo(
        cls,
        repo_root: str | None = None,
        *,
        job_roots: list[str] | tuple[str, ...] | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        poll_interval_s: float = 1.0,
        terminal_poll_interval_s: float = 5.0,
        python_executable: str | None = None,
    ) -> "WebConsoleSettings":
        root = os.path.abspath(repo_root or os.getcwd())
        control_dir = os.path.join(root, ".web_runtime")
        specs_dir = os.path.join(control_dir, "specs")
        results_dir = os.path.join(control_dir, "results")
        frontend_dist_dir = os.path.join(root, "web_console", "frontend", "dist")
        normalized_roots: list[str] = []
        for candidate in list(job_roots or [root]):
            value = os.path.abspath(str(candidate))
            if value not in normalized_roots:
                normalized_roots.append(value)
        root_digest = hashlib.sha1(root.encode("utf-8")).hexdigest()[:12]
        raw_database_uri = os.environ.get("MOBILITY_DB_URI")
        database_uri = normalize_database_uri(
            raw_database_uri,
            default_memory_name=f"web-console-{root_digest}",
        )
        if is_memory_uri(raw_database_uri):
            database_uri = f"{str(raw_database_uri).rstrip('/')}-{root_digest}"
        return cls(
            repo_root=root,
            control_dir=control_dir,
            database_uri=database_uri,
            specs_dir=specs_dir,
            results_dir=results_dir,
            host=host,
            port=int(port),
            poll_interval_s=max(0.25, float(poll_interval_s)),
            terminal_poll_interval_s=max(float(poll_interval_s), float(terminal_poll_interval_s)),
            python_executable=python_executable or sys.executable,
            frontend_dist_dir=frontend_dist_dir,
            job_roots=tuple(normalized_roots),
        )

    def ensure_directories(self) -> None:
        for path in (self.control_dir, self.specs_dir, self.results_dir):
            os.makedirs(path, exist_ok=True)
