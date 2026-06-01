from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_TRACE_LOCK = threading.Lock()
_TRACE_COUNTER = 0


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or "").strip())
    return cleaned.strip("_") or "trace"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")  # type: ignore[no-any-return]
        except Exception:
            return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _active_workdir(explicit: str | None = None) -> str | None:
    candidate = explicit or os.environ.get("MOBILITY_ACTIVE_WORKDIR")
    if not candidate:
        return None
    return os.path.abspath(candidate)


def _runtime_dir(workdir: str | None = None) -> Path | None:
    resolved = _active_workdir(workdir)
    if not resolved:
        return None
    path = Path(resolved) / ".runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def active_workdir_scope(workdir: str | None):
    previous = os.environ.get("MOBILITY_ACTIVE_WORKDIR")
    try:
        if workdir:
            os.environ["MOBILITY_ACTIVE_WORKDIR"] = os.path.abspath(workdir)
        else:
            os.environ.pop("MOBILITY_ACTIVE_WORKDIR", None)
        yield
    finally:
        if previous is None:
            os.environ.pop("MOBILITY_ACTIVE_WORKDIR", None)
        else:
            os.environ["MOBILITY_ACTIVE_WORKDIR"] = previous


def _next_trace_prefix() -> str:
    global _TRACE_COUNTER
    with _TRACE_LOCK:
        _TRACE_COUNTER += 1
        counter = _TRACE_COUNTER
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{counter:04d}"


def progress_enabled() -> bool:
    return _env_bool("MOBILITY_PROGRESS", True)


def llm_trace_enabled() -> bool:
    return _env_bool("LLM_TRACE_ENABLED", True)


def tool_evidence_enabled() -> bool:
    return _env_bool("LLM_TOOL_EVIDENCE_ENABLED", False)


def emit_progress(
    message: str,
    *,
    workdir: str | None = None,
    channel: str = "runtime",
    details: dict[str, Any] | None = None,
) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[mobility {timestamp}] [{channel}] {message}"
    if details:
        summary_parts: list[str] = []
        for key, value in details.items():
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, (list, dict, tuple, set)) and not value:
                continue
            summary_parts.append(f"{key}={value}")
        summary = ", ".join(summary_parts)
        if summary:
            line = f"{line} | {summary}"
    if progress_enabled():
        print(line, file=sys.stderr, flush=True)
    runtime_dir = _runtime_dir(workdir)
    if runtime_dir is None:
        return
    try:
        with (runtime_dir / "runtime_progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        return


def dump_json_trace(
    kind: str,
    payload: dict[str, Any],
    *,
    role: str | None = None,
    workdir: str | None = None,
) -> str | None:
    if not llm_trace_enabled():
        return None
    runtime_dir = _runtime_dir(workdir)
    if runtime_dir is None:
        return None
    trace_dir = runtime_dir / "llm_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    label_parts = [_sanitize_token(role or ""), _sanitize_token(kind)]
    filename = f"{_next_trace_prefix()}_{'_'.join(part for part in label_parts if part and part != 'trace')}.json"
    path = trace_dir / filename
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    except Exception:
        return None
    return str(path)
