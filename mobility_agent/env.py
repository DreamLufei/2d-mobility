from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dotenv_candidates(project_root: str | os.PathLike[str] | None = None) -> list[Path]:
    root = Path(project_root) if project_root is not None else _DEFAULT_PROJECT_ROOT
    return [root / ".env", root / ".env.local"]


def ensure_project_env_loaded(project_root: str | os.PathLike[str] | None = None) -> None:
    """Load `.env` then `.env.local` without overriding existing shell variables."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return

    candidates = _dotenv_candidates(project_root)
    if not any(path.exists() for path in candidates):
        _ENV_LOADED = True
        return

    try:
        from dotenv import dotenv_values  # type: ignore
    except Exception:
        _ENV_LOADED = True
        return

    merged_values: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            values = dotenv_values(path)
        except Exception:
            continue
        for key, value in values.items():
            if not key or value is None:
                continue
            merged_values[str(key)] = str(value)

    for key, value in merged_values.items():
        os.environ.setdefault(key, value)

    _ENV_LOADED = True
