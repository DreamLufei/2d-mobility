from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import SkillLoadResult
from .registry import canonical_skill_name, discover_skills


def load_skill(
    root_dir: str,
    skill_name: str,
    *,
    include_body: bool = True,
    include_resources: bool = True,
    resource_limit: int | None = None,
) -> dict[str, Any]:
    registry = discover_skills(root_dir)
    canonical = canonical_skill_name(skill_name)
    meta = registry.get(canonical)
    if meta is None:
        raise FileNotFoundError(f"skill_not_found:{skill_name}")
    skill_dir = Path(meta["path"])
    skill_text = ""
    if include_body:
        with open(meta["skill_md"], "r", encoding="utf-8") as handle:
            skill_text = handle.read()
    resource_payloads: dict[str, Any] = {}
    resources = list(meta.get("resources", []) or [])
    if include_resources:
        selected_resources = resources[: int(resource_limit)] if resource_limit is not None else resources
        for resource in selected_resources:
            relative_path = str((resource or {}).get("path") or "")
            if not relative_path:
                continue
            path = skill_dir / relative_path
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                try:
                    resource_payloads[relative_path] = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    resource_payloads[relative_path] = path.read_text(encoding="utf-8")
            else:
                resource_payloads[relative_path] = path.read_text(encoding="utf-8")
    payload = SkillLoadResult.model_validate(
        {
            **meta,
            "text": skill_text,
            "resource_payloads": resource_payloads,
        }
    )
    dumped = payload.model_dump(mode="json")
    dumped["support_files"] = dict(resource_payloads)
    return dumped
