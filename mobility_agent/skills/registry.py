from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from .models import SkillManifest, SkillRegistryEntry, SkillResource


def canonical_skill_name(name: str) -> str:
    value = str(name or "").strip().lower().replace("-", "_")
    aliases = {
        "single_material_mobility_skill": "single_material_mobility",
        "batch_mobility_screening_skill": "batch_mobility_screening",
        "planning_skill": "planning",
        "default_scientific_path_skill": "single_material_mobility",
        "recovery_skill": "recovery",
        "recovery_diagnosis_skill": "recovery",
        "strain_refinement_skill": "strain_refinement",
        "physics_validation_skill": "physics_validation",
        "critique_skill": "critique",
        "resource_guardian_skill": "cost_guardian",
        "arbitration_skill": "orchestration",
        "final_reporting_skill": "reporting",
        "reporting_skill": "reporting",
        "orchestration_skill": "orchestration",
        "cost_guardian_skill": "cost_guardian",
        "execution_feasibility_skill": "execution_feasibility",
    }
    return aliases.get(value, value)


def default_skills_root() -> str:
    return str((Path(__file__).resolve().parents[2] / "skills").resolve())


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _normalize_section_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return normalized.strip("_")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    stripped = str(text or "")
    if not stripped.startswith("+++"):
        return {}, stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return {}, stripped
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "+++":
            end_index = index
            break
    if end_index is None:
        return {}, stripped
    try:
        payload = tomllib.loads("\n".join(lines[1:end_index]))
    except Exception:
        return {}, stripped
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return dict(payload or {}), body


def _parse_sections(text: str) -> tuple[str, dict[str, list[str]]]:
    title = ""
    sections: dict[str, list[str]] = {}
    current_key = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        match = _SECTION_RE.match(line)
        if match:
            current_key = _normalize_section_key(match.group(1))
            sections.setdefault(current_key, [])
            continue
        if current_key:
            sections[current_key].append(line)
    return title, sections


def _section_list(lines: list[str]) -> list[str]:
    values: list[str] = []
    for line in list(lines or []):
        stripped = str(line or "").strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ")):
            values.append(stripped[2:].strip())
        else:
            values.append(stripped)
    return [item for item in values if item]


def _section_text(lines: list[str]) -> str:
    return " ".join(_section_list(lines)).strip()


def _resource_kind(path: Path, skill_dir: Path) -> str:
    relative_parts = path.relative_to(skill_dir).parts
    if "scripts" in relative_parts:
        return "script"
    if "templates" in relative_parts:
        return "template"
    if path.suffix.lower() == ".json":
        return "data"
    return "reference"


def _discover_resources(skill_dir: Path) -> list[SkillResource]:
    resources: list[SkillResource] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SKILL.md" or path.name.startswith("."):
            continue
        resources.append(
            SkillResource(
                path=str(path.relative_to(skill_dir)).replace("\\", "/"),
                kind=_resource_kind(path, skill_dir),
                size_bytes=path.stat().st_size,
            )
        )
    return resources


def _manifest_from_skill_md(skill_name: str, skill_md: Path, skill_dir: Path) -> SkillRegistryEntry:
    raw_text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw_text)
    title, sections = _parse_sections(body)
    resources = _discover_resources(skill_dir)
    manifest = SkillManifest(
        name=canonical_skill_name(str(frontmatter.get("name") or title or skill_name)),
        version=str(frontmatter.get("version") or "1"),
        description=str(frontmatter.get("description") or _section_text(sections.get("description", [])) or _section_text(sections.get("purpose", []))),
        purpose=str(frontmatter.get("purpose") or _section_text(sections.get("purpose", []))),
        when_to_use=list(frontmatter.get("when_to_use") or _section_list(sections.get("when_to_use", []))),
        required_inputs=list(frontmatter.get("required_inputs") or _section_list(sections.get("required_inputs", []))),
        relevant_state_fields=list(frontmatter.get("relevant_state_fields") or _section_list(sections.get("relevant_state_fields", []))),
        allowed_tools=list(frontmatter.get("allowed_tools") or _section_list(sections.get("allowed_tools", []))),
        decision_rules=list(frontmatter.get("decision_rules") or _section_list(sections.get("decision_rules", []))),
        stop_conditions=list(frontmatter.get("stop_conditions") or _section_list(sections.get("stop_conditions", []))),
        expected_output_schema=list(frontmatter.get("expected_output_schema") or _section_list(sections.get("expected_output_schema", []))),
        caveats=list(
            frontmatter.get("caveats")
            or frontmatter.get("warnings")
            or _section_list(sections.get("caveats_warnings", []) or sections.get("caveats", []))
        ),
        roles=list(frontmatter.get("roles") or []),
        task_types=list(frontmatter.get("task_types") or []),
        stages=list(frontmatter.get("stages") or []),
        run_statuses=list(frontmatter.get("run_statuses") or []),
        error_patterns=list(frontmatter.get("error_patterns") or []),
        anomaly_patterns=list(frontmatter.get("anomaly_patterns") or []),
        tags=list(frontmatter.get("tags") or []),
        load_strategy=str(frontmatter.get("load_strategy") or "summary_only"),
        resource_roots=list(frontmatter.get("resource_roots") or []),
    )
    summary = manifest.description or manifest.purpose or f"Skill package at {skill_dir.name}"
    return SkillRegistryEntry(
        name=manifest.name,
        path=str(skill_dir.resolve()),
        skill_md=str(skill_md.resolve()),
        description=summary,
        manifest=manifest,
        summary=summary,
        resources=resources,
    )


def discover_skills(root_dir: str) -> dict[str, dict[str, Any]]:
    root = Path(root_dir)
    registry: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return registry
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        entry = _manifest_from_skill_md(canonical_skill_name(child.name), skill_md, child)
        registry[entry.name] = entry.model_dump(mode="json")
    return registry


def list_skill_packages(root_dir: str) -> list[dict[str, Any]]:
    registry = discover_skills(root_dir)
    items: list[dict[str, Any]] = []
    for name in sorted(registry):
        entry = dict(registry.get(name, {}) or {})
        manifest = dict(entry.get("manifest", {}) or {})
        resources = list(entry.get("resources", []) or [])
        items.append(
            {
                "name": name,
                "description": str(entry.get("description") or ""),
                "path": str(entry.get("path") or ""),
                "skill_md": str(entry.get("skill_md") or ""),
                "roles": list(manifest.get("roles", []) or []),
                "task_types": list(manifest.get("task_types", []) or []),
                "stages": list(manifest.get("stages", []) or []),
                "load_strategy": str(manifest.get("load_strategy") or "summary_only"),
                "resource_count": len(resources),
            }
        )
    return items
