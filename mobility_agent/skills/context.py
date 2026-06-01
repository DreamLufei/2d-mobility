from __future__ import annotations

from typing import Any

from .models import SkillResolutionRequest
from .registry import canonical_skill_name
from .resolver import resolve_skills
from ..utils import dedupe_keep_order


def choose_skills(
    *,
    task_type: str,
    stage: str,
    role: str | None = None,
    has_error: bool = False,
    run_status: str | None = None,
    latest_error: str | None = None,
    anomaly_flags: list[str] | None = None,
    explicit_skills: list[str] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
    limit: int = 6,
) -> list[str]:
    normalized_explicit = dedupe_keep_order([canonical_skill_name(skill) for skill in list(explicit_skills or []) if skill])
    if registry:
        selection = resolve_skills(
            registry,
            request=SkillResolutionRequest(
                role=role,
                task_type=task_type,
                stage=stage,
                run_status=run_status,
                has_error=has_error,
                latest_error=latest_error,
                anomaly_flags=list(anomaly_flags or []),
                explicit_skills=normalized_explicit,
                limit=limit,
            ),
        )
        if selection.selected_skills:
            return [canonical_skill_name(skill) for skill in selection.selected_skills]
    skills = []
    if task_type == "single_material":
        skills.append("single_material_mobility")
    if task_type == "batch_database":
        skills.append("batch_mobility_screening")
    if stage in {"relax", "scf", "band", "effective_mass", "strain_loop", "mobility"}:
        skills.append("recovery")
    if stage in {"strain_loop", "refinement"}:
        skills.append("strain_refinement")
    if stage in {"validation", "report"}:
        skills.append("physics_validation")
        skills.append("reporting")
    if has_error:
        skills.append("recovery")
    skills.extend(normalized_explicit)
    return [canonical_skill_name(skill) for skill in skills]
