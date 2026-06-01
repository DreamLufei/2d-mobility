from __future__ import annotations

from typing import Any

from .models import SkillCandidate, SkillManifest, SkillResolutionRequest, SkillSelectionRecord
from .registry import canonical_skill_name


def _matches_any(value: str | None, accepted: list[str]) -> bool:
    if not accepted:
        return True
    current = str(value or "").strip().lower()
    normalized = [str(item or "").strip().lower() for item in accepted]
    return bool(current) and current in normalized


def _text_has_pattern(value: str | None, patterns: list[str]) -> bool:
    current = str(value or "").strip().lower()
    if not current or not patterns:
        return False
    return any(str(pattern or "").strip().lower() in current for pattern in patterns)


def _list_has_pattern(values: list[str], patterns: list[str]) -> bool:
    if not values or not patterns:
        return False
    lowered_values = [str(item or "").strip().lower() for item in values]
    lowered_patterns = [str(item or "").strip().lower() for item in patterns]
    return any(pattern in value for pattern in lowered_patterns for value in lowered_values if pattern)


def _fallback_manifest_hints(name: str, manifest: SkillManifest) -> list[str]:
    hints = list(manifest.tags)
    if name == "recovery":
        hints.extend(["needs_recovery", "error", "failure"])
    if name == "physics_validation":
        hints.extend(["validation", "anomaly"])
    if name == "strain_refinement":
        hints.extend(["refinement", "strain"])
    if name == "reporting":
        hints.extend(["finalize", "report"])
    if name == "single_material_mobility":
        hints.extend(["mainline", "single_material"])
    return hints


def _score_candidate(name: str, manifest: SkillManifest, request: SkillResolutionRequest) -> SkillCandidate:
    reasons: list[str] = []
    score = 0.0
    canonical_name = canonical_skill_name(name)

    if canonical_name in request.explicit_skills:
        score += 100.0
        reasons.append("explicit_skill")
    if _matches_any(request.role, manifest.roles):
        score += 30.0
        reasons.append(f"role:{request.role}")
    if _matches_any(request.task_type, manifest.task_types):
        score += 24.0
        reasons.append(f"task_type:{request.task_type}")
    if _matches_any(request.stage, manifest.stages):
        score += 18.0
        reasons.append(f"stage:{request.stage}")
    if _matches_any(request.run_status, manifest.run_statuses):
        score += 12.0
        reasons.append(f"run_status:{request.run_status}")
    if _text_has_pattern(request.latest_error, manifest.error_patterns):
        score += 22.0
        reasons.append("error_pattern")
    if _list_has_pattern(request.anomaly_flags, manifest.anomaly_patterns):
        score += 18.0
        reasons.append("anomaly_pattern")
    if request.has_error and (canonical_name == "recovery" or "error" in _fallback_manifest_hints(canonical_name, manifest)):
        score += 14.0
        reasons.append("error_context")

    tags = _fallback_manifest_hints(canonical_name, manifest)
    if request.stage and request.stage.lower() in [item.lower() for item in tags]:
        score += 8.0
        reasons.append(f"tag:{request.stage}")
    if request.run_status and request.run_status.lower() in [item.lower() for item in tags]:
        score += 8.0
        reasons.append(f"tag:{request.run_status}")

    if not reasons and canonical_name == "single_material_mobility" and request.task_type == "single_material":
        score += 6.0
        reasons.append("default_mainline")

    return SkillCandidate(
        name=canonical_name,
        score=score,
        selected=False,
        reasons=reasons,
        manifest=manifest,
    )


def resolve_skills(
    registry: dict[str, dict[str, Any]],
    *,
    request: SkillResolutionRequest,
) -> SkillSelectionRecord:
    normalized_request = request.model_copy(
        update={
            "explicit_skills": [canonical_skill_name(skill) for skill in list(request.explicit_skills or []) if skill],
        }
    )
    candidates: list[SkillCandidate] = []
    for name, entry in sorted(dict(registry or {}).items()):
        manifest_payload = dict(entry.get("manifest", {}) or {})
        manifest = SkillManifest.model_validate(manifest_payload | {"name": manifest_payload.get("name") or name})
        candidates.append(_score_candidate(name, manifest, normalized_request))

    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item.reasons),
            float(item.score),
            item.name,
        ),
        reverse=True,
    )
    selected: list[str] = []
    for candidate in ranked:
        if len(selected) >= normalized_request.limit:
            break
        if float(candidate.score) <= 0.0 and candidate.name not in normalized_request.explicit_skills:
            continue
        candidate.selected = True
        selected.append(candidate.name)

    if not selected and "single_material_mobility" in registry and normalized_request.task_type == "single_material":
        selected = ["single_material_mobility"]
        for candidate in ranked:
            if candidate.name == "single_material_mobility":
                candidate.selected = True
                if "default_mainline" not in candidate.reasons:
                    candidate.reasons.append("default_mainline")
                break

    return SkillSelectionRecord(
        role=normalized_request.role,
        task_type=normalized_request.task_type,
        stage=normalized_request.stage,
        run_status=normalized_request.run_status,
        selected_skills=selected,
        candidates=ranked,
        resolution_mode="resolver",
    )
