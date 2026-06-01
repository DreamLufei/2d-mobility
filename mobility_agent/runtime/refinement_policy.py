from __future__ import annotations

from typing import Any

from ..utils import dedupe_keep_order


VALID_REFINEMENT_CHANNELS = ("x", "y")
DEFAULT_REFINEMENT_TARGET_POINTS = 9
DEFAULT_FIT_R2_THRESHOLD = 0.90
_MIDPOINT_STRATEGIES = {
    "midpoint",
    "midpoint_enrichment",
    "midpoint_refinement",
    "midpoint_sampling",
}


def _round_point(value: float) -> float:
    return round(float(value), 6)


def _normalize_points(values: Any) -> list[float]:
    points: list[float] = []
    for item in list(values or []):
        try:
            points.append(_round_point(float(item)))
        except Exception:
            continue
    return sorted(dedupe_keep_order(points))


def _normalize_channels(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    channels = [str(item).strip().lower() for item in list(values or []) if str(item).strip()]
    return [item for item in dedupe_keep_order(channels) if item in VALID_REFINEMENT_CHANNELS]


def _fit_threshold(state: dict[str, Any], default: float) -> float:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    fit = dict(diagnostics.get("fit_diagnostics", {}) or {})
    for value in (
        fit.get("fit_r2_threshold"),
        ((state.get("mission", {}) or {}).get("runtime_constraints", {}) or {}).get("fit_r2_threshold"),
    ):
        try:
            if value is not None:
                return float(value)
        except Exception:
            continue
    return float(default)


def infer_refinement_channels(
    state: dict[str, Any],
    parameters: dict[str, Any] | None,
    *,
    fit_threshold: float = DEFAULT_FIT_R2_THRESHOLD,
) -> list[str]:
    params = dict(parameters or {})
    for key in ("target_channels", "target_directions", "channels", "directions"):
        normalized = _normalize_channels(params.get(key))
        if normalized:
            return normalized

    inferred_from_targets: list[str] = []
    for item in list(params.get("target_fits", []) or []):
        token = str(item or "").strip().lower()
        if token.endswith("_x") or token.endswith(":x") or token.endswith("/x"):
            inferred_from_targets.append("x")
        elif token.endswith("_y") or token.endswith(":y") or token.endswith("/y"):
            inferred_from_targets.append("y")
    normalized_targets = _normalize_channels(inferred_from_targets)
    if normalized_targets:
        return normalized_targets

    diagnostics = dict(state.get("diagnostics", {}) or {})
    fit = dict(diagnostics.get("fit_diagnostics", {}) or {})
    per_direction = dict(fit.get("per_direction", {}) or {})
    weak_channels: list[str] = []
    threshold = _fit_threshold(state, fit_threshold)
    for channel in VALID_REFINEMENT_CHANNELS:
        payload = dict(per_direction.get(channel, {}) or {})
        metric = payload.get("effective_fit_quality", payload.get("edge_fit_r2", fit.get("fit_r2_min", 1.0)))
        try:
            metric_value = float(metric if metric is not None else 1.0)
        except Exception:
            metric_value = 1.0
        if metric_value < threshold:
            weak_channels.append(channel)
    if weak_channels:
        return weak_channels

    accepted = _normalize_channels(((state.get("physics_results", {}) or {}).get("accepted_channels") or []))
    return accepted or list(VALID_REFINEMENT_CHANNELS)


def _midpoint_candidates(existing: list[float]) -> list[float]:
    if len(existing) < 2:
        return []
    candidates: list[float] = []
    existing_set = set(existing)
    for left, right in zip(existing, existing[1:]):
        midpoint = _round_point((float(left) + float(right)) / 2.0)
        if midpoint in existing_set or midpoint <= float(left) or midpoint >= float(right):
            continue
        candidates.append(midpoint)
    return sorted(dedupe_keep_order(candidates), key=lambda value: (abs(value), value))


def resolve_refinement_sampling(
    state: dict[str, Any],
    parameters: dict[str, Any] | None,
    *,
    max_points_per_direction: int = DEFAULT_REFINEMENT_TARGET_POINTS,
    fit_threshold: float = DEFAULT_FIT_R2_THRESHOLD,
) -> dict[str, Any]:
    params = dict(parameters or {})
    plan_by_direction = dict((state.get("physics_results", {}) or {}).get("strain_plan_by_direction", {}) or {})
    target_channels = infer_refinement_channels(state, params, fit_threshold=fit_threshold)
    requested_suggested = params.get("suggested_points")
    suggested_by_direction = dict(requested_suggested or {}) if isinstance(requested_suggested, dict) else {}
    strategy = str(params.get("refinement_strategy") or "").strip().lower()
    applied_points: dict[str, list[float]] = {}
    suggested_points: dict[str, list[float]] = {}
    full_plan_by_direction: dict[str, list[float]] = {}
    limit = max(0, int(max_points_per_direction or DEFAULT_REFINEMENT_TARGET_POINTS))

    for channel in target_channels:
        existing = _normalize_points(plan_by_direction.get(channel, []))
        remaining_slots = max(0, limit - len(existing))
        if remaining_slots <= 0:
            full_plan_by_direction[channel] = [float(item) for item in existing]
            continue
        explicit = _normalize_points(suggested_by_direction.get(channel, []))
        candidates = [value for value in explicit if value not in set(existing)]
        if not candidates and strategy in _MIDPOINT_STRATEGIES:
            candidates = _midpoint_candidates(existing)
        chosen: list[float] = []
        seen = set(existing)
        for value in candidates:
            if value in seen:
                continue
            chosen.append(value)
            seen.add(value)
            if len(chosen) >= remaining_slots:
                break
        if chosen:
            applied_points[channel] = [float(item) for item in chosen]
            suggested_points[channel] = [float(item) for item in chosen]
        full_plan_by_direction[channel] = [float(item) for item in sorted(seen)]

    return {
        "target_channels": target_channels,
        "suggested_points": suggested_points,
        "applied_points": applied_points,
        "full_plan_by_direction": full_plan_by_direction,
        "refinement_strategy": strategy or None,
        "max_points_per_direction": limit,
    }


def validation_requires_refinement(
    state: dict[str, Any],
    *,
    max_points_per_direction: int = DEFAULT_REFINEMENT_TARGET_POINTS,
    fit_threshold: float = DEFAULT_FIT_R2_THRESHOLD,
) -> bool:
    diagnostics = dict(state.get("diagnostics", {}) or {})
    fit = dict(diagnostics.get("fit_diagnostics", {}) or {})
    if not fit:
        return False
    try:
        effective_fit = float(fit.get("effective_fit_quality", fit.get("fit_r2_min", 1.0)) or 0.0)
    except Exception:
        effective_fit = 1.0
    if effective_fit >= _fit_threshold(state, fit_threshold):
        return False
    if list((state.get("blackboard", {}) or {}).get("anomaly_flags", []) or []):
        return False

    workflow = dict(state.get("workflow", {}) or {})
    if int(workflow.get("refinement_rounds", 0) or 0) >= int(workflow.get("max_refinement_rounds", 0) or 0):
        return False

    proposal = resolve_refinement_sampling(
        state,
        {"refinement_strategy": "midpoint_enrichment"},
        max_points_per_direction=max_points_per_direction,
        fit_threshold=fit_threshold,
    )
    return bool(proposal.get("applied_points"))
