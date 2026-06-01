from __future__ import annotations

import math
from typing import Any

from ..tools.physics_validator import validate_physics_window
from ..utils import dedupe_keep_order
from .channel_utils import (
    VALID_CARRIERS,
    VALID_DIRECTIONS,
    canonical_subchannel,
    default_subchannels,
    derive_direction_acceptance,
)
from .refinement_policy import DEFAULT_FIT_R2_THRESHOLD, DEFAULT_REFINEMENT_TARGET_POINTS, resolve_refinement_sampling


SEVERE_E1_FIT_R2_THRESHOLD = 0.20
SEVERE_MOBILITY_CM2_VS = 1.0e7
SUSPICIOUS_MOBILITY_CM2_VS = 1.0e5
FLAT_E1_ABS_EV_THRESHOLD = 0.05
REL_E1_SIGMA_REFINE_THRESHOLD = 0.50
REL_C2D_SIGMA_REFINE_THRESHOLD = 0.20


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _relative_sigma(sigma: float | None, baseline: float | None) -> float | None:
    if sigma is None or baseline is None:
        return None
    if abs(baseline) <= 1.0e-12:
        return None
    return abs(float(sigma) / float(baseline))


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _channel_review(
    *,
    direction: str,
    carrier: str,
    direction_data: dict[str, Any],
    carrier_data: dict[str, Any],
    fit_threshold: float,
) -> dict[str, Any]:
    token = canonical_subchannel(direction, carrier)
    n_points = int(direction_data.get("n_points", 0) or 0)
    mobility = _safe_float(carrier_data.get("mobility_cm2_Vs"))
    raw_mobility = _safe_float(carrier_data.get("raw_mobility_cm2_Vs"))
    if mobility is None and raw_mobility is not None:
        mobility = raw_mobility
    e1 = _safe_float(carrier_data.get("E1_eV"))
    e1_sigma = _safe_float(carrier_data.get("E1_eV_sigma"))
    e1_fit_r2 = _safe_float(carrier_data.get("E1_fit_R2"))
    c2d = _safe_float(carrier_data.get("C2D_J_m2"))
    c2d_sigma = _safe_float(carrier_data.get("C2D_sigma_J_m2"))
    c2d_fit_r2 = _safe_float(carrier_data.get("C2D_fit_R2"))
    rel_e1_sigma = _relative_sigma(e1_sigma, e1)
    rel_c2d_sigma = _relative_sigma(c2d_sigma, c2d)
    mass_status = str(carrier_data.get("mass_status") or "").strip()
    mass_valid = carrier_data.get("mass_valid_for_mobility")
    mass_rejection_reasons = list(carrier_data.get("mass_rejection_reasons", []) or [])
    review = {
        "channel": token,
        "direction": direction,
        "carrier": carrier,
        "n_points": n_points,
        "mobility_cm2_Vs": mobility,
        "E1_eV": e1,
        "E1_eV_sigma": e1_sigma,
        "E1_fit_R2": e1_fit_r2,
        "C2D_J_m2": c2d,
        "C2D_sigma_J_m2": c2d_sigma,
        "C2D_fit_R2": c2d_fit_r2,
        "rel_e1_sigma": rel_e1_sigma,
        "rel_c2d_sigma": rel_c2d_sigma,
        "mass_status": mass_status,
        "mass_valid_for_mobility": mass_valid,
        "mass_rejection_reasons": mass_rejection_reasons,
        "mass_fit_R2": _safe_float(carrier_data.get("mass_fit_R2")),
        "mass_dynamic_band_switch": carrier_data.get("mass_dynamic_band_switch"),
        "mass_curvature_sign_ok": carrier_data.get("mass_curvature_sign_ok"),
        "mass_center_is_extremum": carrier_data.get("mass_center_is_extremum"),
        "mass_fixed_branch_energy_jump_max_eV": _safe_float(carrier_data.get("mass_fixed_branch_energy_jump_max_eV")),
        "mass_dynamic_edge_energy_jump_max_eV": _safe_float(carrier_data.get("mass_dynamic_edge_energy_jump_max_eV")),
        "mass_fit_window_stability_rel": _safe_float(carrier_data.get("mass_fit_window_stability_rel")),
        "warning_reasons": [],
        "status": "accepted",
        "reason": "channel_within_validation_window",
    }
    warning_reasons: list[str] = []
    if mass_valid is False or mass_status == "rejected":
        warning_reasons.append("effective_mass_quality_warning")

    metrics_present = all(
        _is_finite(value)
        for value in (mobility, e1, e1_fit_r2, c2d, c2d_fit_r2)
    )
    if not metrics_present:
        review["status"] = "rejected"
        review["reason"] = "missing_or_non_physical_channel_metrics"
        return review
    if float(mobility or 0.0) <= 0.0:
        warning_reasons.append("non_positive_signed_mobility")

    catastrophic_fit = float(e1_fit_r2 or 0.0) < SEVERE_E1_FIT_R2_THRESHOLD
    catastrophic_mobility = abs(float(mobility or 0.0)) >= SEVERE_MOBILITY_CM2_VS
    flat_response = abs(float(e1 or 0.0)) < FLAT_E1_ABS_EV_THRESHOLD and abs(float(mobility or 0.0)) >= SUSPICIOUS_MOBILITY_CM2_VS
    if catastrophic_mobility:
        review["status"] = "rejected"
        review["reason"] = "extreme_unphysical_mobility"
        return review
    if catastrophic_fit:
        warning_reasons.append("severe_e1_fit_quality_warning")
    if flat_response:
        warning_reasons.append("flat_deformation_potential_with_suspicious_mobility")

    needs_refinement = (
        float(e1_fit_r2 or 1.0) < float(fit_threshold)
        or float(c2d_fit_r2 or 1.0) < float(fit_threshold)
        or (rel_e1_sigma is not None and float(rel_e1_sigma) > REL_E1_SIGMA_REFINE_THRESHOLD)
        or (rel_c2d_sigma is not None and float(rel_c2d_sigma) > REL_C2D_SIGMA_REFINE_THRESHOLD)
    )
    if warning_reasons:
        review["warning_reasons"] = [str(item) for item in dedupe_keep_order(warning_reasons)]
    if needs_refinement:
        review["status"] = "refine_candidate"
        review["reason"] = "fit_quality_or_uncertainty_suggests_additional_sampling"
        return review

    if warning_reasons:
        review["status"] = "accepted_with_warning"
        review["reason"] = str(warning_reasons[0])
        return review

    return review


def _summarize_channel_reviews(reviews: dict[str, dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    retained_subchannels: list[str] = []
    rejected_subchannels: list[str] = []
    refinement_directions: list[str] = []
    for review in reviews.values():
        token = str(review.get("channel") or "").strip()
        status = str(review.get("status") or "").strip()
        direction = str(review.get("direction") or "").strip()
        if status in {"accepted", "accepted_with_warning", "refine_candidate"} and token:
            retained_subchannels.append(token)
        if status == "rejected" and token:
            rejected_subchannels.append(token)
        if status == "refine_candidate" and direction:
            refinement_directions.append(direction)
    return (
        [str(item) for item in dedupe_keep_order(retained_subchannels)],
        [str(item) for item in dedupe_keep_order(rejected_subchannels)],
        [str(item) for item in dedupe_keep_order(refinement_directions)],
    )


def build_validation_report(
    state: dict[str, Any],
    *,
    fit_threshold: float = DEFAULT_FIT_R2_THRESHOLD,
    max_points_per_direction: int = DEFAULT_REFINEMENT_TARGET_POINTS,
    anomaly_flags: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    raw_warnings = [str(item) for item in dedupe_keep_order(list(warnings or []))]
    physics = dict(state.get("physics_results", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    results = dict(physics.get("results", {}) or {})
    fit_metrics = {
        **validate_physics_window(results),
        **dict(diagnostics.get("fit_diagnostics", {}) or {}),
    }
    reviews: dict[str, dict[str, Any]] = {}
    for direction in VALID_DIRECTIONS:
        direction_data = dict((results.get("results_by_direction", {}) or {}).get(direction, {}) or {})
        for carrier in VALID_CARRIERS:
            review = _channel_review(
                direction=direction,
                carrier=carrier,
                direction_data=direction_data,
                carrier_data=dict(direction_data.get(carrier, {}) or {}),
                fit_threshold=float(fit_threshold),
            )
            reviews[str(review["channel"])] = review
    retained_subchannels, rejected_subchannels, refinement_directions = _summarize_channel_reviews(reviews)
    refinement_preview = resolve_refinement_sampling(
        state,
        {"target_channels": refinement_directions, "refinement_strategy": "midpoint_enrichment"},
        max_points_per_direction=max_points_per_direction,
        fit_threshold=fit_threshold,
    ) if refinement_directions else {
        "target_channels": [],
        "applied_points": {},
        "suggested_points": {},
        "full_plan_by_direction": dict(physics.get("strain_plan_by_direction", {}) or {}),
        "refinement_strategy": "midpoint_enrichment",
        "max_points_per_direction": max_points_per_direction,
    }
    if refinement_directions and not refinement_preview.get("applied_points"):
        workflow = dict(state.get("workflow", {}) or {})
        refinement_rounds = int(workflow.get("refinement_rounds", 0) or 0)
        max_rounds = int(workflow.get("max_refinement_rounds", 0) or 0)
        unresolved_reason = "refinement_budget_exhausted" if refinement_rounds >= max_rounds else "no_fresh_refinement_points"
        raw_warnings = dedupe_keep_order(
            list(raw_warnings)
            + [unresolved_reason, f"unresolved_refinement_targets:{','.join(refinement_directions)}"]
        )
        targeted_directions = set(refinement_directions)
        for review in reviews.values():
            if str(review.get("status") or "") != "refine_candidate":
                continue
            if str(review.get("direction") or "") not in targeted_directions:
                continue
            review["status"] = "accepted_with_warning"
            review["reason"] = unresolved_reason
            review["warning_reasons"] = [
                str(item)
                for item in dedupe_keep_order(list(review.get("warning_reasons", []) or []) + [unresolved_reason])
            ]
        retained_subchannels, rejected_subchannels, refinement_directions = _summarize_channel_reviews(reviews)

    recommended_action = "finalize"
    if refinement_directions and refinement_preview.get("applied_points"):
        recommended_action = "refine_sampling"
    accepted_directions, rejected_directions = derive_direction_acceptance(retained_subchannels, rejected_subchannels)
    if not retained_subchannels and rejected_subchannels:
        recommended_action = "finalize"

    decision_warnings = [item for item in raw_warnings if item not in {"dry_run_mode"}]
    decision = "pass"
    if not retained_subchannels and not refinement_preview.get("applied_points"):
        decision = "fail"
    elif recommended_action == "refine_sampling" or rejected_subchannels or list(anomaly_flags or []) or decision_warnings:
        decision = "pass_with_warning"

    return {
        "decision": decision,
        "reason": "physics_validation_review",
        "warnings": raw_warnings,
        "failed_checks": [str(item) for item in dedupe_keep_order(list(anomaly_flags or []))],
        "fit_metrics": fit_metrics,
        "anomaly_flags": [str(item) for item in dedupe_keep_order(list(anomaly_flags or []))],
        "channel_reviews": reviews,
        "retained_subchannels": [str(item) for item in dedupe_keep_order(retained_subchannels)],
        "rejected_subchannels": [str(item) for item in dedupe_keep_order(rejected_subchannels)],
        "accepted_channels": accepted_directions,
        "rejected_channels": rejected_directions,
        "recommended_action": recommended_action,
        "refinement_targets": refinement_directions,
        "refinement_preview": refinement_preview,
        "all_subchannels": default_subchannels(),
    }
