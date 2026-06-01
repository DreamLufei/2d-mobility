from __future__ import annotations

import math

from typing import Any


HIGH_QUALITY_LABEL = "high-quality"
MODERATE_QUALITY_LABEL = "moderate-quality"
NOT_RETAINED_LABEL = "not retained"


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _is_finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def classify_channel_quality(direction_data: dict[str, Any], carrier_data: dict[str, Any]) -> str:
    mobility = _safe_float(carrier_data.get("mobility_cm2_Vs"))
    e1 = _safe_float(carrier_data.get("E1_eV"))
    e1_sigma = _safe_float(carrier_data.get("E1_eV_sigma"))
    e1_fit_r2 = _safe_float(carrier_data.get("E1_fit_R2"))
    c2d = _safe_float(carrier_data.get("C2D_J_m2"))
    c2d_sigma = _safe_float(carrier_data.get("C2D_sigma_J_m2"))
    c2d_fit_r2 = _safe_float(carrier_data.get("C2D_fit_R2"))
    n_points = _safe_float(direction_data.get("n_points"))

    usable = (
        _is_finite_positive(mobility)
        and math.isfinite(e1)
        and math.isfinite(c2d)
        and math.isfinite(e1_fit_r2)
        and math.isfinite(c2d_fit_r2)
    )
    if not usable:
        return "failed"

    rel_e1_sigma = abs(e1_sigma / e1) if math.isfinite(e1_sigma) and abs(e1) > 1e-12 else float("inf")
    rel_c2d_sigma = abs(c2d_sigma / c2d) if math.isfinite(c2d_sigma) and abs(c2d) > 1e-12 else float("inf")
    min_fit_r2 = min(e1_fit_r2, c2d_fit_r2)
    e1_abs = abs(e1)

    if (
        n_points >= 7
        and min_fit_r2 >= 0.985
        and rel_e1_sigma <= 0.25
        and rel_c2d_sigma <= 0.10
        and e1_abs >= 0.10
    ):
        return "filtered"

    if (
        n_points >= 5
        and min_fit_r2 >= 0.95
        and rel_e1_sigma <= 0.50
        and rel_c2d_sigma <= 0.20
        and e1_abs >= 0.05
    ):
        return "caution"

    return "weak"


def classify_material_quality(results: dict[str, Any]) -> str:
    filtered_count = 0
    caution_count = 0

    for direction_data in dict(results.get("results_by_direction", {}) or {}).values():
        payload = dict(direction_data or {})
        for carrier in ("electron", "hole"):
            carrier_data = dict(payload.get(carrier, {}) or {})
            quality = classify_channel_quality(payload, carrier_data)
            if quality == "filtered":
                filtered_count += 1
            elif quality == "caution":
                caution_count += 1

    if filtered_count > 0:
        return HIGH_QUALITY_LABEL
    if caution_count > 0:
        return MODERATE_QUALITY_LABEL
    return NOT_RETAINED_LABEL
