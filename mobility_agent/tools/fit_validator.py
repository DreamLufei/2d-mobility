from __future__ import annotations

import numpy as np
from typing import Any


def summarize_strain_fit_quality(strain_data: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in strain_data if row.get("completed", False)]
    failed = [row for row in strain_data if not row.get("completed", False)]
    per_direction: dict[str, dict[str, Any]] = {}
    overall_fit_values: list[float] = []
    energy_fit_values: list[float] = []
    edge_fit_values: list[float] = []
    directions_needing_refinement: list[str] = []

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    for direction in ["x", "y"]:
        subset = [row for row in completed if row.get("direction") == direction]
        failed_subset = [row for row in failed if row.get("direction") == direction]
        if not subset:
            per_direction[direction] = {
                "n_points": 0,
                "failed_points": len(failed_subset),
                "energy_fit_r2": 0.0,
                "edge_fit_r2": 0.0,
                "overall_fit_quality": 0.0,
                "fit_r2": 0.0,
                "valley_switch_detected": False,
            }
            directions_needing_refinement.append(direction)
            continue

        subset = sorted(subset, key=lambda row: float(row.get("strain", 0.0)))
        strain = np.asarray([float(row.get("strain", 0.0)) for row in subset], dtype=float)
        total_energy = np.asarray([float(row.get("total_energy", 0.0)) for row in subset], dtype=float)
        aligned_vbm = np.asarray([float(row.get("e_vbm", 0.0)) - float(row.get("e_vacuum", 0.0)) for row in subset], dtype=float)
        aligned_cbm = np.asarray([float(row.get("e_cbm", 0.0)) - float(row.get("e_vacuum", 0.0)) for row in subset], dtype=float)
        if len(subset) >= 3:
            coeffs_e = np.polyfit(strain, total_energy, 2)
            fit_e = np.polyval(coeffs_e, strain)
            r2_e = _r2(total_energy, fit_e)
        else:
            r2_e = 0.0
        if len(subset) >= 2:
            coeffs_d_vbm = np.polyfit(strain, aligned_vbm, 1)
            coeffs_d_cbm = np.polyfit(strain, aligned_cbm, 1)
            fit_d_vbm = np.polyval(coeffs_d_vbm, strain)
            fit_d_cbm = np.polyval(coeffs_d_cbm, strain)
            r2_d_vbm = _r2(aligned_vbm, fit_d_vbm)
            r2_d_cbm = _r2(aligned_cbm, fit_d_cbm)
            r2_d = min(float(r2_d_vbm), float(r2_d_cbm))
        else:
            r2_d_vbm = 0.0
            r2_d_cbm = 0.0
            r2_d = 0.0

        valley_switch = any(
            row.get("vbm_kpoint_global") != row.get("vbm_kpoint_ref")
            or row.get("cbm_kpoint_global") != row.get("cbm_kpoint_ref")
            for row in subset
        )
        fit_r2 = min(float(r2_e), float(r2_d))
        per_direction[direction] = {
            "n_points": len(subset),
            "failed_points": len(failed_subset),
            "energy_fit_r2": float(r2_e),
            "edge_fit_r2": float(r2_d),
            "edge_fit_r2_vbm": float(r2_d_vbm),
            "edge_fit_r2_cbm": float(r2_d_cbm),
            "overall_fit_quality": fit_r2,
            "fit_r2": fit_r2,
            "valley_switch_detected": valley_switch,
            "strain_points": [float(v) for v in strain.tolist()],
        }
        overall_fit_values.append(fit_r2)
        energy_fit_values.append(float(r2_e))
        edge_fit_values.append(float(r2_d))
        if len(subset) == 0 or valley_switch or len(failed_subset) > 0:
            directions_needing_refinement.append(direction)

    return {
        "n_points": len(completed),
        "n_points_total": len(completed),
        "n_points_by_direction": {direction: int((per_direction.get(direction) or {}).get("n_points", 0) or 0) for direction in ["x", "y"]},
        "failed_points": len(failed),
        "fit_r2": min(overall_fit_values) if overall_fit_values else 0.0,
        "energy_fit_r2_min": min(energy_fit_values) if energy_fit_values else 0.0,
        "edge_fit_r2_min": min(edge_fit_values) if edge_fit_values else 0.0,
        "overall_fit_quality": min(overall_fit_values) if overall_fit_values else 0.0,
        "valley_switch_detected": any(per_direction.get(d, {}).get("valley_switch_detected", False) for d in per_direction),
        "per_direction": per_direction,
        "directions_needing_refinement": directions_needing_refinement,
    }
