from __future__ import annotations

from typing import Any


def validate_physics_window(results: dict[str, Any]) -> dict[str, Any]:
    anomaly_flags: list[str] = []
    warnings: list[str] = []
    effective_fit_values: list[float] = []
    energy_fit_values: list[float] = []
    edge_fit_values: list[float] = []
    e1_sigmas: list[float] = []
    c2d_sigmas: list[float] = []
    per_direction: dict[str, dict[str, Any]] = {}

    for direction, direction_data in dict(results.get("results_by_direction", {}) or {}).items():
        direction_energy_values: list[float] = []
        direction_edge_values: list[float] = []
        for carrier in ("electron", "hole"):
            carrier_data = dict(direction_data.get(carrier, {}) or {})
            if not carrier_data:
                continue
            edge_r2 = float(carrier_data.get("E1_fit_R2", 1.0) or 0.0)
            edge_fit_values.append(edge_r2)
            direction_edge_values.append(edge_r2)
            sigma = carrier_data.get("E1_eV_sigma")
            if sigma is not None:
                e1_sigmas.append(float(sigma))
            mobility = carrier_data.get("mobility_cm2_Vs")
            if mobility is not None and float(mobility) < 0:
                anomaly_flags.append("negative_mobility")
            if carrier_data.get("mass_valid_for_mobility") is False:
                anomaly_flags.append(f"effective_mass_quality_failed:{carrier}_{direction}")
            if carrier_data.get("mass_dynamic_band_switch") is True and carrier_data.get("mass_valid_for_mobility") is False:
                anomaly_flags.append(f"effective_mass_band_switch:{carrier}_{direction}")
            energy_r2 = carrier_data.get("C2D_fit_R2")
            if energy_r2 is not None:
                energy_r2_f = float(energy_r2)
                energy_fit_values.append(energy_r2_f)
                direction_energy_values.append(energy_r2_f)
        sigma_c2d = direction_data.get("electron", {}).get("C2D_sigma_J_m2") or direction_data.get("hole", {}).get("C2D_sigma_J_m2")
        if sigma_c2d is not None:
            c2d_sigmas.append(float(sigma_c2d))
        energy_min = min(direction_energy_values) if direction_energy_values else 1.0
        edge_min = min(direction_edge_values) if direction_edge_values else 1.0
        per_direction[str(direction)] = {
            "energy_fit_r2": energy_min,
            "edge_fit_r2": edge_min,
            "effective_fit_quality": min(energy_min, edge_min),
            "n_points": int(direction_data.get("n_points", 0) or 0),
        }
        effective_fit_values.append(min(energy_min, edge_min))

    return {
        "fit_r2_min": min(effective_fit_values) if effective_fit_values else 1.0,
        "energy_fit_r2_min": min(energy_fit_values) if energy_fit_values else 1.0,
        "edge_fit_r2_min": min(edge_fit_values) if edge_fit_values else 1.0,
        "effective_fit_quality": min(effective_fit_values) if effective_fit_values else 1.0,
        "e1_sigma_max": max(e1_sigmas) if e1_sigmas else 0.0,
        "c2d_sigma_max": max(c2d_sigmas) if c2d_sigmas else 0.0,
        "warnings": warnings,
        "anomaly_flags": anomaly_flags,
        "per_direction": per_direction,
    }
