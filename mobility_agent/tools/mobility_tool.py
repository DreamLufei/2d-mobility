from __future__ import annotations

import os
import json
import csv
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from pydantic import Field

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .physics_common import ANGSTROM_TO_M, E_CHARGE, EV_TO_J, HBAR, KB, M0


class MobilityToolInput(ToolInputBase):
    strain_data: list[dict[str, Any]] = Field(default_factory=list)
    electron_mass_x: float | None = None
    electron_mass_y: float | None = None
    electron_mass_dos: float | None = None
    hole_mass_x: float | None = None
    hole_mass_y: float | None = None
    hole_mass_dos: float | None = None
    mass_diagnostics: dict[str, Any] = Field(default_factory=dict)


class MobilityToolOutput(ToolOutputBase):
    results: dict[str, Any] = Field(default_factory=dict)
    mobility_summary: dict[str, Any] = Field(default_factory=dict)
    fit_diagnostics: dict[str, Any] = Field(default_factory=dict)


class MobilityTool(DeterministicTool):
    name = "compute_mobility"
    description = "Fit deformation-potential quantities and compute final mobilities"
    input_model = MobilityToolInput
    output_model = MobilityToolOutput

    def __init__(self, executor=None, *, temperature: float = 300.0, c2d_prefac: float = 1.0):
        super().__init__(executor=executor)
        self.temperature = temperature
        self.c2d_prefac = c2d_prefac

    def _resolve_structure_path(self, inputs: MobilityToolInput) -> str | None:
        payload = dict(inputs.state_payload or {})
        physics = dict(payload.get("physics_results", {}) or {})
        material = dict(payload.get("material", {}) or {})
        relaxed = (
            payload.get("relaxed_poscar")
            or payload.get("relaxed_structure_path")
            or physics.get("relaxed_poscar")
            or physics.get("relaxed_structure_path")
        )
        poscar = payload.get("poscar_path") or material.get("poscar_path")
        candidate = str(relaxed or poscar or "").strip()
        return candidate or None

    def _write_artifacts(self, inputs: MobilityToolInput, results: dict[str, Any], fit_diagnostics: dict[str, Any]) -> None:
        with open(os.path.join(inputs.base_dir, "mobility_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(os.path.join(inputs.base_dir, "fit_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(fit_diagnostics, f, ensure_ascii=False, indent=2)

        strain_rows = list(inputs.strain_data or [])
        if strain_rows:
            fieldnames = sorted({key for row in strain_rows for key in row.keys()})
            with open(os.path.join(inputs.base_dir, "strain_data.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in strain_rows:
                    writer.writerow(row)
            status_rows = [
                {
                    "direction": row.get("direction"),
                    "strain": row.get("strain"),
                    "completed": row.get("completed", False),
                    "error": row.get("error"),
                    "folder": row.get("folder"),
                }
                for row in strain_rows
            ]
            with open(os.path.join(inputs.base_dir, "strain_status.csv"), "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["direction", "strain", "completed", "error", "folder"])
                writer.writeheader()
                for row in status_rows:
                    writer.writerow(row)

    def _mass_diagnostics(self, inputs: MobilityToolInput) -> dict[str, Any]:
        if inputs.mass_diagnostics:
            return dict(inputs.mass_diagnostics)
        payload = dict(inputs.state_payload or {})
        physics = dict(payload.get("physics_results", {}) or {})
        summary = dict(physics.get("effective_mass_summary", {}) or {})
        diagnostics = dict(summary.get("mass_diagnostics", {}) or {})
        return diagnostics

    def _mass_quality_for_channel(self, diagnostics: dict[str, Any], carrier: str, direction: str) -> dict[str, Any]:
        token = f"{carrier}_{direction}"
        if not diagnostics:
            return {
                "mass_status": "accepted",
                "mass_valid_for_mobility": True,
                "mass_qc_available": False,
                "mass_rejection_reasons": [],
            }
        current = dict(diagnostics.get(token, {}) or {})
        other_direction = "y" if direction == "x" else "x"
        other = dict(diagnostics.get(f"{carrier}_{other_direction}", {}) or {})
        current_status = str(current.get("status") or "missing")
        other_status = str(other.get("status") or "missing")
        current_ok = current_status == "accepted"
        dos_ok = other_status == "accepted"
        reasons = list(current.get("rejection_reasons", []) or [])
        if not current_ok and not reasons:
            reasons.append(f"mass_status_{current_status}")
        if not dos_ok:
            reasons.append(f"dos_mass_{other_direction}_status_{other_status}")
        return {
            "mass_status": current_status,
            "mass_valid_for_mobility": bool(current_ok and dos_ok),
            "mass_qc_available": True,
            "mass_rejection_reasons": reasons,
            "mass_fit_R2": current.get("mass_fit_R2"),
            "mass_curvature_sign_ok": current.get("curvature_sign_ok"),
            "mass_center_is_extremum": current.get("center_is_extremum"),
            "mass_dynamic_band_switch": current.get("dynamic_band_switch"),
            "mass_fixed_branch_energy_jump_max_eV": current.get("fixed_branch_energy_jump_max_eV"),
            "mass_dynamic_edge_energy_jump_max_eV": current.get("dynamic_edge_energy_jump_max_eV"),
            "mass_fit_window_stability_rel": current.get("fit_window_stability_rel"),
            "mass_fixed_band_index": current.get("fixed_band_index"),
            "mass_dynamic_band_indices": current.get("dynamic_band_indices", []),
        }

    def _execute(self, inputs: MobilityToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        from pymatgen.core import Structure

        all_strain_points = list(inputs.strain_data or [])
        completed_all = [d for d in all_strain_points if d.get("completed", False)]
        failed_all = [d for d in all_strain_points if not d.get("completed", False)]
        problems: list[str] = []
        per_dir_counts: dict[str, int] = {}
        for dir_name in ["x", "y"]:
            subset = [d for d in completed_all if d.get("direction", "x") == dir_name]
            per_dir_counts[dir_name] = len(subset)
            if len(subset) < 5:
                problems.append(f"{dir_name} 方向有效数据不足(<5): {len(subset)}")
                continue
            strains = sorted({float(d.get("strain")) for d in subset})
            if 0.0 not in strains:
                problems.append(f"{dir_name} 方向缺少 ε=0.0 点")
            if not (min(strains) < 0.0 and max(strains) > 0.0):
                problems.append(f"{dir_name} 方向缺少正负两侧应变点")
        if problems:
            diag = {
                "completed_points": len(completed_all),
                "failed_points": len(failed_all),
                "per_direction_completed": per_dir_counts,
                "problems": problems,
            }
            self._write_artifacts(inputs, {}, diag)
            return {
                "errors": ["; ".join(problems)],
                "fit_diagnostics": diag,
                "mobility_summary": {
                    "status_label": "failed",
                    "direction_count": 0,
                    "failed_points": len(failed_all),
                },
            }

        structure_path = self._resolve_structure_path(inputs)
        if not structure_path or not os.path.exists(structure_path):
            return {
                "errors": [f"missing_structure_path_for_mobility:{structure_path or 'none'}"],
                "fit_diagnostics": {
                    "completed_points": len(completed_all),
                    "failed_points": len(failed_all),
                    "per_direction_completed": per_dir_counts,
                    "problems": [f"mobility stage missing structure path: {structure_path or 'none'}"],
                },
                "mobility_summary": {
                    "status_label": "failed",
                    "direction_count": 0,
                    "failed_points": len(failed_all),
                },
            }

        struct = Structure.from_file(str(structure_path))
        a_vec = np.array(struct.lattice.matrix[0], dtype=float)
        b_vec = np.array(struct.lattice.matrix[1], dtype=float)
        area_A2 = float(np.linalg.norm(np.cross(a_vec, b_vec)))
        area_m2 = float(area_A2 * (ANGSTROM_TO_M**2))

        def quad(x, a, b, c):
            return a * x**2 + b * x + c

        def linear(x, a, b):
            return a * x + b

        def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
            return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results_by_dir: dict[str, Any] = {}
        effective_fit_values: list[float] = []
        energy_fit_values: list[float] = []
        edge_fit_values: list[float] = []
        e1_sigma_values: list[float] = []
        c2d_sigma_values: list[float] = []
        per_direction_fit: dict[str, Any] = {}
        mass_diagnostics = self._mass_diagnostics(inputs)
        for dir_name in ["x", "y"]:
            subset = [d for d in completed_all if d.get("direction", "x") == dir_name]
            subset = sorted(subset, key=lambda row: float(row.get("strain", 0.0)))
            strain = np.asarray([float(d.get("strain", 0.0)) for d in subset], dtype=float)
            etot = np.asarray([float(d.get("total_energy", 0.0)) for d in subset], dtype=float)
            e_vbm = np.asarray([float(d.get("e_vbm", 0.0)) for d in subset], dtype=float)
            e_cbm = np.asarray([float(d.get("e_cbm", 0.0)) for d in subset], dtype=float)
            e_vac = np.asarray([float(d.get("e_vacuum", 0.0)) for d in subset], dtype=float)
            vbm_aligned = e_vbm - e_vac
            cbm_aligned = e_cbm - e_vac
            popt_e, pcov_e = curve_fit(quad, strain, etot)
            a_e = float(popt_e[0])
            a_e_err = float(np.sqrt(max(float(pcov_e[0, 0]), 0.0))) if pcov_e is not None else float("nan")
            etot_fit = quad(strain, *popt_e)
            r2_etot = r2_score(etot, etot_fit)
            d2E_dEps2_eV = float(2.0 * a_e)
            d2E_dEps2_eV_err = float(2.0 * a_e_err) if np.isfinite(a_e_err) else float("nan")
            c2d = float(self.c2d_prefac * d2E_dEps2_eV * EV_TO_J / area_m2)
            c2d_err = float(self.c2d_prefac * d2E_dEps2_eV_err * EV_TO_J / area_m2) if np.isfinite(d2E_dEps2_eV_err) else float("nan")
            energy_fit_values.append(float(r2_etot))
            if np.isfinite(c2d_err):
                c2d_sigma_values.append(float(c2d_err))
            results_all = {}
            dir_edge_r2_values: list[float] = []
            for carrier, e_aligned in [("electron", cbm_aligned), ("hole", vbm_aligned)]:
                popt_d, pcov_d = curve_fit(linear, strain, e_aligned)
                e1 = float(popt_d[0])
                e1_err = float(np.sqrt(max(float(pcov_d[0, 0]), 0.0))) if pcov_d is not None else float("nan")
                e_fit = linear(strain, *popt_d)
                r2_e1 = r2_score(e_aligned, e_fit)
                edge_fit_values.append(float(r2_e1))
                dir_edge_r2_values.append(float(r2_e1))
                if np.isfinite(e1_err):
                    e1_sigma_values.append(float(e1_err))
                if dir_name == "x":
                    m_trans = inputs.electron_mass_x if carrier == "electron" else inputs.hole_mass_x
                else:
                    m_trans = inputs.electron_mass_y if carrier == "electron" else inputs.hole_mass_y
                m_dos = inputs.electron_mass_dos if carrier == "electron" else inputs.hole_mass_dos
                mass_quality = self._mass_quality_for_channel(mass_diagnostics, carrier, dir_name)
                if m_trans is None or m_dos is None:
                    results_all[carrier] = {
                        "mobility_cm2_Vs": None,
                        "mobility_m2_Vs": None,
                        "raw_mobility_cm2_Vs": None,
                        "raw_mobility_m2_Vs": None,
                        "m_transport": None if m_trans is None else float(m_trans),
                        "m_dos": None if m_dos is None else float(m_dos),
                        "E1_eV": float(e1),
                        "E1_eV_sigma": float(e1_err) if np.isfinite(e1_err) else None,
                        "E1_fit_R2": float(r2_e1),
                        "C2D_J_m2": float(c2d),
                        "C2D_sigma_J_m2": float(c2d_err) if np.isfinite(c2d_err) else None,
                        "C2D_fit_R2": float(r2_etot),
                        "d2E_dEps2_eV": float(d2E_dEps2_eV),
                        "C2D_prefactor": float(self.c2d_prefac),
                        **mass_quality,
                        "mass_valid_for_mobility": False,
                        "mass_rejection_reasons": list(mass_quality.get("mass_rejection_reasons", []) or []) + ["missing_effective_mass"],
                    }
                    continue
                m_trans_si = float(m_trans) * M0
                m_dos_si = float(m_dos) * M0
                e1_si = abs(e1) * EV_TO_J
                mu_si = float(E_CHARGE * (HBAR**3) * c2d / (KB * self.temperature * m_trans_si * m_dos_si * (e1_si**2)))
                mass_valid = bool(mass_quality.get("mass_valid_for_mobility", True))
                mobility_qc_status = "trusted" if mass_valid else "mass_qc_warning"
                results_all[carrier] = {
                    "mobility_cm2_Vs": float(mu_si * 1e4),
                    "mobility_m2_Vs": float(mu_si),
                    "raw_mobility_cm2_Vs": float(mu_si * 1e4),
                    "raw_mobility_m2_Vs": float(mu_si),
                    "mobility_qc_status": mobility_qc_status,
                    "m_transport": float(m_trans),
                    "m_dos": float(m_dos),
                    "E1_eV": float(e1),
                    "E1_eV_sigma": float(e1_err) if np.isfinite(e1_err) else None,
                    "E1_fit_R2": float(r2_e1),
                    "C2D_J_m2": float(c2d),
                    "C2D_sigma_J_m2": float(c2d_err) if np.isfinite(c2d_err) else None,
                    "C2D_fit_R2": float(r2_etot),
                    "d2E_dEps2_eV": float(d2E_dEps2_eV),
                    "C2D_prefactor": float(self.c2d_prefac),
                    **mass_quality,
                }
            results_by_dir[dir_name] = {
                "electron": results_all.get("electron"),
                "hole": results_all.get("hole"),
                "elastic_modulus_C2D_J_m2": float(c2d),
                "area_A2": float(area_A2),
                "n_points": int(len(subset)),
            }
            dir_effective_fit = min([float(r2_etot)] + dir_edge_r2_values) if dir_edge_r2_values else float(r2_etot)
            effective_fit_values.append(dir_effective_fit)
            per_direction_fit[dir_name] = {
                "energy_fit_r2": float(r2_etot),
                "edge_fit_r2": min(dir_edge_r2_values) if dir_edge_r2_values else 1.0,
                "effective_fit_quality": dir_effective_fit,
                "n_points": int(len(subset)),
            }

        final_results = {
            "material_id": inputs.material_id,
            "temperature_K": float(self.temperature),
            "results_by_direction": results_by_dir,
            "area_A2": float(area_A2),
            "relax_retry_backups": list((inputs.state_payload or {}).get("relax_retry_backups", []) or []),
            "mass_diagnostics": mass_diagnostics,
        }
        fit_diagnostics = {
            "completed_points": len(completed_all),
            "failed_points": len(failed_all),
            "fit_r2_min": min(effective_fit_values) if effective_fit_values else 1.0,
            "energy_fit_r2_min": min(energy_fit_values) if energy_fit_values else 1.0,
            "edge_fit_r2_min": min(edge_fit_values) if edge_fit_values else 1.0,
            "effective_fit_quality": min(effective_fit_values) if effective_fit_values else 1.0,
            "e1_sigma_max": max(e1_sigma_values) if e1_sigma_values else 0.0,
            "c2d_sigma_max": max(c2d_sigma_values) if c2d_sigma_values else 0.0,
            "per_direction": per_direction_fit,
        }
        self._write_artifacts(inputs, final_results, fit_diagnostics)
        return {
            "results": final_results,
            "fit_diagnostics": fit_diagnostics,
            "mobility_summary": {
                "status_label": "completed",
                "confidence_score": (inputs.state_payload or {}).get("confidence_score"),
                "direction_count": len(results_by_dir),
                "failed_points": len(failed_all),
            },
        }

    def _build_output(self, inputs: MobilityToolInput, raw: dict[str, Any], duration_s: float) -> MobilityToolOutput:
        results = dict(raw.get("results", {}) or {})
        fit_diagnostics = dict(raw.get("fit_diagnostics", {}) or {})
        summary = dict(raw.get("mobility_summary", {}) or {})
        diag_path = os.path.join(inputs.base_dir, "fit_diagnostics.json")
        return MobilityToolOutput(
            success=not bool(raw.get("errors")),
            warnings=list(raw.get("warnings", []) or []),
            key_summary={
                "status_label": summary.get("status_label"),
                "confidence_score": summary.get("confidence_score"),
                "results_by_direction": list((results.get("results_by_direction") or {}).keys()),
            },
            artifact_paths=build_artifact_map(
                os.path.join(inputs.base_dir, "mobility_results.json"),
                os.path.join(inputs.base_dir, "strain_data.csv"),
                os.path.join(inputs.base_dir, "strain_status.csv"),
                diag_path,
            ),
            error_summary=("; ".join(raw.get("errors", [])) if raw.get("errors") else None),
            duration_s=duration_s,
            state_updates=raw,
            results=results,
            mobility_summary=summary or {
                "status_label": raw.get("status_label"),
                "confidence_score": raw.get("confidence_score"),
                "direction_count": len((results.get("results_by_direction") or {})),
            },
            fit_diagnostics=fit_diagnostics,
        )
