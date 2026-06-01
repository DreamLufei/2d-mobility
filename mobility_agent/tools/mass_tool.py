from __future__ import annotations

import os
import json
import shutil
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field
from pymatgen.io.vasp.inputs import Kpoints

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .physics_common import (
    calculate_effective_mass,
    frac_to_cart_k,
    read_eigenval_with_occupations,
    read_fermi_energy_eV,
)
from .vasp_common import (
    policy_stage_planning_allowed,
    prune_dir_keep_files,
    read_chgcar_compatible_incar_overrides,
    run_vasp,
    summarize_vasp_failure,
    symlink_force,
    write_incar,
)

if TYPE_CHECKING:
    from ..policy.engine import AgenticPolicyEngine

class MassToolInput(ToolInputBase):
    poscar_path: str | None = None
    potcar_path: str | None = None
    reciprocal_lattice: list[list[float]]
    vbm_kpoint: list[float]
    cbm_kpoint: list[float]
    vbm_band_index: int
    cbm_band_index: int
    vbm_spin: int | None = None
    cbm_spin: int | None = None
    fermi_energy: float | None = None


class MassToolOutput(ToolOutputBase):
    electron_mass_x: float | None = None
    electron_mass_y: float | None = None
    electron_mass_dos: float | None = None
    hole_mass_x: float | None = None
    hole_mass_y: float | None = None
    hole_mass_dos: float | None = None
    mass_summary: dict[str, object] = Field(default_factory=dict)
    mass_diagnostics: dict[str, object] = Field(default_factory=dict)


class MassTool(DeterministicTool):
    name = "compute_effective_mass"
    description = "Compute transport and DOS effective masses from band-edge sampling"
    input_model = MassToolInput
    output_model = MassToolOutput

    def __init__(
        self,
        executor=None,
        *,
        vasp_cmd: str = "mpirun -np 4 vasp_std > sout 2>&1",
        consider_spin: bool = False,
        effmass_npoints: int = 21,
        effmass_delta_k_cart: float = 0.01,
        effmass_fit_kmax: float = 0.03,
        min_mass_fit_r2: float = 0.95,
        min_effective_mass_m0: float = 0.02,
        max_effective_mass_m0: float = 20.0,
        max_branch_energy_jump_eV: float = 0.50,
        max_fit_window_stability_rel: float = 1.0,
        fermi_tolerance_eV: float = 1.0e-3,
        policy_engine: AgenticPolicyEngine | None = None,
    ):
        super().__init__(executor=executor)
        self.vasp_cmd = vasp_cmd
        self.consider_spin = consider_spin
        self.effmass_npoints = effmass_npoints
        self.effmass_delta_k_cart = effmass_delta_k_cart
        self.effmass_fit_kmax = effmass_fit_kmax
        self.min_mass_fit_r2 = min_mass_fit_r2
        self.min_effective_mass_m0 = min_effective_mass_m0
        self.max_effective_mass_m0 = max_effective_mass_m0
        self.max_branch_energy_jump_eV = max_branch_energy_jump_eV
        self.max_fit_window_stability_rel = max_fit_window_stability_rel
        self.fermi_tolerance_eV = fermi_tolerance_eV
        self.policy_engine = policy_engine

    def _resolve_fermi_energy(self, inputs: MassToolInput, work_dir: str) -> tuple[float | None, str]:
        if inputs.fermi_energy is not None:
            return float(inputs.fermi_energy), "scf_state"
        scf_dir = os.path.join(inputs.base_dir, "02_scf")
        try:
            return float(read_fermi_energy_eV(scf_dir)), "scf_dir"
        except Exception:
            pass
        try:
            return float(read_fermi_energy_eV(work_dir)), "effective_mass_dir"
        except Exception:
            return None, "unavailable"

    def _dynamic_band_edge(
        self,
        energies: np.ndarray,
        occ: np.ndarray,
        spin_idx: int,
        carrier_type: str,
        fermi_energy: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        e_edge = np.zeros(energies.shape[1], dtype=float)
        picked_band = np.full(energies.shape[1], -1, dtype=int)
        selector_source = "fermi_energy" if fermi_energy is not None else "occupancy"
        ef = float(fermi_energy) if fermi_energy is not None else None
        tol = abs(float(self.fermi_tolerance_eV))
        for ik in range(energies.shape[1]):
            e_k = energies[spin_idx, ik, :]
            o_k = occ[spin_idx, ik, :]
            if carrier_type == "hole":
                if ef is not None:
                    mask = e_k <= ef - tol
                    if not np.any(mask):
                        mask = e_k <= ef + tol
                else:
                    mask = o_k >= 0.5
                if not np.any(mask):
                    raise ValueError(f"有效质量({carrier_type}) k点{ik} 无费米能级以下价带态")
                e_mask = np.where(mask, e_k, -np.inf)
                ib = int(np.argmax(e_mask))
            else:
                if ef is not None:
                    mask = e_k >= ef + tol
                    if not np.any(mask):
                        mask = e_k >= ef - tol
                else:
                    mask = o_k <= 0.5
                if not np.any(mask):
                    raise ValueError(f"有效质量({carrier_type}) k点{ik} 无费米能级以上导带态")
                e_mask = np.where(mask, e_k, np.inf)
                ib = int(np.argmin(e_mask))
            e_edge[ik] = float(e_mask[ib])
            picked_band[ik] = ib
        return e_edge, picked_band, selector_source

    def _fit_mass_branch(
        self,
        *,
        carrier_type: str,
        direction: str,
        k_array: np.ndarray,
        branch_energy: np.ndarray,
        fixed_band_index: int,
        dynamic_band_indices: np.ndarray,
        dynamic_edge_energy: np.ndarray,
        fixed_branch_occupations: np.ndarray | None = None,
        fermi_energy: float | None = None,
        fermi_source: str | None = None,
        dynamic_selector_source: str = "occupancy",
    ) -> tuple[float | None, dict[str, Any]]:
        center_index = int(len(k_array) // 2)
        k_centered = np.asarray(k_array, dtype=float) - float(k_array[center_index])
        fit_mask = np.abs(k_centered) <= float(self.effmass_fit_kmax)
        if np.sum(fit_mask) < 5:
            fit_mask = np.ones_like(k_centered, dtype=bool)

        fit_k = k_centered[fit_mask]
        fit_e = np.asarray(branch_energy, dtype=float)[fit_mask]
        fit_dynamic_bands = np.asarray(dynamic_band_indices, dtype=int)[fit_mask]
        fit_dynamic_e = np.asarray(dynamic_edge_energy, dtype=float)[fit_mask]
        fit_occ = np.asarray(fixed_branch_occupations, dtype=float)[fit_mask] if fixed_branch_occupations is not None else None
        energy_jumps = np.abs(np.diff(fit_e)) if len(fit_e) > 1 else np.asarray([], dtype=float)
        dynamic_energy_jumps = np.abs(np.diff(fit_dynamic_e)) if len(fit_dynamic_e) > 1 else np.asarray([], dtype=float)
        fixed_band_indices = [int(fixed_band_index)] * int(np.sum(fit_mask))
        rejection_reasons: list[str] = []
        warnings: list[str] = []

        try:
            m_eff, a_fit, r_sq = calculate_effective_mass(fit_k, fit_e, carrier_type)
        except Exception as exc:
            diag = {
                "channel": f"{carrier_type}_{direction}",
                "carrier": carrier_type,
                "direction": direction,
                "status": "rejected",
                "rejection_reasons": ["mass_fit_failed"],
                "error": str(exc),
                "fixed_band_index": int(fixed_band_index),
                "dynamic_band_indices": [int(v) for v in fit_dynamic_bands.tolist()],
            }
            return None, diag

        curvature_sign_ok = bool(a_fit > 0.0) if carrier_type == "electron" else bool(a_fit < 0.0)
        center_e = float(branch_energy[center_index])
        if carrier_type == "electron":
            center_is_extremum = bool(center_e <= float(np.min(fit_e)) + 1.0e-3)
        else:
            center_is_extremum = bool(center_e >= float(np.max(fit_e)) - 1.0e-3)

        dynamic_band_switch = bool(len(set(int(v) for v in fit_dynamic_bands.tolist())) > 1)
        fixed_energy_jump_max = float(np.max(energy_jumps)) if energy_jumps.size else 0.0
        dynamic_energy_jump_max = float(np.max(dynamic_energy_jumps)) if dynamic_energy_jumps.size else 0.0
        fixed_branch_crosses_fermi: bool | None = None
        fixed_branch_fermi_side_ok: bool | None = None
        if fermi_energy is not None:
            ef = float(fermi_energy)
            tol = abs(float(self.fermi_tolerance_eV))
            if carrier_type == "hole":
                fixed_branch_crosses_fermi = bool(np.any(fit_e > ef + tol))
            else:
                fixed_branch_crosses_fermi = bool(np.any(fit_e < ef - tol))
            fixed_branch_fermi_side_ok = not fixed_branch_crosses_fermi

        partial_occupation_in_fit_window: bool | None = None
        fixed_branch_occ_min: float | None = None
        fixed_branch_occ_max: float | None = None
        if fit_occ is not None and len(fit_occ):
            fixed_branch_occ_min = float(np.min(fit_occ))
            fixed_branch_occ_max = float(np.max(fit_occ))
            partial_occupation_in_fit_window = bool(np.any((fit_occ > 1.0e-3) & (fit_occ < 1.0 - 1.0e-3)))

        fit_window_stability_rel: float | None = None
        inner_limit = min(float(self.effmass_fit_kmax) * 2.0 / 3.0, float(self.effmass_fit_kmax) - 1.0e-12)
        inner_mask = np.abs(k_centered) <= inner_limit
        if np.sum(inner_mask) >= 5:
            try:
                inner_m_eff, _inner_a, _inner_r2 = calculate_effective_mass(k_centered[inner_mask], np.asarray(branch_energy, dtype=float)[inner_mask], carrier_type)
                if np.isfinite(inner_m_eff) and abs(float(m_eff)) > 1.0e-12:
                    fit_window_stability_rel = abs(float(inner_m_eff) - float(m_eff)) / abs(float(m_eff))
            except Exception:
                warnings.append("inner_window_mass_fit_failed")

        if dynamic_band_switch:
            warnings.append(f"{dynamic_selector_source}_edge_selector_switch")
        if fixed_energy_jump_max > float(self.max_branch_energy_jump_eV):
            rejection_reasons.append("large_fixed_branch_energy_jump")
        if dynamic_energy_jump_max > float(self.max_branch_energy_jump_eV):
            warnings.append("large_dynamic_edge_energy_jump")
        if fixed_branch_crosses_fermi is True:
            rejection_reasons.append("fixed_branch_crosses_fermi")
        if partial_occupation_in_fit_window is True:
            warnings.append("fixed_branch_partial_occupation_in_fit_window")
        if float(r_sq) < float(self.min_mass_fit_r2):
            rejection_reasons.append("mass_fit_r2_below_threshold")
        if not curvature_sign_ok:
            rejection_reasons.append("wrong_curvature_sign")
        if not center_is_extremum:
            rejection_reasons.append("center_not_band_extremum")
        if float(m_eff) < float(self.min_effective_mass_m0):
            rejection_reasons.append("anomalously_small_effective_mass")
        if float(m_eff) > float(self.max_effective_mass_m0):
            rejection_reasons.append("unusually_large_effective_mass")
        if fit_window_stability_rel is not None and fit_window_stability_rel > float(self.max_fit_window_stability_rel):
            rejection_reasons.append("mass_fit_window_unstable")

        status = "accepted" if not rejection_reasons else "rejected"
        diag = {
            "channel": f"{carrier_type}_{direction}",
            "carrier": carrier_type,
            "direction": direction,
            "status": status,
            "rejection_reasons": rejection_reasons,
            "warnings": warnings,
            "mass_m0": float(m_eff),
            "a_fit_eVA2": float(a_fit),
            "mass_fit_R2": float(r_sq),
            "curvature_sign_ok": curvature_sign_ok,
            "center_is_extremum": center_is_extremum,
            "fixed_band_index": int(fixed_band_index),
            "fixed_band_indices": fixed_band_indices,
            "dynamic_band_indices": [int(v) for v in fit_dynamic_bands.tolist()],
            "dynamic_band_switch": dynamic_band_switch,
            "dynamic_selector_source": dynamic_selector_source,
            "fermi_energy_eV": float(fermi_energy) if fermi_energy is not None else None,
            "fermi_source": fermi_source,
            "fixed_branch_fermi_side_ok": fixed_branch_fermi_side_ok,
            "fixed_branch_crosses_fermi": fixed_branch_crosses_fermi,
            "fixed_branch_occ_min": fixed_branch_occ_min,
            "fixed_branch_occ_max": fixed_branch_occ_max,
            "partial_occupation_in_fit_window": partial_occupation_in_fit_window,
            "fixed_branch_energy_jump_max_eV": fixed_energy_jump_max,
            "dynamic_edge_energy_jump_max_eV": dynamic_energy_jump_max,
            "fit_window_kmax_Ainv": float(np.max(np.abs(fit_k))) if len(fit_k) else None,
            "fit_point_count": int(len(fit_k)),
            "fit_window_stability_rel": fit_window_stability_rel,
            "thresholds": {
                "min_mass_fit_r2": float(self.min_mass_fit_r2),
                "min_effective_mass_m0": float(self.min_effective_mass_m0),
                "max_effective_mass_m0": float(self.max_effective_mass_m0),
                "max_branch_energy_jump_eV": float(self.max_branch_energy_jump_eV),
                "max_fit_window_stability_rel": float(self.max_fit_window_stability_rel),
            },
        }
        return float(m_eff), diag

    def _execute(self, inputs: MassToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        if not inputs.poscar_path or not inputs.potcar_path:
            raise ValueError("mass tool requires poscar_path and potcar_path")

        rec_lattice = np.asarray(inputs.reciprocal_lattice, dtype=float)
        mass_diagnostics: dict[str, Any] = {}
        chgcar_compatible_overrides = read_chgcar_compatible_incar_overrides(inputs.base_dir)
        effmass_incar_template: dict[str, Any] = {
            "ISTART": 0,
            "ICHARG": 11,
            "ISMEAR": 0,
            "SIGMA": 0.01,
            "LREAL": "Auto",
            "PREC": "Normal",
            "EDIFF": 1e-6,
            "ENCUT": 600,
            "NELM": 300,
            "IVDW": 12,
            "NELMIN": 4,
            "ALGO": "Normal",
            "LCHARG": False,
            "LWAVE": False,
            "LELF": False,
        }
        effmass_incar_template.update(chgcar_compatible_overrides)
        if self.consider_spin and "ISPIN" not in effmass_incar_template:
            effmass_incar_template["ISPIN"] = 2
        policy_updates: dict[str, Any] = {}
        if self.policy_engine is not None and policy_stage_planning_allowed(inputs.state_payload, "effective_mass"):
            plan = self.policy_engine.plan_stage(
                stage="effective_mass",
                state_payload=dict(inputs.state_payload or {}),
                default_incar=effmass_incar_template,
                default_kpoints_policy={
                    "line_mode_density": int(self.effmass_npoints),
                },
                extra_context={
                    "charge_density_source": os.path.join(inputs.base_dir, "02_scf", "CHGCAR"),
                    "nscf_requires_chgcar_grid_compatibility": True,
                },
            )
            effmass_incar_template.update(dict(plan.incar_overrides or {}))
            effmass_incar_template.update(chgcar_compatible_overrides)
            policy_updates = {
                "services": {
                    "parameter_plans": {"effective_mass": plan.model_dump(mode="json")},
                    "retrieval_trace": [
                        {
                            "kind": "parameter_plan",
                            "stage": "effective_mass",
                            "source": plan.source,
                            "confidence": plan.confidence,
                            "evidence": [item.model_dump(mode="json") for item in list(plan.evidence_items or [])],
                            "rationale": plan.rationale,
                        }
                    ],
                }
            }
        carriers = {
            "electron": {
                "k_frac": list(inputs.cbm_kpoint),
                "band_idx": int(inputs.cbm_band_index),
                "spin_idx": int(inputs.cbm_spin or 0),
                "masses": {},
            },
            "hole": {
                "k_frac": list(inputs.vbm_kpoint),
                "band_idx": int(inputs.vbm_band_index),
                "spin_idx": int(inputs.vbm_spin or 0),
                "masses": {},
            },
        }
        for carrier_type, cinfo in carriers.items():
            k0_frac = np.array(cinfo["k_frac"], dtype=float)
            k0_cart = frac_to_cart_k(k0_frac, rec_lattice)
            for direction, dir_name in [(0, "x"), (1, "y")]:
                work_dir = os.path.join(inputs.base_dir, f"04_effmass_{carrier_type}_{dir_name}")
                os.makedirs(work_dir, exist_ok=True)
                shutil.copy(str(inputs.poscar_path), os.path.join(work_dir, "POSCAR"))
                shutil.copy(str(inputs.potcar_path), os.path.join(work_dir, "POTCAR"))
                chgcar_src = os.path.join(inputs.base_dir, "02_scf", "CHGCAR")
                if os.path.exists(chgcar_src):
                    symlink_force(chgcar_src, os.path.join(work_dir, "CHGCAR"))

                b_vec = rec_lattice[direction]
                b_length = np.linalg.norm(b_vec)
                if b_length < 1e-8:
                    raise ValueError(f"倒格矢长度异常: direction={direction}, |b|={b_length}")
                delta_k_frac = float(self.effmass_delta_k_cart) / float(b_length)
                npoints = int(self.effmass_npoints)
                if npoints % 2 == 0:
                    npoints += 1
                kpoints_frac: list[list[float]] = []
                k_cart_values: list[float] = []
                b_unit = b_vec / b_length
                for idx in range(npoints):
                    dk_idx = idx - npoints // 2
                    k_frac = k0_frac.copy()
                    k_frac[direction] += dk_idx * delta_k_frac
                    kpoints_frac.append(k_frac.tolist())
                    k_cart = frac_to_cart_k(k_frac.tolist(), rec_lattice)
                    dk_signed = float(np.dot(k_cart - k0_cart, b_unit))
                    k_cart_values.append(dk_signed)

                kpts_obj = Kpoints(
                    comment=f"Effective mass {carrier_type} {dir_name}",
                    num_kpts=npoints,
                    style=Kpoints.supported_modes.Reciprocal,
                    kpts=kpoints_frac,
                    kpts_weights=[1] * npoints,
                )
                kpts_obj.write_file(os.path.join(work_dir, "KPOINTS"))
                incar_params = dict(effmass_incar_template)
                incar_params.update({
                    "SYSTEM": f"Effective Mass {carrier_type} {dir_name}",
                    "ICHARG": 11,
                    "LCHARG": False,
                    "LWAVE": False,
                    "LELF": False,
                })
                if self.consider_spin:
                    incar_params["ISPIN"] = 2
                write_incar(work_dir, incar_params)
                if not run_vasp(cwd=work_dir, vasp_cmd=self.vasp_cmd, check_convergence=False):
                    recovery_summary = summarize_vasp_failure(
                        work_dir,
                        stage="effective_mass",
                        default_error=f"{carrier_type} {dir_name} 有效质量计算失败",
                    )
                    return {
                        "errors": [str(recovery_summary["error_summary"])],
                        "recovery_summary": recovery_summary,
                        **policy_updates,
                    }

                eigenval = os.path.join(work_dir, "EIGENVAL")
                _kpts_frac_read, energies, occ, _ = read_eigenval_with_occupations(eigenval)
                spin_idx = int(cinfo["spin_idx"])
                if spin_idx >= energies.shape[0]:
                    spin_idx = 0
                fermi_energy, fermi_source = self._resolve_fermi_energy(inputs, work_dir)
                dynamic_edge, dynamic_picked_band, dynamic_selector_source = self._dynamic_band_edge(
                    energies,
                    occ,
                    spin_idx,
                    carrier_type,
                    fermi_energy=fermi_energy,
                )
                fixed_band_idx = max(0, min(int(cinfo["band_idx"]), int(energies.shape[2]) - 1))
                e_branch = np.asarray(energies[spin_idx, :, fixed_band_idx], dtype=float)
                fixed_occ = np.asarray(occ[spin_idx, :, fixed_band_idx], dtype=float)
                k_array = np.asarray(k_cart_values, dtype=float)
                center_index = int(len(k_array) // 2)
                k_array_centered = k_array - float(k_array[center_index])
                m_eff, channel_diag = self._fit_mass_branch(
                    carrier_type=carrier_type,
                    direction=dir_name,
                    k_array=k_array,
                    branch_energy=e_branch,
                    fixed_band_index=fixed_band_idx,
                    dynamic_band_indices=dynamic_picked_band,
                    dynamic_edge_energy=dynamic_edge,
                    fixed_branch_occupations=fixed_occ,
                    fermi_energy=fermi_energy,
                    fermi_source=fermi_source,
                    dynamic_selector_source=dynamic_selector_source,
                )
                channel_name = f"{carrier_type}_{dir_name}"
                mass_diagnostics[channel_name] = channel_diag
                if m_eff is None:
                    return {"errors": [f"{carrier_type} {dir_name} 有效质量拟合失败: {channel_diag.get('error', 'unknown')}"], **policy_updates}
                cinfo["masses"][f"m{dir_name}"] = float(m_eff)
                np.savetxt(
                    os.path.join(work_dir, "ek_data.txt"),
                    np.column_stack([k_array, k_array_centered, e_branch, fixed_occ, np.full_like(k_array, fixed_band_idx), dynamic_edge, dynamic_picked_band]),
                    header="k(1/Å) k_center(1/Å) E_fixed_branch(eV) fixed_branch_occ fixed_band_index dynamic_edge_E(eV) dynamic_picked_band_index",
                    fmt="%.6e",
                )
                prune_dir_keep_files(work_dir, {"INCAR", "KPOINTS", "POSCAR", "POTCAR", "EIGENVAL", "OUTCAR", "ek_data.txt"})

        for carrier_type, cinfo in carriers.items():
            mx = float(cinfo["masses"]["mx"])
            my = float(cinfo["masses"]["my"])
            cinfo["masses"]["m_dos"] = float(np.sqrt(mx * my))

        with open(os.path.join(inputs.base_dir, "mass_diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(mass_diagnostics, handle, ensure_ascii=False, indent=2)

        return {
            "effmass_completed": True,
            "electron_mass_x": float(carriers["electron"]["masses"]["mx"]),
            "electron_mass_y": float(carriers["electron"]["masses"]["my"]),
            "electron_mass_dos": float(carriers["electron"]["masses"]["m_dos"]),
            "hole_mass_x": float(carriers["hole"]["masses"]["mx"]),
            "hole_mass_y": float(carriers["hole"]["masses"]["my"]),
            "hole_mass_dos": float(carriers["hole"]["masses"]["m_dos"]),
            "mass_diagnostics": mass_diagnostics,
            **policy_updates,
        }

    def _build_output(self, inputs: MassToolInput, raw: dict[str, object], duration_s: float) -> MassToolOutput:
        base_dir = inputs.base_dir
        artifacts = build_artifact_map(
            os.path.join(base_dir, "04_effmass_electron_x", "ek_data.txt"),
            os.path.join(base_dir, "04_effmass_electron_y", "ek_data.txt"),
            os.path.join(base_dir, "04_effmass_hole_x", "ek_data.txt"),
            os.path.join(base_dir, "04_effmass_hole_y", "ek_data.txt"),
            os.path.join(base_dir, "mass_diagnostics.json"),
        )
        mass_diagnostics = dict(raw.get("mass_diagnostics", {}) or {})
        mass_quality = {
            channel: dict(payload or {}).get("status")
            for channel, payload in mass_diagnostics.items()
        }
        return MassToolOutput(
            success=not bool(raw.get("errors")),
            warnings=list(raw.get("warnings", []) or []),
            key_summary={
                "effmass_completed": bool(raw.get("effmass_completed", False)),
                "electron_mass_dos": raw.get("electron_mass_dos"),
                "hole_mass_dos": raw.get("hole_mass_dos"),
                "mass_quality": mass_quality,
                "mass_diagnostics": mass_diagnostics,
            },
            artifact_paths=artifacts,
            error_summary=("; ".join(raw.get("errors", [])) if raw.get("errors") else None),
            duration_s=duration_s,
            state_updates=raw,
            electron_mass_x=raw.get("electron_mass_x"),
            electron_mass_y=raw.get("electron_mass_y"),
            electron_mass_dos=raw.get("electron_mass_dos"),
            hole_mass_x=raw.get("hole_mass_x"),
            hole_mass_y=raw.get("hole_mass_y"),
            hole_mass_dos=raw.get("hole_mass_dos"),
            mass_diagnostics=mass_diagnostics,
            mass_summary={
                "electron_mass_x": raw.get("electron_mass_x"),
                "electron_mass_y": raw.get("electron_mass_y"),
                "electron_mass_dos": raw.get("electron_mass_dos"),
                "hole_mass_x": raw.get("hole_mass_x"),
                "hole_mass_y": raw.get("hole_mass_y"),
                "hole_mass_dos": raw.get("hole_mass_dos"),
                "mass_quality": mass_quality,
                "mass_diagnostics": mass_diagnostics,
            },
        )
