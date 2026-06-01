from __future__ import annotations

import os
import re
import warnings
from typing import Any, Optional

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1e-10
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
KB = 1.380649e-23
M0 = 9.1093837015e-31


def _load_structure(path: str):
    from pymatgen.core import Structure

    return Structure.from_file(path)


def _load_kpoints(*, comment: str, num_kpts: int, kpts: list[list[float]]):
    from pymatgen.io.vasp.inputs import Kpoints

    return Kpoints(
        comment=comment,
        num_kpts=num_kpts,
        style=Kpoints.supported_modes.Reciprocal,
        kpts=kpts,
        kpts_weights=[1] * num_kpts,
    )


def _coerce_fortran_float(token: str) -> float:
    s = str(token).strip()
    if not s:
        raise ValueError("empty float token")
    s = s.replace("D", "E").replace("d", "E")
    if ("E" not in s) and ("e" not in s):
        m = re.match(r"^([+-]?(?:\d+\.\d*|\.\d+|\d+))([+-]\d{2,3})$", s)
        if m:
            s = f"{m.group(1)}E{m.group(2)}"
    return float(s)


def read_final_total_energy_eV(work_dir: str) -> float:
    from pymatgen.io.vasp import Oszicar, Vasprun

    oszicar_path = os.path.join(work_dir, "OSZICAR")
    if os.path.exists(oszicar_path):
        try:
            return float(Oszicar(oszicar_path).final_energy)
        except Exception:
            try:
                with open(oszicar_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.read().splitlines()
                for line in reversed(lines):
                    if "F=" in line:
                        m = re.search(r"F=\s*([^\s]+)", line)
                        if m:
                            return _coerce_fortran_float(m.group(1))
            except Exception:
                pass

    outcar_path = os.path.join(work_dir, "OUTCAR")
    if os.path.exists(outcar_path):
        try:
            last_toten_token: Optional[str] = None
            with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "free  energy   TOTEN" in line:
                        parts = line.strip().split()
                        if "=" in parts:
                            eq_idx = parts.index("=")
                            if eq_idx + 1 < len(parts):
                                last_toten_token = parts[eq_idx + 1]
                        else:
                            m = re.search(r"TOTEN\s*=\s*([^\s]+)", line)
                            if m:
                                last_toten_token = m.group(1)
            if last_toten_token is not None:
                return _coerce_fortran_float(last_toten_token)
        except Exception:
            pass

    vasprun_path = os.path.join(work_dir, "vasprun.xml")
    if os.path.exists(vasprun_path):
        return float(Vasprun(vasprun_path).final_energy)
    raise FileNotFoundError(f"No OSZICAR/OUTCAR/vasprun.xml found under: {work_dir}")


def read_fermi_energy_eV(work_dir: str) -> float:
    from pymatgen.io.vasp import Vasprun

    outcar_path = os.path.join(work_dir, "OUTCAR")
    if os.path.exists(outcar_path):
        try:
            last_fermi_token: Optional[str] = None
            with open(outcar_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "E-fermi" not in line:
                        continue
                    m = re.search(r"E-fermi\s*:\s*([^\s]+)", line)
                    if m:
                        last_fermi_token = m.group(1)
            if last_fermi_token is not None:
                return _coerce_fortran_float(last_fermi_token)
        except Exception:
            pass

    vasprun_path = os.path.join(work_dir, "vasprun.xml")
    if os.path.exists(vasprun_path):
        try:
            return float(Vasprun(vasprun_path).efermi)
        except Exception:
            pass

    doscar_path = os.path.join(work_dir, "DOSCAR")
    if os.path.exists(doscar_path):
        try:
            with open(doscar_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [next(f) for _ in range(6)]
            parts = lines[5].split()
            if len(parts) >= 4:
                return _coerce_fortran_float(parts[3])
        except Exception:
            pass

    raise FileNotFoundError(f"No OUTCAR/vasprun.xml/DOSCAR Fermi energy found under: {work_dir}")


def get_reciprocal_lattice(poscar_path: str) -> np.ndarray:
    struct = _load_structure(poscar_path)
    return np.array(struct.lattice.reciprocal_lattice.matrix)


def frac_to_cart_k(k_frac: list[float], rec_lattice: np.ndarray) -> np.ndarray:
    return np.asarray(k_frac, dtype=float) @ np.asarray(rec_lattice, dtype=float)


def read_eigenval_with_occupations(eigenval_path: str):
    with open(eigenval_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 7:
        raise ValueError(f"Malformed EIGENVAL: too few lines in {eigenval_path}")
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    probe_idx = 6
    while probe_idx < len(lines) and len(lines[probe_idx].split()) == 0:
        probe_idx += 1
    while probe_idx < len(lines):
        parts = lines[probe_idx].split()
        if len(parts) >= 4:
            try:
                [float(x) for x in parts[:4]]
                break
            except Exception:
                pass
        probe_idx += 1
    probe_idx += 1
    while probe_idx < len(lines) and len(lines[probe_idx].split()) == 0:
        probe_idx += 1
    first_band_line = lines[probe_idx].split() if probe_idx < len(lines) else []
    is_spin_polarized = len(first_band_line) >= 5
    nspins = 2 if is_spin_polarized else 1
    kpoints = np.zeros((nkpts, 3), dtype=float)
    energies = np.zeros((nspins, nkpts, nbands), dtype=float)
    occ = np.zeros((nspins, nkpts, nbands), dtype=float)
    line_idx = 7
    for ik in range(nkpts):
        while line_idx < len(lines) and len(lines[line_idx].split()) == 0:
            line_idx += 1
        k_parts = lines[line_idx].split()
        kpoints[ik, :] = [float(x) for x in k_parts[:3]]
        line_idx += 1
        for ib in range(nbands):
            while line_idx < len(lines) and len(lines[line_idx].split()) == 0:
                line_idx += 1
            parts = lines[line_idx].split()
            if not is_spin_polarized:
                energies[0, ik, ib] = float(parts[1])
                occ[0, ik, ib] = float(parts[2])
            else:
                energies[0, ik, ib] = float(parts[1])
                energies[1, ik, ib] = float(parts[2])
                occ[0, ik, ib] = float(parts[3])
                occ[1, ik, ib] = float(parts[4])
            line_idx += 1
    return kpoints, energies, occ, is_spin_polarized


def find_band_edges_from_eigenval_occupancy(eigenval_path: str, occ_threshold: float = 0.5) -> tuple:
    kpts, energies, occ, _ = read_eigenval_with_occupations(eigenval_path)
    occ_mask = occ >= float(occ_threshold)
    if not np.any(occ_mask):
        raise ValueError("未找到占据态")
    masked_vbm = np.where(occ_mask, energies, -np.inf)
    v_spin, v_k, v_b = np.unravel_index(int(np.argmax(masked_vbm)), masked_vbm.shape)
    unocc_mask = occ <= float(1.0 - occ_threshold)
    if not np.any(unocc_mask):
        raise ValueError("未找到非占据态")
    masked_cbm = np.where(unocc_mask, energies, np.inf)
    c_spin, c_k, c_b = np.unravel_index(int(np.argmin(masked_cbm)), masked_cbm.shape)
    return (
        float(masked_vbm[v_spin, v_k, v_b]),
        kpts[v_k].tolist(),
        int(v_b),
        int(v_spin),
        float(masked_cbm[c_spin, c_k, c_b]),
        kpts[c_k].tolist(),
        int(c_b),
        int(c_spin),
    )


def find_band_edges_from_eigenval_fermi(
    eigenval_path: str,
    *,
    fermi_energy: float,
    fermi_tolerance_eV: float = 1.0e-3,
) -> tuple:
    kpts, energies, _occ, _ = read_eigenval_with_occupations(eigenval_path)
    ef = float(fermi_energy)
    tol = abs(float(fermi_tolerance_eV))

    occupied_mask = energies <= ef - tol
    if not np.any(occupied_mask):
        occupied_mask = energies <= ef + tol
    if not np.any(occupied_mask):
        raise ValueError("未找到费米能级以下的价带态")
    masked_vbm = np.where(occupied_mask, energies, -np.inf)
    v_spin, v_k, v_b = np.unravel_index(int(np.argmax(masked_vbm)), masked_vbm.shape)

    unoccupied_mask = energies >= ef + tol
    if not np.any(unoccupied_mask):
        unoccupied_mask = energies >= ef - tol
    if not np.any(unoccupied_mask):
        raise ValueError("未找到费米能级以上的导带态")
    masked_cbm = np.where(unoccupied_mask, energies, np.inf)
    c_spin, c_k, c_b = np.unravel_index(int(np.argmin(masked_cbm)), masked_cbm.shape)

    return (
        float(masked_vbm[v_spin, v_k, v_b]),
        kpts[v_k].tolist(),
        int(v_b),
        int(v_spin),
        float(masked_cbm[c_spin, c_k, c_b]),
        kpts[c_k].tolist(),
        int(c_b),
        int(c_spin),
    )


def _nearest_kpoint_index(kpts_frac: np.ndarray, target_k_frac: list[float], tol: float = 1e-6) -> int:
    tgt = np.asarray(target_k_frac, dtype=float).reshape(1, 3)
    d2 = np.sum((np.asarray(kpts_frac, dtype=float) - tgt) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return idx


def extract_edge_energy_at_fixed_kpoint(eigenval_path: str, *, target_k_frac: list[float], carrier_type: str, reference_energy: Optional[float] = None, spin_hint: Optional[int] = None, occ_threshold: float = 0.5) -> float:
    kpts, energies, occ, _ = read_eigenval_with_occupations(eigenval_path)
    ik = _nearest_kpoint_index(np.asarray(kpts, dtype=float), target_k_frac, tol=1e-6)
    nspins = int(np.asarray(energies).shape[0])
    spins_to_try = [int(spin_hint)] if spin_hint is not None and 0 <= int(spin_hint) < nspins else list(range(nspins))
    best_energy: Optional[float] = None
    best_cost = float("inf")
    for sp in spins_to_try:
        e_k = np.asarray(energies)[sp, ik, :]
        o_k = np.asarray(occ)[sp, ik, :]
        if carrier_type == "hole":
            mask = o_k >= float(occ_threshold)
            if not np.any(mask):
                continue
            candidates = e_k[mask]
            picked = float(np.max(candidates)) if reference_energy is None else float(candidates[int(np.argmin(np.abs(candidates - float(reference_energy))))])
        else:
            mask = o_k <= float(1.0 - occ_threshold)
            if not np.any(mask):
                continue
            candidates = e_k[mask]
            picked = float(np.min(candidates)) if reference_energy is None else float(candidates[int(np.argmin(np.abs(candidates - float(reference_energy))))])
        cost = 0.0 if reference_energy is None else float(abs(picked - float(reference_energy)))
        if cost < best_cost:
            best_cost = cost
            best_energy = picked
    if best_energy is None:
        raise ValueError(f"Failed to extract {carrier_type} edge at fixed kpoint")
    return float(best_energy)


def load_strain_reference_from_band(base_strain_dir: str, direction: str) -> Optional[dict[str, Any]]:
    folder0 = os.path.join(base_strain_dir, direction, f"strain_{0.0:+.4f}")
    eigenval0 = os.path.join(folder0, "03_band", "EIGENVAL")
    if not os.path.exists(eigenval0) or os.path.getsize(eigenval0) <= 0:
        return None
    try:
        vbm, vbm_kpt, _vbm_b, vbm_sp, cbm, cbm_kpt, _cbm_b, cbm_sp = find_band_edges_from_eigenval_occupancy(
            eigenval0,
            occ_threshold=0.5,
        )
    except Exception:
        return None
    return {
        "vbm_energy": float(vbm),
        "vbm_kpoint": list(vbm_kpt),
        "vbm_spin": int(vbm_sp),
        "cbm_energy": float(cbm),
        "cbm_kpoint": list(cbm_kpt),
        "cbm_spin": int(cbm_sp),
    }


def calculate_effective_mass(k_cart_array: np.ndarray, e_band: np.ndarray, carrier_type: str) -> tuple[float, float, float]:
    def parabola(k, a, b, c):
        return a * k**2 + b * k + c

    k_au = np.asarray(k_cart_array, dtype=float) * BOHR_TO_ANGSTROM
    e_au = np.asarray(e_band, dtype=float) * EV_TO_HARTREE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt_au, _ = curve_fit(parabola, k_au, e_au)
    a_fit_au = float(popt_au[0])
    e_fit_au = parabola(k_au, *popt_au)
    ss_res = float(np.sum((e_au - e_fit_au) ** 2))
    ss_tot = float(np.sum((e_au - float(np.mean(e_au))) ** 2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if abs(a_fit_au) < 1e-18:
        raise ValueError("曲率接近零，能带可能为平带")
    m_eff_ratio = abs(1.0 / (2.0 * a_fit_au))
    a_fit_eVA2 = a_fit_au * HARTREE_TO_EV * (BOHR_TO_ANGSTROM**2)
    return float(m_eff_ratio), float(a_fit_eVA2), float(r_squared)


def read_vacuum_level_from_locpot(path: str, vacuum_direction: int = 2) -> float:
    from pymatgen.io.vasp import Locpot

    locpot = Locpot.from_file(path)
    avg_pot = np.array(locpot.get_average_along_axis(vacuum_direction))
    if avg_pot.size < 10:
        return float(np.max(avg_pot))
    n_top = max(1, int(0.10 * avg_pot.size))
    return float(np.mean(np.sort(avg_pot)[-n_top:]))
