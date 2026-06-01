from __future__ import annotations

import csv
import os
import shutil
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .vasp_common import (
    build_incar_band,
    build_incar_relax,
    build_incar_scf,
    policy_stage_planning_allowed,
    prune_dir_keep_files,
    read_chgcar_compatible_incar_overrides,
    run_vasp,
    summarize_vasp_failure,
    symlink_force,
    write_band_kpoints,
    write_incar,
    write_relax_scf_kpoints,
)
from .physics_common import (
    extract_edge_energy_at_fixed_kpoint,
    find_band_edges_from_eigenval_occupancy,
    load_strain_reference_from_band,
    read_final_total_energy_eV,
    read_vacuum_level_from_locpot,
)
from .relax_retry import RelaxRetryFatal, relax_retry_enabled, run_relax_vasp_with_retry
from ..runtime.telemetry import emit_progress

if TYPE_CHECKING:
    from ..policy.engine import AgenticPolicyEngine


def _strain_key(row: dict[str, Any]) -> tuple[str, float] | None:
    direction = str(row.get("direction") or "").strip()
    if not direction:
        return None
    try:
        strain = round(float(row.get("strain", 0.0) or 0.0), 6)
    except Exception:
        return None
    return direction, strain


def _merge_strain_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index_by_key: dict[tuple[str, float], int] = {}
    for row in list(existing_rows or []) + list(new_rows or []):
        candidate = dict(row or {})
        key = _strain_key(candidate)
        if key is None:
            merged.append(candidate)
            continue
        if key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append(candidate)
            continue
        previous = dict(merged[index_by_key[key]] or {})
        previous_completed = bool(previous.get("completed", False))
        current_completed = bool(candidate.get("completed", False))
        if current_completed or not previous_completed:
            merged[index_by_key[key]] = candidate
    return merged


def _planned_strains(plan: dict[str, list[float]] | None, direction: str) -> list[float]:
    normalized: set[float] = {0.0}
    for value in list((plan or {}).get(direction, []) or []):
        try:
            normalized.add(round(float(value), 6))
        except Exception:
            continue
    return sorted(normalized)


def _is_zero_strain(value: float, *, tol: float = 1.0e-12) -> bool:
    return abs(float(value)) <= float(tol)


def _ordered_strains(values: list[float]) -> list[float]:
    ordered: list[float] = []
    if any(_is_zero_strain(value) for value in values):
        ordered.append(0.0)
    ordered.extend([float(value) for value in list(values or []) if not _is_zero_strain(value)])
    return ordered


def _direction_fit_readiness(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    completed_strains = sorted(
        {
            round(float(row.get("strain", 0.0) or 0.0), 6)
            for row in list(rows or [])
            if str(row.get("direction") or "") == direction and bool(row.get("completed", False))
        }
    )
    has_zero_reference = any(_is_zero_strain(value) for value in completed_strains)
    has_negative_side = any(float(value) < 0.0 for value in completed_strains)
    has_positive_side = any(float(value) > 0.0 for value in completed_strains)
    return {
        "has_zero_reference": has_zero_reference,
        "has_negative_side": has_negative_side,
        "has_positive_side": has_positive_side,
        "fit_ready": len(completed_strains) >= 5 and has_zero_reference and has_negative_side and has_positive_side,
    }


def _serialize_strain_keys(keys: set[tuple[str, float]] | list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [
        {"direction": direction, "strain": strain}
        for direction, strain in sorted(
            {(str(direction), round(float(strain), 6)) for direction, strain in list(keys or [])},
            key=lambda item: (item[0], item[1]),
        )
    ]


def _status_completed_rows(base_strain_dir: str) -> list[tuple[str, float, str]]:
    status_path = os.path.join(os.path.dirname(os.path.abspath(base_strain_dir)), "strain_status.csv")
    if not os.path.exists(status_path):
        return []
    rows: list[tuple[str, float, str]] = []
    try:
        with open(status_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("completed") or "").strip().lower() not in {"1", "true", "yes"}:
                    continue
                direction = str(row.get("direction") or "").strip()
                if direction not in {"x", "y"}:
                    continue
                try:
                    strain = round(float(row.get("strain", 0.0) or 0.0), 6)
                except Exception:
                    continue
                folder = str(row.get("folder") or "").strip()
                if not folder:
                    folder = os.path.join(base_strain_dir, direction, f"strain_{strain:+.4f}")
                rows.append((direction, strain, folder))
    except Exception:
        return []
    return rows


def _recover_completed_strain_rows_from_disk(base_strain_dir: str, *, vacuum_direction: int) -> list[dict[str, Any]]:
    candidates = _status_completed_rows(base_strain_dir)
    if not candidates:
        return []
    recovered: list[dict[str, Any]] = []
    refs: dict[str, dict[str, Any] | None] = {}
    for direction, strain, folder in candidates:
        if direction not in refs:
            refs[direction] = load_strain_reference_from_band(base_strain_dir, direction)
        ref = refs.get(direction)
        if ref is None:
            continue
        scf_dir = os.path.join(folder, "02_scf")
        band_dir = os.path.join(folder, "03_band")
        eigenval = os.path.join(band_dir, "EIGENVAL")
        if not os.path.exists(eigenval) or os.path.getsize(eigenval) <= 0:
            continue
        try:
            total_e = float(read_final_total_energy_eV(scf_dir))
            e_vacuum = float(read_vacuum_level_from_locpot(os.path.join(scf_dir, "LOCPOT"), vacuum_direction=vacuum_direction))
            vbm_g, vbm_kpt_g, vbm_b_g, vbm_sp_g, cbm_g, cbm_kpt_g, cbm_b_g, cbm_sp_g = find_band_edges_from_eigenval_occupancy(eigenval, occ_threshold=0.5)
            vbm_fixed = extract_edge_energy_at_fixed_kpoint(
                eigenval,
                target_k_frac=ref["vbm_kpoint"],
                carrier_type="hole",
                reference_energy=ref.get("vbm_energy"),
                spin_hint=int(ref.get("vbm_spin", 0)),
                occ_threshold=0.5,
            )
            cbm_fixed = extract_edge_energy_at_fixed_kpoint(
                eigenval,
                target_k_frac=ref["cbm_kpoint"],
                carrier_type="electron",
                reference_energy=ref.get("cbm_energy"),
                spin_hint=int(ref.get("cbm_spin", 0)),
                occ_threshold=0.5,
            )
        except Exception:
            continue
        recovered.append(
            {
                "direction": direction,
                "strain": float(strain),
                "total_energy": float(total_e),
                "e_vbm": float(vbm_fixed),
                "e_cbm": float(cbm_fixed),
                "e_vacuum": float(e_vacuum),
                "e_vbm_global": float(vbm_g),
                "e_cbm_global": float(cbm_g),
                "vbm_kpoint_global": list(vbm_kpt_g),
                "cbm_kpoint_global": list(cbm_kpt_g),
                "vbm_band_global": int(vbm_b_g),
                "cbm_band_global": int(cbm_b_g),
                "vbm_spin_global": int(vbm_sp_g),
                "cbm_spin_global": int(cbm_sp_g),
                "vbm_kpoint_ref": list(ref.get("vbm_kpoint")),
                "cbm_kpoint_ref": list(ref.get("cbm_kpoint")),
                "folder": folder,
                "completed": True,
                "recovered_from_disk": True,
            }
        )
    return recovered


class StrainToolInput(ToolInputBase):
    relaxed_poscar: str
    potcar_path: str
    strain_plan_by_direction: dict[str, list[float]]
    no_relax_retry: bool = False
    relax_retry_backups: list[str] = Field(default_factory=list)


class StrainToolOutput(ToolOutputBase):
    strain_data: list[dict[str, Any]] = Field(default_factory=list)
    retry_backups: list[str] = Field(default_factory=list)
    strain_summary: dict[str, Any] = Field(default_factory=dict)


class StrainTool(DeterministicTool):
    name = "run_strain_campaign"
    description = "Run strain campaign with configurable sampling plan and summarize fit readiness"
    input_model = StrainToolInput
    output_model = StrainToolOutput

    def __init__(
        self,
        executor=None,
        *,
        vasp_cmd: str = "mpirun -np 4 vasp_std > sout 2>&1",
        consider_spin: bool = False,
        vacuum_direction: int = 2,
        policy_engine: AgenticPolicyEngine | None = None,
    ):
        super().__init__(executor=executor)
        self.vasp_cmd = vasp_cmd
        self.consider_spin = consider_spin
        self.vacuum_direction = vacuum_direction
        self.policy_engine = policy_engine

    def _execute(self, inputs: StrainToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        directions = {"x": 0, "y": 1}
        planned_strains_by_direction = {direction: _planned_strains(inputs.strain_plan_by_direction, direction) for direction in directions}
        state_payload = dict(inputs.state_payload or {})
        physics_results = dict(state_payload.get("physics_results", {}) or {})
        diagnostics = dict(state_payload.get("diagnostics", {}) or {})
        prior_summary = dict(diagnostics.get("strain_summary", {}) or {})
        base_strain_dir = os.path.join(inputs.base_dir, "05_strain")
        os.makedirs(base_strain_dir, exist_ok=True)
        existing_rows = list(state_payload.get("strain_data", []) or [])
        if not existing_rows:
            existing_rows = list(physics_results.get("strain_data", []) or [])
        disk_rows = _recover_completed_strain_rows_from_disk(base_strain_dir, vacuum_direction=self.vacuum_direction)
        if disk_rows:
            existing_rows = _merge_strain_rows(existing_rows, disk_rows)
        completed = set()
        for row in existing_rows:
            key = _strain_key(row)
            if key is not None and row.get("completed", False):
                completed.add(key)

        strain_results: list[dict[str, Any]] = []
        all_backups = list(inputs.relax_retry_backups or [])
        strain_recovery_events: list[dict[str, Any]] = []
        policy_plan_records: dict[str, Any] = {}
        retrieval_trace: list[dict[str, Any]] = []
        for dir_name, dir_idx in directions.items():
            strains = _ordered_strains(list(planned_strains_by_direction.get(dir_name, [0.0])))
            lattice_constraints = ".FALSE. .TRUE. .FALSE." if dir_name == "x" else ".TRUE. .FALSE. .FALSE."
            ref = load_strain_reference_from_band(base_strain_dir, dir_name)
            for s in strains:
                point_key = (dir_name, round(float(s), 6))
                emit_progress(
                    "strain point started",
                    channel="stage",
                    workdir=inputs.base_dir,
                    details={
                        "stage": "strain_loop",
                        "direction": dir_name,
                        "strain": f"{float(s):+.4f}",
                        "substage": "relax",
                    },
                )
                if point_key in completed:
                    emit_progress(
                        "strain point skipped (already completed)",
                        channel="stage",
                        workdir=inputs.base_dir,
                        details={
                            "stage": "strain_loop",
                            "direction": dir_name,
                            "strain": f"{float(s):+.4f}",
                        },
                    )
                    if ref is None:
                        ref = load_strain_reference_from_band(base_strain_dir, dir_name)
                    continue
                folder = os.path.join(base_strain_dir, dir_name, f"strain_{s:+.4f}")
                relax_dir = os.path.join(folder, "01_relax")
                scf_dir = os.path.join(folder, "02_scf")
                band_dir = os.path.join(folder, "03_band")
                os.makedirs(relax_dir, exist_ok=True)
                os.makedirs(scf_dir, exist_ok=True)
                os.makedirs(band_dir, exist_ok=True)
                poscar_in_relax = os.path.join(relax_dir, "POSCAR")
                contcar_in_relax = os.path.join(relax_dir, "CONTCAR")
                try:
                    from pymatgen.core import Structure

                    struct = Structure.from_file(str(inputs.relaxed_poscar))
                    strain_tensor = [0.0, 0.0, 0.0]
                    strain_tensor[dir_idx] = float(s)
                    struct.apply_strain(strain_tensor)
                    struct.to(fmt="POSCAR", filename=os.path.join(relax_dir, "POSCAR"))
                    shutil.copy(str(inputs.potcar_path), os.path.join(relax_dir, "POTCAR"))
                    incar_relax = build_incar_relax(
                        f"Strain({dir_name}) {s:+.3f} Relax",
                        ediff=1e-6,
                        isif=3,
                        lattice_constraints=lattice_constraints,
                        consider_spin=self.consider_spin,
                    )
                    relax_kpoints_policy = {"target_ka": 50.0, "gamma_centered": False}
                    if self.policy_engine is not None and policy_stage_planning_allowed(state_payload, "relax", stage_aliases=("strain_loop",)):
                        relax_plan = self.policy_engine.plan_stage(
                            stage="relax",
                            state_payload=state_payload,
                            default_incar=incar_relax,
                            default_kpoints_policy=relax_kpoints_policy,
                            extra_context={"task_scope": "strain_loop", "direction": dir_name, "strain": float(s), "substage": "relax"},
                        )
                        incar_relax.update(dict(relax_plan.incar_overrides or {}))
                        if dict(relax_plan.kpoints_policy or {}):
                            relax_kpoints_policy.update(dict(relax_plan.kpoints_policy or {}))
                        plan_key = f"strain_loop::{dir_name}::{float(s):+.4f}::relax"
                        policy_plan_records[plan_key] = relax_plan.model_dump(mode="json")
                        retrieval_trace.append(
                            {
                                "kind": "parameter_plan",
                                "stage": "strain_loop",
                                "substage": "relax",
                                "direction": dir_name,
                                "strain": float(s),
                                "source": relax_plan.source,
                                "confidence": relax_plan.confidence,
                                "evidence": [item.model_dump(mode="json") for item in list(relax_plan.evidence_items or [])],
                            }
                        )
                    write_incar(relax_dir, incar_relax)
                    write_relax_scf_kpoints(
                        relax_dir,
                        material_name=inputs.material_id,
                        target_ka=float(relax_kpoints_policy.get("target_ka", 50.0) or 50.0),
                        gamma_centered=bool(relax_kpoints_policy.get("gamma_centered", False)),
                    )
                    retry_log = os.path.join(inputs.base_dir, "vasp_relax_retry.log")
                    enabled = relax_retry_enabled(cli_no_relax_retry=bool(inputs.no_relax_retry))
                    try:
                        ok, backups, recovery_summary = run_relax_vasp_with_retry(
                            workdir=relax_dir,
                            material_id=str(inputs.material_id),
                            vasp_cmd=self.vasp_cmd,
                            retry_log_path=retry_log,
                            enabled=enabled,
                            check_convergence=True,
                        )
                    except RelaxRetryFatal as e:
                        emit_progress(
                            "strain point relax failed",
                            channel="stage",
                            workdir=inputs.base_dir,
                            details={
                                "stage": "strain_loop",
                                "direction": dir_name,
                                "strain": f"{float(s):+.4f}",
                                "substage": "relax",
                                "error": str(e),
                            },
                        )
                        recovery_summary = {
                            **dict(getattr(e, "summary", {}) or {}),
                            "stage": "strain_relax",
                            "task_scope": "strain_relax",
                            "direction": dir_name,
                            "strain": float(s),
                            "relax_dir": relax_dir,
                            "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                            "has_contcar": os.path.exists(contcar_in_relax) and os.path.getsize(contcar_in_relax) > 0,
                        }
                        strain_recovery_events.append({
                            "direction": dir_name,
                            "strain": float(s),
                            **recovery_summary,
                        })
                        strain_results.append({
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": str(e) or "RELAX_RETRY_FATAL",
                            "error_type": recovery_summary.get("error_type", "retry_limit_reached"),
                            "recovery_action": recovery_summary.get("applied_action", "skip_material"),
                            "has_poscar": recovery_summary.get("has_poscar"),
                            "has_contcar": recovery_summary.get("has_contcar"),
                            "recovery_summary": recovery_summary,
                        })
                        continue
                    recovery_summary = {
                        **dict(recovery_summary or {}),
                        "stage": "strain_relax",
                        "task_scope": "strain_relax",
                        "direction": dir_name,
                        "strain": float(s),
                        "relax_dir": relax_dir,
                        "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                        "has_contcar": os.path.exists(contcar_in_relax) and os.path.getsize(contcar_in_relax) > 0,
                    }
                    if backups:
                        all_backups.extend(list(backups))
                    if not ok:
                        emit_progress(
                            "strain point relax failed",
                            channel="stage",
                            workdir=inputs.base_dir,
                            details={
                                "stage": "strain_loop",
                                "direction": dir_name,
                                "strain": f"{float(s):+.4f}",
                                "substage": "relax",
                                "error": "RELAX_FAILED",
                            },
                        )
                        strain_recovery_events.append({
                            "direction": dir_name,
                            "strain": float(s),
                            **dict(recovery_summary or {}),
                        })
                        strain_results.append({
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": "RELAX_FAILED",
                            "error_type": dict(recovery_summary or {}).get("error_type", "relax_failed"),
                            "recovery_action": dict(recovery_summary or {}).get("applied_action", "retry"),
                            "has_poscar": recovery_summary.get("has_poscar"),
                            "has_contcar": recovery_summary.get("has_contcar"),
                            "recovery_summary": dict(recovery_summary or {}),
                        })
                        continue
                    contcar = os.path.join(relax_dir, "CONTCAR")
                    if not os.path.exists(contcar):
                        emit_progress(
                            "strain point relax output missing",
                            channel="stage",
                            workdir=inputs.base_dir,
                            details={
                                "stage": "strain_loop",
                                "direction": dir_name,
                                "strain": f"{float(s):+.4f}",
                                "substage": "relax",
                                "error": "CONTCAR_MISSING",
                            },
                        )
                        strain_recovery_events.append(
                            {
                                "stage": "strain_relax",
                                "task_scope": "strain_relax",
                                "direction": dir_name,
                                "strain": float(s),
                                "relax_dir": relax_dir,
                                "error_type": "missing_output",
                                "trigger_pattern": "CONTCAR_MISSING",
                                "retries_used": int(recovery_summary.get("retries_used", 0) or 0),
                                "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                                "has_contcar": False,
                                "applied_action": "skip_point",
                                "final_outcome": "failed",
                            }
                        )
                        strain_results.append({
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": "CONTCAR_MISSING",
                            "error_type": "missing_output",
                            "recovery_action": "skip_point",
                            "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                            "has_contcar": False,
                        })
                        continue
                    prune_dir_keep_files(relax_dir, {"INCAR", "KPOINTS", "POSCAR", "CONTCAR", "POTCAR"})

                    shutil.copy(contcar, os.path.join(scf_dir, "POSCAR"))
                    shutil.copy(str(inputs.potcar_path), os.path.join(scf_dir, "POTCAR"))
                    incar_scf = build_incar_scf(f"Strain({dir_name}) {s:+.3f} SCF", consider_spin=self.consider_spin, ediff=1e-6, lvtot=True, lvhar=True)
                    scf_kpoints_policy = {"target_ka": 50.0, "gamma_centered": False}
                    if self.policy_engine is not None and policy_stage_planning_allowed(state_payload, "scf", stage_aliases=("strain_loop",)):
                        scf_plan = self.policy_engine.plan_stage(
                            stage="scf",
                            state_payload=state_payload,
                            default_incar=incar_scf,
                            default_kpoints_policy=scf_kpoints_policy,
                            extra_context={"task_scope": "strain_loop", "direction": dir_name, "strain": float(s), "substage": "scf"},
                        )
                        incar_scf.update(dict(scf_plan.incar_overrides or {}))
                        if dict(scf_plan.kpoints_policy or {}):
                            scf_kpoints_policy.update(dict(scf_plan.kpoints_policy or {}))
                        plan_key = f"strain_loop::{dir_name}::{float(s):+.4f}::scf"
                        policy_plan_records[plan_key] = scf_plan.model_dump(mode="json")
                        retrieval_trace.append(
                            {
                                "kind": "parameter_plan",
                                "stage": "strain_loop",
                                "substage": "scf",
                                "direction": dir_name,
                                "strain": float(s),
                                "source": scf_plan.source,
                                "confidence": scf_plan.confidence,
                                "evidence": [item.model_dump(mode="json") for item in list(scf_plan.evidence_items or [])],
                            }
                        )
                    write_incar(scf_dir, incar_scf)
                    write_relax_scf_kpoints(
                        scf_dir,
                        material_name=inputs.material_id,
                        target_ka=float(scf_kpoints_policy.get("target_ka", 50.0) or 50.0),
                        gamma_centered=bool(scf_kpoints_policy.get("gamma_centered", False)),
                    )
                    emit_progress(
                        "strain point substage started",
                        channel="stage",
                        workdir=inputs.base_dir,
                        details={
                            "stage": "strain_loop",
                            "direction": dir_name,
                            "strain": f"{float(s):+.4f}",
                            "substage": "scf",
                        },
                    )
                    if not run_vasp(cwd=scf_dir, vasp_cmd=self.vasp_cmd, check_convergence=True):
                        recovery_summary = summarize_vasp_failure(scf_dir, stage="strain_loop", default_error="SCF_FAILED")
                        emit_progress(
                            "strain point scf failed",
                            channel="stage",
                            workdir=inputs.base_dir,
                            details={
                                "stage": "strain_loop",
                                "direction": dir_name,
                                "strain": f"{float(s):+.4f}",
                                "substage": "scf",
                                "error": recovery_summary["error_summary"],
                            },
                        )
                        strain_results.append({
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": recovery_summary["error_summary"],
                            "error_type": recovery_summary["error_type"],
                            "recovery_summary": recovery_summary,
                        })
                        continue
                    total_e = float(read_final_total_energy_eV(scf_dir))
                    e_vacuum = float(read_vacuum_level_from_locpot(os.path.join(scf_dir, "LOCPOT"), vacuum_direction=self.vacuum_direction))

                    shutil.copy(os.path.join(scf_dir, "POSCAR"), os.path.join(band_dir, "POSCAR"))
                    shutil.copy(str(inputs.potcar_path), os.path.join(band_dir, "POTCAR"))
                    chgcar_src = os.path.join(scf_dir, "CHGCAR")
                    if os.path.exists(chgcar_src):
                        symlink_force(chgcar_src, os.path.join(band_dir, "CHGCAR"))
                    incar_band = build_incar_band(f"Strain({dir_name}) {s:+.3f} Band", consider_spin=self.consider_spin, ediff=1e-6)
                    chgcar_compatible_overrides = read_chgcar_compatible_incar_overrides(folder)
                    incar_band.update(chgcar_compatible_overrides)
                    band_kpoints_policy = {"line_mode_density": 40}
                    if self.policy_engine is not None and policy_stage_planning_allowed(state_payload, "band", stage_aliases=("strain_loop",)):
                        band_plan = self.policy_engine.plan_stage(
                            stage="band",
                            state_payload=state_payload,
                            default_incar=incar_band,
                            default_kpoints_policy=band_kpoints_policy,
                            extra_context={"task_scope": "strain_loop", "direction": dir_name, "strain": float(s), "substage": "band"},
                        )
                        incar_band.update(dict(band_plan.incar_overrides or {}))
                        incar_band.update(chgcar_compatible_overrides)
                        if dict(band_plan.kpoints_policy or {}):
                            band_kpoints_policy.update(dict(band_plan.kpoints_policy or {}))
                        plan_key = f"strain_loop::{dir_name}::{float(s):+.4f}::band"
                        policy_plan_records[plan_key] = band_plan.model_dump(mode="json")
                        retrieval_trace.append(
                            {
                                "kind": "parameter_plan",
                                "stage": "strain_loop",
                                "substage": "band",
                                "direction": dir_name,
                                "strain": float(s),
                                "source": band_plan.source,
                                "confidence": band_plan.confidence,
                                "evidence": [item.model_dump(mode="json") for item in list(band_plan.evidence_items or [])],
                            }
                        )
                    write_incar(band_dir, incar_band)
                    write_band_kpoints(
                        os.path.join(band_dir, "KPOINTS"),
                        npoints_per_segment=int(band_kpoints_policy.get("line_mode_density", 40) or 40),
                    )
                    emit_progress(
                        "strain point substage started",
                        channel="stage",
                        workdir=inputs.base_dir,
                        details={
                            "stage": "strain_loop",
                            "direction": dir_name,
                            "strain": f"{float(s):+.4f}",
                            "substage": "band",
                        },
                    )
                    if not run_vasp(cwd=band_dir, vasp_cmd=self.vasp_cmd, check_convergence=False):
                        recovery_summary = summarize_vasp_failure(band_dir, stage="strain_loop", default_error="BAND_FAILED")
                        emit_progress(
                            "strain point band failed",
                            channel="stage",
                            workdir=inputs.base_dir,
                            details={
                                "stage": "strain_loop",
                                "direction": dir_name,
                                "strain": f"{float(s):+.4f}",
                                "substage": "band",
                                "error": recovery_summary["error_summary"],
                            },
                        )
                        strain_results.append({
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": recovery_summary["error_summary"],
                            "error_type": recovery_summary["error_type"],
                            "recovery_summary": recovery_summary,
                        })
                        continue
                    eigenval = os.path.join(band_dir, "EIGENVAL")
                    vbm_g, vbm_kpt_g, vbm_b_g, vbm_sp_g, cbm_g, cbm_kpt_g, cbm_b_g, cbm_sp_g = find_band_edges_from_eigenval_occupancy(eigenval, occ_threshold=0.5)
                    if ref is None and abs(float(s)) <= 1e-15:
                        ref = {
                            "vbm_energy": float(vbm_g),
                            "vbm_kpoint": list(vbm_kpt_g),
                            "vbm_spin": int(vbm_sp_g),
                            "cbm_energy": float(cbm_g),
                            "cbm_kpoint": list(cbm_kpt_g),
                            "cbm_spin": int(cbm_sp_g),
                        }
                    if ref is None:
                        raise RuntimeError(f"Missing 0% reference for direction={dir_name}")
                    vbm_fixed = extract_edge_energy_at_fixed_kpoint(eigenval, target_k_frac=ref["vbm_kpoint"], carrier_type="hole", reference_energy=ref.get("vbm_energy"), spin_hint=int(ref.get("vbm_spin", 0)), occ_threshold=0.5)
                    cbm_fixed = extract_edge_energy_at_fixed_kpoint(eigenval, target_k_frac=ref["cbm_kpoint"], carrier_type="electron", reference_energy=ref.get("cbm_energy"), spin_hint=int(ref.get("cbm_spin", 0)), occ_threshold=0.5)
                    sr = {
                        "direction": dir_name,
                        "strain": float(s),
                        "total_energy": float(total_e),
                        "e_vbm": float(vbm_fixed),
                        "e_cbm": float(cbm_fixed),
                        "e_vacuum": float(e_vacuum),
                        "e_vbm_global": float(vbm_g),
                        "e_cbm_global": float(cbm_g),
                        "vbm_kpoint_global": list(vbm_kpt_g),
                        "cbm_kpoint_global": list(cbm_kpt_g),
                        "vbm_band_global": int(vbm_b_g),
                        "cbm_band_global": int(cbm_b_g),
                        "vbm_spin_global": int(vbm_sp_g),
                        "cbm_spin_global": int(cbm_sp_g),
                        "vbm_kpoint_ref": list(ref.get("vbm_kpoint")),
                        "cbm_kpoint_ref": list(ref.get("cbm_kpoint")),
                        "folder": folder,
                        "completed": True,
                    }
                    strain_results.append(sr)
                    emit_progress(
                        "strain point completed",
                        channel="stage",
                        workdir=inputs.base_dir,
                        details={
                            "stage": "strain_loop",
                            "direction": dir_name,
                            "strain": f"{float(s):+.4f}",
                        },
                    )
                    prune_dir_keep_files(scf_dir, {"INCAR", "KPOINTS", "POSCAR", "POTCAR"})
                    prune_dir_keep_files(band_dir, {"INCAR", "KPOINTS", "POSCAR", "POTCAR", "EIGENVAL"})
                except Exception as e:
                    emit_progress(
                        "strain point failed",
                        channel="stage",
                        workdir=inputs.base_dir,
                        details={
                            "stage": "strain_loop",
                            "direction": dir_name,
                            "strain": f"{float(s):+.4f}",
                            "error": str(e),
                        },
                    )
                    strain_recovery_events.append(
                        {
                            "stage": "strain_relax",
                            "task_scope": "strain_relax",
                            "direction": dir_name,
                            "strain": float(s),
                            "relax_dir": relax_dir,
                            "error_type": "unknown_failure",
                            "trigger_pattern": str(e),
                            "retries_used": 0,
                            "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                            "has_contcar": os.path.exists(contcar_in_relax) and os.path.getsize(contcar_in_relax) > 0,
                            "applied_action": "skip_point",
                            "final_outcome": "failed",
                        }
                    )
                    strain_results.append(
                        {
                            "direction": dir_name,
                            "strain": float(s),
                            "completed": False,
                            "folder": folder,
                            "error": str(e),
                            "error_type": "unknown_failure",
                            "recovery_action": "skip_point",
                            "has_poscar": os.path.exists(poscar_in_relax) and os.path.getsize(poscar_in_relax) > 0,
                            "has_contcar": os.path.exists(contcar_in_relax) and os.path.getsize(contcar_in_relax) > 0,
                        }
                    )

        combined = _merge_strain_rows(existing_rows, strain_results)
        completed_rows = [row for row in combined if row.get("completed", False)]
        failed_rows = [row for row in combined if not row.get("completed", False)]
        completed_keys = {_strain_key(row) for row in completed_rows}
        failed_keys = {_strain_key(row) for row in failed_rows}
        active_plan_keys = {
            (direction, strain)
            for direction, planned_values in planned_strains_by_direction.items()
            for strain in list(planned_values or [])
        }
        active_completed_keys = {key for key in completed_keys if key is not None and key in active_plan_keys}
        active_failed_keys = {key for key in failed_keys if key is not None and key in active_plan_keys}
        missing_keys: set[tuple[str, float]] = set()
        prior_historical_keys = {
            key
            for key in [
                _strain_key(item)
                for item in list(prior_summary.get("historical_failed_keys", []) or [])
            ]
            if key is not None
        }
        seeded_historical_rows = (
            []
            if prior_summary
            else [row for row in list(existing_rows or []) if not row.get("completed", False)]
        )
        new_failure_rows = [row for row in strain_results if not row.get("completed", False)]
        historical_failed_keys = prior_historical_keys | {
            key
            for key in [
                _strain_key(row)
                for row in list(seeded_historical_rows or []) + list(new_failure_rows or [])
            ]
            if key is not None
        }
        prior_failure_attempts = int(
            prior_summary.get("historical_failure_attempts", prior_summary.get("historical_failed_points", 0)) or 0
        )
        historical_failure_attempts = (
            prior_failure_attempts + len(new_failure_rows)
            if prior_summary
            else len([row for row in list(existing_rows or []) + list(new_failure_rows or []) if not row.get("completed", False)])
        )
        per_direction_summary = {}
        for direction in ["x", "y"]:
            subset = [row for row in completed_rows if row.get("direction") == direction]
            planned_keys = [(direction, strain) for strain in planned_strains_by_direction.get(direction, [0.0])]
            missing_for_direction = [key for key in planned_keys if key not in active_completed_keys and key not in active_failed_keys]
            missing_keys.update(missing_for_direction)
            fit_readiness = _direction_fit_readiness(completed_rows, direction)
            per_direction_summary[direction] = {
                "completed_points": len(subset),
                "failed_points": len([key for key in active_failed_keys if key[0] == direction]),
                "missing_points": len(missing_for_direction),
                "planned_points": len(planned_keys),
                "historical_failed_points": len([key for key in historical_failed_keys if key[0] == direction]),
                **fit_readiness,
            }
        active_failed_count = len(active_failed_keys)
        active_missing_count = len(missing_keys)
        strain_completed = (
            all(bool(summary.get("fit_ready", False)) for summary in per_direction_summary.values())
            and active_failed_count == 0
            and active_missing_count == 0
        )
        return {
            # Keep cumulative strain data across rounds so mobility/refinement
            # sees all completed points instead of only the latest incremental batch.
            "strain_data": combined,
            "relax_retry_backups": all_backups,
            "strain_completed": strain_completed,
            "strain_summary": {
                "completed_points": len(completed_rows),
                "failed_points": active_failed_count,
                "missing_points": active_missing_count,
                "missing_keys": _serialize_strain_keys(missing_keys),
                "active_plan_completed_points": len(active_completed_keys),
                "active_plan_failed_points": active_failed_count,
                "active_plan_missing_points": active_missing_count,
                "active_plan_failed_keys": _serialize_strain_keys(active_failed_keys),
                "active_plan_missing_keys": _serialize_strain_keys(missing_keys),
                "historical_failed_points": len(historical_failed_keys),
                "historical_failed_keys": _serialize_strain_keys(historical_failed_keys),
                "historical_failure_attempts": historical_failure_attempts,
                "strain_completed": strain_completed,
                "per_direction_summary": per_direction_summary,
                "recovery_events": strain_recovery_events,
                "strain_relax_recovery_event_count": len(strain_recovery_events),
                "strain_relax_recovery_actions": sorted(
                    {
                        str(event.get("applied_action") or event.get("recovery_action"))
                        for event in strain_recovery_events
                        if event.get("applied_action") or event.get("recovery_action")
                    }
                ),
            },
            "services": {
                "parameter_plans": policy_plan_records,
                "retrieval_trace": retrieval_trace,
            },
        }

    def _build_output(self, inputs: StrainToolInput, raw: dict[str, Any], duration_s: float) -> StrainToolOutput:
        strain_data = list(raw.get("strain_data", []) or [])
        completed = [row for row in strain_data if row.get("completed", False)]
        failed = [row for row in strain_data if not row.get("completed", False)]
        summary = dict(raw.get("strain_summary", {}) or {})
        summary.setdefault("completed_points", len(completed))
        summary.setdefault("active_plan_failed_points", summary.get("failed_points", len(failed)))
        summary.setdefault("active_plan_missing_points", summary.get("missing_points", 0))
        summary.setdefault("failed_points", int(summary.get("active_plan_failed_points", len(failed)) or 0))
        summary.setdefault("missing_points", int(summary.get("active_plan_missing_points", 0) or 0))
        summary.setdefault("historical_failed_points", len(failed))
        summary.setdefault("historical_failure_attempts", int(summary.get("historical_failed_points", len(failed)) or 0))
        summary.setdefault(
            "strain_completed",
            bool(
                summary.get("active_plan_failed_points", summary.get("failed_points", len(failed))) == 0
                and summary.get("active_plan_missing_points", summary.get("missing_points", 0)) == 0
            ),
        )
        warnings = list(raw.get("warnings", []) or [])
        errors = list(raw.get("errors", []) or [])
        failed_points = int(summary.get("active_plan_failed_points", summary.get("failed_points", len(failed))) or 0)
        missing_points = int(summary.get("active_plan_missing_points", summary.get("missing_points", 0)) or 0)
        fit_ready = bool(summary.get("strain_completed", False))
        if failed_points > 0:
            warnings.append(f"strain_campaign_incomplete:{failed_points}_failed_points")
            errors.append(f"strain_campaign_incomplete:{failed_points}_failed_points")
        if missing_points > 0:
            warnings.append(f"strain_campaign_incomplete:{missing_points}_missing_points")
            errors.append(f"strain_campaign_incomplete:{missing_points}_missing_points")
        if not fit_ready and not errors:
            warnings.append("strain_campaign_not_fit_ready")
            errors.append("strain_campaign_not_fit_ready")
        warnings = list(dict.fromkeys(warnings))
        errors = list(dict.fromkeys(errors))
        return StrainToolOutput(
            success=not errors and fit_ready,
            warnings=warnings,
            key_summary={
                "completed_points": summary.get("completed_points", len(completed)),
                "failed_points": failed_points,
                "missing_points": missing_points,
                "strain_completed": fit_ready,
            },
            artifact_paths=build_artifact_map(os.path.join(inputs.base_dir, "05_strain")),
            error_summary=("; ".join(errors) if errors else None),
            duration_s=duration_s,
            state_updates=raw,
            strain_data=strain_data,
            retry_backups=list(raw.get("relax_retry_backups", []) or []),
            strain_summary={**summary, "per_direction_plan": inputs.strain_plan_by_direction},
        )
