from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .vasp_common import (
    build_incar_band,
    copy_poscar_potcar,
    policy_stage_planning_allowed,
    prune_dir_keep_files,
    read_chgcar_compatible_incar_overrides,
    reuse_completed_vasp_stages_enabled,
    run_vasp,
    summarize_vasp_failure,
    symlink_force,
    write_band_kpoints,
    write_incar,
)
from .physics_common import find_band_edges_from_eigenval_fermi, find_band_edges_from_eigenval_occupancy

if TYPE_CHECKING:
    from ..policy.engine import AgenticPolicyEngine


class BandToolInput(ToolInputBase):
    poscar_path: str
    potcar_path: str
    chgcar_path: str | None = None
    fermi_energy: float | None = None


class BandToolOutput(ToolOutputBase):
    vbm_energy: float | None = None
    cbm_energy: float | None = None
    vbm_kpoint: list[float] | None = None
    cbm_kpoint: list[float] | None = None
    vbm_band_index: int | None = None
    cbm_band_index: int | None = None
    vbm_spin: int | None = None
    cbm_spin: int | None = None
    band_summary: dict[str, object] = Field(default_factory=dict)


class BandTool(DeterministicTool):
    name = "run_band"
    description = "Run line-mode band calculation and extract band-edge summary"
    input_model = BandToolInput
    output_model = BandToolOutput

    def __init__(
        self,
        executor=None,
        *,
        vasp_cmd: str = "mpirun -np 4 vasp_std > sout 2>&1",
        consider_spin: bool = False,
        npoints_per_segment: int = 40,
        policy_engine: AgenticPolicyEngine | None = None,
    ):
        super().__init__(executor=executor)
        self.vasp_cmd = vasp_cmd
        self.consider_spin = consider_spin
        self.npoints_per_segment = npoints_per_segment
        self.policy_engine = policy_engine

    def _extract_band_edges(self, eigenval: str, fermi_energy: float | None) -> tuple[tuple, str]:
        if fermi_energy is not None:
            return (
                find_band_edges_from_eigenval_fermi(
                    eigenval,
                    fermi_energy=float(fermi_energy),
                    fermi_tolerance_eV=1.0e-3,
                ),
                "fermi_energy",
            )
        return find_band_edges_from_eigenval_occupancy(eigenval, occ_threshold=0.5), "occupancy"

    def _execute(self, inputs: BandToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        work_dir = os.path.join(inputs.base_dir, "03_band")
        os.makedirs(work_dir, exist_ok=True)
        eigenval = os.path.join(work_dir, "EIGENVAL")
        reuse_warnings: list[str] = []
        if reuse_completed_vasp_stages_enabled() and os.path.exists(eigenval):
            try:
                edges, edge_source = self._extract_band_edges(eigenval, inputs.fermi_energy)
                vbm_b, vbm_kpt_b, vbm_idx_b, vbm_spin_b, cbm_b, cbm_kpt_b, cbm_idx_b, cbm_spin_b = edges
            except Exception as e:
                reuse_warnings.append(f"ignored_invalid_reused_band_eigenval:{str(e)}")
                try:
                    os.remove(eigenval)
                except OSError:
                    pass
            else:
                return {
                    "band_completed": True,
                    "vbm_energy": float(vbm_b),
                    "cbm_energy": float(cbm_b),
                    "vbm_kpoint": vbm_kpt_b,
                    "cbm_kpoint": cbm_kpt_b,
                    "vbm_band_index": vbm_idx_b,
                    "cbm_band_index": cbm_idx_b,
                    "vbm_spin": vbm_spin_b,
                    "cbm_spin": cbm_spin_b,
                    "band_edge_source": edge_source,
                    "warnings": ["reused_completed_stage:band", f"band_edge_source:{edge_source}"],
                    "_tool_source": "native_tool_reuse",
                }

        copy_poscar_potcar(inputs.poscar_path, inputs.potcar_path, work_dir)
        chgcar_src = inputs.chgcar_path or os.path.join(inputs.base_dir, "02_scf", "CHGCAR")
        if chgcar_src and os.path.exists(chgcar_src):
            symlink_force(chgcar_src, os.path.join(work_dir, "CHGCAR"))
        chgcar_compatible_overrides = read_chgcar_compatible_incar_overrides(inputs.base_dir)
        incar_params = build_incar_band("Band Structure", consider_spin=self.consider_spin, ediff=1e-6)
        incar_params.update(chgcar_compatible_overrides)
        kpoints_policy = {"line_mode_density": self.npoints_per_segment}
        if self.policy_engine is not None and policy_stage_planning_allowed(inputs.state_payload, "band"):
            plan = self.policy_engine.plan_stage(
                stage="band",
                state_payload=dict(inputs.state_payload or {}),
                default_incar=incar_params,
                default_kpoints_policy=kpoints_policy,
            )
            incar_params.update(dict(plan.incar_overrides or {}))
            incar_params.update(chgcar_compatible_overrides)
            if dict(plan.kpoints_policy or {}):
                kpoints_policy.update(dict(plan.kpoints_policy or {}))
            policy_updates = {
                "services": {
                    "parameter_plans": {"band": plan.model_dump(mode="json")},
                    "retrieval_trace": [
                        {
                            "kind": "parameter_plan",
                            "stage": "band",
                            "source": plan.source,
                            "confidence": plan.confidence,
                            "evidence": [item.model_dump(mode="json") for item in list(plan.evidence_items or [])],
                            "rationale": plan.rationale,
                        }
                    ],
                }
            }
        else:
            policy_updates = {}
        write_incar(work_dir, incar_params)
        write_band_kpoints(
            os.path.join(work_dir, "KPOINTS"),
            npoints_per_segment=int(kpoints_policy.get("line_mode_density", self.npoints_per_segment) or self.npoints_per_segment),
        )
        if not run_vasp(cwd=work_dir, vasp_cmd=self.vasp_cmd, check_convergence=False):
            recovery_summary = summarize_vasp_failure(work_dir, stage="band", default_error="BAND 失败")
            return {"errors": [str(recovery_summary["error_summary"])], "warnings": reuse_warnings, "recovery_summary": recovery_summary, **policy_updates}

        try:
            edges, edge_source = self._extract_band_edges(eigenval, inputs.fermi_energy)
            vbm_b, vbm_kpt_b, vbm_idx_b, vbm_spin_b, cbm_b, cbm_kpt_b, cbm_idx_b, cbm_spin_b = edges
        except Exception as e:
            return {"errors": [f"03_band 能带解析失败: {str(e)}"], **policy_updates}
        prune_dir_keep_files(work_dir, {"INCAR", "KPOINTS", "POSCAR", "POTCAR", "EIGENVAL"})
        return {
            "band_completed": True,
            "vbm_energy": float(vbm_b),
            "cbm_energy": float(cbm_b),
            "vbm_kpoint": vbm_kpt_b,
            "cbm_kpoint": cbm_kpt_b,
            "vbm_band_index": vbm_idx_b,
            "cbm_band_index": cbm_idx_b,
            "vbm_spin": vbm_spin_b,
            "cbm_spin": cbm_spin_b,
            "band_edge_source": edge_source,
            "warnings": reuse_warnings + [f"band_edge_source:{edge_source}"],
            "_tool_source": "native_tool",
            **policy_updates,
        }

    def _build_output(self, inputs: BandToolInput, raw: dict[str, object], duration_s: float) -> BandToolOutput:
        band_dir = os.path.join(inputs.base_dir, "03_band")
        vbm = raw.get("vbm_energy")
        cbm = raw.get("cbm_energy")
        bandgap = None
        if vbm is not None and cbm is not None:
            bandgap = float(cbm) - float(vbm)
        return BandToolOutput(
            success=not bool(raw.get("errors")),
            warnings=list(raw.get("warnings", []) or []),
            key_summary={
                "band_completed": bool(raw.get("band_completed", False)),
                "bandgap_eV": bandgap,
                "vbm_energy": vbm,
                "cbm_energy": cbm,
                "band_edge_source": raw.get("band_edge_source"),
            },
            artifact_paths=build_artifact_map(band_dir, os.path.join(band_dir, "EIGENVAL")),
            error_summary=("; ".join(raw.get("errors", [])) if raw.get("errors") else None),
            duration_s=duration_s,
            state_updates=raw,
            vbm_energy=float(vbm) if vbm is not None else None,
            cbm_energy=float(cbm) if cbm is not None else None,
            vbm_kpoint=raw.get("vbm_kpoint"),
            cbm_kpoint=raw.get("cbm_kpoint"),
            vbm_band_index=raw.get("vbm_band_index"),
            cbm_band_index=raw.get("cbm_band_index"),
            vbm_spin=raw.get("vbm_spin"),
            cbm_spin=raw.get("cbm_spin"),
            band_summary={
                "bandgap_eV": bandgap,
                "band_dir": band_dir,
                "vbm_kpoint": raw.get("vbm_kpoint"),
                "cbm_kpoint": raw.get("cbm_kpoint"),
                "band_edge_source": raw.get("band_edge_source"),
            },
        )
