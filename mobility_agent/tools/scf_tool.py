from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .vasp_common import (
    build_incar_scf,
    copy_poscar_potcar,
    policy_stage_planning_allowed,
    prune_dir_keep_files,
    reuse_completed_vasp_stages_enabled,
    run_vasp,
    summarize_vasp_failure,
    write_incar,
    write_relax_scf_kpoints,
)
from .physics_common import read_fermi_energy_eV

if TYPE_CHECKING:
    from ..policy.engine import AgenticPolicyEngine


class ScfToolInput(ToolInputBase):
    poscar_path: str
    potcar_path: str
    material_name: str


class ScfToolOutput(ToolOutputBase):
    fermi_energy: float | None = None
    chgcar_path: str | None = None
    scf_summary: dict[str, object] = Field(default_factory=dict)


class ScfTool(DeterministicTool):
    name = "run_scf"
    description = "Run SCF and summarize charge-density artifacts"
    input_model = ScfToolInput
    output_model = ScfToolOutput

    def __init__(
        self,
        executor=None,
        *,
        vasp_cmd: str = "mpirun -np 4 vasp_std > sout 2>&1",
        consider_spin: bool = False,
        target_ka: float = 50.0,
        policy_engine: AgenticPolicyEngine | None = None,
    ):
        super().__init__(executor=executor)
        self.vasp_cmd = vasp_cmd
        self.consider_spin = consider_spin
        self.target_ka = target_ka
        self.policy_engine = policy_engine

    def _execute(self, inputs: ScfToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        work_dir = os.path.join(inputs.base_dir, "02_scf")
        os.makedirs(work_dir, exist_ok=True)
        chgcar_path = os.path.join(work_dir, "CHGCAR")
        warnings: list[str] = []
        if reuse_completed_vasp_stages_enabled() and os.path.exists(chgcar_path):
            warnings.append("reused_completed_stage:scf")
            try:
                fermi = float(read_fermi_energy_eV(work_dir))
            except Exception:
                warnings.append("reused_scf_without_fermi_energy")
                warnings.append("reran_scf_to_restore_fermi_energy")
            else:
                return {
                    "scf_completed": True,
                    "fermi_energy": fermi,
                    "chgcar_path": chgcar_path,
                    "warnings": warnings,
                    "_tool_source": "native_tool_reuse",
                }

        copy_poscar_potcar(inputs.poscar_path, inputs.potcar_path, work_dir)
        incar_params = build_incar_scf("SCF Calculation", consider_spin=self.consider_spin, ediff=1e-6, lvtot=False, lvhar=False)
        kpoints_policy = {"target_ka": self.target_ka, "gamma_centered": False}
        if self.policy_engine is not None and policy_stage_planning_allowed(inputs.state_payload, "scf"):
            plan = self.policy_engine.plan_stage(
                stage="scf",
                state_payload=dict(inputs.state_payload or {}),
                default_incar=incar_params,
                default_kpoints_policy=kpoints_policy,
            )
            incar_params.update(dict(plan.incar_overrides or {}))
            if dict(plan.kpoints_policy or {}):
                kpoints_policy.update(dict(plan.kpoints_policy or {}))
            policy_updates = {
                "services": {
                    "parameter_plans": {"scf": plan.model_dump(mode="json")},
                    "retrieval_trace": [
                        {
                            "kind": "parameter_plan",
                            "stage": "scf",
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
        write_relax_scf_kpoints(
            work_dir,
            material_name=inputs.material_name,
            target_ka=float(kpoints_policy.get("target_ka", self.target_ka) or self.target_ka),
            gamma_centered=bool(kpoints_policy.get("gamma_centered", False)),
        )
        if not run_vasp(cwd=work_dir, vasp_cmd=self.vasp_cmd, check_convergence=True):
            recovery_summary = summarize_vasp_failure(work_dir, stage="scf", default_error="SCF 失败")
            return {
                "errors": [str(recovery_summary["error_summary"])],
                "warnings": warnings,
                "recovery_summary": recovery_summary,
                **policy_updates,
            }
        try:
            fermi = float(read_fermi_energy_eV(work_dir))
        except Exception:
            return {"errors": ["无法读取费米能级(OUTCAR/vasprun.xml/DOSCAR)"], "warnings": warnings, **policy_updates}
        prune_dir_keep_files(work_dir, {"INCAR", "KPOINTS", "POSCAR", "POTCAR", "CHGCAR", "OUTCAR", "OSZICAR"})
        return {
            "scf_completed": True,
            "fermi_energy": fermi,
            "chgcar_path": chgcar_path,
            "warnings": warnings,
            "_tool_source": "native_tool",
            **policy_updates,
        }

    def _build_output(self, inputs: ScfToolInput, raw: dict[str, object], duration_s: float) -> ScfToolOutput:
        scf_dir = os.path.join(inputs.base_dir, "02_scf")
        chgcar_path = os.path.join(scf_dir, "CHGCAR")
        fermi_energy = raw.get("fermi_energy")
        return ScfToolOutput(
            success=not bool(raw.get("errors")),
            warnings=list(raw.get("warnings", []) or []),
            key_summary={
                "scf_completed": bool(raw.get("scf_completed", False)),
                "fermi_energy": fermi_energy,
                "chgcar_path": chgcar_path if os.path.exists(chgcar_path) else None,
            },
            artifact_paths=build_artifact_map(scf_dir, chgcar_path),
            error_summary=("; ".join(raw.get("errors", [])) if raw.get("errors") else None),
            duration_s=duration_s,
            state_updates=raw,
            fermi_energy=float(fermi_energy) if fermi_energy is not None else None,
            chgcar_path=(chgcar_path if os.path.exists(chgcar_path) else None),
            scf_summary={
                "fermi_energy": fermi_energy,
                "scf_dir": scf_dir,
            },
        )
