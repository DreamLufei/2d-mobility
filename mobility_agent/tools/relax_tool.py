from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .base import DeterministicTool, ToolInputBase, ToolOutputBase, build_artifact_map
from .vasp_common import (
    build_incar_relax,
    copy_poscar_potcar,
    policy_stage_planning_allowed,
    prune_dir_keep_files,
    reuse_completed_vasp_stages_enabled,
    write_incar,
    write_relax_scf_kpoints,
)
from .physics_common import get_reciprocal_lattice
from .relax_retry import RelaxRetryFatal, relax_retry_enabled, run_relax_vasp_with_retry

if TYPE_CHECKING:
    from ..policy.engine import AgenticPolicyEngine


class RelaxToolInput(ToolInputBase):
    poscar_path: str
    potcar_path: str
    no_relax_retry: bool = False
    recovery_param_updates: dict[str, Any] = Field(default_factory=dict)


class RelaxToolOutput(ToolOutputBase):
    relaxed_poscar: str | None = None
    reciprocal_lattice: list[list[float]] | None = None
    retry_backups: list[str] = Field(default_factory=list)
    recovery_summary: dict[str, Any] = Field(default_factory=dict)


class RelaxTool(DeterministicTool):
    name = "run_relax"
    description = "Run structural relaxation with retry-aware recovery summary"
    input_model = RelaxToolInput
    output_model = RelaxToolOutput

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

    def _execute(self, inputs: RelaxToolInput) -> dict[str, Any]:
        if self.executor is not None:
            return super()._execute(inputs)

        work_dir = os.path.join(inputs.base_dir, "01_relax")
        os.makedirs(work_dir, exist_ok=True)
        relaxed_poscar = os.path.join(work_dir, "CONTCAR")
        reuse_warnings: list[str] = []
        if reuse_completed_vasp_stages_enabled() and os.path.exists(relaxed_poscar):
            try:
                if os.path.getsize(relaxed_poscar) <= 0:
                    raise ValueError("empty CONTCAR")
                rec_lattice = get_reciprocal_lattice(relaxed_poscar)
            except Exception as exc:
                reuse_warnings.append(f"ignored_invalid_reused_relax_contcar:{exc}")
            else:
                existing_backups = list((inputs.state_payload or {}).get("relax_retry_backups") or [])
                return {
                    "relax_completed": True,
                    "relaxed_poscar": relaxed_poscar,
                    "reciprocal_lattice": rec_lattice.tolist(),
                    "relax_retry_backups": existing_backups,
                    "recovery_summary": {
                        "stage": "relax",
                        "applied_action": "reuse_completed_stage",
                        "final_outcome": "success",
                    },
                    "warnings": ["reused_completed_stage:relax"],
                    "_tool_source": "native_tool_reuse",
                }

        relax_overrides = dict(inputs.recovery_param_updates or {})
        reuse_existing_workdir_poscar = bool(relax_overrides.pop("__use_existing_workdir_poscar__", False))
        if reuse_existing_workdir_poscar and os.path.exists(os.path.join(work_dir, "POSCAR")):
            shutil.copy(str(inputs.potcar_path), os.path.join(work_dir, "POTCAR"))
        else:
            copy_poscar_potcar(inputs.poscar_path, inputs.potcar_path, work_dir)

        incar_params = build_incar_relax(
            "Structure Relaxation",
            ediff=float(relax_overrides.get("EDIFF", 1e-5)),
            isif=int(relax_overrides.get("ISIF", 3)),
            lattice_constraints=".TRUE. .TRUE. .FALSE.",
            consider_spin=self.consider_spin,
        )
        for key, value in relax_overrides.items():
            if key not in {"EDIFF", "ISIF"}:
                incar_params[key] = value
        kpoints_policy = {"target_ka": self.target_ka, "gamma_centered": False}
        if self.policy_engine is not None and policy_stage_planning_allowed(inputs.state_payload, "relax"):
            plan = self.policy_engine.plan_stage(
                stage="relax",
                state_payload=dict(inputs.state_payload or {}),
                default_incar=incar_params,
                default_kpoints_policy=kpoints_policy,
            )
        else:
            plan = None
        if plan is not None:
            incar_params.update(dict(plan.incar_overrides or {}))
            if dict(plan.kpoints_policy or {}):
                kpoints_policy.update(dict(plan.kpoints_policy or {}))
            policy_updates = {
                "services": {
                    "parameter_plans": {"relax": plan.model_dump(mode="json")},
                    "retrieval_trace": [
                        {
                            "kind": "parameter_plan",
                            "stage": "relax",
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
            material_name=inputs.material_id,
            target_ka=float(kpoints_policy.get("target_ka", self.target_ka) or self.target_ka),
            gamma_centered=bool(kpoints_policy.get("gamma_centered", False)),
        )

        retry_log = os.path.join(inputs.base_dir, "vasp_relax_retry.log")
        enabled = relax_retry_enabled(cli_no_relax_retry=bool(inputs.no_relax_retry))
        existing_backups = list((inputs.state_payload or {}).get("relax_retry_backups") or [])
        try:
            ok, backups, recovery_summary = run_relax_vasp_with_retry(
                workdir=work_dir,
                material_id=str(inputs.material_id),
                vasp_cmd=self.vasp_cmd,
                retry_log_path=retry_log,
                enabled=enabled,
                check_convergence=True,
            )
        except RelaxRetryFatal as e:
            recovery_summary = dict(getattr(e, "summary", {}) or {})
            error_text = str(e)
            return {
                "errors": ["SKIP_AFTER_3_RETRIES" if error_text == "SKIP_AFTER_3_RETRIES" else f"RELAX_FATAL:{error_text}"],
                "warnings": reuse_warnings,
                "relax_retry_backups": existing_backups,
                "recovery_summary": recovery_summary or {
                    "stage": "relax",
                    "error_type": "relax_fatal",
                    "trigger_pattern": error_text,
                    "retries_used": len(existing_backups),
                    "max_retries": 3,
                    "recommended_action": "skip_material",
                    "applied_action": "skip_material",
                    "final_outcome": "failed",
                },
                **policy_updates,
            }

        if not ok:
            error_message = str(dict(recovery_summary or {}).get("error_summary") or "结构弛豫失败")
            return {
                "errors": [error_message],
                "warnings": reuse_warnings,
                "relax_retry_backups": existing_backups + list(backups or []),
                "recovery_summary": recovery_summary or {
                    "stage": "relax",
                    "error_type": "relax_failed",
                    "trigger_pattern": "run_relax_vasp_with_retry_returned_false",
                    "retries_used": len(existing_backups) + len(list(backups or [])),
                    "max_retries": 3,
                    "recommended_action": "skip_material" if len(existing_backups) + len(list(backups or [])) >= 3 else "retry",
                    "applied_action": "skip_material" if len(existing_backups) + len(list(backups or [])) >= 3 else "retry",
                    "final_outcome": "failed",
                },
                **policy_updates,
            }

        if not os.path.exists(relaxed_poscar):
            return {
                "errors": ["CONTCAR 不存在"],
                "warnings": reuse_warnings,
                "recovery_summary": {
                    "stage": "relax",
                    "error_type": "missing_output",
                    "trigger_pattern": "CONTCAR_MISSING",
                    "retries_used": len(existing_backups) + len(list(backups or [])),
                    "max_retries": 3,
                    "recommended_action": "skip_material",
                    "applied_action": "skip_material",
                    "final_outcome": "failed",
                },
                **policy_updates,
            }

        rec_lattice = get_reciprocal_lattice(relaxed_poscar)
        result = {
            "relax_completed": True,
            "relaxed_poscar": relaxed_poscar,
            "reciprocal_lattice": rec_lattice.tolist(),
            "relax_retry_backups": existing_backups + list(backups or []),
            "recovery_summary": recovery_summary,
            "warnings": reuse_warnings,
            "_tool_source": "native_tool",
            **policy_updates,
        }
        prune_dir_keep_files(work_dir, {"INCAR", "KPOINTS", "POSCAR", "CONTCAR", "POTCAR"})
        return result

    def _build_output(self, inputs: RelaxToolInput, raw: dict[str, Any], duration_s: float) -> RelaxToolOutput:
        base_dir = inputs.base_dir
        relax_dir = os.path.join(base_dir, "01_relax")
        relaxed_poscar = raw.get("relaxed_poscar")
        retry_backups = list(raw.get("relax_retry_backups", []) or [])
        return RelaxToolOutput(
            success=not bool(raw.get("errors")),
            warnings=list(raw.get("warnings", []) or []),
            key_summary={
                "relax_completed": bool(raw.get("relax_completed", False)),
                "relaxed_poscar": relaxed_poscar,
                "retry_backups": retry_backups,
                "recovery_summary": dict(raw.get("recovery_summary", {}) or {}),
            },
            artifact_paths=build_artifact_map(relax_dir, os.path.join(base_dir, "vasp_relax_retry.log")),
            error_summary=("; ".join(raw.get("errors", [])) if raw.get("errors") else None),
            duration_s=duration_s,
            state_updates=raw,
            relaxed_poscar=relaxed_poscar,
            reciprocal_lattice=raw.get("reciprocal_lattice"),
            retry_backups=retry_backups,
            recovery_summary=dict(raw.get("recovery_summary", {}) or {}),
        )
