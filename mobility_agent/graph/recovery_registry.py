from __future__ import annotations


STAGE_ALLOWED_ACTIONS: dict[str, list[str]] = {
    "prepare": ["retry_current_stage", "manual_fix_resume", "skip_material", "abort_task"],
    "relax": [
        "retry_current_stage",
        "modify_params_and_retry",
        "copy_contcar_to_poscar_and_retry",
        "manual_fix_resume",
        "skip_material",
        "abort_task",
    ],
    "scf": ["retry_current_stage", "rerun_previous_stage", "manual_fix_resume", "skip_material", "abort_task"],
    "band": ["retry_current_stage", "rerun_previous_stage", "manual_fix_resume", "skip_material", "abort_task"],
    "effective_mass": ["retry_current_stage", "rerun_previous_stage", "manual_fix_resume", "skip_material", "abort_task"],
    "strain_loop": [
        "retry_current_stage",
        "rerun_previous_stage",
        "manual_fix_resume",
        "skip_point",
        "skip_material",
        "abort_task",
    ],
    "mobility": ["retry_current_stage", "rerun_previous_stage", "manual_fix_resume", "skip_material", "abort_task"],
}


def allowed_actions_for_stage(stage: str) -> list[str]:
    return list(STAGE_ALLOWED_ACTIONS.get(stage, ["skip_material", "abort_task"]))

