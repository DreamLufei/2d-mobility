from __future__ import annotations

import os

from ..graph.state import ExternalEventRecord, MaterialRunOutcome
from .agentic_controller import AgenticMaterialController, run_agentic_material, run_agentic_material_external_event
from .context import RuntimeContext
from .telemetry import emit_progress


def default_material_workdir(root_path: str) -> str:
    return os.path.abspath(os.path.join(root_path, "mobility_calculation"))


def _build_nodes(runtime: RuntimeContext) -> dict[str, object]:
    """Compatibility helper for tests and tooling that compile the graph directly."""
    controller = AgenticMaterialController(runtime)
    return {
        "observe_state": controller.observe_node,
        "proposal_phase": controller.proposal_node,
        "critique_phase": controller.critique_node,
        "arbitration_phase": controller.arbitration_node,
        "execute_selected_action": controller.execute_node,
        "reflect_round": controller.reflect_node,
        "check_termination": controller.check_termination_node,
        "final_report": controller.final_report_node,
    }


def run_single_material(
    *,
    runtime: RuntimeContext,
    material_id: str,
    root_path: str,
    workdir: str | None = None,
    poscar_path: str | None = None,
    potcar_path: str | None = None,
    user_goal: str = "calculate_2d_mobility",
    parent_batch_id: str | None = None,
    fresh: bool = False,
    thread_id: str | None = None,
) -> MaterialRunOutcome:
    runtime.require_llm_ready()
    root_abs = os.path.abspath(root_path)
    workdir_abs = os.path.abspath(workdir or default_material_workdir(root_abs))
    poscar_abs = os.path.abspath(poscar_path or os.path.join(root_abs, "POSCAR"))
    potcar_abs = os.path.abspath(potcar_path or os.path.join(root_abs, "POTCAR"))
    os.environ["MOBILITY_ACTIVE_ROOT_PATH"] = root_abs
    os.environ["MOBILITY_ACTIVE_WORKDIR"] = workdir_abs
    os.environ["MOBILITY_ACTIVE_MATERIAL_ID"] = material_id

    os.makedirs(workdir_abs, exist_ok=True)
    emit_progress(
        "starting single-material runtime",
        workdir=workdir_abs,
        details={
            "material_id": material_id,
            "fresh": fresh,
            "dry_run": runtime.dry_run,
            "workdir": workdir_abs,
        },
    )
    return run_agentic_material(
        runtime=runtime,
        material_id=material_id,
        root_path=root_abs,
        workdir=workdir_abs,
        poscar_path=poscar_abs,
        potcar_path=potcar_abs,
        user_goal=user_goal,
        parent_batch_id=parent_batch_id,
        fresh=fresh,
        thread_id=thread_id,
    )


def run_single_material_external_event(
    *,
    runtime: RuntimeContext,
    workdir: str,
    event: ExternalEventRecord | dict[str, object],
    thread_id: str | None = None,
) -> MaterialRunOutcome:
    runtime.require_llm_ready()
    workdir_abs = os.path.abspath(workdir)
    os.environ["MOBILITY_ACTIVE_WORKDIR"] = workdir_abs
    emit_progress(
        "resuming single-material runtime from external event",
        workdir=workdir_abs,
        details={"thread_id": thread_id},
    )
    return run_agentic_material_external_event(
        runtime=runtime,
        workdir=workdir_abs,
        event=event,
        thread_id=thread_id,
    )
