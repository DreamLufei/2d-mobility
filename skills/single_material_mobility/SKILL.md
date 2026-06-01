+++
name = "single_material_mobility"
version = "1"
description = "Coordinate the canonical single-material mobility workflow without changing deterministic physics logic."
load_strategy = "summary_only"
roles = ["planner", "orchestrator", "executor", "reporter"]
task_types = ["single_material"]
stages = ["observe_state", "proposal_phase", "arbitration_phase", "execute_selected_action", "final_report"]
run_statuses = ["running", "ready_to_finalize"]
tags = ["mainline", "single_material", "mobility", "workflow"]
+++
# single_material_mobility

## purpose
Coordinate the canonical single-material mobility workflow without changing deterministic physics logic.

## when_to_use
- Single-folder local execution
- Per-material execution inside batch mode

## required_inputs
- `task.root_path`
- `material.poscar_path`
- `material.potcar_path`
- runtime configuration

## relevant_state_fields
- `workflow.current_stage`
- `workflow.stage_status`
- `execution.workdir`
- `diagnostics.*`
- `physics_results.*`

## allowed_tools
- `prepare`
- `relax`
- `scf`
- `band`
- `effective_mass`
- `strain_loop`
- `mobility`
- artifact writers

## decision_rules
- Preserve successful upstream stages.
- Route deterministic failures into recovery instead of inventing new tool behavior.
- Only escalate when confidence is low or retry logic is exhausted.

## stop_conditions
- final report written
- rejected by admission
- skipped by HITL fallback
- terminated by validation or all-channel rejection

## expected_output_schema
- `MaterialRunOutcome`

## caveats / warnings
- Agents must not replace native VASP tools.
- Shared state is the primary communication channel.
---
# placeholder
---
