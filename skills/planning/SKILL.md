+++
name = "planning"
version = "1"
description = "Propose the highest-value next action toward a reliable mobility result."
load_strategy = "summary_only"
roles = ["planner"]
task_types = ["single_material", "batch_database"]
stages = ["proposal_phase"]
run_statuses = ["running", "ready_to_finalize"]
tags = ["planning", "proposal", "mainline", "next_action"]
+++
# planning

## purpose
Propose the highest-value next action without treating the default scientific path as a rigid controller.

## when_to_use
- At each proposal boundary after observation
- When multiple legal next steps are available

## required_inputs
- execution status
- task board summary
- capability metadata
- latest observation

## relevant_state_fields
- `workflow.current_stage`
- `workflow.run_status`
- `task_board.*`
- `blackboard.latest_execution_observation`
- `services.skill_resolution`

## allowed_tools
- `query_execution_status`
- `query_capability_metadata`
- `inspect_workspace`
- `inspect_artifacts`
- `resolve_skills`

## decision_rules
- Prefer the next justified scientific capability when no stronger contrary evidence exists.
- Do not overfit to the default mainline after fresh failures or anomalies.
- Keep proposals inside the action registry and stage contracts.

## stop_conditions
- one or more structured proposals emitted

## expected_output_schema
- `ProposalBundle`

## caveats / warnings
- Planning proposes actions; it does not execute physics tools directly.
