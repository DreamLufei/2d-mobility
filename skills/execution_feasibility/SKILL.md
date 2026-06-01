+++
name = "execution_feasibility"
version = "1"
description = "Detect missing inputs and restore execution feasibility before a stage is run."
load_strategy = "summary_only"
roles = ["executor"]
task_types = ["single_material", "batch_database"]
stages = ["proposal_phase", "execute_selected_action"]
run_statuses = ["running", "needs_recovery"]
tags = ["executor", "feasibility", "missing_inputs", "repair"]
+++
# execution_feasibility

## purpose
Detect missing required inputs for a capability and propose bounded execution-context repair when possible.

## when_to_use
- Before running a stage with missing dependencies or artifacts
- When current action feasibility is uncertain

## required_inputs
- current target capability
- required input metadata
- current state snapshot

## relevant_state_fields
- `task_board.*`
- `execution.artifact_registry`
- `execution.latest_execution_observation`
- `physics_results.*`

## allowed_tools
- `query_capability_metadata`
- `inspect_artifacts`
- `check_action_legality`
- `resolve_skills`

## decision_rules
- Prefer restoring required inputs over blind rerun when the missing context is explicit.
- Keep repairs bounded to orchestration state and artifact context.
- Escalate when the missing execution context cannot be repaired safely.

## stop_conditions
- bounded repair proposal emitted
- no repair needed

## expected_output_schema
- `ProposalBundle`

## caveats / warnings
- Execution-feasibility logic must not fabricate physics outputs.
