+++
name = "critique"
version = "1"
description = "Challenge proposals on missing evidence, unsafe assumptions, and weak reasoning."
load_strategy = "summary_only"
roles = ["critic"]
task_types = ["single_material", "batch_database"]
stages = ["critique_phase"]
run_statuses = ["running", "needs_recovery", "ready_to_finalize"]
tags = ["critique", "review", "objection", "preference"]
+++
# critique

## purpose
Review competing proposals and call out illegal, under-evidenced, or strategically weak choices.

## when_to_use
- After proposals are generated
- Before orchestration/arbitration picks a final action

## required_inputs
- proposal bundle
- legality context
- artifact inspection
- task-board status

## relevant_state_fields
- `deliberation.proposals`
- `execution.latest_execution_observation`
- `workflow.run_status`
- `task_board.*`

## allowed_tools
- `check_action_legality`
- `query_capability_metadata`
- `inspect_artifacts`
- `resolve_skills`

## decision_rules
- Object when proposals violate dependencies, budgets, or evidence requirements.
- Prefer conservative, well-supported actions when multiple options are viable.
- Keep critiques tied to observable state and guardrail context.

## stop_conditions
- structured critiques and preferences emitted

## expected_output_schema
- `ReviewBundle`

## caveats / warnings
- Critique should challenge proposals, not silently replace orchestration.
