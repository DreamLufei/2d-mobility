+++
name = "cost_guardian"
version = "1"
description = "Constrain wasteful retries, redundant recompute, and low-yield refinement."
load_strategy = "summary_only"
roles = ["cost_guardian"]
task_types = ["single_material", "batch_database"]
stages = ["critique_phase"]
run_statuses = ["running", "needs_recovery", "ready_to_finalize"]
tags = ["cost", "budget", "retry", "refinement"]
+++
# cost_guardian

## purpose
Evaluate whether proposed retries, reruns, and refinement steps justify their runtime and compute cost.

## when_to_use
- When proposals include retry, rerun, refinement, or termination decisions

## required_inputs
- retry counts
- retry budget
- refinement budget
- proposal cost and risk class

## relevant_state_fields
- `workflow.retry_counts`
- `workflow.retry_budget`
- `workflow.refinement_rounds`
- `workflow.max_refinement_rounds`

## allowed_tools
- `query_execution_status`
- `query_capability_metadata`
- `check_action_legality`
- `resolve_skills`

## decision_rules
- Object to repeated retries with weak new evidence.
- Distinguish necessary recovery from wasteful recompute.
- Allow higher-cost actions when they are the last credible scientific path.

## stop_conditions
- cost-aware critiques and preferences emitted

## expected_output_schema
- `ReviewBundle`

## caveats / warnings
- Cost review should constrain waste, not veto every expensive but necessary step.
