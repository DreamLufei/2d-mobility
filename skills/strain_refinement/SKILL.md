+++
name = "strain_refinement"
version = "1"
description = "Interpret strain-fit quality and decide whether refinement is needed before mobility fitting is accepted."
load_strategy = "summary_only"
roles = ["physics_judge", "planner", "orchestrator"]
task_types = ["single_material"]
stages = ["proposal_phase", "critique_phase", "arbitration_phase"]
run_statuses = ["running"]
anomaly_patterns = ["valley_switch", "fit_quality", "failed_points"]
tags = ["strain", "refinement", "sampling", "fit_quality"]
+++
# strain_refinement

## purpose
Interpret strain-fit quality and decide whether refinement is needed before mobility fitting is accepted.

## when_to_use
- After a full strain loop
- Before mobility fitting is finalized

## required_inputs
- strain data
- fit metrics
- accepted/rejected channels
- refinement budget

## relevant_state_fields
- `physics_results.strain_data`
- `physics_results.strain_plan_by_direction`
- `physics_results.accepted_channels`
- `physics_results.rejected_channels`
- `workflow.refinement_rounds`
- `workflow.max_refinement_rounds`

## allowed_tools
- strain loop rerun
- channel rejection
- termination
- human escalation

## decision_rules
- Prefer accept when fit quality is already above threshold.
- Suggest midpoint enrichment when more points are likely useful and budget remains.
- Reject channels explicitly instead of silently dropping them.

## stop_conditions
- accept
- refine_more_points
- reject_channel
- terminate
- escalate

## expected_output_schema
- `RefinementDecision`

## caveats / warnings
- Refinement must not mutate upstream deterministic physics results outside the strain loop.
