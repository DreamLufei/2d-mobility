+++
name = "orchestration"
version = "1"
description = "Integrate planner, recovery, critique, physics, and cost opinions into one selected action."
load_strategy = "summary_only"
roles = ["orchestrator"]
task_types = ["single_material", "batch_database"]
stages = ["arbitration_phase"]
run_statuses = ["running", "needs_recovery", "ready_to_finalize", "waiting_external"]
tags = ["orchestration", "arbitration", "selection", "guardrails"]
+++
# orchestration

## purpose
Act as the chief decision layer that selects one legal, justified action from competing proposals.

## when_to_use
- At every arbitration boundary
- When guardrail review and agent preferences disagree

## required_inputs
- proposal bundle
- critique bundle
- legality checks
- latest execution status
- memory hints

## relevant_state_fields
- `deliberation.proposals`
- `deliberation.critiques`
- `deliberation.preferences`
- `workflow.run_status`
- `memory.*`

## allowed_tools
- `check_action_legality`
- `query_execution_status`
- `query_capability_metadata`
- `synthesize_observation`
- `resolve_skills`

## decision_rules
- Never choose an illegal proposal.
- Prefer actions with stronger support and lower objection load when otherwise comparable.
- Escalate or noop when the runtime is waiting or when no legal proposal survives.

## stop_conditions
- one selected action emitted
- deliberate noop returned

## expected_output_schema
- `ArbitrationDecisionPayload`

## caveats / warnings
- Arbitration integrates opinions; it should not invent a hidden execution path.
