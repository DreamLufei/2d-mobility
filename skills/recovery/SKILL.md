+++
name = "recovery"
version = "1"
description = "Choose only supported, stage-scoped recovery actions for deterministic workflow failures."
load_strategy = "summary_only"
roles = ["recovery", "planner", "orchestrator", "cost_guardian"]
task_types = ["single_material", "batch_database"]
stages = ["proposal_phase", "critique_phase", "arbitration_phase", "execute_selected_action"]
run_statuses = ["needs_recovery", "running"]
error_patterns = ["failed", "missing", "nonconverged", "zbrent", "returncode"]
tags = ["recovery", "error", "retry", "manual_fix"]
+++
# recovery

## purpose
Choose only supported, stage-scoped recovery actions for deterministic workflow failures.

## when_to_use
- Any stage failure after `prepare`
- Recovery retry or escalation planning

## required_inputs
- stage name
- normalized error summary
- retry counts
- raw evidence references
- allowed actions registry

## relevant_state_fields
- `workflow.current_stage`
- `workflow.retry_counts`
- `diagnostics.last_error`
- `diagnostics.recovery_summary`
- `diagnostics.raw_evidence`
- `execution.pending_parameter_updates`

## allowed_tools
- cleanup policy preview/apply
- existing native retry paths
- manual-fix resume protocol

## decision_rules
- Never invent unsupported actions.
- Respect the stage-scoped allowed-actions registry.
- Prefer deterministic retry logic before escalation.
- Escalate on low confidence, unknown failure, or retry exhaustion.

## stop_conditions
- supported recovery action selected
- escalation requested
- skip/abort chosen

## expected_output_schema
- `RecoveryDecision`

## caveats / warnings
- Recovery changes orchestration only; physics tools remain deterministic.
