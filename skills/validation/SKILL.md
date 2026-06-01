+++
name = "validation"
version = "1"
description = "Turn post-processing diagnostics into a bounded final validation decision without replacing deterministic checks."
load_strategy = "summary_only"
roles = ["validation"]
task_types = ["single_material"]
stages = ["validation", "final_report"]
run_statuses = ["running", "ready_to_finalize"]
tags = ["validation", "postprocess", "acceptance", "quality_gate"]
+++
# validation

## purpose
Convert anomaly flags, fit quality, and accepted-channel status into a final validation decision that can gate final reporting.

## when_to_use
- After mobility fitting and post-processing
- Before final acceptance is written

## required_inputs
- accepted and rejected channel summary
- anomaly flags
- effective fit quality
- historical validation heuristics

## relevant_state_fields
- `physics_results.accepted_channels`
- `physics_results.rejected_channels`
- `diagnostics.validation_report`
- `diagnostics.fit_diagnostics`
- `memory.validation_hints`

## allowed_tools
- `synthesize_observation`
- `inspect_artifacts`
- `resolve_skills`

## decision_rules
- Fail when no credible accepted channels remain.
- Escalate when heuristics and current evidence materially disagree.
- Preserve deterministic validation outputs and only summarize or classify them.

## stop_conditions
- pass
- pass_with_warning
- fail
- escalate

## expected_output_schema
- `ValidationDecision`

## caveats / warnings
- Validation is an evidence-bound gate, not a free-form replacement for deterministic post-processing.
