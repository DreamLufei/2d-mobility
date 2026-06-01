+++
name = "physics_validation"
version = "1"
description = "Perform bounded, physics-aware final validation on mobility results and diagnostics."
load_strategy = "summary_only"
roles = ["physics_judge", "critic", "reporter", "orchestrator"]
task_types = ["single_material"]
stages = ["critique_phase", "arbitration_phase", "final_report", "validation"]
run_statuses = ["running", "ready_to_finalize"]
anomaly_patterns = ["anomaly", "negative_mobility", "fit_quality"]
tags = ["validation", "physics", "anomaly", "mobility"]
+++
# physics_validation

## purpose
Perform bounded, physics-aware final validation on mobility results and diagnostics.

## when_to_use
- After mobility fitting

## required_inputs
- mobility output
- fit diagnostics
- warnings
- anomaly checks

## relevant_state_fields
- `physics_results.mobility`
- `physics_results.results`
- `diagnostics.fit_diagnostics`
- `diagnostics.validation_report`
- `material.warnings`

## allowed_tools
- physics validators
- anomaly detectors
- final reporting

## decision_rules
- Return `pass`, `pass_with_warning`, `fail`, or `escalate`.
- Keep validation structured and evidence-backed.

## stop_conditions
- final validation decision emitted

## expected_output_schema
- `ValidationDecision`

## caveats / warnings
- Validation is bounded reasoning, not a replacement for deterministic post-processing.
