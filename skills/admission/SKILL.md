+++
name = "admission"
version = "1"
description = "Apply bounded preflight screening before a material enters the deterministic mobility workflow."
load_strategy = "summary_only"
roles = ["admission"]
task_types = ["single_material", "batch_database"]
stages = ["admission"]
run_statuses = ["created", "running"]
tags = ["admission", "preflight", "screening", "scope"]
+++
# admission

## purpose
Screen candidate materials for required inputs and obvious out-of-scope conditions before deterministic execution begins.

## when_to_use
- At the first preflight/admission boundary
- When structure metadata or warnings suggest the task may be out of scope

## required_inputs
- material input paths
- structure metadata
- warning list
- atom count summary

## relevant_state_fields
- `material.poscar_path`
- `material.potcar_path`
- `material.structure_metadata`
- `material.atom_count`
- `material.warnings`

## allowed_tools
- `inspect_workspace`
- `resolve_skills`

## decision_rules
- Reject when required inputs are missing or the system is clearly out of scope.
- Prefer `continue_with_warning` over silent acceptance when preflight warnings remain.
- Keep admission decisions bounded to scope, input readiness, and obvious material eligibility.

## stop_conditions
- continue
- continue_with_warning
- reject

## expected_output_schema
- `AdmissionDecision`

## caveats / warnings
- Admission may gate execution, but it must not fabricate downstream physics judgments.
