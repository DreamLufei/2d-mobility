+++
name = "reporting"
version = "1"
description = "Generate concise human-readable summaries and stable artifact references for material and batch outcomes."
load_strategy = "summary_only"
roles = ["reporter", "orchestrator", "report"]
task_types = ["single_material", "batch_database"]
stages = ["final_report", "report"]
run_statuses = ["ready_to_finalize", "completed", "failed", "skipped", "aborted"]
tags = ["report", "summary", "artifacts", "finalization"]
+++
# reporting

## purpose
Generate concise human-readable summaries and stable artifact references for material and batch outcomes.

## when_to_use
- Final single-material reporting
- Final batch reporting

## required_inputs
- final shared state
- validation report
- artifact paths
- aggregated batch outcomes when applicable

## relevant_state_fields
- `execution.artifact_paths`
- `diagnostics.validation_report`
- `diagnostics.confidence_score`
- `material.warnings`
- `batch.global_statistics`

## allowed_tools
- artifact writers
- report summarizers

## decision_rules
- Surface warnings and final status clearly.
- Report canonical artifact locations.
- Preserve trace references for auditing.

## stop_conditions
- report summary written

## expected_output_schema
- `ReportSummary`
- `BatchSummary`

## caveats / warnings
- Reporting must not overwrite deterministic scientific results.
