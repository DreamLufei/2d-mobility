+++
name = "batch_mobility_screening"
version = "1"
description = "Run Mongo-backed batch screening by reusing the canonical single-material runtime."
load_strategy = "summary_only"
roles = ["planner", "orchestrator", "reporter"]
task_types = ["batch_database"]
stages = ["observe_state", "proposal_phase", "arbitration_phase", "final_report"]
run_statuses = ["running", "ready_to_finalize"]
tags = ["batch", "screening", "sequential", "mongo"]
+++
# batch_mobility_screening

## purpose
Run database or collection screening by reusing the canonical single-material runtime for every claimed material.

## when_to_use
- Mongo batch execution
- Dataset-wide screening

## required_inputs
- batch collection metadata
- queue claim capability
- per-material structure payloads

## relevant_state_fields
- `task.collection_name`
- `batch.queue`
- `batch.running_items`
- `batch.completed_items`
- `batch.failed_items`
- `batch.skipped_items`
- `batch.global_statistics`

## allowed_tools
- material enumeration
- POSCAR/POTCAR preparation
- canonical single-material runner
- database update helpers

## decision_rules
- Keep scheduling conservative and sequential by default.
- Treat the single-material runner as the only execution authority.
- Aggregate outcomes directly from `MaterialRunOutcome`.

## stop_conditions
- no pending materials remain
- batch abort policy is triggered

## expected_output_schema
- `BatchSummary`

## caveats / warnings
- Batch mode must not fork its own independent scientific workflow.
