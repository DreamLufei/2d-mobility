# Batch Mode

## Design
- Batch mode is conservative and sequential by default.
- The batch runner prepares material folders, builds input files, invokes the canonical single-material runner, and aggregates canonical outcomes.
- There is no subprocess shell-out to `mobality.py`; batch directly reuses the runner API.
- Batch configuration is limited to scheduler/source/runtime-relevant inputs.
- Old subprocess-era fields such as `MOBALITY_SCRIPT` are no longer active configuration and only survive as deprecated aliases in env normalization.

## Current Flow
1. Claim next material from MongoDB.
2. Write `POSCAR`.
3. Build `POTCAR`.
4. Run `run_single_material(...)`.
5. Update MongoDB with completed, failed, or skipped outcome.
6. Update batch statistics and summary.

## Outputs
- Per-material runtime artifacts live under `<RUNS_ROOT>/<material_id>/mobility_calculation/`.
- Batch summary is written to `<RUNS_ROOT>/batch_summary_<BATCH_TAG>.json`.
- Batch init records normalized/deprecated config warnings in its execution environment summary when legacy aliases are encountered.
