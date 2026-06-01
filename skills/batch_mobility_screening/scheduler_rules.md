# Scheduler Rules

- Default scheduler is sequential.
- Each claimed material gets its own folder under `RUNS_ROOT/<material_id>/`.
- The canonical runtime workdir remains `<material_root>/mobility_calculation`.
- Skipped or failed materials do not stop the full batch by default.
