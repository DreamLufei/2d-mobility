from __future__ import annotations

# Shared default guidance when no role-specific skill package is discovered.
DEFAULT_SCIENCE_MAINLINE = (
    "For 2D mobility calculations, the default scientific mainline is: "
    "admission or preflight, then prepare, relax, scf, band-edge analysis, effective mass, "
    "strain sampling, mobility estimation, physics validation, and final summary. "
    "This is a preferred default scientific path, not a rigid workflow. "
    "Task-board progression and capability ordering are contextual reminders, not automatic path controllers. "
    "You may insert recovery, parameter adjustment, selective recompute, refinement, validation, "
    "channel invalidation, human escalation, skipping, or termination when the observed state justifies it."
)
