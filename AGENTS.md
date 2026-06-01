 # Repository Agent Guidelines

## Scope
- This repository is a real scientific-computing codebase for first-principles 2D carrier mobility.
- Preserve deterministic physics and VASP-native tools whenever possible.
- Refactor orchestration, state, recovery, reporting, and runtime integration aggressively when needed.

## Architecture Expectations
- The canonical execution backbone is the LangGraph single-material runtime in `mobility_agent/graph/` and `mobility_agent/runtime/`.
- Batch mode must reuse the same single-material runner instead of duplicating workflow logic.
- Agents are bounded decision makers only. They must not replace native tools with free-form LLM execution.
- Shared state is the primary communication mechanism. All state updates should be history-preserving unless an overwrite is explicitly intended.
- Recovery and action legality must come from the stage-scoped registries in `mobility_agent/graph/recovery_registry.py` and `mobility_agent/runtime/action_registry.py`.

## Change Discipline
- Keep stage contracts explicit: inputs, canonical outputs, failure outputs, invalidation boundaries.
- Keep tool wrappers typed and preserve both normalized summaries and raw evidence.
- Preserve compatibility artifacts when practical: `mobility_results.json`, `fit_diagnostics.json`, `decision_trace.json`, `tool_trace.json`, `recovery_trace.json`, `validation_report.json`, `final_summary.json`, `material_outcome.json`, and `checkpoint.pkl`.
- Compatibility checkpoint exports belong only at stable stage boundaries, escalation points, and finalization points.

## Validation
- Run static checks or tests after edits when the environment allows it.
- Prefer dry-run architecture tests for orchestration logic instead of fabricating VASP internals.
- Document what changed and what stayed physically unchanged.
