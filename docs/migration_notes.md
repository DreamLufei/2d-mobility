# Migration Notes

## What Changed
- The old split between entrypoint-owned workflow logic and runtime-owned workflow logic has been replaced by one canonical LangGraph runtime.
- Shared state is typed and organized by task, material, workflow, execution, diagnostics, physics results, agent, and batch sections.
- Recovery, refinement, validation, and reporting are implemented as bounded agents with structured outputs.
- Batch mode now reuses the canonical single-material runner directly.
- HITL now uses graph interrupt plus runner-side timeout and resume handling.
- LangGraph `PostgresSaver` is the durable checkpoint source of truth, and LangGraph `PostgresStore` is the long-term memory backend.
- Policy retrieval now uses Postgres + pgvector for bundled house-policy and synced VASP Wiki evidence, with the web console exposing dedicated Wiki query/reindex flows.
- `.runtime/shared_state.json` and `checkpoint.pkl` remain as readable exports, but restore no longer falls back to them ahead of LangGraph persistence.
- `runtime/runner.py` consumes normalized HITL decisions instead of parsing raw human-response payloads directly.
- Manual-fix is a contract-driven runtime protocol: preview, legality, and invalidation all flow from `stage_contracts.py`.
- Old scientist/planning/reflection experimental runtime paths have been removed from the main package surface.

## What Stayed The Same
- Native physics tools for relax, SCF, band, effective mass, strain, and mobility remain the execution backbone.
- Existing artifact names are preserved where practical.
- The runtime now has a single decision philosophy: `LLM-required`.
- `PLANNER_MODE` is no longer a supported runtime switch.
- `LLM_ENABLED` is no longer a supported runtime switch.
- If LLM provider/model credentials are missing, startup fails fast before any material run begins.
- Single-material execution remains the canonical runtime truth.

## Old To New Mapping
- Old single-material orchestration in `mobality.py`
  -> new thin CLI over `mobility_agent/runtime/runner.py`
- Old batch shell-out and separate workflow ownership
  -> new direct batch reuse of `run_single_material`
- Old ad hoc human review polling
  -> `escalate_human` interrupt dispatch inside `execute_selected_action` plus `mobility_agent/hitl/`
- Old loose state dictionaries
  -> `mobility_agent/graph/state.py`
- Old subprocess-era config fields such as `MOBALITY_SCRIPT`
  -> deprecated aliases that are normalized and recorded as warnings, not active runtime configuration

## Extension Guidance
- Add new deterministic computation stages by first extending `mobility_agent/graph/stage_contracts.py`, then runtime nodes, then graph routing.
- Add new decision logic by extending typed agent schemas and stage-scoped skills, not by injecting free-form chat into tool execution.
- If future experimental orchestration layers are explored, keep them out of the default runtime surface unless they become part of the deliverable architecture.
