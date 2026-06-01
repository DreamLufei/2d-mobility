# Architecture Overview

## Goal
The repository centers on a shared-memory LangGraph runtime for 2D mobility calculation.

## Runtime Layers
1. `mobility_agent/graph/`
   - shared state schema
   - stage boundary contracts
   - single-material graph
   - runtime nodes
2. `mobility_agent/tools/`
   - deterministic VASP-native execution primitives
   - normalized tool result models
   - raw execution evidence capture
3. `mobility_agent/agents/`
   - admission, recovery, refinement, validation, reporting, batch supervision
   - structured typed outputs only
4. `mobility_agent/skills/` and top-level `skills/`
   - disk-backed skill registry and prompt context assets
5. `mobility_agent/hitl/`
   - escalation payloads
   - timeout and resume helpers
   - manual-fix protocol
6. `mobility_agent/memory/`
   - LangGraph `PostgresStore` helpers
7. `mobility_agent/runtime/`
   - single-material runner
   - batch runner
   - Mongo/POTCAR/structure adapters
   - Postgres-backed LangGraph checkpointing
   - compatibility snapshot exports
8. `mobility_agent/rag/`
   - VASP Wiki sync and cleaning
   - house-policy document normalization
   - pgvector indexing and retrieval

## Execution Story
- Single-material execution always enters through `run_single_material`.
- Batch execution always enters through `run_mongo_batch`.
- Batch mode writes inputs per material, then invokes the exact same single-material runner and aggregates the returned `MaterialRunOutcome`.
- Batch orchestration is implemented through a LangGraph Functional API entrypoint in `runtime/entrypoints.py`.
- Each material run gets a persistent thread id in the form `material::{task_id}::{material_id}::{run_id}` and stores it under `.runtime/thread_id.txt`.
- `runtime/runner.py` owns graph compile/invoke/stream, interrupt dispatch, checkpointer/store wiring, and canonical outcome return.

## Checkpointing
- LangGraph persistence uses `PostgresSaver` against `MOBILITY_DB_URI`.
- LangGraph long-term memory uses `PostgresStore`.
- Restore and continuation always consult LangGraph Postgres state first.
- `.runtime/shared_state.json` is a debug/export snapshot, not a recovery input.
- `.runtime/langgraph_checkpoint.json` stores redacted thread/backend metadata so workdirs stay portable while recovery truth stays in Postgres.
- Compatibility `checkpoint.pkl` exports remain available for legacy consumers, but they are not part of the canonical restore path.

## HITL
- The graph emits a LangGraph `interrupt(...)` from `execute_selected_action` when the selected action family is `escalate_human`.
- The runner consumes normalized HITL decision objects only.
- `mobility_agent/hitl/` handles prompt/wait/timeout behavior, manual-fix interaction, and resume payload construction outside the graph.
- Manual-fix recovery creates a typed preview schema with:
  - `modified_files`
  - `requested_resume_strategy`
  - `computed_resume_stage`
  - `cleanup_policy`
  - `invalidated_stages`
  - `invalidated_artifacts`
  - `warnings`

## Stage Contracts
- `mobility_agent/graph/stage_contracts.py` is the single source of truth for:
  - stage dependencies
  - canonical outputs
  - failure outputs
  - resume validation
  - cleanup policy interpretation
  - downstream invalidation boundaries
  - affected artifact discovery
- Manual-fix helpers and cleanup helpers only consume that contract; they do not maintain parallel invalidation logic.
