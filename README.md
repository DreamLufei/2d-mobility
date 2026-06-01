# 2D Mobility LangGraph Runtime

This repository provides a LangGraph-based runtime for first-principles 2D carrier mobility calculation.

For manuscript and archival reproduction, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). That document describes the public
release scope, external VASP requirements, dry-run checks, single-material
execution and batch execution.

The delivered system is centered on:
- one canonical single-material runtime
- batch reuse of that same runtime
- a typed shared state model
- LangGraph `StateGraph`, `PostgresSaver`, `PostgresStore`, and `interrupt`/`Command(resume=...)`
- deterministic VASP-native tools for the physics backbone
- LLM-driven multi-agent deliberation for planning, critique, recovery, orchestration, and reporting
- retrieval-backed agentic parameter policy plus VASP Wiki RAG for VASP-writing stages
- Anthropic-style disk-backed skill packages with on-demand loading and auditable skill traces
- human-in-the-loop escalation and manual-fix resume

## Project Purpose
- Preserve the deterministic VASP-native workflow.
- Make the single-material runtime the only execution truth.
- Reuse that runtime in batch/database screening.
- Support contract-driven recovery, HITL, timeout fallback, and manual-fix resume.
- Require a configured LLM provider before startup; this runtime does not support rule-only fallback.
- Keep batch/runtime/CLI behavior clear enough to install, test, and operate in a fresh environment.

## Current Architecture
- `mobility_agent/graph/`
  - shared state and stage contracts
  - single-material graph
  - runtime nodes
  - no compatibility batch subgraph in the runtime path
- `mobility_agent/tools/`
  - deterministic execution primitives
  - typed tool results
  - raw error evidence capture
- `mobility_agent/agents/`
  - admission
  - recovery
  - refinement
  - validation
  - reporting
  - batch supervision
- `mobility_agent/hitl/`
  - escalation payloads
  - timeout handling
  - manual-fix protocol
  - resume command helpers
- `mobility_agent/memory/`
  - LangGraph store helpers for recovery cases, validation heuristics, batch statistics, and skill registry
- `mobility_agent/runtime/`
  - canonical single-material runner
  - batch runner
  - Postgres-backed checkpoint/store helpers
  - Mongo/POTCAR/structure adapters
- `mobility_agent/policy/`
  - retrieval-backed parameter planning
  - retrieval-backed failure diagnosis
  - stage probe assembly
- `mobility_agent/rag/`
  - MediaWiki sync
  - house-policy + VASP Wiki document loading
  - pgvector indexing
  - retrieval and QA chains

## Agentic Policy Layer
The runtime now includes a bounded agentic policy layer for VASP-writing stages.

What stays fixed:
- the LangGraph multi-agent structure
- deterministic stage execution
- stage ordering and physics/post-processing flow

What becomes more agentic:
- stage-scoped `INCAR` overrides
- stage-scoped `KPOINTS` policy
- failure diagnosis and bounded recovery suggestions
- retrieval-backed evidence using Postgres + pgvector for bundled house-policy plus synced VASP Wiki pages

Useful environment knobs:

```bash
export MOBILITY_DB_URI=postgresql://postgres:postgres@127.0.0.1:5432/mobility_agent
export EMBEDDING_MODEL=text-embedding-3-large
export EMBEDDING_BASE_URL=https://your-openai-compatible-host/v1
export EMBEDDING_API_KEY=...
export WIKI_QA_MODEL=...
export AGENTIC_POLICY_ENABLED=true
export POLICY_ALLOWLIST_MODE=restricted
export POLICY_RETRIEVAL_TOP_K=5
export POLICY_TRACE_ENABLED=true
export RAG_TOP_K=6
```

Single-switch full-autonomy profile:

```bash
export MOBILITY_PROFILE=full_autonomy
python mobality.py --fresh --json
```

The `full_autonomy` profile auto-applies runtime defaults unless you explicitly override them:
- `FULL_AUTONOMY=true`
- `ALLOW_EXTERNAL_WAIT=false`
- `RAG_REQUIRED=true`
- `AGENTIC_POLICY_ENABLED=true`
- `ENABLE_HUMAN_REVIEW=true`
- `HUMAN_REVIEW_TIMEOUT_SECONDS=300`
- `HITL_POLICY=interactive`

Shortcut launcher:

```bash
./scripts/run_full_autonomy.sh --fresh --json
```

To sync and index the VASP Wiki corpus:

```bash
python scripts/sync_vasp_wiki.py --mode full
python scripts/reindex_vasp_wiki.py
```

Additional details live in:
- `docs/agentic_policy.md`

Top-level project directories are:
- `docs/`
- `mobility_agent/`
- `skills/`
- `tests/`

## Anthropic-Style Skills On LangGraph
This repository does not claim that LangGraph core has native Anthropic `Agent Skills`.
Instead, it implements an Anthropic-style skill layer on top of the LangGraph runtime:
- each skill package lives under `skills/<skill_name>/`
- each package is defined by `SKILL.md` frontmatter plus optional nested resources
- agents receive summary-first skill context and can lazily load more detail
- runtime writes `skill_trace.json` and related selection metadata for auditing

Useful runtime knobs:

```bash
export MOBILITY_SKILLS_ROOT=/absolute/path/to/skills
export SKILL_AUTO_RESOLVE_LIMIT=6
export SKILL_INLINE_BODY_LIMIT=2400
```

You can also override them per run:

```bash
python mobality.py --skills-root /absolute/path/to/skills
python mobality.py --skill-auto-resolve-limit 8 --skill-inline-body-limit 3200
python run_mongo_batch.py --skills-root /absolute/path/to/skills
python mobality.py --list-skills --json
python run_mongo_batch.py --list-skills
```

## Single-Material Mode
Prepare a folder with:
- `POSCAR`
- `POTCAR`

Then run:

```bash
python mobality.py
```

Useful examples:

```bash
python mobality.py --fresh
python mobality.py --dry-run --json
python mobality.py --dry-run --dry-run-fail-stages scf --hitl-policy non_interactive_skip_on_timeout
python mobality.py --dry-run --dry-run-fail-stages scf --hitl-policy non_interactive_abort_on_timeout
python mobality.py --dry-run --skills-root /absolute/path/to/skills --json
```

Default workdir:

```text
<material-root>/mobility_calculation
```

## Batch Mode
Mongo batch mode directly reuses `run_single_material(...)`.

```bash
python run_mongo_batch.py
python run_mongo_batch.py --dry-run --fresh-materials
```

Both single-material and batch entrypoints are LLM-required. If `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, or the effective model selection is missing, startup fails fast with a configuration error.
Legacy toggles such as `PLANNER_MODE` and `LLM_ENABLED` are no longer accepted.

Batch summary:

```text
<RUNS_ROOT>/batch_summary_<BATCH_TAG>.json
```

## Web Console
The repository now includes a localhost-only control plane for launching, monitoring, and intervening in runtime jobs.

Backend:

```bash
python -m mobility_agent.web_console --host 127.0.0.1 --port 8765 --job-root /absolute/path/to/runs
```

Frontend development:

```bash
cd web_console/frontend
npm install
npm run dev
```

Frontend production build:

```bash
cd web_console/frontend
npm run build
```

Key behaviors:
- worker processes start in a dedicated process group and cancellation targets the full group
- `.runtime/ui_state.json` and `.runtime/ui_events.jsonl` provide stable structured UI data
- batch view exposes parent-child runs instead of a flat summary only
- settings now expose Postgres/RAG configuration instead of local corpus file paths
- a dedicated `Wiki` page supports query, citation preview, health checks, and reindex jobs
- `?presentation=1` enables screenshot-friendly presentation mode

## Human In The Loop
- Graph-side pause happens when arbitration selects `escalate_human`; `execute_selected_action` emits the LangGraph `interrupt(...)` payload for the runner to handle.
- Runner-side handling consumes normalized HITL decisions and resumes with `Command(resume=...)`.
- Supported runtime policies:
  - `interactive`
  - `non_interactive_skip_on_timeout`
  - `non_interactive_abort_on_timeout`
- Legacy aliases `non_interactive_wait` and `non_interactive_skip` are still accepted and normalized with deprecation warnings.

## Manual-Fix Recovery
Manual-fix preview always shows:
- `modified_files`
- `requested_resume_strategy`
- `computed_resume_stage`
- `cleanup_policy`
- `invalidated_stages`
- `invalidated_artifacts`
- `warnings`

Default business rules remain:
- `INCAR -> current_stage + retry_current_stage_only`
- `KPOINTS -> scf + invalidate_downstream`
- `POSCAR -> relax + restart_from_stage`
- `multiple -> explicit confirmation + conservative restart`
- `custom -> validated custom stage`

`mobility_agent/graph/stage_contracts.py` is the single source of truth for:
- resume legality
- downstream invalidation
- artifact invalidation
- cleanup-policy consequences

## Checkpointing And Store
- LangGraph Postgres checkpointing is the only recovery truth.
- `.runtime/shared_state.json` is a readable export only.
- `checkpoint.pkl` is a compatibility export only.
- `.runtime/skill_trace.json` captures selected and loaded skills for the run.
- LangGraph `PostgresStore` is used for reusable long-term memory categories such as:
  - `recovery_cases`
  - `validation_heuristics`
  - `batch_statistics`
  - `skill_registry`

## Installation
`pyproject.toml` is the canonical dependency source. `requirements.txt` is a pinned export for cluster/bootstrap use.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Optional dev/test extras
pip install -e ".[dev]"
```

Then configure a real OpenAI-compatible endpoint (used through LangChain OpenAI adapters):

```bash
export LLM_PROVIDER=openai
export LLM_BASE_URL=https://your-openai-compatible-host/v1
export LLM_API_KEY=...
export LLM_MODEL=...
export LLM_USE_RESPONSES_API=false
export LLM_REASONING_EFFORT=
export MOBILITY_DB_URI=postgresql://postgres:postgres@127.0.0.1:5432/mobility_agent
export EMBEDDING_MODEL=text-embedding-3-large
```

Zhipu GLM also works through the same OpenAI-compatible path:

```bash
export LLM_PROVIDER=openai
export LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
export LLM_API_KEY=...
export LLM_MODEL=glm-5.1
export LLM_USE_RESPONSES_API=false
```

Structured agent outputs are now forced through LangChain `json_mode`, which maps to the OpenAI-compatible JSON-object response path that Zhipu documents as `response_format={"type":"json_object"}`. This applies globally to the runtime, including OpenRouter-backed chat models, so the runtime no longer relies on LangChain's default `json_schema` path for agent deliberation.

For OpenAI Responses-compatible gateways, you can opt in explicitly:

```bash
export LLM_PROVIDER=openai
export LLM_BASE_URL=http://your-openai-compatible-gateway
export LLM_API_KEY=...
export LLM_MODEL=gpt-5.4
export LLM_USE_RESPONSES_API=true
export LLM_REASONING_EFFORT=xhigh
```

If you already have a working embedding endpoint, you can keep `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, and `EMBEDDING_API_KEY` on that existing provider while switching only the chat/decision model to Zhipu.

`bson` does not need a separate package in this project; it is provided by `pymongo`.

## New Cluster Quickstart
For a fresh cluster migration, the shortest path is:

```bash
git clone <your-private-repo-url>
cd script_new
./scripts/bootstrap_cluster.sh
cp .env.example .env.local
```

Then edit `.env.local` with:
- your LLM endpoint and API key
- your cluster-specific `VASP_CMD`
- optional HITL/email settings
- optional agentic-policy settings and local VASP Wiki corpus path

For Zhipu GLM, the minimum LLM block is:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-5.1
```

When using Zhipu GLM, keep the agent prompts in plain JSON-object mode. The runtime already does this internally for planner/judge/orchestrator/reporter structured calls, so no extra `.env` toggle is required.

Recommended first verification:

```bash
source .venv/bin/activate
python mobality.py --root-path /absolute/path/to/material --dry-run --fresh --json
```

Then run a real material:

```bash
python mobality.py --root-path /absolute/path/to/material --fresh --json
```

Portable deployment details live in:
- `docs/cluster_migration_guide.md`
- `docs/web_console.md`

## Web Console Security And Lifecycle
The web console has no built-in authentication, so the safest default is to bind it to `127.0.0.1` and access it through SSH tunneling when needed.

Temporary start/stop:

```bash
./scripts/start_web_console.sh
./scripts/stop_web_console.sh
./scripts/status_web_console.sh
```

Optional user-systemd install:

```bash
./scripts/install_web_console_service.sh
```

By default, the helper writes a unit but does not enable persistent auto-start. If you want the service to survive logout/reboot, enable it explicitly after reviewing the exposure risk.

## Dry-Run Mode
Dry-run mode exercises:
- graph routing
- state updates
- recovery/HITL behavior
- batch aggregation

Examples:

```bash
python mobality.py --dry-run
python mobality.py --dry-run --dry-run-fail-stages scf --hitl-policy non_interactive_skip_on_timeout
python run_mongo_batch.py --dry-run
```

## More Docs
- `docs/architecture_overview.md`
- `docs/migration_notes.md`
- `docs/cluster_migration_guide.md`
- `docs/human_in_the_loop.md`
- `docs/skills.md`
- `docs/batch_mode.md`
- `docs/web_console.md`
