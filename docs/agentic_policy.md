# Agentic Policy Layer

This project now includes a first-pass agentic policy layer for VASP-writing stages.

The design goal is:

- keep the multi-agent LangGraph structure unchanged
- keep the deterministic physics execution backbone unchanged
- move more of the parameter choice and failure diagnosis into retrieval-backed LLM decisions

## What It Currently Controls

In v1, the agentic policy layer only affects stages that write VASP inputs:

- `relax`
- `scf`
- `band`
- VASP-writing substages inside `strain_loop`

It can make bounded decisions about:

- selected `INCAR` overrides
- selected `KPOINTS` policy choices
- failure diagnosis evidence and bounded recovery suggestions

It does not change:

- stage order
- downstream invalidation rules
- mobility/effective-mass/validation physics logic
- arbitrary shell execution

## Evidence Sources

The policy layer can retrieve from two corpora:

1. bundled `house_policy`
   - distilled from the current project defaults and recovery priors
2. synced `vasp_wiki`
   - cleaned MediaWiki pages stored in Postgres and indexed with pgvector

The bundled corpus lives at:

```text
mobility_agent/policy/corpus/house_policy.json
```

## Environment Knobs

Set these in `.env.local` when you want the policy layer active:

```dotenv
MOBILITY_DB_URI=postgresql://postgres:postgres@127.0.0.1:5432/mobility_agent
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_BASE_URL=https://your-openai-compatible-host/v1
EMBEDDING_API_KEY=__REAL_KEY__
WIKI_QA_MODEL=__OPTIONAL_QA_MODEL__
AGENTIC_POLICY_ENABLED=true
POLICY_ALLOWLIST_MODE=restricted
POLICY_RETRIEVAL_TOP_K=5
POLICY_TRACE_ENABLED=true
RAG_TOP_K=6
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=180
RAG_REINDEX_BATCH_SIZE=64
```

Notes:

- `POLICY_ALLOWLIST_MODE=restricted` is the only supported mode in v1.
- If evidence is weak or the LLM response is unusable, the runtime falls back to the existing deterministic templates.

## Sync And Index The VASP Wiki Corpus

Use the included helper:

```bash
python scripts/sync_vasp_wiki.py --mode full
python scripts/reindex_vasp_wiki.py
```

That default command fetches a curated set of pages useful for:

- `INCAR`
- `KPOINTS`
- common relaxation controls
- common SCF controls
- common band-structure controls
- parallel/runtime hints

If you want a broader corpus:

```bash
python scripts/sync_vasp_wiki.py --mode full --all-pages --max-pages 300
python scripts/reindex_vasp_wiki.py
```

The web console `Wiki` page can trigger the same reindex flow through the control plane.

## New Trace Artifacts

When the policy layer is active, runs can emit additional artifacts:

- `retrieval_trace.json`
- `parameter_plan.json`
- `recovery_diagnosis.json`

`retrieval_trace.json` now records chunk-level citations, metadata filters, similarity scores, and source links from the Postgres/pgvector retriever.

## Current Safety Boundary

This layer is intentionally not fully free-form.

The model is still constrained to:

- a stage-scoped parameter allowlist
- bounded recovery actions
- deterministic renderers and deterministic stage tools

That keeps the benchmark focused on decision quality, not on who is most willing to generate risky commands.
