# Cluster Migration Guide

This guide is the portable path for moving `script_new` onto a new cluster.

## 1. What Is Actually Cluster-Specific

Most of the project is already portable.
The parts that usually change from cluster to cluster are:

- Python environment
- Node/Vite frontend build environment
- LLM endpoint and API key
- VASP launch command
- MPI and compiler runtime environment
- optional Mongo connection settings
- optional email/SMTP settings
- optional web-console bind address and job-root scan paths

The goal is to keep all of those differences in configuration, not in source edits.

## 2. Copy The Repository

```bash
git clone <your-private-repo-url>
cd script_new
```

If you are moving from an old machine instead of cloning from GitHub, copy the repository directory as-is, but do not copy:

- `.venv/`
- `.closure_venv/`
- `.web_runtime/`
- `.env.local`
- runtime output directories such as `mobility_calculation/`

## 3. Bootstrap The New Cluster

Use the bootstrap helper:

```bash
./scripts/bootstrap_cluster.sh
```

This script:

- creates `.venv` if needed
- upgrades `pip`
- installs the project from `pyproject.toml`
- installs frontend dependencies with `npm`
- builds the production frontend bundle

If the new cluster does not have `npm`, the backend can still run, but the production web UI will not be available until you install Node.js and build `web_console/frontend`.

## 4. Create Your Local Runtime Config

Create a machine-local config file:

```bash
cp .env.example .env.local
```

Then edit `.env.local`.

At minimum, set:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://your-llm-endpoint/v1
LLM_API_KEY=__REAL_KEY__
LLM_MODEL=__REAL_MODEL__
MOBILITY_DB_URI=postgresql://postgres:postgres@db-host:5432/mobility_agent
EMBEDDING_MODEL=__REAL_EMBEDDING_MODEL__
VASP_CMD=mpirun -np 4 vasp_std > sout 2>&1
```

For Qwen on the DashScope OpenAI-compatible endpoint, use:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=__REAL_DASHSCOPE_KEY__
LLM_MODEL=qwen3.6-plus
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_API_KEY=__REAL_DASHSCOPE_KEY__
EMBEDDING_MODEL=text-embedding-v4
```

The runtime's structured agent calls now use JSON mode globally. In practice this means LangChain is steered with `method=\"json_mode\"`, which aligns with OpenAI-compatible JSON-object guidance instead of depending on OpenAI `json_schema` semantics.

For loop3, keep chat, decision, and embeddings on DashScope. Keep OpenRouter credentials as a manual fallback route rather than mixing providers during one run.

Optional but common cluster-specific values:

```dotenv
OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0
HITL_POLICY=interactive
HUMAN_REVIEW_TIMEOUT_SECONDS=300
HUMAN_REVIEW_DEFAULT_ACTION=skip_material
```

Optional agentic-policy settings:

```dotenv
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
EMBEDDING_API_KEY=__REAL_EMBEDDING_KEY__
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

Optional email escalation settings:

```dotenv
ENABLE_EMAIL_NOTIFICATIONS=true
EMAIL_NOTIFY_TO=you@example.com
EMAIL_DRY_RUN=false
SMTP_HOST=smtp.qq.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=your_mailbox@example.com
SMTP_FROM=your_mailbox@example.com
SMTP_PASSWORD=your_smtp_password
```

Optional web-console helper defaults:

```dotenv
WEB_CONSOLE_HOST=127.0.0.1
WEB_CONSOLE_PORT=8765
WEB_CONSOLE_JOB_ROOTS=/absolute/path/to/runs
```

## 4A. Optional: Sync And Index The VASP Wiki Corpus

If you want retrieval-backed parameter planning, recovery, and the web-console Wiki page to use VASP Wiki evidence on the new cluster, sync documents into Postgres and build the pgvector index:

```bash
source .venv/bin/activate
python scripts/sync_vasp_wiki.py --mode full
python scripts/reindex_vasp_wiki.py
```

For a broader import:

```bash
python scripts/sync_vasp_wiki.py --mode full --all-pages --max-pages 300
python scripts/reindex_vasp_wiki.py
```

If you prefer not to run CLI helpers, the web console also exposes `POST /api/wiki/reindex` and a dedicated `Wiki` page that triggers the same control-plane job.

## 5. Verify The CLI First

Before bringing up the web console, verify the runtime on one material from the command line.

Dry-run sanity check:

```bash
source .venv/bin/activate
python mobality.py --root-path /absolute/path/to/material --dry-run --fresh --json
```

Real run:

```bash
python mobality.py --root-path /absolute/path/to/material --fresh --json
```

Batch mode:

```bash
python run_mongo_batch.py --dry-run --fresh-materials --json
```

## 6. Bring Up The Web Console

Temporary foreground-style management through helper scripts:

```bash
./scripts/start_web_console.sh
./scripts/status_web_console.sh
./scripts/stop_web_console.sh
```

The shell helper defaults to:

- host `127.0.0.1`
- port `8765`
- job root scan set containing the repository root

If you want extra scan roots, set `WEB_CONSOLE_JOB_ROOTS` as a colon-separated list.

The runtime settings page now manages `MOBILITY_DB_URI`, embedding credentials, and RAG chunking knobs directly, and the `Wiki` page lets you query indexed VASP pages with citations.

Example:

```bash
export WEB_CONSOLE_JOB_ROOTS=/data/mobility_runs:/data/batch_runs
./scripts/start_web_console.sh
```

## 7. Optional User-Systemd Service

If you want a reusable user service, install one:

```bash
./scripts/install_web_console_service.sh
```

This writes a unit into:

```text
~/.config/systemd/user/script-new-web-console.service
```

By default, the install helper only writes the unit file.
It does not automatically enable persistent auto-start.

Useful lifecycle commands:

```bash
systemctl --user start script-new-web-console.service
systemctl --user stop script-new-web-console.service
systemctl --user disable script-new-web-console.service
systemctl --user status script-new-web-console.service
```

If you enable the unit and also enable user lingering, the port can stay up after logout.
That is convenient, but it also increases exposure risk.

## 8. Security Recommendation

The web console currently has no built-in authentication.

Recommended pattern:

- bind to `127.0.0.1`
- reach it through SSH tunneling

Example:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<cluster-host>
```

Then open:

```text
http://127.0.0.1:8765/
```

Only use `WEB_CONSOLE_HOST=0.0.0.0` when you intentionally want network access and already have firewall or proxy protections in place.

## 9. GitHub Publishing Flow

Recommended sequence:

1. Initialize git locally if this directory is not already a repository.
2. Commit the migration-friendly baseline.
3. Create or choose a private GitHub repository.
4. Add that repository as `origin`.
5. Push the branch.

Example:

```bash
git init
git add .
git commit -m "Initial import of script_new"
git remote add origin <your-private-repo-url>
git push -u origin main
```

Secrets should stay out of git:

- `.env`
- `.env.local`
- `.web_runtime/`
- virtual environments
- runtime outputs

Those paths are already covered by `.gitignore`.
