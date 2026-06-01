# Reproducibility guide

This repository contains the deterministic first-principles mobility runtime
used by InvDesMobility. It executes VASP-based deformation-potential mobility
calculations, wraps them in a bounded multi-agent control layer, and records
stage-level evidence for validation and recovery.

## Scope

The code here can reproduce the mobility-evaluation part of the study:

- relaxation, SCF and band-structure stages;
- effective-mass fitting at band edges;
- strain-loop total-energy and band-edge shifts;
- deformation-potential mobility calculations;
- channel-level reliability labels used as feedback.

The numerical mobility values are produced by deterministic VASP calculations
and deterministic Python post-processing. LLM calls are used for planning,
validation, bounded recovery and reporting decisions; they do not generate
mobility labels.

## External requirements

Required software:

- Python 3.11 or newer;
- PostgreSQL with pgvector if checkpoint/store persistence and RAG are used;
- MongoDB for batch mode;
- a licensed VASP installation available on the target machine;
- VASP pseudopotentials prepared locally as `POTCAR` files;
- an OpenAI-compatible LLM endpoint for the agentic policy layer.

VASP itself and `POTCAR` files are not distributed in this repository.

## Install

```bash
git clone https://github.com/DreamLufei/2d-mobility.git
cd 2d-mobility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

For a quick non-VASP smoke test:

```bash
python mobality.py --list-skills --json
python mobality.py --dry-run --json --hitl-policy non_interactive_skip_on_timeout
python run_mongo_batch.py --dry-run --json
```

## Runtime configuration

Create a local `.env.local` file or export variables in the shell. Do not commit
local credentials.

Minimum variables for a real single-material or batch run:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://your-provider.example/v1
export LLM_API_KEY=...
export LLM_MODEL=...
export MOBILITY_DB_URI=postgresql://user:password@host:5432/mobility_agent
export MONGO_URI=mongodb://host:27017
export VASP_COMMAND="srun vasp_std"
```

The exact VASP launch command is site-specific. Use the command and scheduler
wrapper approved for the local cluster.

## Single-material reproduction

Prepare a material directory:

```text
example_material/
  POSCAR
  POTCAR
```

Run:

```bash
cd 2d-mobility
python mobality.py \
  --root-path /absolute/path/to/example_material \
  --material-id example_material \
  --fresh \
  --json
```

The default output directory is:

```text
/absolute/path/to/example_material/mobility_calculation
```

Important outputs include stage directories, `checkpoint.pkl` compatibility
exports when enabled, channel mobility reports, validation evidence and the
final JSON outcome printed by the CLI.

## Batch reproduction

Batch mode reads candidates from MongoDB and reuses the same single-material
runtime for every material:

```bash
python run_mongo_batch.py --fresh-materials --json
```

Dry-run mode verifies orchestration without launching VASP:

```bash
python run_mongo_batch.py --dry-run --fresh-materials --json
```

## Public-release policy

This repository intentionally excludes local credentials, local database state,
VASP raw outputs and licensed VASP input files. If a manuscript claim requires
raw VASP outputs, deposit them in a separate archival record with access terms
compatible with the VASP license and cite that record from the manuscript Data
Availability statement.

Recommended manuscript Code Availability wording:

```text
The first-principles mobility workflow code is available at
https://github.com/DreamLufei/2d-mobility. VASP is proprietary software and is
not distributed with the code.
```
