# Web Console

The web console adds a localhost-only control plane on top of the canonical runtime.

## Goals
- launch single-material jobs from the browser without creating a second workflow engine
- launch batch jobs while preserving the same canonical per-material runner
- stream structured runtime state, logs, artifacts, HITL, and external-event recovery
- provide presentation-ready overview and detail pages for demos or paper figures

## Runtime Contract
Each material workdir now exports:
- `.runtime/shared_state.json`
- `.runtime/ui_state.json`
- `.runtime/ui_events.jsonl`

`ui_state.json` is the latest stable UI snapshot.
`ui_events.jsonl` is the append-only structured timeline used by the detail page.

HITL payloads and responses remain in:
- `human_escalation_payload.json`
- `human_escalation_response.json`
- `human_escalation_log.json`

The response file is now written through the normalized HITL schema helper with atomic write semantics.

## Control Plane
The backend lives under `mobility_agent/web_console/`.

Core pieces:
- `registry.py`: Postgres-backed control-plane registry using the shared runtime database
- `worker_main.py`: worker process wrapper around `run_single_material(...)` and `run_mongo_batch(...)`
- `service.py`: reconciliation, polling aggregation, process-group cancellation, REST detail assembly, WS broadcast, Wiki RAG health/query/reindex helpers
- `api.py`: FastAPI app and WebSocket endpoints

Cancellation is process-group based:
- `SIGTERM` first
- wait up to 10 seconds
- `SIGKILL` if still alive

## Frontend
The frontend lives in `web_console/frontend/`.

Pages:
- landing page
- overview page
- single-material detail page
- batch parent-child detail page
- Wiki RAG page with query, citations, and reindex controls

Presentation mode:
- add `?presentation=1`
- or use the in-app toggle
- PNG export uses a fixed screenshot region
- PDF export uses print styles

## Running
Backend:

```bash
python -m mobility_agent.web_console --host 127.0.0.1 --port 8765 --job-root /absolute/path/to/runs
```

Shell helper:

```bash
./scripts/start_web_console.sh
./scripts/status_web_console.sh
./scripts/stop_web_console.sh
```

User-systemd install helper:

```bash
./scripts/install_web_console_service.sh
```

Important safety note:
- the web console does not ship with authentication
- the safer default is `127.0.0.1`
- only bind `0.0.0.0` if you intentionally want remote access and already have SSH tunneling, a reverse proxy, a firewall rule, or another access-control layer in front of it

Frontend development:

```bash
cd web_console/frontend
npm install
npm run dev
```

Frontend build:

```bash
cd web_console/frontend
npm run build
```

## API Summary
- `POST /api/jobs/single`
- `POST /api/jobs/batch`
- `GET /api/wiki/health`
- `POST /api/wiki/query`
- `POST /api/wiki/reindex`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/state`
- `GET /api/jobs/{job_id}/timeline`
- `GET /api/jobs/{job_id}/logs`
- `GET /api/jobs/{job_id}/artifacts`
- `GET /api/jobs/{job_id}/download/{artifact_name}`
- `POST /api/jobs/{job_id}/cancel`
- `POST /api/jobs/{job_id}/hitl/respond`
- `POST /api/jobs/{job_id}/events/resume`
- `GET /api/health`
- `WS /ws/jobs`
- `WS /ws/jobs/{job_id}`
