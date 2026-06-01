# Web Console Detailed User Guide

This document includes examples from one deployed server.
For portable new-cluster setup, use `docs/cluster_migration_guide.md` first.

This document explains what the `script_new` web console is, how to open it on the current server, how to use each page, and what happens behind the scenes when you click a button.

## 1. What This Web Console Actually Is

This web console is a control plane layered on top of the canonical mobility runtime.

It is not a second workflow engine.

That means:

- When you launch a single material from the browser, it still runs the same canonical `run_single_material(...)` backend path.
- When you launch a batch job from the browser, it still runs the same canonical `run_mongo_batch(...)` backend path.
- The browser is mainly responsible for starting jobs, showing live state, showing logs, showing artifacts, and allowing human intervention.

The core logic is:

- Frontend page -> calls FastAPI backend
- FastAPI backend -> creates a control-plane job record in SQLite
- Backend -> starts a worker process
- Worker -> calls the real mobility runtime
- Runtime -> writes `.runtime/ui_state.json`, `.runtime/ui_events.jsonl`, `material_outcome.json`, logs, artifacts
- Web console backend -> continuously rescans those files and updates the page

## 2. Current Deployment On This Server

The current web console is already deployed and running on this machine.

Access URLs:

- Local on server: `http://127.0.0.1:8765/`
- LAN access: `http://10.20.5.51:8765/`

The current service is managed by a user-level systemd unit:

- `/home/wkli/.config/systemd/user/script-new-web-console.service`

Helper scripts:

- `/home/wkli/agent/script_new/scripts/start_web_console.sh`
- `/home/wkli/agent/script_new/scripts/stop_web_console.sh`
- `/home/wkli/agent/script_new/scripts/status_web_console.sh`

Log file:

- `/home/wkli/agent/script_new/.web_runtime/web_console.log`

Management commands:

```bash
/home/wkli/agent/script_new/scripts/status_web_console.sh
/home/wkli/agent/script_new/scripts/stop_web_console.sh
/home/wkli/agent/script_new/scripts/start_web_console.sh
```

Realtime service log:

```bash
journalctl --user -u script-new-web-console.service -f
```

## 3. Before You Use It

The web console can show old runs even if you do nothing, because it imports historical run directories automatically.

On this machine, the service is scanning these roots:

- `/home/wkli/agent/script_new`
- `/home/wkli/agent/new_test_script`

So when you open the page, you may already see archived jobs that were not created from the current browser session.

For real new runs, these prerequisites matter:

- LLM config must exist in `.env` or `.env.local`
- Single-material run root must contain `POSCAR`
- Single-material run root should usually contain `POTCAR` too, unless you use a workflow that generates it elsewhere
- Batch mode needs Mongo connection info
- Real production run needs the VASP-side environment to be available on the server

Important default behavior:

- The Single Run form defaults to `Dry run = true`
- If you do not manually uncheck `Dry run`, it will not launch a real VASP production run

## 4. What You See When You Open The Website

There are three main page types:

- Home page
- Overview page
- Job detail page

### Home page

This is the landing page. It is mainly a summary page.

It shows:

- Backend health
- Registry health
- Active job count
- WebSocket client count
- Recent jobs

The Home page is not where you operate in detail. It is mainly the entry point.

### Overview page

This is the main operational page.

It contains:

- `Launch Single Material`
- `Launch Batch`
- `Runtime Snapshot`
- Job card list

This is the page you will use most often.

### Job detail page

When you click a job card, you enter the detail page for that one job.

For a single-material job, this page shows:

- Runtime status
- Control-plane status
- Current stage
- Root path
- Workdir
- Thread id
- Timeline
- Stage status
- Artifacts
- Runtime state JSON
- Runtime logs
- Control actions like cancel or HITL response

For a batch parent job, the detail page shows:

- Parent-child relationship
- Failure taxonomy
- List of child jobs

It does not directly expose all single-job controls on the batch parent itself. You usually click into the child job you want to inspect.

## 5. Single-Material Run: Exact Step-By-Step Usage

This is the most important workflow if you want to run one material from the browser.

### Step 1: Prepare your material directory

Your directory should look conceptually like this:

```text
/some/material_root/
  POSCAR
  POTCAR
```

Real example on this server:

```text
/home/wkli/agent/new_test_script/2dm-1610/
  POSCAR
  POTCAR
```

Important:

- In the web UI, `Root path` should be the material root directory
- It should not be the `mobility_calculation` directory

Why:

- The backend automatically turns `root_path` into `workdir = <root_path>/mobility_calculation`

So if you enter:

```text
/home/wkli/agent/new_test_script/2dm-1610
```

the runtime workdir becomes:

```text
/home/wkli/agent/new_test_script/2dm-1610/mobility_calculation
```

### Step 2: Open Overview

Open:

- `http://10.20.5.51:8765/`

Then click:

- `Overview`

### Step 3: Fill in the Single Run form

Fields in `Launch Single Material`:

- `Display name`
- `Root path`
- `Material ID`
- `Dry-run fail stages`
- `Dry run`
- `Fresh`

What each means:

- `Display name`: only affects what the card is called in the UI
- `Root path`: material folder containing `POSCAR` and usually `POTCAR`
- `Material ID`: optional human-readable identifier; if left blank, the folder name is used
- `Dry-run fail stages`: used only in dry-run mode to simulate failures, for example `scf,band`
- `Dry run`: if checked, no real VASP production launch is performed
- `Fresh`: if checked, ignore saved checkpoints and restart from scratch

Recommended first real run:

- `Display name`: optional
- `Root path`: your material directory
- `Material ID`: optional
- `Dry-run fail stages`: leave empty
- `Dry run`: uncheck it if you want a real run
- `Fresh`: keep checked if you want a clean restart

### Step 4: Click `Start Single Run`

After you click the button:

- The backend creates a job id
- It writes a control-plane row to SQLite
- It spawns a worker process in a new process group
- The browser is redirected to that job's detail page

### Step 5: Watch the job detail page

You will see several areas.

#### Runtime / Control Plane / Current Stage

These top fields are the fastest way to understand what is happening.

- `Runtime` tells you what the scientific runtime thinks is happening
- `Control Plane` tells you what the web manager thinks is happening
- `Current Stage` tells you which step of the mobility workflow is active

Typical stage order:

- `prepare`
- `relax`
- `scf`
- `band`
- `effective_mass`
- `strain_loop`
- `mobility`
- `validation`
- `final_report`

#### Timeline

This is built from:

- `.runtime/ui_events.jsonl`

It shows structured event history rather than only plain text logs.

Use this when you want a concise event story.

#### Stage Status

This usually reflects the current known status of each scientific stage.

Typical values include:

- `success`
- `failed`
- possibly still missing for stages not yet reached

Use this when you want to quickly answer:

- Did `band` succeed yet?
- Did `scf` fail?
- Has validation finished?

#### Runtime Logs

This is the human-readable progress stream, usually from:

- `.runtime/runtime_progress.log`

Use this when the job looks stuck or you want the latest textual clue.

#### Artifacts

This section gives downloadable outputs collected from the workdir.

Typical important artifacts include:

- `mobility_results.json`
- `fit_diagnostics.json`
- `validation_report.json`
- `final_summary.json`
- `material_outcome.json`
- `human_escalation_payload.json`
- `human_escalation_response.json`
- `human_escalation_log.json`

### Step 6: Understand the result

When a run finishes normally, the most important result files are usually:

- `material_outcome.json`
- `final_summary.json`
- `mobility_results.json`
- `validation_report.json`

If you only want the final answer, start with:

- `material_outcome.json`

If you want the mobility data itself, inspect:

- `mobility_results.json`

If you want to know whether the fit and validation were acceptable, inspect:

- `validation_report.json`
- `fit_diagnostics.json`

## 6. Single-Material Run: What The Buttons Actually Do

The single-job detail page exposes several operational controls.

### `Cancel Process Group`

This sends:

- `SIGTERM` first
- waits up to 10 seconds
- then `SIGKILL` if still alive

This is process-group based, so it tries to terminate the whole launched job tree, not just one PID.

After this:

- The control-plane status becomes `cancelled`

### `Submit HITL Response`

The current UI supports these actions:

- `retry_current_stage`
- `skip_material`
- `abort_task`

This writes a normalized HITL response file into the material workdir and lets the runtime resume.

Current limitation:

- The full `manual_fix_resume` path exists in the backend ecosystem
- But the present web UI does not expose a dedicated `manual_fix_resume` form yet

So if you need that specific advanced flow, you currently need a backend-side/manual route rather than the current page buttons.

### `Inject External Event`

This is an advanced feature.

It is for situations where a paused or externally coordinated flow should be resumed with an explicit event payload.

Current UI fields let you send:

- `event_type`
- `job_id`
- `target_capability`
- `action_family`
- `error_summary`

This is not part of the normal beginner workflow. Use it only if you know you are resuming a workflow using an external event.

## 7. Batch Run: Exact Step-By-Step Usage

Use batch mode when you want to pull materials from MongoDB and process multiple materials under one batch tag.

### What the batch form needs

Fields:

- `Display name`
- `Batch tag`
- `Runs root`
- `Mongo URI`
- `Mongo DB`
- `Mongo Collection`
- `Dry run`
- `Fresh materials`

What they mean:

- `Display name`: UI label for the batch parent job
- `Batch tag`: logical name for this batch; also affects the summary file name
- `Runs root`: directory where per-material run folders will be created
- `Mongo URI`: MongoDB connection string
- `Mongo DB`: database name
- `Mongo Collection`: collection name
- `Dry run`: if checked, perform dry-run logic instead of real production execution
- `Fresh materials`: if checked, each material run ignores its old checkpoint and restarts

Hardcoded batch defaults used by the current UI:

- `potcar_method = vaspkit`
- `vaspkit_cmd = vaspkit`
- `vaspkit_task = 103`
- `retry_failed = false`
- `running_stale_s = 43200`

These defaults are in the current frontend payload and are not currently editable from the page.

### Example batch workflow

1. Open `Overview`
2. Fill `Batch tag`, for example `demo_batch_001`
3. Set `Runs root`, for example `/home/wkli/agent/new_batch_runs`
4. Fill `Mongo URI`
5. Fill `Mongo DB`
6. Fill `Mongo Collection`
7. Choose `Dry run` or real mode
8. Click `Start Batch Run`

What happens next:

- The UI creates a batch parent job
- The backend spawns the canonical batch worker
- Child single-material runs appear underneath that parent
- The batch detail page shows children instead of pretending everything is a single flat job

Useful output:

- `batch_summary_<batch_tag>.json`

## 8. Status Fields: How To Read Them Correctly

This is one of the most important concepts in the console.

There are two status systems:

- `runtime_run_status`
- `control_plane_status`

### `runtime_run_status`

This represents what the scientific runtime believes.

Possible values include:

- `pending`
- `running`
- `waiting_external`
- `needs_human`
- `completed`
- `failed`
- `aborted`
- `skipped`

Interpretation:

- `completed`: workflow finished normally
- `failed`: workflow ended in failure
- `aborted`: runtime considered the run aborted
- `skipped`: runtime intentionally skipped the material
- `needs_human`: human input is needed
- `waiting_external`: an external event is required

### `control_plane_status`

This represents what the web manager knows about the worker process.

Possible values include:

- `queued`
- `starting`
- `live`
- `disconnected`
- `cancelled`
- `archived`

Interpretation:

- `starting`: worker was launched but runtime evidence is still limited
- `live`: worker process appears alive
- `disconnected`: control plane no longer sees the process alive, but runtime may not yet have a terminal artifact
- `cancelled`: web console cancelled the process group
- `archived`: terminal or imported historical record

Why this separation matters:

- A job can be `cancelled` from the control-plane point of view without the runtime claiming a normal scientific completion
- A job can be `aborted` scientifically while still being just an archived record in the web registry

## 9. What The Workflow Logic Is Under The Hood

The browser does not invent scientific decisions on its own.

The canonical graph is:

- `observe_state`
- `proposal_phase`
- `critique_phase`
- `arbitration_phase`
- `execute_selected_action`
- `reflect_round`
- `check_termination`
- `final_report`

Inside that process:

- The runtime observes current state
- Planner/recovery/refinement/executor-style roles propose what to do
- A critique step evaluates proposals
- Orchestrator arbitrates and selects the action
- The selected action is executed
- The system reflects on the round
- It decides whether to continue or terminate

For scientific execution, the action normally becomes one of the canonical stages:

- `prepare`
- `relax`
- `scf`
- `band`
- `effective_mass`
- `strain_loop`
- `mobility`
- `validation`
- `final_report`

This means the high-level logic is:

- LLM agents decide what should happen next
- The actual scientific stage execution is still grounded in the canonical runtime and stage contracts

## 10. How Historical Jobs Appear In The Page

The Overview page is not limited to jobs started from this browser.

The backend periodically scans configured roots and imports old runs if it finds:

- `.runtime`
- or `material_outcome.json`

So the page can show:

- jobs launched from the current UI
- jobs launched earlier from CLI
- jobs left behind by previous runs

This is why the Overview is both a launcher and a run browser.

## 11. Important Files Written By A Material Run

A typical material workdir contains these important files:

- `.runtime/shared_state.json`
- `.runtime/ui_state.json`
- `.runtime/ui_events.jsonl`
- `.runtime/runtime_progress.log`
- `material_outcome.json`
- `final_summary.json`
- `validation_report.json`
- `mobility_results.json`

What they mean:

- `shared_state.json`: large canonical runtime state
- `ui_state.json`: latest UI-facing structured snapshot
- `ui_events.jsonl`: append-only timeline used by the detail page
- `runtime_progress.log`: human-readable log stream
- `material_outcome.json`: terminal material-level outcome
- `final_summary.json`: concise final summary
- `validation_report.json`: validation result
- `mobility_results.json`: final mobility data

## 12. Recommended Beginner Usage Pattern

If you are using the console for the first time, use this exact sequence:

1. Open `http://10.20.5.51:8765/`
2. Click `Overview`
3. In `Launch Single Material`, set `Root path` to a known-good material folder
4. Keep `Fresh` checked
5. Leave `Dry run` checked for your very first smoke test
6. Click `Start Single Run`
7. Confirm the detail page updates and that Timeline, Stage Status, and Logs are changing
8. If the dry run behavior looks correct, go back and run again with `Dry run` unchecked for a real production run
9. When complete, download or inspect `material_outcome.json`, `mobility_results.json`, and `validation_report.json`

This gives you the safest first experience.

## 13. Common Mistakes

### Mistake 1: Putting the wrong `Root path`

Wrong:

- setting `Root path` to `.../mobility_calculation`

Right:

- set `Root path` to the material folder above it

### Mistake 2: Forgetting that `Dry run` is checked by default

If you expected a real VASP production run but nothing physical happened, check whether:

- `Dry run` was still enabled

### Mistake 3: Expecting batch mode to work without Mongo

Batch mode needs:

- `Mongo URI`
- `Mongo DB`
- `Mongo Collection`

### Mistake 4: Thinking the page only shows jobs launched from the browser

It also imports historical runs from scanned directories.

### Mistake 5: Confusing `cancelled` with `aborted`

These are not the same:

- `cancelled` is a control-plane action
- `aborted` is a runtime outcome

## 14. Troubleshooting

If the page does not open:

```bash
/home/wkli/agent/script_new/scripts/status_web_console.sh
```

If the service is down:

```bash
/home/wkli/agent/script_new/scripts/start_web_console.sh
```

If the page loads but does not update:

- refresh once
- then check logs:

```bash
journalctl --user -u script-new-web-console.service -f
```

If a launched run immediately fails:

- check whether the material folder really contains `POSCAR`
- check whether `POTCAR` exists if your flow expects it
- check runtime logs in the job detail page
- check `material_outcome.json`

If batch launch fails immediately:

- verify `Mongo URI`
- verify database and collection names
- verify the server can access MongoDB

## 15. Short Practical Summary

If you only remember one version, remember this:

Single-material usage:

1. Open `http://10.20.5.51:8765/`
2. Go to `Overview`
3. Fill `Root path` with the material folder
4. Uncheck `Dry run` if you want a real run
5. Click `Start Single Run`
6. Watch Timeline, Stage Status, Logs
7. Download `material_outcome.json` and `mobility_results.json`

Batch usage:

1. Go to `Overview`
2. Fill `Runs root`, `Mongo URI`, `Mongo DB`, `Mongo Collection`, `Batch tag`
3. Click `Start Batch Run`
4. Open the batch parent
5. Click child jobs to inspect details

That is the core workflow.
