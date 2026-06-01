from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback
from dataclasses import replace
from typing import Any

from ..rag import VaspWikiRagService
from ..runtime.batch_config import BatchConfig, load_config
from ..runtime.checkpointing import write_json_atomic
from ..runtime.context import RuntimeContext
from ..runtime.runner import default_material_workdir, run_single_material
from ..runtime.batch_runner import run_mongo_batch
from .models import BatchJobRequest, SingleJobRequest, WikiReindexRequest, WorkerJobSpec


_TERMINATION_SIGNAL: str | None = None


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        global _TERMINATION_SIGNAL
        try:
            _TERMINATION_SIGNAL = signal.Signals(signum).name
        except Exception:
            _TERMINATION_SIGNAL = str(signum)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except Exception:
            continue


def _result_path(job_id: str) -> str:
    results_dir = os.environ.get("MOBILITY_WEB_RESULTS_DIR") or os.path.join(os.getcwd(), ".web_runtime", "results")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{job_id}.json")


def _apply_runtime_overrides(runtime: RuntimeContext, overrides: dict[str, Any]) -> RuntimeContext:
    updates: dict[str, Any] = {}
    for key in (
        "dry_run",
        "dry_run_fail_stages",
        "hitl_policy",
        "compatibility_export_enabled",
        "compatibility_export_pickle",
        "checkpoint_subdir",
        "db_uri",
        "skills_root",
        "skill_auto_resolve_limit",
        "skill_inline_body_limit",
    ):
        if key in overrides and overrides[key] is not None:
            updates[key] = overrides[key]
    if "dry_run_fail_stages" in updates:
        updates["dry_run_fail_stages"] = tuple(str(item).strip() for item in list(updates["dry_run_fail_stages"] or []) if str(item).strip())
    if "skills_root" in updates:
        updates["skills_root"] = os.path.abspath(str(updates["skills_root"]))
    return replace(runtime, **updates) if updates else runtime


def _build_batch_config(request: BatchJobRequest) -> BatchConfig:
    if request.config is not None:
        payload = request.config.model_dump(mode="python", exclude_none=True)
        return BatchConfig(**payload)
    cfg = load_config()
    overrides = request.config_overrides.model_dump(mode="python", exclude_none=True)
    return replace(cfg, **overrides) if overrides else cfg


def _run_single(spec: WorkerJobSpec) -> tuple[int, dict[str, Any]]:
    request = SingleJobRequest.model_validate(spec.request)
    runtime = _apply_runtime_overrides(RuntimeContext.from_env(), request.runtime.model_dump(mode="python"))
    root_path = os.path.abspath(request.root_path)
    workdir = os.path.abspath(request.workdir or default_material_workdir(root_path))
    material_id = request.material_id or os.path.basename(root_path) or "2D_Material"
    outcome = run_single_material(
        runtime=runtime,
        material_id=material_id,
        root_path=root_path,
        workdir=workdir,
        poscar_path=request.poscar_path,
        potcar_path=request.potcar_path,
        user_goal=request.user_goal,
        fresh=bool(request.fresh),
    )
    payload = outcome.model_dump(mode="json")
    return (0 if outcome.status in {"completed", "skipped"} else 1), payload


def _run_batch(spec: WorkerJobSpec) -> tuple[int, dict[str, Any]]:
    request = BatchJobRequest.model_validate(spec.request)
    runtime = _apply_runtime_overrides(RuntimeContext.from_env(), request.runtime.model_dump(mode="python"))
    cfg = _build_batch_config(request)
    final_state = run_mongo_batch(
        cfg=cfg,
        runtime=runtime,
        thread_id=request.thread_id,
        fresh_materials=bool(request.fresh_materials),
    )
    payload = {
        "status": "completed",
        "runs_root": cfg.runs_root,
        "batch_tag": cfg.batch_tag,
        "final_state": final_state,
    }
    return 0, payload


def _run_wiki_reindex(spec: WorkerJobSpec) -> tuple[int, dict[str, Any]]:
    request = WikiReindexRequest.model_validate(spec.request)
    runtime = RuntimeContext.from_env()
    service = VaspWikiRagService.from_runtime(runtime)
    sync_stats = service.sync(
        mode=request.mode,
        include_all_pages=bool(request.include_all_pages),
        max_pages=request.max_pages,
        delay_seconds=max(0.0, float(request.delay_seconds or 0.0)),
    )
    reindex_stats = service.rebuild_index()
    return 0, {
        "status": "completed",
        "sync": sync_stats,
        "reindex": reindex_stats,
    }


def main(argv: list[str] | None = None) -> int:
    _install_signal_handlers()
    parser = argparse.ArgumentParser(description="Web console worker process")
    parser.add_argument("--job-spec", required=True, help="Path to a JSON worker job spec.")
    args = parser.parse_args(argv)

    spec = WorkerJobSpec.model_validate_json(open(args.job_spec, "r", encoding="utf-8").read())
    exit_code = 1
    result: dict[str, Any] = {
        "job_id": spec.job_id,
        "job_type": spec.job_type,
        "status": "failed",
    }
    try:
        if spec.job_type == "single_material":
            exit_code, payload = _run_single(spec)
        elif spec.job_type == "wiki_reindex":
            exit_code, payload = _run_wiki_reindex(spec)
        else:
            exit_code, payload = _run_batch(spec)
        result.update(payload)
        result["status"] = payload.get("status") or payload.get("final_status") or ("completed" if exit_code == 0 else "failed")
        return exit_code
    except SystemExit as exc:
        if _TERMINATION_SIGNAL:
            exit_code = int(getattr(exc, "code", 143) or 143)
            result.update(
                {
                    "status": "terminated",
                    "signal": _TERMINATION_SIGNAL,
                    "exit_code": exit_code,
                }
            )
            return exit_code
        raise
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return exit_code
    finally:
        result["exit_code"] = exit_code
        if _TERMINATION_SIGNAL and "signal" not in result:
            result["signal"] = _TERMINATION_SIGNAL
        write_json_atomic(_result_path(spec.job_id), result)


if __name__ == "__main__":
    sys.exit(main())
