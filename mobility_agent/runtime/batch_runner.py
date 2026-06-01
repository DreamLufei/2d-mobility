from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime
from typing import Any

from ..agents import BatchSupervisorAgent
from ..graph.state import (
    MaterialRunOutcome,
    derive_compute_status_from_outcome_payload,
)
from .agent_tools import AgentToolGateway
from .checkpointing import build_batch_thread_id, open_runtime_checkpointer, save_checkpoint_metadata, save_thread_id
from .batch_config import BatchConfig
from .entrypoints import build_batch_entrypoint
from .mongo_batch import claim_next_material, connect, mark_completed, mark_failed
from .potcar import build_potcar
from .store import open_memory_store, record_batch_statistics
from .structure_io import structure_from_mongo_doc, write_poscar
from .context import RuntimeContext
from .quality_label import classify_material_quality
from .runner import default_material_workdir, run_single_material
from .telemetry import active_workdir_scope


def _material_root(cfg: BatchConfig, material_id: str) -> str:
    return os.path.abspath(os.path.join(cfg.runs_root, material_id))


def _ensure_batch_runs_root(cfg: BatchConfig, runtime: RuntimeContext) -> BatchConfig:
    target = os.path.abspath(cfg.runs_root)
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write_probe")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok\n")
        os.remove(probe)
        return cfg
    except OSError:
        if not runtime.dry_run:
            raise
        fallback = os.path.abspath(os.path.join(tempfile.gettempdir(), "mobility_runtime_batch_runs", cfg.batch_tag))
        os.makedirs(fallback, exist_ok=True)
        return replace(cfg, runs_root=fallback)


def _coerce_mongo_doc_id(doc_id: Any) -> Any:
    try:
        from bson import ObjectId  # type: ignore

        if isinstance(doc_id, ObjectId):
            return doc_id
        if isinstance(doc_id, str) and ObjectId.is_valid(doc_id):
            return ObjectId(doc_id)
    except Exception:
        return doc_id
    return doc_id


def _msgpack_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        from bson import ObjectId  # type: ignore
        from bson.decimal128 import Decimal128  # type: ignore

        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, Decimal128):
            try:
                return float(obj.to_decimal())
            except Exception:
                return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _msgpack_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_msgpack_safe(item) for item in obj]
    return str(obj)


def _synthetic_outcome(
    *,
    material_id: str,
    workdir: str,
    status: str,
    termination_reason: str,
    error_message: str = "",
) -> dict[str, Any]:
    outcome = MaterialRunOutcome(
        task_id=f"synthetic::{material_id}",
        material_id=material_id,
        status=status,
        final_status=status,
        termination_reason=termination_reason,
        workdir=workdir,
        warnings=[],
        errors=([error_message] if error_message else []),
        final_summary={
            "material_id": material_id,
            "run_status": status,
            "termination_reason": termination_reason,
        },
    )
    return outcome.model_dump(mode="json")


def _failure_stage_from_outcome(outcome: dict[str, Any]) -> str:
    stage_status = dict(outcome.get("stage_status", {}) or {})
    for stage, status in stage_status.items():
        if status == "failed":
            return str(stage)
    reason = str(outcome.get("termination_reason") or "")
    if reason:
        return reason
    return "unknown"


def _normalize_outcome_for_persistence(outcome: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(outcome or {})
    normalized["final_status"] = str(
        normalized.get("final_status") or normalized.get("status") or "failed"
    )
    normalized["status"] = derive_compute_status_from_outcome_payload(normalized)
    final_summary = dict(normalized.get("final_summary", {}) or {})
    if final_summary and not str(final_summary.get("run_status") or "").strip():
        final_summary["run_status"] = str(normalized.get("final_status") or "")
        normalized["final_summary"] = final_summary
    return normalized


def _mark_skipped(
    collection: Any,
    *,
    doc_id: Any,
    outcome: dict[str, Any],
    runtime: RuntimeContext,
    round_index: Any = None,
    round_id: Any = None,
    pipeline_run_id: Any = None,
) -> None:
    validation = dict(outcome.get("validation_report", {}) or {})
    final_acceptance = str(outcome.get("final_acceptance") or validation.get("decision") or "").strip() or None
    quality_grade = str(validation.get("quality_grade") or "").strip() or None
    accepted_channels = list(outcome.get("accepted_channels", []) or [])
    rejected_channels = list(outcome.get("rejected_channels", []) or [])
    unset_fields = {
        "mobility_calc.results": "",
        "mobility_calc.quality_label": "",
        "mobility_calc.scientific_decision": "",
        "mobility_calc.quality_grade": "",
        "mobility_calc.accepted_channels": "",
        "mobility_calc.rejected_channels": "",
        "mobility_calc.channel_labels": "",
        "mobility_calc.failed_at": "",
        "mobility_agent.error": "",
        "mobility_agent.failed_at": "",
    }
    if final_acceptance is None:
        unset_fields["mobility_agent.final_acceptance"] = ""
    if quality_grade is None:
        unset_fields["mobility_agent.quality_grade"] = ""
    calc_round_fields: dict[str, Any] = {}
    try:
        if round_index is not None and str(round_index).strip() != "":
            calc_round_fields["mobility_calc.round_index"] = int(round_index)
    except Exception:
        pass
    if round_id is not None and str(round_id).strip():
        calc_round_fields["mobility_calc.round_id"] = str(round_id).strip()
    if pipeline_run_id is not None and str(pipeline_run_id).strip():
        calc_round_fields["mobility_calc.pipeline_run_id"] = str(pipeline_run_id).strip()

    collection.update_one(
        {"_id": doc_id},
        {
            "$set": {
                "mobility_calc.status": "skipped",
                "mobility_calc.completed_at": datetime.utcnow(),
                "mobility_calc.run_dir": outcome.get("workdir"),
                "mobility_calc.error": "; ".join(outcome.get("errors", []) or [])
                or str(outcome.get("termination_reason") or "skipped"),
                "mobility_agent.status": "skipped",
                "mobility_agent.final_status": str(outcome.get("final_status") or outcome.get("status") or "skipped"),
                "mobility_agent.completed_at": datetime.utcnow(),
                "mobility_agent.summary": outcome.get("final_summary", {}),
                "mobility_agent.validation": validation,
                "mobility_agent.confidence_score": outcome.get("confidence_score"),
                "mobility_agent.decision_engine": runtime.agent_runtime.decision_engine.value,
                "mobility_agent.llm_required": True,
                "mobility_agent.decision_trace_path": outcome.get("artifact_paths", {}).get("decision_trace_path"),
                "mobility_agent.final_summary_path": outcome.get("artifact_paths", {}).get("final_summary_path"),
                **({"mobility_agent.termination_reason": outcome.get("termination_reason")} if outcome.get("termination_reason") is not None else {}),
                **({"mobility_agent.final_acceptance": final_acceptance} if final_acceptance is not None else {}),
                **({"mobility_agent.quality_grade": quality_grade} if quality_grade is not None else {}),
                "mobility_agent.accepted_channels": accepted_channels,
                "mobility_agent.rejected_channels": rejected_channels,
                **calc_round_fields,
            },
            "$unset": unset_fields,
        },
    )


def _write_batch_summary(cfg: BatchConfig, summary: dict[str, Any]) -> str:
    os.makedirs(cfg.runs_root, exist_ok=True)
    path = os.path.join(cfg.runs_root, f"batch_summary_{cfg.batch_tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return path


def _merged_batch(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    base = dict(state.get("batch", {}) or {})
    merged = {**base, **updates}
    return merged


def _batch_nodes(
    cfg: BatchConfig,
    runtime: RuntimeContext,
    *,
    fresh_materials: bool,
    material_runner=run_single_material,
) -> dict[str, object]:
    runtime_ctx = runtime
    skills_root = os.path.abspath(runtime.skills_root)
    supervisor = BatchSupervisorAgent(runtime, skills_root)
    tool_gateway = AgentToolGateway()

    def batch_init() -> dict[str, Any]:
        return {
            "task": {
                "task_type": "batch_database",
                "collection_name": cfg.mongo_collection,
                "root_path": cfg.runs_root,
            },
            "execution": {
                "workdir": cfg.runs_root,
                "environment_summary": {
                    "deprecation_warnings": list(dict.fromkeys(list(runtime.deprecation_warnings) + list(cfg.deprecation_warnings)))
                },
            },
            "batch": {
                "done": False,
                "queue": [],
                "running_items": [],
                "completed_items": [],
                "failed_items": [],
                "skipped_items": [],
                "outcomes": [],
                "global_statistics": {
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "skipped": 0,
                    "scientifically_passed": 0,
                    "scientifically_warning": 0,
                    "scientifically_failed": 0,
                    "scientifically_unknown": 0,
                    "common_failure_stages": {},
                },
            },
        }

    def fetch_next(state: dict[str, Any]) -> dict[str, Any]:
        handles = connect(cfg.mongo_uri, cfg.mongo_db, cfg.mongo_collection)
        try:
            doc = claim_next_material(
                handles.collection,
                batch_tag=cfg.batch_tag,
                retry_failed=cfg.retry_failed,
                running_stale_s=cfg.running_stale_s,
            )
        finally:
            try:
                handles.client.close()
            except Exception:
                pass
        if not doc:
            return {"batch": _merged_batch(state, {"done": True})}
        material_id = str(doc.get("material_id") or doc.get("_id"))
        material_root = _material_root(cfg, material_id)
        return {
            "batch": _merged_batch(
                state,
                {
                "done": False,
                "running_items": list(state.get("batch", {}).get("running_items", []) or [])
                + [{"material_id": material_id, "root_path": material_root}],
                },
            ),
            "current_doc": _msgpack_safe(
                {
                    "_id": doc.get("_id"),
                    "structure": doc.get("structure"),
                    "material_id": doc.get("material_id"),
                    "loop_metadata": doc.get("loop_metadata"),
                }
            ),
            "current_doc_id": str(doc.get("_id")),
            "current_material_id": material_id,
            "current_material_root": material_root,
            "current_workdir": default_material_workdir(material_root),
        }

    def prepare_item(state: dict[str, Any]) -> dict[str, Any]:
        doc = dict(state.get("current_doc", {}) or {})
        material_id = str(state.get("current_material_id") or "")
        material_root = str(state.get("current_material_root") or "")
        try:
            os.makedirs(material_root, exist_ok=True)
            struct = structure_from_mongo_doc(doc)
            poscar_path = os.path.join(material_root, "POSCAR")
            potcar_path = os.path.join(material_root, "POTCAR")
            write_poscar(struct, poscar_path)
            potcar_used = None
            if not os.path.exists(potcar_path):
                if cfg.potcar_method == "vaspkit":
                    log_path = os.path.join(material_root, "potcar_gen.log")
                    with open(log_path, "w", encoding="utf-8") as handle:
                        proc = subprocess.Popen(
                            [cfg.vaspkit_cmd, "-task", str(cfg.vaspkit_task)],
                            cwd=material_root,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                        )
                        proc.wait()
                    if proc.returncode != 0 or not os.path.exists(potcar_path):
                        raise RuntimeError(f"vaspkit_potcar_failed:returncode={proc.returncode}")
                else:
                    potcar_used = build_potcar(
                        struct,
                        potcar_root=cfg.potcar_root or "",
                        dest_path=potcar_path,
                        potcar_map_path=cfg.potcar_map_path,
                    )
            return {
                "current_poscar_path": poscar_path,
                "current_potcar_path": potcar_path,
                "current_potcar_used": potcar_used,
                "current_prepare_error": None,
            }
        except Exception as exc:
            return {
                "current_prepare_error": str(exc),
                "current_outcome": _synthetic_outcome(
                    material_id=material_id or "unknown",
                    workdir=str(state.get("current_workdir") or material_root),
                    status="failed",
                    termination_reason="prepare_failed",
                    error_message=str(exc),
                ),
            }

    def run_item(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("current_outcome"):
            return {"current_outcome": dict(state.get("current_outcome", {}) or {})}
        material_id = str(state.get("current_material_id") or "unknown")
        material_root = str(state.get("current_material_root") or cfg.runs_root)
        workdir = str(state.get("current_workdir") or default_material_workdir(material_root))
        try:
            outcome = material_runner(
                runtime=runtime,
                material_id=material_id,
                root_path=material_root,
                workdir=workdir,
                poscar_path=str(state.get("current_poscar_path") or os.path.join(material_root, "POSCAR")),
                potcar_path=str(state.get("current_potcar_path") or os.path.join(material_root, "POTCAR")),
                user_goal="batch_mobility_screening",
                parent_batch_id=cfg.batch_tag,
                fresh=fresh_materials,
            )
            return {"current_outcome": outcome.model_dump(mode="json")}
        except Exception as exc:
            return {
                "current_outcome": _synthetic_outcome(
                    material_id=material_id,
                    workdir=workdir,
                    status="failed",
                    termination_reason="runner_exception",
                    error_message=f"{type(exc).__name__}:{exc}",
                )
            }

    def aggregate_item(state: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        batch = dict(state.get("batch", {}) or {})
        outcome = _normalize_outcome_for_persistence(dict(state.get("current_outcome", {}) or {}))
        material_id = str(state.get("current_material_id") or outcome.get("material_id") or "unknown")
        completed_items = list(batch.get("completed_items", []) or [])
        failed_items = list(batch.get("failed_items", []) or [])
        skipped_items = list(batch.get("skipped_items", []) or [])
        running_items = [
            item for item in list(batch.get("running_items", []) or []) if item.get("material_id") != material_id
        ]
        outcomes = list(batch.get("outcomes", []) or []) + [outcome]
        aggregate = tool_gateway.call("summarize_batch_outcomes", {"outcomes": outcomes})
        stats = {
            "processed": int(aggregate.get("processed", 0) or 0),
            "succeeded": int(aggregate.get("succeeded", 0) or 0),
            "failed": int(aggregate.get("failed", 0) or 0),
            "skipped": int(aggregate.get("skipped", 0) or 0),
            "scientifically_passed": int(aggregate.get("scientifically_passed", 0) or 0),
            "scientifically_warning": int(aggregate.get("scientifically_warning", 0) or 0),
            "scientifically_failed": int(aggregate.get("scientifically_failed", 0) or 0),
            "scientifically_unknown": int(aggregate.get("scientifically_unknown", 0) or 0),
            "common_failure_stages": dict(aggregate.get("common_failure_stages", {}) or {}),
        }
        status = str(outcome.get("status") or outcome.get("final_status") or "failed")

        if status == "completed":
            completed_items.append({"material_id": material_id, "status": status})
        elif status == "skipped":
            skipped_items.append({"material_id": material_id, "status": status})
        else:
            failed_items.append({"material_id": material_id, "status": status})

        current_doc = dict(state.get("current_doc", {}) or {})
        loop_metadata = dict(current_doc.get("loop_metadata", {}) or {})
        round_index = loop_metadata.get("round_index")
        round_id = loop_metadata.get("round_id")
        pipeline_run_id = loop_metadata.get("pipeline_run_id")

        handles = connect(cfg.mongo_uri, cfg.mongo_db, cfg.mongo_collection)
        try:
            doc_id = _coerce_mongo_doc_id(state.get("current_doc_id"))
            if status == "completed":
                results_payload = dict(outcome.get("results", {}) or {})
                mark_completed(
                    handles.collection,
                    doc_id=doc_id,
                    results=results_payload,
                    run_dir=str(outcome.get("workdir") or state.get("current_workdir") or ""),
                    quality_label=classify_material_quality(results_payload),
                    potcar_used=state.get("current_potcar_used"),
                    decision_engine=runtime_ctx.agent_runtime.decision_engine.value,
                    llm_required=True,
                    agent_summary=dict(outcome.get("final_summary", {}) or {}),
                    validation=dict(outcome.get("validation_report", {}) or {}),
                    confidence_score=outcome.get("confidence_score"),
                    warnings=list(outcome.get("warnings", []) or []),
                    decision_trace_path=outcome.get("artifact_paths", {}).get("decision_trace_path"),
                    final_summary_path=outcome.get("artifact_paths", {}).get("final_summary_path"),
                    recovery_summary={"errors": list(outcome.get("errors", []) or [])},
                    refinement_summary=dict(outcome.get("validation_report", {}).get("fit_metrics", {}) or {}),
                    final_status=str(outcome.get("final_status") or outcome.get("status") or "completed"),
                    termination_reason=outcome.get("termination_reason"),
                    final_acceptance=outcome.get("final_acceptance"),
                    quality_grade=outcome.get("validation_report", {}).get("quality_grade"),
                    accepted_channels=list(outcome.get("accepted_channels", []) or []),
                    rejected_channels=list(outcome.get("rejected_channels", []) or []),
                    round_index=round_index,
                    round_id=round_id,
                    pipeline_run_id=pipeline_run_id,
                )
            elif status == "skipped":
                _mark_skipped(
                    handles.collection,
                    doc_id=doc_id,
                    outcome=outcome,
                    runtime=runtime_ctx,
                    round_index=round_index,
                    round_id=round_id,
                    pipeline_run_id=pipeline_run_id,
                )
            else:
                mark_failed(
                    handles.collection,
                    doc_id=doc_id,
                    error="; ".join(outcome.get("errors", []) or []) or str(outcome.get("termination_reason") or "failed"),
                    run_dir=str(outcome.get("workdir") or state.get("current_workdir") or ""),
                    decision_engine=runtime_ctx.agent_runtime.decision_engine.value,
                    llm_required=True,
                    agent_summary=dict(outcome.get("final_summary", {}) or {}),
                    validation=dict(outcome.get("validation_report", {}) or {}),
                    confidence_score=outcome.get("confidence_score"),
                    decision_trace_path=outcome.get("artifact_paths", {}).get("decision_trace_path"),
                    final_summary_path=outcome.get("artifact_paths", {}).get("final_summary_path"),
                    final_status=str(outcome.get("final_status") or outcome.get("status") or "failed"),
                    termination_reason=outcome.get("termination_reason"),
                    final_acceptance=outcome.get("final_acceptance"),
                    quality_grade=outcome.get("validation_report", {}).get("quality_grade"),
                    accepted_channels=list(outcome.get("accepted_channels", []) or []),
                    rejected_channels=list(outcome.get("rejected_channels", []) or []),
                    round_index=round_index,
                    round_id=round_id,
                    pipeline_run_id=pipeline_run_id,
                )
        finally:
            try:
                handles.client.close()
            except Exception:
                pass

        try:
            with active_workdir_scope(cfg.runs_root):
                batch_summary = supervisor.summarize(outcomes=outcomes).model_dump(mode="json")
        except Exception as exc:
            # Keep batch progressing even if reporter LLM is temporarily unavailable.
            batch_summary = {
                "processed": int(stats.get("processed", 0) or 0),
                "succeeded": int(stats.get("succeeded", 0) or 0),
                "failed": int(stats.get("failed", 0) or 0),
                "skipped": int(stats.get("skipped", 0) or 0),
                "scientifically_passed": int(stats.get("scientifically_passed", 0) or 0),
                "scientifically_warning": int(stats.get("scientifically_warning", 0) or 0),
                "scientifically_failed": int(stats.get("scientifically_failed", 0) or 0),
                "scientifically_unknown": int(stats.get("scientifically_unknown", 0) or 0),
                "common_failure_stages": dict(stats.get("common_failure_stages", {}) or {}),
                "llm_summary_unavailable": True,
                "llm_summary_error": f"{type(exc).__name__}:{exc}",
            }
        batch_summary["common_failure_stages"] = dict(stats.get("common_failure_stages", {}) or {})
        with open_memory_store(runtime_ctx.resolved_db_uri) as store:
            record_batch_statistics(
                store,
                collection_name=cfg.mongo_collection,
                payload={
                    "collection_name": cfg.mongo_collection,
                    "batch_tag": cfg.batch_tag,
                    **stats,
                },
            )
        return {
            "batch": _merged_batch(
                state,
                {
                "done": False,
                "running_items": running_items,
                "completed_items": completed_items,
                "failed_items": failed_items,
                "skipped_items": skipped_items,
                "outcomes": outcomes,
                "summary": batch_summary,
                "global_statistics": stats,
                },
            ),
            "current_outcome": {},
            "current_prepare_error": None,
        }

    def finalize(state: dict[str, Any]) -> dict[str, Any]:
        batch = dict(state.get("batch", {}) or {})
        outcomes = list(batch.get("outcomes", []) or [])
        aggregate = tool_gateway.call("summarize_batch_outcomes", {"outcomes": outcomes})
        try:
            if batch.get("summary"):
                summary = dict(batch.get("summary", {}) or {})
            else:
                with active_workdir_scope(cfg.runs_root):
                    summary = supervisor.summarize(outcomes=outcomes).model_dump(mode="json")
        except Exception as exc:
            summary = {
                "processed": int(aggregate.get("processed", 0) or 0),
                "succeeded": int(aggregate.get("succeeded", 0) or 0),
                "failed": int(aggregate.get("failed", 0) or 0),
                "skipped": int(aggregate.get("skipped", 0) or 0),
                "scientifically_passed": int(aggregate.get("scientifically_passed", 0) or 0),
                "scientifically_warning": int(aggregate.get("scientifically_warning", 0) or 0),
                "scientifically_failed": int(aggregate.get("scientifically_failed", 0) or 0),
                "scientifically_unknown": int(aggregate.get("scientifically_unknown", 0) or 0),
                "llm_summary_unavailable": True,
                "llm_summary_error": f"{type(exc).__name__}:{exc}",
            }
        summary["common_failure_stages"] = dict(aggregate.get("common_failure_stages", {}) or {})
        summary["batch_tag"] = cfg.batch_tag
        summary["collection_name"] = cfg.mongo_collection
        summary_path = _write_batch_summary(cfg, summary)
        return {
            "batch": _merged_batch(
                state,
                {
                    "done": True,
                    "summary": summary,
                    "summary_path": summary_path,
                },
            )
        }

    return {
        "batch_init": batch_init,
        "fetch_next": fetch_next,
        "prepare_item": prepare_item,
        "run_item": run_item,
        "aggregate_item": aggregate_item,
        "finalize": finalize,
    }


def run_mongo_batch(
    *,
    cfg: BatchConfig,
    runtime: RuntimeContext,
    thread_id: str | None = None,
    fresh_materials: bool = False,
    material_runner=run_single_material,
) -> dict[str, Any]:
    runtime.require_llm_ready()
    cfg = _ensure_batch_runs_root(cfg, runtime)
    resolved_thread_id = thread_id or build_batch_thread_id(batch_id=cfg.batch_tag)
    config = {"configurable": {"thread_id": resolved_thread_id}}
    nodes = _batch_nodes(cfg, runtime, fresh_materials=fresh_materials, material_runner=material_runner)
    save_thread_id(workdir=cfg.runs_root, thread_id=resolved_thread_id, checkpoint_subdir=runtime.checkpoint_subdir)
    save_checkpoint_metadata(
        workdir=cfg.runs_root,
        thread_id=resolved_thread_id,
        database_uri=runtime.resolved_db_uri,
        checkpoint_subdir=runtime.checkpoint_subdir,
    )
    with open_runtime_checkpointer(database_uri=runtime.resolved_db_uri) as checkpointer:
        with open_memory_store(runtime.resolved_db_uri) as store:
            app = build_batch_entrypoint(
                checkpointer=checkpointer,
                store=store,
                batch_init=nodes["batch_init"],
                fetch_next=nodes["fetch_next"],
                prepare_item=nodes["prepare_item"],
                run_item=nodes["run_item"],
                aggregate_item=nodes["aggregate_item"],
                finalize=nodes["finalize"],
            )
            final_state = dict(app.invoke({}, config=config) or {})
    return final_state
