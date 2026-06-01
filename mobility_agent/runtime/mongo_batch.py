from __future__ import annotations

import json
import os

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .channel_utils import VALID_CARRIERS, VALID_DIRECTIONS, canonical_subchannel

try:
    from pymongo import MongoClient, ReturnDocument
except Exception:  # pragma: no cover - used for low-dependency unit tests
    MongoClient = None  # type: ignore

    class _ReturnDocumentFallback:
        AFTER = "after"

    ReturnDocument = _ReturnDocumentFallback()  # type: ignore


@dataclass(frozen=True)
class MongoHandles:
    client: MongoClient
    collection: Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _terminal_success_unset_fields() -> dict[str, str]:
    return {
        "mobility_calc.error": "",
        "mobility_calc.failed_at": "",
        "mobility_agent.error": "",
        "mobility_agent.failed_at": "",
    }


def _terminal_non_success_unset_fields() -> dict[str, str]:
    return {
        "mobility_calc.results": "",
        "mobility_calc.quality_label": "",
        "mobility_calc.scientific_decision": "",
        "mobility_calc.quality_grade": "",
        "mobility_calc.accepted_channels": "",
        "mobility_calc.rejected_channels": "",
        "mobility_calc.channel_labels": "",
        "mobility_calc.completed_at": "",
        "mobility_agent.completed_at": "",
    }


def _round_metadata_set_fields(
    *,
    round_index: Any = None,
    round_id: Any = None,
    pipeline_run_id: Any = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    normalized_round_index = _safe_int(round_index)
    normalized_round_id = str(round_id).strip() if round_id is not None else ""
    normalized_pipeline_run_id = str(pipeline_run_id).strip() if pipeline_run_id is not None else ""
    if normalized_round_index is not None:
        fields["mobility_calc.round_index"] = normalized_round_index
    if normalized_round_id:
        fields["mobility_calc.round_id"] = normalized_round_id
    if normalized_pipeline_run_id:
        fields["mobility_calc.pipeline_run_id"] = normalized_pipeline_run_id
    return fields


def _project_channel_labels(validation: Optional[Dict[str, Any]]) -> dict[str, dict[str, Any]]:
    reviews = dict((validation or {}).get("channel_reviews", {}) or {})
    projected: dict[str, dict[str, Any]] = {}
    for direction in VALID_DIRECTIONS:
        for carrier in VALID_CARRIERS:
            token = canonical_subchannel(direction, carrier)
            review = dict(reviews.get(token, {}) or {})
            projected[token] = {
                "status": str(review.get("status") or "unknown"),
                "reason": str(review.get("reason") or "").strip() or None,
                "direction": str(review.get("direction") or direction),
                "carrier": str(review.get("carrier") or carrier),
                "n_points": _safe_int(review.get("n_points")),
                "mobility_cm2_Vs": _safe_float(review.get("mobility_cm2_Vs")),
                "E1_fit_R2": _safe_float(review.get("E1_fit_R2")),
                "C2D_fit_R2": _safe_float(review.get("C2D_fit_R2")),
            }
    return projected


def connect(mongo_uri: str, db_name: str, collection_name: str) -> MongoHandles:
    if MongoClient is None:
        raise ModuleNotFoundError("pymongo is required for MongoDB connectivity")
    timeout_ms = int(os.environ.get("MONGO_TIMEOUT_MS", "30000") or "30000")
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=timeout_ms,
        connectTimeoutMS=timeout_ms,
        socketTimeoutMS=timeout_ms,
    )
    # 立刻验证连通性，便于给出清晰报错（Atlas IP 白名单/网络阻断时常见）
    client.admin.command("ping")
    col = client[db_name][collection_name]
    return MongoHandles(client=client, collection=col)


def load_claim_filter_from_env() -> Optional[Dict[str, Any]]:
    raw = str(os.environ.get("MONGO_CLAIM_FILTER_JSON") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MONGO_CLAIM_FILTER_JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MONGO_CLAIM_FILTER_JSON must decode to a JSON object")
    return payload


def claim_next_material(
    col,
    *,
    batch_tag: str,
    retry_failed: bool,
    running_stale_s: int,
) -> Optional[Dict[str, Any]]:
    """原子化领取一个待处理材料，设置为 running 并返回文档。"""

    # 允许字段不存在或 status==pending
    pending_query: Dict[str, Any] = {
        "$or": [
            {"mobility_calc": {"$exists": False}},
            {"mobility_calc.status": {"$exists": False}},
            {"mobility_calc.status": "pending"},
        ]
    }

    if retry_failed:
        # Only reclaim failures from previous batches.
        # Without this guard, a document that fails in the current batch is
        # immediately eligible again, so the smallest _id can be claimed over
        # and over and starve the rest of the collection.
        pending_query["$or"].append(
            {
                "mobility_calc.status": "failed",
                "mobility_calc.batch_tag": {"$ne": batch_tag},
            }
        )

    # running 状态可能来自上次异常退出；超过阈值允许重新领取
    if running_stale_s and running_stale_s > 0:
        stale_before = utc_now().timestamp() - float(running_stale_s)
        # pymongo 存的是 datetime，这里用 datetime 进行比较
        stale_dt = datetime.fromtimestamp(stale_before, tz=timezone.utc)
        pending_query["$or"].append(
            {
                "mobility_calc.status": "running",
                "mobility_calc.started_at": {"$lt": stale_dt},
            }
        )

    update = {
        "$set": {
            "mobility_calc.status": "running",
            "mobility_calc.batch_tag": batch_tag,
            "mobility_calc.started_at": utc_now(),
        }
    }
    extra_filter = load_claim_filter_from_env()
    final_filter: Dict[str, Any] = pending_query
    if extra_filter:
        final_filter = {"$and": [extra_filter, pending_query]}

    doc = col.find_one_and_update(
        filter=final_filter,
        update=update,
        sort=[("_id", 1)],
        return_document=ReturnDocument.AFTER,
    )
    return doc


def mark_completed(
    col,
    *,
    doc_id,
    results: Dict[str, Any],
    run_dir: str,
    quality_label: Optional[str] = None,
    potcar_used: Optional[list[str]] = None,
    decision_engine: Optional[str] = None,
    llm_required: Optional[bool] = None,
    agent_summary: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    confidence_score: Optional[float] = None,
    warnings: Optional[list[str]] = None,
    decision_trace_path: Optional[str] = None,
    final_summary_path: Optional[str] = None,
    recovery_summary: Optional[Dict[str, Any]] = None,
    refinement_summary: Optional[Dict[str, Any]] = None,
    final_status: Optional[str] = None,
    termination_reason: Optional[str] = None,
    final_acceptance: Optional[str] = None,
    quality_grade: Optional[str] = None,
    accepted_channels: Optional[list[str]] = None,
    rejected_channels: Optional[list[str]] = None,
    round_index: Any = None,
    round_id: Any = None,
    pipeline_run_id: Any = None,
) -> None:
    if validation is not None:
        if final_acceptance is None:
            final_acceptance = str(validation.get("decision") or "").strip() or None
        if quality_grade is None:
            quality_grade = str(validation.get("quality_grade") or "").strip() or None

    calc_set: Dict[str, Any] = {
        "mobility_calc.status": "completed",
        "mobility_calc.completed_at": utc_now(),
        "mobility_calc.run_dir": run_dir,
        "mobility_calc.results": results,
    }
    calc_unset = _terminal_success_unset_fields()
    if potcar_used is not None:
        calc_set["mobility_calc.potcar_used"] = potcar_used
    if quality_label is not None:
        calc_set["mobility_calc.quality_label"] = str(quality_label)
    else:
        calc_unset["mobility_calc.quality_label"] = ""
    if validation is not None:
        calc_set["mobility_calc.channel_labels"] = _project_channel_labels(validation)
    else:
        calc_unset["mobility_calc.channel_labels"] = ""
    if final_acceptance is not None:
        calc_set["mobility_calc.scientific_decision"] = final_acceptance
    else:
        calc_unset["mobility_calc.scientific_decision"] = ""
    if quality_grade is not None:
        calc_set["mobility_calc.quality_grade"] = quality_grade
    else:
        calc_unset["mobility_calc.quality_grade"] = ""
    if accepted_channels is not None:
        calc_set["mobility_calc.accepted_channels"] = list(accepted_channels)
    else:
        calc_unset["mobility_calc.accepted_channels"] = ""
    if rejected_channels is not None:
        calc_set["mobility_calc.rejected_channels"] = list(rejected_channels)
    else:
        calc_unset["mobility_calc.rejected_channels"] = ""
    calc_set.update(
        _round_metadata_set_fields(
            round_index=round_index,
            round_id=round_id,
            pipeline_run_id=pipeline_run_id,
        )
    )

    update = {
        "$set": calc_set,
        "$unset": calc_unset,
    }

    agent_set: Dict[str, Any] = {
        "mobility_agent.completed_at": utc_now(),
        "mobility_agent.status": "completed",
        "mobility_agent.final_status": str(final_status or "completed"),
    }
    if decision_engine is not None:
        agent_set["mobility_agent.decision_engine"] = decision_engine
    if llm_required is not None:
        agent_set["mobility_agent.llm_required"] = bool(llm_required)
    if agent_summary is not None:
        agent_set["mobility_agent.summary"] = agent_summary
        admission = agent_summary.get("admission") if isinstance(agent_summary, dict) else None
        if admission is not None:
            agent_set["mobility_agent.admission"] = admission
        warnings_value = agent_summary.get("warnings") if isinstance(agent_summary, dict) else None
        if warnings_value is not None:
            agent_set["mobility_agent.warnings"] = warnings_value
    if validation is not None:
        agent_set["mobility_agent.validation"] = validation
    if confidence_score is not None:
        agent_set["mobility_agent.confidence_score"] = float(confidence_score)
    if warnings is not None:
        agent_set["mobility_agent.warnings"] = warnings
    if decision_trace_path is not None:
        agent_set["mobility_agent.decision_trace_path"] = decision_trace_path
    if final_summary_path is not None:
        agent_set["mobility_agent.final_summary_path"] = final_summary_path
    if recovery_summary is not None:
        agent_set["mobility_agent.recovery_summary"] = recovery_summary
    if refinement_summary is not None:
        agent_set["mobility_agent.refinement_summary"] = refinement_summary
    if termination_reason is not None:
        agent_set["mobility_agent.termination_reason"] = termination_reason
    if final_acceptance is not None:
        agent_set["mobility_agent.final_acceptance"] = final_acceptance
    if quality_grade is not None:
        agent_set["mobility_agent.quality_grade"] = quality_grade
    if accepted_channels is not None:
        agent_set["mobility_agent.accepted_channels"] = list(accepted_channels)
    if rejected_channels is not None:
        agent_set["mobility_agent.rejected_channels"] = list(rejected_channels)

    update["$set"].update(agent_set)

    col.update_one({"_id": doc_id}, update)


def mark_failed(
    col,
    *,
    doc_id,
    error: str,
    run_dir: str,
    decision_engine: Optional[str] = None,
    llm_required: Optional[bool] = None,
    agent_summary: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
    confidence_score: Optional[float] = None,
    decision_trace_path: Optional[str] = None,
    final_summary_path: Optional[str] = None,
    final_status: Optional[str] = None,
    termination_reason: Optional[str] = None,
    final_acceptance: Optional[str] = None,
    quality_grade: Optional[str] = None,
    accepted_channels: Optional[list[str]] = None,
    rejected_channels: Optional[list[str]] = None,
    round_index: Any = None,
    round_id: Any = None,
    pipeline_run_id: Any = None,
) -> None:
    if validation is not None:
        if final_acceptance is None:
            final_acceptance = str(validation.get("decision") or "").strip() or None
        if quality_grade is None:
            quality_grade = str(validation.get("quality_grade") or "").strip() or None
    failure_unset = _terminal_non_success_unset_fields()
    if validation is None:
        failure_unset["mobility_agent.validation"] = ""
    if final_acceptance is None:
        failure_unset["mobility_agent.final_acceptance"] = ""
    if quality_grade is None:
        failure_unset["mobility_agent.quality_grade"] = ""
    if accepted_channels is None:
        failure_unset["mobility_agent.accepted_channels"] = ""
    if rejected_channels is None:
        failure_unset["mobility_agent.rejected_channels"] = ""
    col.update_one(
        {"_id": doc_id},
        {
            "$set": {
                "mobility_calc.status": "failed",
                "mobility_calc.failed_at": utc_now(),
                "mobility_calc.run_dir": run_dir,
                "mobility_calc.error": error,
                "mobility_agent.status": "failed",
                "mobility_agent.failed_at": utc_now(),
                "mobility_agent.error": error,
                "mobility_agent.final_status": str(final_status or "failed"),
                **({"mobility_agent.decision_engine": decision_engine} if decision_engine is not None else {}),
                **({"mobility_agent.llm_required": bool(llm_required)} if llm_required is not None else {}),
                **({"mobility_agent.summary": agent_summary} if agent_summary is not None else {}),
                **({"mobility_agent.validation": validation} if validation is not None else {}),
                **({"mobility_agent.confidence_score": float(confidence_score)} if confidence_score is not None else {}),
                **({"mobility_agent.decision_trace_path": decision_trace_path} if decision_trace_path is not None else {}),
                **({"mobility_agent.final_summary_path": final_summary_path} if final_summary_path is not None else {}),
                **({"mobility_agent.termination_reason": termination_reason} if termination_reason is not None else {}),
                **({"mobility_agent.final_acceptance": final_acceptance} if final_acceptance is not None else {}),
                **({"mobility_agent.quality_grade": quality_grade} if quality_grade is not None else {}),
                **({"mobility_agent.accepted_channels": list(accepted_channels)} if accepted_channels is not None else {}),
                **({"mobility_agent.rejected_channels": list(rejected_channels)} if rejected_channels is not None else {}),
                **_round_metadata_set_fields(
                    round_index=round_index,
                    round_id=round_id,
                    pipeline_run_id=pipeline_run_id,
                ),
            },
            "$unset": failure_unset,
        },
    )
