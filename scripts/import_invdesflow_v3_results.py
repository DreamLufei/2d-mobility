from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pymongo import MongoClient, UpdateOne


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import completed mobility results from a run root into MongoDB, optionally updating "
            "existing source documents in-place."
        )
    )
    parser.add_argument("--mongo-uri", default=os.environ.get("MONGO_URI", ""))
    parser.add_argument("--db", default=os.environ.get("MONGO_DB", "materials_database"))
    parser.add_argument("--source-collection", default="invdesmobility_v3")
    parser.add_argument("--target-collection", default="invdesflow_v3_result")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Use $set upserts so completed results can be written back into an existing collection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _file_mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _material_dirs(runs_root: Path) -> list[Path]:
    return sorted(
        path
        for path in runs_root.iterdir()
        if path.is_dir() and (path / "mobility_calculation").exists()
    )


@dataclass
class ImportRow:
    material_id: str
    doc: dict[str, Any]


def _base_doc_from_source(source_doc: dict[str, Any]) -> dict[str, Any]:
    doc = copy.deepcopy(source_doc)
    doc.pop("_id", None)
    doc.pop("mobility_calc", None)
    doc.pop("mobility_agent", None)
    return doc


def _build_mobility_calc(
    *,
    source_doc: dict[str, Any],
    workdir: Path,
    mobility_results: dict[str, Any],
) -> dict[str, Any]:
    existing_calc = dict(source_doc.get("mobility_calc", {}) or {})
    loop_metadata = dict(source_doc.get("loop_metadata", {}) or {})
    batch_tag = str(existing_calc.get("batch_tag") or "").strip() or None
    round_id = str(
        loop_metadata.get("round_id")
        or existing_calc.get("round_id")
        or ""
    ).strip() or None
    pipeline_run_id = str(
        loop_metadata.get("pipeline_run_id")
        or existing_calc.get("pipeline_run_id")
        or ""
    ).strip() or None
    round_index = loop_metadata.get("round_index", existing_calc.get("round_index"))
    started_at = _normalize_datetime(existing_calc.get("started_at"))
    completed_at = _normalize_datetime(existing_calc.get("completed_at"))
    if started_at is None:
        started_at = _file_mtime(workdir / "execution_checkpoint.json") or _file_mtime(workdir / "mobility_results.json")
    if completed_at is None:
        completed_at = _file_mtime(workdir / "material_outcome.json") or _file_mtime(workdir / "mobility_results.json")
    calc = {
        "status": "completed",
        "run_dir": str(workdir),
        "results": mobility_results,
    }
    if batch_tag is not None:
        calc["batch_tag"] = batch_tag
    if round_index is not None and str(round_index).strip():
        calc["round_index"] = int(round_index)
    if round_id is not None:
        calc["round_id"] = round_id
    if pipeline_run_id is not None:
        calc["pipeline_run_id"] = pipeline_run_id
    if started_at is not None:
        calc["started_at"] = started_at
    if completed_at is not None:
        calc["completed_at"] = completed_at
    return calc


def _prepare_rows(
    *,
    runs_root: Path,
    source_collection: Any,
) -> list[ImportRow]:
    rows: list[ImportRow] = []
    for material_dir in _material_dirs(runs_root):
        material_id = material_dir.name
        workdir = material_dir / "mobility_calculation"
        mobility_results_path = workdir / "mobility_results.json"
        outcome_path = workdir / "material_outcome.json"
        if not mobility_results_path.exists():
            raise FileNotFoundError(f"{material_id}: missing {mobility_results_path}")
        mobility_results = _load_json(mobility_results_path)
        results_by_direction = dict(mobility_results.get("results_by_direction", {}) or {})
        if not results_by_direction:
            raise ValueError(f"{material_id}: mobility_results.json has no results_by_direction")
        if not outcome_path.exists():
            raise FileNotFoundError(f"{material_id}: missing {outcome_path}")
        outcome = _load_json(outcome_path)
        stage_status = dict(outcome.get("stage_status", {}) or {})
        failed_compute = [
            stage
            for stage in ("prepare", "relax", "scf", "band", "effective_mass", "strain_loop", "mobility")
            if str(stage_status.get(stage) or "") == "failed"
        ]
        if failed_compute:
            raise ValueError(f"{material_id}: compute stages still failed in outcome: {failed_compute}")
        source_doc = source_collection.find_one({"material_id": material_id})
        if source_doc is None:
            raise KeyError(f"{material_id}: missing source document in source collection")
        doc = _base_doc_from_source(source_doc)
        doc["mobility_calc"] = _build_mobility_calc(
            source_doc=source_doc,
            workdir=workdir,
            mobility_results=mobility_results,
        )
        rows.append(ImportRow(material_id=material_id, doc=doc))
    return rows


def main() -> int:
    args = _parse_args()
    if not args.mongo_uri:
        raise SystemExit("MONGO_URI is required via --mongo-uri or environment")

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        raise SystemExit(f"runs root does not exist: {runs_root}")

    client = MongoClient(args.mongo_uri)
    db = client[args.db]
    source = db[args.source_collection]
    target = db[args.target_collection]

    rows = _prepare_rows(runs_root=runs_root, source_collection=source)
    material_ids = [row.material_id for row in rows]
    existing_ids = set(target.distinct("material_id", {"material_id": {"$in": material_ids}}))

    print(
        json.dumps(
            {
                "db": args.db,
                "source_collection": args.source_collection,
                "target_collection": args.target_collection,
                "runs_root": str(runs_root),
                "materials_found": len(rows),
                "existing_in_target": sorted(existing_ids),
                "update_existing": bool(args.update_existing),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    if args.dry_run:
        preview = {row.material_id: row.doc["mobility_calc"] for row in rows}
        print(json.dumps(preview, indent=2, ensure_ascii=False, default=str))
        return 0

    target.create_index("material_id", unique=True, name="uniq_material_id")
    update_operator = "$set" if args.update_existing else "$setOnInsert"
    ops = [
        UpdateOne({"material_id": row.material_id}, {update_operator: row.doc}, upsert=True)
        for row in rows
    ]
    result = target.bulk_write(ops, ordered=True)

    print(
        json.dumps(
            {
                "inserted_or_upserted": int(result.upserted_count),
                "matched_existing": int(result.matched_count),
                "modified_existing": int(result.modified_count),
                "target_count_after": int(target.count_documents({})),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
