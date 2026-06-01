from __future__ import annotations

from typing import Any

from .schemas import StageProbe


def build_stage_probe_from_state(state_payload: dict[str, Any], *, stage: str, extra_context: dict[str, Any] | None = None) -> StageProbe:
    state = dict(state_payload or {})
    material = dict(state.get("material", {}) or {})
    execution = dict(state.get("execution", {}) or {})
    diagnostics = dict(state.get("diagnostics", {}) or {})
    latest = dict(execution.get("latest_execution_observation", {}) or {})
    structure_summary = dict(material.get("structure_summary", {}) or material.get("preflight_summary", {}) or {})
    atom_count = int(material.get("atom_count") or structure_summary.get("atom_count") or 0)
    target_ka = 50.0 if stage in {"relax", "scf"} else None
    return StageProbe(
        stage=stage,
        material_id=str(material.get("material_id") or ""),
        atom_count=atom_count,
        composition=material.get("composition"),
        structure_summary=structure_summary,
        resource_summary=dict(execution.get("environment_summary", {}) or {}),
        kpoint_summary={k: v for k, v in {"target_ka": target_ka}.items() if v is not None},
        prior_execution_summary={
            "latest_status": latest.get("status"),
            "latest_error": diagnostics.get("last_error") or latest.get("error_summary"),
            "warnings": list(material.get("warnings", []) or []),
        },
        extra_context=dict(extra_context or {}),
    )
