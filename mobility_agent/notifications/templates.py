from __future__ import annotations

from typing import Any


def build_escalation_subject(payload: dict[str, Any]) -> str:
    return (
        f"[mobility-agent] escalation: "
        f"{payload.get('material_id', 'unknown')} / {payload.get('current_stage', 'unknown')}"
    )


def build_escalation_body(payload: dict[str, Any]) -> str:
    return (
        "Human escalation requested.\n\n"
        f"task_id: {payload.get('task_id')}\n"
        f"material_id: {payload.get('material_id')}\n"
        f"stage: {payload.get('current_stage')}\n"
        f"issue: {payload.get('error_summary')}\n"
        f"recommended_options: {payload.get('recommended_options')}\n"
        f"timeout_policy: {payload.get('timeout_policy')}\n"
        f"working_directory: {payload.get('working_directory')}\n"
        f"log_paths: {payload.get('log_paths')}\n"
    )

