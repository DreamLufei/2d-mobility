from __future__ import annotations

import json
import os
import time
from typing import Any

from ..agents.schemas import HITLDecision


def wait_for_response_file(*, response_path: str, timeout_s: int) -> dict[str, Any] | None:
    deadline = time.time() + max(0, int(timeout_s))
    while time.time() < deadline:
        if os.path.exists(response_path):
            try:
                with open(response_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return None
        time.sleep(1.0)
    return None


def timeout_decision(*, policy: str, default_action: str | None = None) -> HITLDecision:
    if policy == "non_interactive_abort_on_timeout":
        action = "abort_task"
    else:
        action = str(default_action or "skip_material")
    if action not in {
        "retry_current_stage",
        "rerun_previous_stage",
        "modify_params_and_retry",
        "copy_contcar_to_poscar_and_retry",
        "skip_point",
        "skip_material",
        "abort_task",
    }:
        action = "skip_material"
    return HITLDecision(
        action=action,  # type: ignore[arg-type]
        reason=f"{policy}_timeout",
        source="timeout_default",
        warnings=[],
    )
