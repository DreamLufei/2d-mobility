from __future__ import annotations

from typing import Any


def should_escalate_recovery(summary: dict[str, Any], decision: dict[str, Any]) -> bool:
    retries_used = int(summary.get("retries_used", 0) or 0)
    confidence = float(decision.get("confidence", 1.0) or 0.0)
    error_type = str(summary.get("error_type") or "")
    if decision.get("decision") == "manual_fix_resume":
        return True
    if retries_used >= int(summary.get("max_retries", 2) or 2):
        return True
    if error_type in {"unknown_failure", "missing_output"} and confidence < 0.75:
        return True
    return bool(decision.get("should_escalate", False))


def should_escalate_validation(validation: dict[str, Any]) -> bool:
    decision = str(validation.get("decision") or "")
    confidence = float(validation.get("confidence", validation.get("confidence_score", 1.0)) or 0.0)
    return decision == "escalate" or confidence < 0.45


def should_escalate_refinement(refinement: dict[str, Any]) -> bool:
    return str(refinement.get("decision") or "") == "escalate"

