from __future__ import annotations


def detect_basic_anomalies(state: dict) -> list[str]:
    flags: list[str] = []
    if state.get("errors") and (state.get("run_status") == "failed" or state.get("status_label") in {"failed", "rejected"}):
        flags.append("workflow_errors_present")
    if state.get("confidence_score") is not None and float(state.get("confidence_score") or 0.0) < 0.5:
        flags.append("low_confidence_score")
    return flags