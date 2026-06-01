from __future__ import annotations

def resolve_terminal_status(
    *,
    final_acceptance: str | None,
    termination_reason: str | None,
    errors: list[str] | None,
    current_status: str | None = None,
) -> str:
    if current_status == "skipped":
        return "skipped"
    if termination_reason in {"abort_task", "abort_batch"}:
        return "failed"
    if termination_reason in {"skip_material", "skip_point"}:
        return "skipped"
    if final_acceptance in {"pass", "accepted"}:
        return "completed"
    if final_acceptance in {"pass_with_warning", "accepted_with_warning"}:
        return "completed"
    if final_acceptance in {"fail", "rejected"}:
        return "failed"
    if errors:
        return "failed"
    return "completed"
