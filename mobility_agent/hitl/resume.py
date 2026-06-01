from __future__ import annotations

from typing import Any

from ..agents.schemas import HITLDecision, ManualFixInstruction


def normalize_hitl_decision(action_payload: dict[str, Any], *, source: str = "precomputed") -> HITLDecision:
    action = str(action_payload.get("action") or action_payload.get("decision") or "").strip()
    if action == "manual_fix_resume":
        instruction_payload = action_payload.get("instruction") if isinstance(action_payload.get("instruction"), dict) else action_payload
        instruction = ManualFixInstruction.model_validate(instruction_payload)
        return HITLDecision(
            action="manual_fix_resume",
            instruction=instruction,
            reason=str(action_payload.get("reason") or instruction.reason or "").strip(),
            source=source,  # type: ignore[arg-type]
            warnings=list(action_payload.get("warnings", []) or []),
        )
    return HITLDecision(
        action=action,  # type: ignore[arg-type]
        reason=str(action_payload.get("reason") or "").strip(),
        source=source,  # type: ignore[arg-type]
        warnings=list(action_payload.get("warnings", []) or []),
    )


def build_resume_command(decision: HITLDecision | dict[str, Any]) -> dict[str, Any]:
    if isinstance(decision, dict):
        decision = normalize_hitl_decision(decision)
    if decision.action == "manual_fix_resume":
        instruction = ManualFixInstruction.model_validate(
            decision.instruction.model_dump(mode="json") if decision.instruction is not None else {}
        )
        return {"action": "manual_fix_resume", "instruction": instruction.model_dump(mode="json")}
    payload = {"action": decision.action}
    if decision.reason:
        payload["reason"] = decision.reason
    if decision.warnings:
        payload["warnings"] = list(decision.warnings)
    return payload
