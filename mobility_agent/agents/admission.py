from __future__ import annotations

from typing import Any

from .base import SkillAwareAgent
from .schemas import AdmissionDecision
from ..utils import dedupe_keep_order


class AdmissionAgent(SkillAwareAgent):
    agent_name = "admission"
    llm_role = "admission"

    def decide(self, state: dict[str, Any]) -> AdmissionDecision:
        material = dict(state.get("material", {}) or {})
        warnings = dedupe_keep_order(list(material.get("warnings", []) or []))
        atom_count = int(material.get("atom_count", 0) or 0)
        metadata = dict(material.get("structure_metadata", {}) or {})
        if not material.get("poscar_path") or not material.get("potcar_path"):
            rule = AdmissionDecision(decision="reject", reason="required_input_missing", warnings=warnings, confidence=0.99)
        elif bool(metadata.get("is_metal", False)):
            rule = AdmissionDecision(decision="reject", reason="metallicity_detected", warnings=warnings, confidence=0.99)
        elif bool(metadata.get("is_magnetic", False)):
            rule = AdmissionDecision(decision="reject", reason="magnetic_system_outside_scope", warnings=warnings, confidence=0.98)
        elif atom_count <= 0:
            rule = AdmissionDecision(decision="continue_with_warning", reason="structure_summary_missing", warnings=warnings + ["structure_summary_missing"], confidence=0.70)
        elif warnings:
            rule = AdmissionDecision(decision="continue_with_warning", reason="preflight_contains_warnings", warnings=warnings, confidence=0.75)
        else:
            rule = AdmissionDecision(decision="continue", reason="passed_preflight", warnings=[], confidence=0.90)
        llm = self._maybe_call_llm(
            kind="admission",
            schema=AdmissionDecision,
            task_type=str(state.get("task", {}).get("task_type") or "single_material"),
            stage="admission",
            summary={"material": material, "warnings": warnings},
            rule_payload=rule.model_dump(mode="json"),
            allowed_actions=["continue", "continue_with_warning", "reject"],
            has_error=bool(warnings),
            explicit_skills=["admission"],
        )
        return AdmissionDecision.model_validate({**rule.model_dump(mode="json"), **(llm or {})})
