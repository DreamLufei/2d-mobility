from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator

from ..utils import dedupe_keep_order


_UTC = timezone.utc
_ACTION_FAMILY_VALUES = (
    "run_capability",
    "retry_capability",
    "rerun_from_capability",
    "repair_execution_context",
    "refine_sampling",
    "revalidate_result",
    "invalidate_channel",
    "skip_channel",
    "escalate_human",
    "finalize_material",
    "abort_material",
)


def _strip_markdown_formatting(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"[*`]+", "", value)
    value = re.sub(r"^#+\s*", "", value)
    return value.strip()


def _extract_json_payload_from_text(text: str) -> Any:
    candidate = _strip_markdown_formatting(text)
    if not candidate:
        return None
    decoder = json.JSONDecoder()
    start_positions = [index for index, char in enumerate(candidate) if char in "[{"]
    for start in start_positions:
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _normalized_markdown_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        cleaned = _strip_markdown_formatting(raw).strip().strip("|").strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    label_lookup = [str(label or "").strip().lower() for label in labels if str(label or "").strip()]
    if not label_lookup:
        return None
    for line in _normalized_markdown_lines(text):
        parts = [part.strip() for part in line.split("|") if part.strip()]
        normalized = " | ".join(parts) if parts else line
        line_lower = normalized.lower()
        for label in label_lookup:
            if parts and parts[0].lower() == label and len(parts) > 1:
                return parts[1].strip()
            if line_lower.startswith(label):
                remainder = normalized[len(label) :].lstrip(" :|-").strip()
                if remainder:
                    return remainder
    return None


def _join_text_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _extract_action_family(text: str) -> str | None:
    lowered = str(text or "").lower()
    for action in _ACTION_FAMILY_VALUES:
        if action in lowered:
            return action
    if re.search(r"\brun\s+`?[a-z_]+`?\s+capability\b", lowered):
        return "run_capability"
    return None


def _extract_target_capability(text: str) -> str | None:
    labeled = _extract_labeled_value(text, ["Target Capability", "Capability"])
    if labeled:
        token = re.sub(r"[^A-Za-z0-9_:-].*$", "", labeled).strip("` ").strip()
        if token:
            return token
    patterns = (
        r"\brun\s+`?([a-z_][a-z0-9_]*)`?\s+capability\b",
        r"[→>-]\s*`?([a-z_][a-z0-9_]*)`?\b",
        r"\bnext pending capability\b[^A-Za-z0-9_`-]*`?([a-z_][a-z0-9_]*)`?\b",
    )
    lowered = str(text or "").lower()
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            token = str(match.group(1) or "").strip()
            if token:
                return token
    return None


def _extract_proposal_id(text: str) -> str:
    labeled = _extract_labeled_value(text, ["Selected Proposal", "Source Proposal", "Proposal", "ID"])
    if labeled:
        token = str(labeled).strip().strip("`")
        if token:
            return token
    match = re.search(r"`([^`]*::[^`]*)`", str(text or ""))
    return str(match.group(1) or "").strip() if match else ""


def _extract_channels(text: str) -> list[str]:
    channels: list[str] = []
    for token in re.findall(r"(?:^|[^A-Za-z])([xy])(?:[^A-Za-z]|$)", str(text or "").lower()):
        item = str(token or "").strip()
        if item in {"x", "y"} and item not in channels:
            channels.append(item)
    return channels


def _proposal_payload_from_text(text: str) -> dict[str, Any] | None:
    payload_text = str(text or "").strip()
    if not payload_text:
        return None
    action_family = _extract_action_family(payload_text)
    target_capability = _extract_target_capability(payload_text)
    if not action_family and target_capability:
        action_family = "run_capability"
    if not action_family and not target_capability:
        return None
    rationale = _extract_labeled_value(payload_text, ["Rationale", "Decision Rationale"]) or _strip_markdown_formatting(
        payload_text
    )[:1200]
    selected_skill = _extract_labeled_value(payload_text, ["Selected Skill", "Skill"])
    proposal_id = _extract_proposal_id(payload_text)
    return {
        "proposal_id": proposal_id,
        "action_family": action_family,
        "target_capability": target_capability,
        "selected_skill": str(selected_skill or "").strip() or None,
        "rationale": rationale,
    }


def _review_bundle_from_text(text: str) -> dict[str, Any]:
    payload_text = str(text or "").strip()
    if not payload_text:
        return {"critiques": [], "preferences": []}
    lowered = payload_text.lower()
    stance = "critique"
    if any(marker in lowered for marker in ("approved", "approve", "accepted", "support")):
        stance = "support"
    elif any(marker in lowered for marker in ("reject", "rejected", "objection", "block")):
        stance = "objection"
    proposal_id = _extract_proposal_id(payload_text)
    critique = {
        "agent_name": "critic",
        "proposal_id": proposal_id,
        "message_type": stance,
        "stance": stance,
        "concerns": [],
        "recommendation": _strip_markdown_formatting(payload_text)[:1200],
        "confidence": 0.5,
    }
    preference_id = ""
    if stance == "support":
        preference_id = proposal_id
    return {
        "critiques": [critique],
        "preferences": (
            [
                {
                    "agent_name": "critic",
                    "preferred_proposal_id": preference_id,
                    "preference_strength": 0.5,
                    "reason": critique["recommendation"],
                    "confidence": 0.5,
                }
            ]
            if preference_id
            else []
        ),
    }


def _arbitration_payload_from_text(text: str) -> dict[str, Any] | None:
    payload_text = str(text or "").strip()
    if not payload_text:
        return None
    selected_proposal_id = _extract_proposal_id(payload_text)
    action_family = _extract_action_family(payload_text)
    target_capability = _extract_target_capability(payload_text)
    rationale = _extract_labeled_value(payload_text, ["Rationale", "Decision Rationale"]) or _strip_markdown_formatting(
        payload_text
    )[:1500]
    if not selected_proposal_id and not action_family and not target_capability:
        return None
    selected_action: dict[str, Any] | None = None
    if action_family or target_capability:
        selected_action = {
            "action_family": action_family or ("run_capability" if target_capability else None),
            "target_capability": target_capability,
            "source_proposal_id": selected_proposal_id or None,
            "rationale": rationale,
        }
    return {
        "selected_proposal_id": selected_proposal_id or None,
        "selected_action": selected_action,
        "rationale": rationale,
    }


def _refinement_payload_from_text(text: str) -> dict[str, Any] | None:
    payload_text = str(text or "").strip()
    if not payload_text:
        return None
    lowered = payload_text.lower()
    decision = None
    if "refine_more_points" in lowered or "refine more" in lowered or "supply points" in lowered:
        decision = "refine_more_points"
    elif "reject_channel" in lowered or ("reject" in lowered and "channel" in lowered):
        decision = "reject_channel"
    elif "terminate" in lowered or "stop" in lowered:
        decision = "terminate"
    elif "escalate" in lowered:
        decision = "escalate"
    elif "accept" in lowered:
        decision = "accept"
    if not decision:
        return None
    return {
        "decision": decision,
        "target_channels": _extract_channels(payload_text),
        "reason": _strip_markdown_formatting(payload_text)[:800],
    }


def _report_summary_from_text(text: str) -> dict[str, Any]:
    payload_text = _strip_markdown_formatting(text)
    return {
        "reason": "report_summary_recovered_from_text",
        "confidence": 0.5,
        "final_summary": {"narrative_report": payload_text[:4000]},
        "key_findings": [],
        "artifact_references": {},
    }


def _batch_summary_from_text(text: str) -> dict[str, Any]:
    payload_text = _strip_markdown_formatting(text)
    fields = {
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "scientifically_passed": 0,
        "scientifically_warning": 0,
        "scientifically_failed": 0,
        "scientifically_unknown": 0,
    }
    for key in list(fields.keys()):
        match = re.search(rf"\b{re.escape(key)}\b[^0-9-]*(-?\d+)", payload_text, flags=re.IGNORECASE)
        if match:
            fields[key] = int(match.group(1))
    return {
        **fields,
        "common_failure_stages": {},
        "outcomes": [],
    }


def _now_iso() -> str:
    return datetime.now(_UTC).isoformat().replace("+00:00", "Z")


MessageType = Literal[
    "proposal",
    "critique",
    "support",
    "objection",
    "preference",
    "arbitration",
    "reflection",
    "execution_command",
    "execution_observation",
]


ActionFamily = Literal[
    "run_capability",
    "retry_capability",
    "rerun_from_capability",
    "repair_execution_context",
    "refine_sampling",
    "revalidate_result",
    "invalidate_channel",
    "skip_channel",
    "escalate_human",
    "finalize_material",
    "abort_material",
]


class AgentDecisionBase(BaseModel):
    reason: str = ""
    confidence: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    should_escalate: bool = False

    @model_validator(mode="after")
    def _normalize(self):
        self.reason = str(self.reason or "").strip()
        self.warnings = [str(item) for item in dedupe_keep_order(self.warnings or [])]
        return self


class AgentMessage(BaseModel):
    agent_name: str
    message_type: MessageType
    round_id: int = 0
    target_task_id: str = ""
    content: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    priority: int = 0
    cost_estimate: float = 0.0
    risk_estimate: float = 0.0
    timestamp: str = Field(default_factory=_now_iso)

    @model_validator(mode="after")
    def _normalize(self):
        self.agent_name = str(self.agent_name or "").strip()
        self.target_task_id = str(self.target_task_id or "").strip()
        self.content = dict(self.content or {})
        self.evidence_refs = [str(item) for item in dedupe_keep_order(self.evidence_refs or [])]
        self.priority = int(self.priority or 0)
        return self


class Proposal(AgentMessage):
    message_type: Literal["proposal"] = "proposal"
    agent_name: str = ""
    proposal_id: str = ""
    action_family: ActionFamily = "run_capability"
    target_capability: str | None = Field(default=None, validation_alias=AliasChoices("target_capability", "capability"))
    selected_skill: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_benefit: str = ""
    expected_risk: str = ""
    rationale: str = ""
    expected_observation: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    fallback_if_failed: list[str] = Field(default_factory=list)
    submit_external_job: bool = False
    wait_for_event_after_submission: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_proposal_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                parsed = _proposal_payload_from_text(data)
                return parsed or data
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "action_family" not in payload and payload.get("action") is not None:
            payload["action_family"] = payload.get("action")
        if "target_capability" not in payload and payload.get("capability") is not None:
            payload["target_capability"] = payload.get("capability")
        if "selected_skill" not in payload and payload.get("skill") is not None:
            payload["selected_skill"] = payload.get("skill")
        if "target_capability" not in payload:
            payload["target_capability"] = payload.get("target") or payload.get("selected_capability")
        if "action_family" not in payload and payload.get("target_capability"):
            payload["action_family"] = "run_capability"
        if isinstance(payload.get("rationale"), list):
            payload["rationale"] = _join_text_list(payload.get("rationale"))
        if isinstance(payload.get("expected_observation"), list):
            payload["expected_observation"] = _join_text_list(payload.get("expected_observation"))
        if not payload.get("success_criteria") and payload.get("expected_artifacts") is not None:
            payload["success_criteria"] = _coerce_string_list(payload.get("expected_artifacts"))
        if not payload.get("fallback_if_failed") and payload.get("fallback") is not None:
            payload["fallback_if_failed"] = _coerce_string_list(payload.get("fallback"))
        parameters = dict(payload.get("parameters") or {}) if isinstance(payload.get("parameters"), dict) else {}
        if not parameters:
            channel_alias = payload.get("target_channels") or payload.get("channels")
            if channel_alias is not None:
                parameters["target_channels"] = _coerce_string_list(channel_alias)
            suggested_points = payload.get("suggested_points") or payload.get("suggested_strain_points")
            if isinstance(suggested_points, dict):
                parameters["suggested_points"] = suggested_points
        if parameters:
            payload["parameters"] = parameters
        return payload

    @model_validator(mode="after")
    def _normalize_proposal(self):
        self.parameters = dict(self.parameters or {})
        self.expected_benefit = str(self.expected_benefit or "").strip()
        self.expected_risk = str(self.expected_risk or "").strip()
        self.rationale = str(self.rationale or "").strip()
        self.expected_observation = str(self.expected_observation or "").strip()
        self.success_criteria = [str(item) for item in dedupe_keep_order(self.success_criteria or [])]
        self.fallback_if_failed = [str(item) for item in dedupe_keep_order(self.fallback_if_failed or [])]
        self.submit_external_job = bool(self.submit_external_job)
        self.wait_for_event_after_submission = bool(self.wait_for_event_after_submission)
        return self


class Critique(AgentMessage):
    message_type: Literal["critique", "support", "objection"] = "critique"
    agent_name: str = ""
    proposal_id: str = ""
    stance: Literal["support", "objection", "critique"] = "critique"
    concerns: list[str] = Field(default_factory=list)
    recommendation: str = ""

    @model_validator(mode="after")
    def _normalize_critique(self):
        self.concerns = [str(item) for item in dedupe_keep_order(self.concerns or [])]
        self.recommendation = str(self.recommendation or "").strip()
        if self.message_type == "support":
            self.stance = "support"
        elif self.message_type == "objection":
            self.stance = "objection"
        return self


class Preference(AgentMessage):
    message_type: Literal["preference"] = "preference"
    agent_name: str = ""
    preferred_proposal_id: str = ""
    preference_strength: float = 0.5
    reason: str = ""

    @model_validator(mode="after")
    def _normalize_preference(self):
        self.reason = str(self.reason or "").strip()
        return self


class SelectedAction(BaseModel):
    action_family: ActionFamily
    target_capability: str | None = None
    selected_skill: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_proposal_id: str | None = None
    rationale: str = ""
    cost_class: str = "medium"
    risk_class: str = "medium"
    expected_observation: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    fallback_if_failed: list[str] = Field(default_factory=list)
    submit_external_job: bool = False
    wait_for_event_after_submission: bool = False
    supporting_agent_opinions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_selected_action_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                parsed = _arbitration_payload_from_text(data)
                if parsed and parsed.get("selected_action") is not None:
                    return parsed.get("selected_action")
                return data
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "action_family" not in payload and payload.get("action") is not None:
            payload["action_family"] = payload.get("action")
        if "target_capability" not in payload and payload.get("capability") is not None:
            payload["target_capability"] = payload.get("capability")
        if "selected_skill" not in payload and payload.get("skill") is not None:
            payload["selected_skill"] = payload.get("skill")
        if "action_family" not in payload and payload.get("target_capability"):
            payload["action_family"] = "run_capability"
        if isinstance(payload.get("rationale"), list):
            payload["rationale"] = _join_text_list(payload.get("rationale"))
        if isinstance(payload.get("expected_observation"), list):
            payload["expected_observation"] = _join_text_list(payload.get("expected_observation"))
        parameters = dict(payload.get("parameters") or {}) if isinstance(payload.get("parameters"), dict) else {}
        if not parameters:
            channel_alias = payload.get("target_channels") or payload.get("channels")
            if channel_alias is not None:
                parameters["target_channels"] = _coerce_string_list(channel_alias)
            suggested_points = payload.get("suggested_points") or payload.get("suggested_strain_points")
            if isinstance(suggested_points, dict):
                parameters["suggested_points"] = suggested_points
        if parameters:
            payload["parameters"] = parameters
        return payload

    @model_validator(mode="after")
    def _normalize(self):
        self.parameters = dict(self.parameters or {})
        self.rationale = str(self.rationale or "").strip()
        self.expected_observation = str(self.expected_observation or "").strip()
        self.success_criteria = [str(item) for item in dedupe_keep_order(self.success_criteria or [])]
        self.fallback_if_failed = [str(item) for item in dedupe_keep_order(self.fallback_if_failed or [])]
        self.supporting_agent_opinions = [str(item) for item in dedupe_keep_order(self.supporting_agent_opinions or [])]
        self.submit_external_job = bool(self.submit_external_job)
        self.wait_for_event_after_submission = bool(self.wait_for_event_after_submission)
        return self


class ArbitrationRecord(AgentMessage):
    message_type: Literal["arbitration"] = "arbitration"
    selected_proposal_id: str | None = None
    selected_action: SelectedAction | None = None
    rejected_proposals: list[str] = Field(default_factory=list)
    guardrail_notes: list[str] = Field(default_factory=list)
    rationale: str = ""
    disagreement_summary: list[str] = Field(default_factory=list)
    whether_noop: bool = False
    whether_waiting_external: bool = False
    whether_ready_to_finalize: bool = False

    @model_validator(mode="after")
    def _normalize_arbitration(self):
        self.rejected_proposals = [str(item) for item in dedupe_keep_order(self.rejected_proposals or [])]
        self.guardrail_notes = [str(item) for item in dedupe_keep_order(self.guardrail_notes or [])]
        self.disagreement_summary = [str(item) for item in dedupe_keep_order(self.disagreement_summary or [])]
        self.rationale = str(self.rationale or "").strip()
        self.whether_noop = bool(self.whether_noop)
        self.whether_waiting_external = bool(self.whether_waiting_external)
        self.whether_ready_to_finalize = bool(self.whether_ready_to_finalize)
        return self


class ExecutionCommand(AgentMessage):
    message_type: Literal["execution_command"] = "execution_command"
    action_family: ActionFamily
    target_capability: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_artifacts: list[str] = Field(default_factory=list)
    dependency_snapshot: dict[str, Any] = Field(default_factory=dict)
    submit_external_job: bool = False
    wait_for_event_after_submission: bool = False

    @model_validator(mode="after")
    def _normalize_command(self):
        self.parameters = dict(self.parameters or {})
        self.expected_artifacts = [str(item) for item in dedupe_keep_order(self.expected_artifacts or [])]
        self.dependency_snapshot = dict(self.dependency_snapshot or {})
        self.submit_external_job = bool(self.submit_external_job)
        self.wait_for_event_after_submission = bool(self.wait_for_event_after_submission)
        return self


class ExecutionObservation(AgentMessage):
    message_type: Literal["execution_observation"] = "execution_observation"
    action_family: ActionFamily
    target_capability: str | None = None
    status: Literal["success", "failed", "skipped", "completed", "running"] = "success"
    error_summary: str | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    result_summary: dict[str, Any] = Field(default_factory=dict)
    raw_evidence: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_observation(self):
        self.artifact_paths = {str(k): str(v) for k, v in dict(self.artifact_paths or {}).items() if v}
        self.result_summary = dict(self.result_summary or {})
        self.raw_evidence = dict(self.raw_evidence or {})
        if self.error_summary is not None:
            self.error_summary = str(self.error_summary)
        return self


class ReflectionRecord(AgentMessage):
    message_type: Literal["reflection"] = "reflection"
    selected_action: dict[str, Any] = Field(default_factory=dict)
    tradeoff_summary: list[str] = Field(default_factory=list)
    failure_pattern: str | None = None
    follow_up_tasks: list[dict[str, Any]] = Field(default_factory=list)
    continue_deliberation: bool = True

    @model_validator(mode="after")
    def _normalize_reflection(self):
        self.selected_action = dict(self.selected_action or {})
        self.tradeoff_summary = [str(item) for item in dedupe_keep_order(self.tradeoff_summary or [])]
        self.follow_up_tasks = [dict(item) for item in list(self.follow_up_tasks or [])]
        if self.failure_pattern is not None:
            self.failure_pattern = str(self.failure_pattern)
        return self


class ProposalBundle(BaseModel):
    proposals: list[Proposal] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_bundle(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                parsed = _proposal_payload_from_text(data)
                return {"proposals": [parsed]} if parsed else {"proposals": []}
        if isinstance(data, list):
            return {"proposals": list(data)}
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "proposals" in payload:
            proposals = payload.get("proposals")
            if isinstance(proposals, dict):
                payload["proposals"] = [proposals]
            elif not isinstance(proposals, list):
                payload["proposals"] = []
            return payload
        if any(
            key in payload
            for key in ("action_family", "action", "target_capability", "capability", "selected_skill", "skill")
        ):
            return {"proposals": [payload]}
        return payload


class ReviewBundle(BaseModel):
    critiques: list[Critique] = Field(default_factory=list)
    preferences: list[Preference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_review_bundle(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                return _review_bundle_from_text(data)
        if isinstance(data, list):
            preference_like = []
            critique_like = []
            for item in list(data or []):
                if not isinstance(item, dict):
                    continue
                if any(key in item for key in ("preferred_proposal_id", "selected_proposal", "preference_strength")):
                    preference_like.append(item)
                else:
                    critique_like.append(item)
            return {"critiques": critique_like, "preferences": preference_like}
        if not isinstance(data, dict):
            return {"critiques": [], "preferences": []}
        payload = dict(data)
        raw_critiques = payload.get("critiques")
        if isinstance(raw_critiques, dict):
            raw_critiques = [raw_critiques]
        if not isinstance(raw_critiques, list):
            raw_critiques = []

        normalized_critiques: list[dict[str, Any]] = []
        for item in raw_critiques:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            verdict = str(candidate.get("verdict") or candidate.get("stance") or "").strip().lower()
            if verdict in {"support", "approve", "approved", "proceed", "pass"}:
                stance = "support"
            elif verdict in {"objection", "reject", "rejected", "fail", "block"}:
                stance = "objection"
            else:
                stance = "critique"
            concerns: list[str] = []
            for entry in list(candidate.get("concerns", []) or []):
                text = str(entry or "").strip()
                if text:
                    concerns.append(text)
            for entry in list(candidate.get("obvious_concerns", []) or []):
                text = str(entry or "").strip()
                if text:
                    concerns.append(text)
            evidence_assessment = dict(candidate.get("evidence_assessment", {}) or {})
            for key in ("missing_evidence", "weak_inference", "assumption_hazards"):
                for entry in list(evidence_assessment.get(key, []) or []):
                    text = str(entry or "").strip()
                    if text:
                        concerns.append(f"{key}:{text}")
            normalized_critiques.append(
                {
                    "agent_name": str(candidate.get("agent_name") or "critic"),
                    "proposal_id": str(candidate.get("proposal_id") or candidate.get("target_proposal_id") or ""),
                    "message_type": str(candidate.get("message_type") or stance),
                    "stance": stance,
                    "concerns": dedupe_keep_order(concerns),
                    "recommendation": str(candidate.get("recommendation") or candidate.get("rationale") or ""),
                    "confidence": candidate.get("confidence", 0.5),
                }
            )

        raw_preferences = payload.get("preferences")
        if isinstance(raw_preferences, dict):
            raw_preferences = [raw_preferences]
        if not isinstance(raw_preferences, list):
            raw_preferences = []

        normalized_preferences: list[dict[str, Any]] = []
        for item in raw_preferences:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            preferred = str(
                candidate.get("preferred_proposal_id")
                or candidate.get("selected_proposal")
                or candidate.get("proposal_id")
                or ""
            ).strip()
            strength = candidate.get("preference_strength", candidate.get("conservative_confidence", 0.5))
            normalized_preferences.append(
                {
                    "agent_name": str(candidate.get("agent_name") or "critic"),
                    "preferred_proposal_id": preferred,
                    "preference_strength": strength,
                    "reason": str(
                        candidate.get("reason")
                        or candidate.get("tiebreaker_rationale")
                        or candidate.get("recommendation")
                        or ""
                    ),
                    "confidence": candidate.get("confidence", 0.5),
                }
            )

        payload["critiques"] = normalized_critiques
        payload["preferences"] = normalized_preferences
        return payload


class ArbitrationDecisionPayload(BaseModel):
    selected_proposal_id: str | None = None
    selected_action: SelectedAction | None = None
    rejected_proposal_ids: list[str] = Field(default_factory=list)
    guardrail_notes: list[str] = Field(default_factory=list)
    rationale: str = ""
    disagreement_summary: list[str] = Field(default_factory=list)
    whether_noop: bool = False
    whether_waiting_external: bool = False
    whether_ready_to_finalize: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_arbitration_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                return _arbitration_payload_from_text(data) or data
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "selected_proposal_id" not in payload and payload.get("selected_proposal") is not None:
            payload["selected_proposal_id"] = payload.get("selected_proposal")
        if "rejected_proposal_ids" not in payload and payload.get("rejected_proposals") is not None:
            payload["rejected_proposal_ids"] = payload.get("rejected_proposals")
        if "rationale" not in payload and payload.get("reason") is not None:
            payload["rationale"] = payload.get("reason")
        if "selected_action" not in payload and any(
            key in payload for key in ("action_family", "action", "target_capability", "capability")
        ):
            payload["selected_action"] = {
                "action_family": payload.get("action_family") or payload.get("action"),
                "target_capability": payload.get("target_capability") or payload.get("capability"),
                "selected_skill": payload.get("selected_skill") or payload.get("skill"),
                "source_proposal_id": payload.get("selected_proposal_id"),
                "rationale": payload.get("rationale") or payload.get("reason") or "",
            }
        return payload


class AdmissionDecision(AgentDecisionBase):
    decision: Literal["continue", "continue_with_warning", "reject"]
    preflight_tags: list[str] = Field(default_factory=list)


class RecoveryDecision(AgentDecisionBase):
    decision: Literal[
        "retry_current_stage",
        "rerun_previous_stage",
        "modify_params_and_retry",
        "copy_contcar_to_poscar_and_retry",
        "manual_fix_resume",
        "skip_point",
        "skip_material",
        "abort_task",
    ]
    parameter_updates: dict[str, Any] = Field(default_factory=dict)
    file_updates: dict[str, Any] = Field(default_factory=dict)
    return_stage: str | None = None
    cleanup_policy: str | None = None


class RefinementDecision(AgentDecisionBase):
    decision: Literal["accept", "refine_more_points", "reject_channel", "terminate", "escalate"]
    target_channels: list[str] = Field(default_factory=list)
    suggested_points: dict[str, list[float]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_refinement_payload(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                return _refinement_payload_from_text(data) or data
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "decision" not in payload and payload.get("action") is not None:
            payload["decision"] = payload.get("action")
        if "target_channels" not in payload:
            payload["target_channels"] = (
                payload.get("channels")
                or payload.get("target_directions")
                or payload.get("directions")
                or payload.get("target_channel")
            )
        if "suggested_points" not in payload and payload.get("suggested_strain_points") is not None:
            payload["suggested_points"] = payload.get("suggested_strain_points")
        if "suggested_points" not in payload and payload.get("supply_points") is not None:
            payload["suggested_points"] = payload.get("supply_points")
        if "reason" not in payload and payload.get("rationale") is not None:
            payload["reason"] = payload.get("rationale")
        return payload


class ValidationDecision(AgentDecisionBase):
    decision: Literal["pass", "pass_with_warning", "fail", "escalate"]
    failed_checks: list[str] = Field(default_factory=list)


class HumanEscalationDecision(AgentDecisionBase):
    should_interrupt: bool = True
    recommended_options: list[str] = Field(default_factory=list)
    default_timeout_action: str = "skip_material"


class ReportSummary(AgentDecisionBase):
    final_summary: dict[str, Any] = Field(default_factory=dict)
    key_findings: list[str] = Field(default_factory=list)
    artifact_references: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_report_summary(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                return _report_summary_from_text(data)
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "final_summary" not in payload:
            known = {"reason", "confidence", "warnings", "should_escalate", "key_findings", "artifact_references"}
            inferred = {key: value for key, value in payload.items() if key not in known}
            payload["final_summary"] = dict(inferred or {})
        if "artifact_references" not in payload and isinstance(payload.get("final_summary"), dict):
            payload["artifact_references"] = dict(payload["final_summary"].get("artifact_paths", {}) or {})
        return payload


class ManualFixPreview(BaseModel):
    modified_files: list[str] = Field(default_factory=list)
    requested_resume_strategy: str
    computed_resume_stage: str
    cleanup_policy: str
    invalidated_stages: list[str] = Field(default_factory=list)
    invalidated_artifacts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self):
        self.modified_files = [str(item) for item in dedupe_keep_order(self.modified_files or [])]
        self.invalidated_stages = [str(item) for item in dedupe_keep_order(self.invalidated_stages or [])]
        self.invalidated_artifacts = [str(item) for item in dedupe_keep_order(self.invalidated_artifacts or [])]
        self.warnings = [str(item) for item in dedupe_keep_order(self.warnings or [])]
        return self


class ManualFixInstruction(BaseModel):
    action: Literal["manual_fix_resume"] = "manual_fix_resume"
    modified_files: list[str] = Field(default_factory=list)
    modification_type: str = "custom"
    requested_resume_strategy: str = "default_rule"
    resume_stage: str
    cleanup_policy: str
    invalidated_stages: list[str] = Field(default_factory=list)
    invalidated_artifacts: list[str] = Field(default_factory=list)
    preview: ManualFixPreview | None = None
    reason: str = "user_manual_fix"

    @model_validator(mode="after")
    def _normalize(self):
        self.modified_files = [str(item) for item in dedupe_keep_order(self.modified_files or [])]
        self.invalidated_stages = [str(item) for item in dedupe_keep_order(self.invalidated_stages or [])]
        self.invalidated_artifacts = [str(item) for item in dedupe_keep_order(self.invalidated_artifacts or [])]
        if self.preview is not None:
            if not self.invalidated_stages:
                self.invalidated_stages = list(self.preview.invalidated_stages)
            if not self.invalidated_artifacts:
                self.invalidated_artifacts = list(self.preview.invalidated_artifacts)
            if not self.modified_files:
                self.modified_files = list(self.preview.modified_files)
            if not self.requested_resume_strategy:
                self.requested_resume_strategy = self.preview.requested_resume_strategy
            if not self.resume_stage:
                self.resume_stage = self.preview.computed_resume_stage
            if not self.cleanup_policy:
                self.cleanup_policy = self.preview.cleanup_policy
        return self


class HITLDecision(BaseModel):
    action: Literal[
        "retry_current_stage",
        "rerun_previous_stage",
        "modify_params_and_retry",
        "copy_contcar_to_poscar_and_retry",
        "manual_fix_resume",
        "skip_point",
        "skip_material",
        "abort_task",
    ]
    instruction: ManualFixInstruction | None = None
    reason: str = ""
    source: Literal["interactive", "response_file", "timeout_default", "precomputed"] = "precomputed"
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self):
        self.warnings = [str(item) for item in dedupe_keep_order(self.warnings or [])]
        self.reason = str(self.reason or "").strip()
        if self.action == "manual_fix_resume" and self.instruction is None:
            raise ValueError("manual_fix_resume_requires_instruction")
        return self


class BatchSummary(BaseModel):
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    scientifically_passed: int = 0
    scientifically_warning: int = 0
    scientifically_failed: int = 0
    scientifically_unknown: int = 0
    common_failure_stages: dict[str, int] = Field(default_factory=dict)
    outcomes: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_batch_summary(cls, data: Any) -> Any:
        if isinstance(data, str):
            extracted = _extract_json_payload_from_text(data)
            if extracted is not None:
                data = extracted
            else:
                return _batch_summary_from_text(data)
        return data
