from __future__ import annotations

import json
import math
import os
import re
from json import JSONDecoder
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..agents.llm_client import build_llm_client, llm_request_guard
from ..rag.wiki_sync import load_house_policy_documents
from ..runtime.context import RuntimeContext
from .probe import build_stage_probe_from_state
from .retrieval import PolicyKnowledgeBase
from .schemas import FailureDiagnosis, ParameterPlan, RetrievedEvidence, StageProbe

_INT_FIELDS = {"ENCUT", "NELM", "IBRION", "ISIF", "NSW", "ISMEAR", "ISYM", "KPAR", "NCORE", "NPAR", "line_mode_density", "mpi_ranks", "omp_threads"}
_FLOAT_FIELDS = {"EDIFF", "SIGMA", "POTIM", "EDIFFG", "target_ka"}
_BOOL_FIELDS = {"LASPH", "ADDGRID", "LVTOT", "LVHAR", "gamma_centered"}
_STRING_FIELDS = {"PREC", "ALGO", "LREAL", "gpu_binding"}
_COMMON_INCAR_KEYS = {"ENCUT", "EDIFF", "NELM", "PREC", "ALGO", "ISMEAR", "SIGMA", "LASPH", "ADDGRID", "LREAL", "ISYM", "KPAR", "NCORE", "NPAR"}
_STAGE_ALLOWED_KEYS = {
    "relax": _COMMON_INCAR_KEYS | {"IBRION", "ISIF", "NSW", "POTIM", "EDIFFG"},
    "scf": _COMMON_INCAR_KEYS | {"LVTOT", "LVHAR"},
    "band": _COMMON_INCAR_KEYS,
    "effective_mass": _COMMON_INCAR_KEYS,
}
_DEFAULT_ALLOWED_ACTIONS = ["retry_capability", "rerun_from_capability", "repair_execution_context", "escalate_human", "abort_material"]
_RELAX_FAILURE_EXACT = {
    "relax_failed",
    "relax_retry_fatal",
    "relax_retry_limit_reached",
    "relax_nonconverged",
    "zbrent_fatal",
}
_RELAX_FAILURE_MARKERS = (
    "RELAX_FAILED",
    "RELAX_RETRY_FATAL",
    "CONTCAR_MISSING",
    "结构弛豫失败",
    "弛豫失败",
)


def _knowledge_base_from_runtime(runtime: RuntimeContext) -> PolicyKnowledgeBase:
    house_policy_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "corpus", "house_policy.json")
    )
    fallback_documents = [item.model_dump(mode="json") for item in load_house_policy_documents(house_policy_path)]
    return PolicyKnowledgeBase(
        database_uri=runtime.resolved_db_uri,
        embedding_model=runtime.embedding_model,
        embedding_base_url=runtime.embedding_base_url,
        embedding_api_key=runtime.embedding_api_key,
        qa_model=runtime.wiki_qa_model or runtime.agent_runtime.llm_model,
        qa_base_url=runtime.agent_runtime.llm_base_url or "",
        qa_api_key=runtime.agent_runtime.llm_api_key or "",
        rag_top_k=runtime.rag_top_k,
        chunk_size=runtime.rag_chunk_size,
        chunk_overlap=runtime.rag_chunk_overlap,
        reindex_batch_size=runtime.rag_reindex_batch_size,
        strict_rag=bool(runtime.rag_required),
        fallback_documents=fallback_documents,
    )

def _extract_first_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    if not candidate:
        raise ValueError("empty_llm_response")
    decoder = JSONDecoder()
    start_positions = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0]
    if not start_positions:
        raise ValueError("json_payload_not_found")
    start = min(start_positions)
    parsed, _ = decoder.raw_decode(candidate[start:])
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("json_payload_not_dict")


def _query_from_probe(probe: StageProbe) -> str:
    parts = [
        probe.stage,
        probe.material_id,
        str(probe.composition or ""),
        json.dumps(probe.structure_summary, ensure_ascii=False),
        json.dumps(probe.extra_context, ensure_ascii=False),
        json.dumps(probe.prior_execution_summary, ensure_ascii=False),
    ]
    return "\n".join([part for part in parts if str(part).strip()])


def _historical_evidence(state_payload: dict[str, Any], *, stage: str) -> list[RetrievedEvidence]:
    state = dict(state_payload or {})
    recovered = list((state.get("memory", {}) or {}).get("recovered_case_patterns", []) or [])
    evidence: list[RetrievedEvidence] = []
    for idx, item in enumerate(recovered[:3], start=1):
        if not isinstance(item, dict):
            continue
        item_stage = str(item.get("stage") or "")
        if item_stage and item_stage != stage:
            continue
        snippet = json.dumps(item, ensure_ascii=False)
        evidence.append(
            RetrievedEvidence(
                corpus="historical_case",
                source_id=str(item.get("task_id") or f"case-{idx}"),
                title=f"Historical recovery case {idx}",
                url_or_path="memory://recovered_case_patterns",
                heading=item_stage or stage,
                snippet=snippet[:420],
                score=1.0,
                tags=[item_stage] if item_stage else [],
            )
        )
    return evidence


def _string_has_relax_failure(value: str) -> bool:
    text = str(value or "")
    if not text:
        return False
    if any(marker in text for marker in _RELAX_FAILURE_MARKERS):
        return True
    lowered = text.lower()
    if any(marker in lowered for marker in _RELAX_FAILURE_EXACT):
        return True
    if "relax" in lowered and "failed" in lowered:
        return True
    if "relax" in lowered and "missing" in lowered and "contcar" in lowered:
        return True
    if "contcar" in lowered and ("missing" in lowered or "不存在" in text):
        return True
    return False


def _dict_has_relax_failure(value: dict[str, Any]) -> bool:
    payload = dict(value or {})
    stage_hint = " ".join(
        str(payload.get(key) or "")
        for key in ("stage", "substage", "task_scope", "target_capability", "capability")
    ).lower()
    error_hint = " ".join(
        str(payload.get(key) or "")
        for key in ("error", "error_type", "error_summary", "trigger_pattern", "message")
    )
    if _string_has_relax_failure(error_hint):
        return True
    if "relax" in stage_hint and any(
        str(payload.get(key) or "").strip().lower() in _RELAX_FAILURE_EXACT
        for key in ("error", "error_type", "trigger_pattern")
    ):
        return True
    if "relax" in stage_hint and "contcar" in error_hint.lower():
        return True
    return False


def _payload_has_relax_failure(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> bool:
    if depth > 8:
        return False
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple, set)):
        marker = id(value)
        if marker in seen:
            return False
        seen.add(marker)
    if isinstance(value, dict):
        if _dict_has_relax_failure(value):
            return True
        return any(_payload_has_relax_failure(item, depth=depth + 1, seen=seen) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_payload_has_relax_failure(item, depth=depth + 1, seen=seen) for item in value)
    if isinstance(value, str):
        return _string_has_relax_failure(value)
    return False


def has_relax_failure_signature(
    *,
    stage: str,
    latest_failure: dict[str, Any] | None,
    state_payload: dict[str, Any] | None,
) -> bool:
    """Return True only for execution evidence that points to relaxation failure."""

    normalized_stage = str(stage or "").strip()
    state = dict(state_payload or {})
    failure = dict(latest_failure or {})
    candidates: list[Any] = [failure]
    for section_name in ("blackboard", "execution"):
        section = dict(state.get(section_name, {}) or {})
        observation = section.get("latest_execution_observation")
        if observation:
            candidates.append(observation)
    if state.get("last_observation"):
        candidates.append(state.get("last_observation"))

    if normalized_stage in {"relax", "strain_loop"}:
        physics = dict(state.get("physics_results", {}) or {})
        for key in ("strain_data", "strain_recovery_events", "strain_results"):
            if physics.get(key):
                candidates.append(physics.get(key))
        diagnostics = dict(state.get("diagnostics", {}) or {})
        for key in ("last_error", "recovery_summary", "recovery_events"):
            if diagnostics.get(key):
                candidates.append(diagnostics.get(key))

    return any(_payload_has_relax_failure(candidate) for candidate in candidates)


class AgenticPolicyEngine:
    def __init__(self, runtime: RuntimeContext, *, knowledge_base: PolicyKnowledgeBase | None = None):
        self.runtime = runtime
        self.knowledge_base = knowledge_base or _knowledge_base_from_runtime(runtime)
        self._planner_llm, self._planner_reason = build_llm_client(runtime.agent_runtime, role="planner", require_real=True)
        self._recovery_llm, self._recovery_reason = build_llm_client(runtime.agent_runtime, role="recovery", require_real=True)

    def _sanitize_map(self, payload: dict[str, Any], *, allowed: set[str]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in dict(payload or {}).items():
            name = str(key or "").strip()
            if not name or name not in allowed:
                continue
            try:
                if name in _INT_FIELDS:
                    clean[name] = int(value)
                elif name in _FLOAT_FIELDS:
                    clean[name] = float(value)
                elif name in _BOOL_FIELDS:
                    clean[name] = bool(value)
                elif name in _STRING_FIELDS:
                    clean[name] = str(value)
                else:
                    clean[name] = value
            except Exception:
                continue
        return clean

    def _call_json_schema(self, *, llm: Any, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError("llm_unavailable")
        with llm_request_guard(self.runtime.agent_runtime, role="planner"):
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
                ]
            )
        return _extract_first_json_object(getattr(response, "content", ""))

    def plan_stage(
        self,
        *,
        stage: str,
        state_payload: dict[str, Any],
        default_incar: dict[str, Any],
        default_kpoints_policy: dict[str, Any],
        extra_context: dict[str, Any] | None = None,
    ) -> ParameterPlan:
        probe = build_stage_probe_from_state(state_payload, stage=stage, extra_context=extra_context)
        retrieval_error_type = ""
        try:
            evidence = self.knowledge_base.retrieve(
                query=_query_from_probe(probe),
                stage=stage,
                top_k=self.runtime.policy_retrieval_top_k,
                corpora=("house_policy", "vasp_wiki"),
            )
        except Exception as exc:
            retrieval_error_type = type(exc).__name__
            evidence = _historical_evidence(state_payload, stage=stage)
        fallback = ParameterPlan(
            stage=stage,
            source="fallback",
            incar_overrides={},
            kpoints_policy=dict(default_kpoints_policy or {}),
            runtime_policy={},
            evidence_refs=[item.reference for item in evidence],
            evidence_items=evidence,
            house_rule_refs=[item.reference for item in evidence if item.corpus == "house_policy"],
            confidence=0.35 if evidence else (0.2 if retrieval_error_type else 0.15),
            rationale=(
                f"fallback_to_deterministic_stage_templates:retrieval_error:{retrieval_error_type}"
                if retrieval_error_type
                else "fallback_to_deterministic_stage_templates"
            ),
        )
        if not self.runtime.agentic_policy_enabled:
            fallback.rationale = (
                f"agentic_policy_disabled:retrieval_error:{retrieval_error_type}"
                if retrieval_error_type
                else "agentic_policy_disabled"
            )
            return fallback
        llm = self._planner_llm
        if llm is None:
            fallback.rationale = (
                f"planner_llm_unavailable:{self._planner_reason or 'unknown'}:retrieval_error:{retrieval_error_type}"
                if retrieval_error_type
                else f"planner_llm_unavailable:{self._planner_reason or 'unknown'}"
            )
            return fallback
        allowed_incar = _STAGE_ALLOWED_KEYS.get(stage, _COMMON_INCAR_KEYS)
        try:
            payload = {
                "stage": stage,
                "probe": probe.model_dump(mode="json"),
                "default_incar": dict(default_incar or {}),
                "default_kpoints_policy": dict(default_kpoints_policy or {}),
                "allowed_incar_keys": sorted(allowed_incar),
                "allowed_kpoints_keys": ["target_ka", "gamma_centered", "line_mode_density"],
                "allowed_runtime_keys": ["mpi_ranks", "omp_threads", "gpu_binding"],
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "instruction": (
                    "Return a single JSON object with keys: stage, source, incar_overrides, kpoints_policy, runtime_policy, "
                    "evidence_refs, house_rule_refs, confidence, rationale. "
                    "Keep non-default changes minimal. Only use allowed keys. If evidence is weak, keep overrides empty."
                ),
            }
            raw = self._call_json_schema(
                llm=llm,
                system_prompt=(
                    "You are a VASP parameter policy agent. "
                    "Your job is to recommend cautious, evidence-backed overrides for a single VASP stage. "
                    "Do not invent unsupported tags. Prefer minimal changes over aggressive tuning."
                ),
                user_payload=payload,
            )
            plan = ParameterPlan.model_validate(raw)
            plan.stage = stage
            plan.source = plan.source or "llm"
            plan.incar_overrides = self._sanitize_map(plan.incar_overrides, allowed=allowed_incar)
            plan.kpoints_policy = self._sanitize_map(plan.kpoints_policy, allowed={"target_ka", "gamma_centered", "line_mode_density"})
            plan.runtime_policy = self._sanitize_map(plan.runtime_policy, allowed={"mpi_ranks", "omp_threads", "gpu_binding"})
            plan.evidence_items = evidence
            plan.evidence_refs = [item.reference for item in evidence]
            plan.house_rule_refs = [item.reference for item in evidence if item.corpus == "house_policy"]
            if not plan.kpoints_policy:
                plan.kpoints_policy = dict(default_kpoints_policy or {})
            if not plan.incar_overrides and not plan.kpoints_policy:
                return fallback
            return plan
        except Exception as exc:
            fallback.rationale = (
                f"planner_fallback:{type(exc).__name__}:retrieval_error:{retrieval_error_type}"
                if retrieval_error_type
                else f"planner_fallback:{type(exc).__name__}"
            )
            return fallback

    def diagnose_failure(
        self,
        *,
        stage: str,
        state_payload: dict[str, Any],
        latest_failure: dict[str, Any],
        allowed_actions: list[str] | None = None,
    ) -> FailureDiagnosis:
        failure = dict(latest_failure or {})
        query = "\n".join(
            [
                stage,
                str(failure.get("error_summary") or ""),
                str(failure.get("error_category") or ""),
                json.dumps(failure.get("artifact_paths", {}) or {}, ensure_ascii=False),
            ]
        )
        try:
            evidence = self.knowledge_base.retrieve(
                query=query,
                stage=stage,
                top_k=self.runtime.policy_retrieval_top_k,
                corpora=("house_policy", "vasp_wiki"),
            )
        except Exception:
            evidence = []
        evidence.extend(_historical_evidence(state_payload, stage=stage))
        evidence = evidence[: max(1, self.runtime.policy_retrieval_top_k)]
        available_actions = [str(item) for item in list(allowed_actions or _DEFAULT_ALLOWED_ACTIONS) if str(item)]
        fallback_action = "retry_capability" if "retry_capability" in available_actions else available_actions[0]
        if has_relax_failure_signature(stage=stage, latest_failure=failure, state_payload=state_payload):
            action = "escalate_human" if "escalate_human" in available_actions else fallback_action
            return FailureDiagnosis(
                stage=stage,
                source="deterministic_relax_failure_policy",
                hypotheses=[
                    "relaxation_failure_requires_human_intervention",
                    str(failure.get("error_summary") or "relax_failure_detected"),
                ],
                recommended_action=action,
                parameter_patch={},
                needs_human=action == "escalate_human",
                evidence_refs=[item.reference for item in evidence],
                evidence_items=evidence,
                confidence=0.98,
                rationale=(
                    "deterministic_relax_failure_requires_human: "
                    "RELAX_FAILED/relax_failed/CONTCAR relaxation-output failure evidence was found; "
                    "non-relax failures remain under normal recovery policy."
                ),
            )
        fallback = FailureDiagnosis(
            stage=stage,
            source="fallback",
            hypotheses=[f"stage={stage}", str(failure.get("error_summary") or "unknown_failure")],
            recommended_action=fallback_action,
            parameter_patch={},
            needs_human=fallback_action == "escalate_human",
            evidence_refs=[item.reference for item in evidence],
            evidence_items=evidence,
            confidence=0.45 if evidence else 0.2,
            rationale="fallback_failure_diagnosis",
        )
        if not self.runtime.agentic_policy_enabled:
            fallback.rationale = "agentic_policy_disabled"
            return fallback
        llm = self._recovery_llm
        if llm is None:
            fallback.rationale = f"recovery_llm_unavailable:{self._recovery_reason or 'unknown'}"
            return fallback
        try:
            raw = self._call_json_schema(
                llm=llm,
                system_prompt=(
                    "You are a VASP recovery diagnosis agent. "
                    "Read the failure evidence, retrieved VASP Wiki guidance, and house-policy hints. "
                    "Return one cautious JSON diagnosis. Prefer bounded recovery actions."
                ),
                user_payload={
                    "stage": stage,
                    "latest_failure": failure,
                    "allowed_actions": available_actions,
                    "evidence": [item.model_dump(mode="json") for item in evidence],
                    "instruction": (
                        "Return a JSON object with keys: stage, source, hypotheses, recommended_action, parameter_patch, "
                        "needs_human, evidence_refs, confidence, rationale. "
                        "recommended_action must be one of allowed_actions."
                    ),
                },
            )
            diagnosis = FailureDiagnosis.model_validate(raw)
            diagnosis.stage = stage
            if diagnosis.recommended_action not in available_actions:
                diagnosis.recommended_action = fallback_action
            diagnosis.parameter_patch = self._sanitize_map(diagnosis.parameter_patch, allowed=_STAGE_ALLOWED_KEYS.get(stage, _COMMON_INCAR_KEYS))
            diagnosis.evidence_items = evidence
            diagnosis.evidence_refs = [item.reference for item in evidence]
            if not diagnosis.hypotheses:
                diagnosis.hypotheses = fallback.hypotheses
            return diagnosis
        except Exception as exc:
            fallback.rationale = f"recovery_fallback:{type(exc).__name__}"
            return fallback
