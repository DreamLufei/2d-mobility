from __future__ import annotations

import json
import os
import select
import sys
from contextlib import suppress
from typing import Any

from ..agents.schemas import HITLDecision
from ..notifications.emailer import EmailNotificationBackend
from ..runtime.checkpointing import append_ui_event, write_json_atomic
from ..runtime.context import normalize_hitl_policy
from .manual_fix import build_manual_fix_instruction, can_interactively_prompt
from .resume import build_resume_command, normalize_hitl_decision
from .resume_rules import validate_stage_name
from ..graph.stage_contracts import default_cleanup_policy_for_resume_stage
from .timeout import timeout_decision, wait_for_response_file


def escalation_paths(workdir: str) -> dict[str, str]:
    return {
        "payload_path": os.path.join(workdir, "human_escalation_payload.json"),
        "response_path": os.path.join(workdir, "human_escalation_response.json"),
        "log_path": os.path.join(workdir, "human_escalation_log.json"),
    }


def write_escalation_payload(
    *,
    workdir: str,
    payload: dict[str, Any],
    checkpoint_subdir: str = ".runtime",
) -> dict[str, str]:
    paths = escalation_paths(workdir)
    os.makedirs(workdir, exist_ok=True)
    write_json_atomic(paths["payload_path"], payload)
    append_ui_event(
        workdir=workdir,
        event_type="hitl_payload_written",
        checkpoint_subdir=checkpoint_subdir,
        extra={
            "current_stage": str(payload.get("current_stage") or ""),
            "wait_reason": str(payload.get("wait_reason") or "needs_human") or None,
            "hitl_pending": True,
        },
    )
    return paths


def write_human_response(
    *,
    workdir: str,
    response: dict[str, Any],
    checkpoint_subdir: str = ".runtime",
    source: str = "precomputed",
) -> dict[str, Any]:
    paths = escalation_paths(workdir)
    normalized = normalize_hitl_decision(dict(response or {}), source=source)
    resume_payload = build_resume_command(normalized)
    write_json_atomic(paths["response_path"], resume_payload)
    append_ui_event(
        workdir=workdir,
        event_type="hitl_response_written",
        checkpoint_subdir=checkpoint_subdir,
        extra={
            "hitl_pending": False,
            "human_action": normalized.action,
            "human_reason": normalized.reason or None,
        },
    )
    return {
        "paths": paths,
        "decision": normalized.model_dump(mode="json"),
        "resume_payload": resume_payload,
    }


def notify_escalation(payload: dict[str, Any]) -> dict[str, object]:
    backend = EmailNotificationBackend()
    return backend.send_payload(payload)


def _is_valid_resume_stage(stage: str | None) -> bool:
    text = str(stage or "").strip()
    if not text:
        return False
    try:
        validate_stage_name(text)
    except Exception:
        return False
    return True


def _normalize_stage_alias(raw: str) -> str | None:
    text = raw.strip().lower()
    if not text:
        return None
    aliases = {
        "prepare": "prepare",
        "00_prepare": "prepare",
        "relax": "relax",
        "01_relax": "relax",
        "scf": "scf",
        "02_scf": "scf",
        "band": "band",
        "03_band": "band",
        "effective_mass": "effective_mass",
        "effmass": "effective_mass",
        "04_effmass": "effective_mass",
        "strain": "strain_loop",
        "05_strain": "strain_loop",
        "strain_loop": "strain_loop",
        "refinement": "refinement",
        "mobility": "mobility",
        "validation": "validation",
        "report": "report",
        "final_report": "report",
    }
    return aliases.get(text)


def _infer_stage_from_error_summary(payload: dict[str, Any]) -> str | None:
    summary = str(payload.get("error_summary") or "").lower()
    if not summary:
        return None
    if "band" in summary:
        return "band"
    if "scf" in summary:
        return "scf"
    if "relax" in summary:
        return "relax"
    if "eff" in summary and "mass" in summary:
        return "effective_mass"
    if "strain" in summary:
        return "strain_loop"
    if "mobility" in summary:
        return "mobility"
    if "validation" in summary:
        return "validation"
    if "report" in summary:
        return "report"
    return None


def _infer_stage_from_log_paths(payload: dict[str, Any]) -> str | None:
    paths = [str(path).lower() for path in list(payload.get("log_paths", []) or [])]
    for token, stage in [
        ("/03_band", "band"),
        ("/02_scf", "scf"),
        ("/01_relax", "relax"),
        ("effmass", "effective_mass"),
        ("strain", "strain_loop"),
        ("mobility", "mobility"),
        ("validation", "validation"),
    ]:
        if any(token in item for item in paths):
            return stage
    return None


def _resolve_current_stage(payload: dict[str, Any]) -> str:
    candidates: list[str] = []
    for key in ["current_stage", "recovery_stage", "resume_stage", "target_stage"]:
        value = str(payload.get(key) or "").strip()
        if value:
            candidates.append(value)

    history = list(payload.get("recovery_history_summary", []) or [])
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        for key in ["target_stage", "stage", "current_stage"]:
            value = str(item.get(key) or "").strip()
            if value:
                candidates.append(value)

    for stage in candidates:
        alias = _normalize_stage_alias(stage)
        if alias and _is_valid_resume_stage(alias):
            return alias
        if _is_valid_resume_stage(stage):
            return stage

    inferred = _infer_stage_from_error_summary(payload)
    if inferred and _is_valid_resume_stage(inferred):
        return inferred

    inferred = _infer_stage_from_log_paths(payload)
    if inferred and _is_valid_resume_stage(inferred):
        return inferred

    return "relax"

def _open_terminal_streams() -> tuple[Any, Any, bool]:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return sys.stdin, sys.stdout, False
    tty_in = open("/dev/tty", "r", encoding="utf-8")
    tty_out = open("/dev/tty", "w", encoding="utf-8")
    return tty_in, tty_out, True


def _write_line(stream: Any, message: str = "") -> None:
    stream.write(f"{message}\n")
    stream.flush()


def _timed_input(prompt: str, *, timeout_seconds: int, stdin: Any, stdout: Any) -> tuple[str, bool]:
    if timeout_seconds <= 0:
        stdout.write(prompt)
        stdout.flush()
        line = stdin.readline()
        if line == "":
            return "", False
        return line.rstrip("\n"), False
    stdout.write(prompt)
    stdout.flush()
    ready, _, _ = select.select([stdin], [], [], float(timeout_seconds))
    if ready:
        line = stdin.readline()
        if line == "":
            return "", False
        return line.rstrip("\n"), False
    return "", True


def _manual_fix_continue_decision(*, current_stage: str, workdir: str, raw_command: str) -> HITLDecision:
    command = str(raw_command or "").strip()
    lowered = command.lower()
    if lowered in {"skip", "skip_material"}:
        return HITLDecision(action="skip_material", reason="interactive_skip_after_manual_pause", source="interactive")
    if lowered in {"abort", "abort_task"}:
        return HITLDecision(action="abort_task", reason="interactive_abort_after_manual_pause", source="interactive")

    resume_stage = current_stage
    requested_resume_strategy = None
    selected_resume_stage = None
    if lowered.startswith("continue"):
        tail = command[8:].strip()
        selected_cleanup_policy = None
        if tail:
            alias = _normalize_stage_alias(tail) or tail
            if not _is_valid_resume_stage(alias):
                raise ValueError(f"invalid_resume_stage:{tail}")
            resume_stage = alias
            requested_resume_strategy = "custom_stage"
            selected_resume_stage = alias
            selected_cleanup_policy = default_cleanup_policy_for_resume_stage(
                current_stage=current_stage,
                resume_stage=alias,
            )
        instruction = build_manual_fix_instruction(
            current_stage=current_stage,
            workdir=workdir,
            modification_type="custom",
            requested_resume_strategy=requested_resume_strategy,
            selected_resume_stage=selected_resume_stage,
            selected_cleanup_policy=selected_cleanup_policy,
        )
        return HITLDecision(
            action="manual_fix_resume",
            instruction=instruction,
            reason=f"user_manual_fix_continue:{resume_stage}",
            source="interactive",
        )
    raise ValueError(f"unsupported_manual_command:{command}")


def _interactive_decision(payload: dict[str, Any], *, timeout_seconds: int, default_action: str) -> HITLDecision:
    resolved_stage = _resolve_current_stage(payload)
    payload = dict(payload)
    payload["current_stage"] = resolved_stage

    workdir = str(payload.get("working_directory") or os.getcwd())
    recommended = list(
        payload.get("recommended_options", []) or ["manual_fix_resume", "retry_current_stage", "skip_material", "abort_task"]
    )
    log_paths = [str(path) for path in list(payload.get("log_paths", []) or []) if str(path).strip()]
    recovery_history = list(payload.get("recovery_history_summary", []) or [])
    stdin, stdout, close_streams = _open_terminal_streams()
    try:
        _write_line(stdout, "")
        _write_line(stdout, "Human escalation requested.")
        _write_line(stdout, f"material_id: {payload.get('material_id')}")
        _write_line(stdout, f"stage: {payload.get('current_stage')}")
        if payload.get("graph_node"):
            _write_line(stdout, f"graph_node: {payload.get('graph_node')}")
        _write_line(stdout, f"issue: {payload.get('error_summary')}")
        _write_line(stdout, f"working_directory: {workdir}")
        if log_paths:
            _write_line(stdout, "log_paths:")
            for idx, path in enumerate(log_paths[:8], start=1):
                _write_line(stdout, f"  [{idx}] {path}")
            if len(log_paths) > 8:
                _write_line(stdout, f"  ... and {len(log_paths) - 8} more")
        if recovery_history:
            _write_line(stdout, "recent_recovery_history:")
            for item in recovery_history[-3:]:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action") or item.get("decision") or "unknown")
                target_stage = str(item.get("target_stage") or item.get("stage") or "")
                reason = str(item.get("reason") or "")
                snippet = f"{action}"
                if target_stage:
                    snippet += f" -> {target_stage}"
                if reason:
                    snippet += f" ({reason})"
                _write_line(stdout, f"  - {snippet}")
        _write_line(stdout, "recommended options:")
        for idx, option in enumerate(recommended, start=1):
            _write_line(stdout, f"[{idx}] {option}")
        raw, timed_out = _timed_input("select [1]: ", timeout_seconds=timeout_seconds, stdin=stdin, stdout=stdout)
        if timed_out:
            selected = default_action if default_action in recommended else recommended[0]
            _write_line(stdout, f"")
            _write_line(stdout, f"No response in {timeout_seconds}s, auto-selecting: {selected}")
        else:
            raw = raw.strip() or "1"
            try:
                selected = recommended[int(raw) - 1]
            except Exception:
                selected = recommended[0]
        if selected == "manual_fix_resume":
            _write_line(stdout, "")
            _write_line(stdout, "Manual intervention selected.")
            _write_line(stdout, "Edit files in the working directory, then return here and type one of:")
            _write_line(stdout, "  continue")
            _write_line(stdout, "  continue <stage>")
            _write_line(stdout, "  skip")
            _write_line(stdout, "  abort")
            while True:
                raw_command, _ = _timed_input("command> ", timeout_seconds=0, stdin=stdin, stdout=stdout)
                raw_command = raw_command.strip()
                if not raw_command:
                    continue
                try:
                    return _manual_fix_continue_decision(
                        current_stage=str(payload.get("current_stage") or "relax"),
                        workdir=workdir,
                        raw_command=raw_command,
                    )
                except Exception as exc:
                    _write_line(stdout, f"invalid command: {exc}")
                    _write_line(stdout, "Use: continue | continue <stage> | skip | abort")
        return HITLDecision(
            action=selected,  # type: ignore[arg-type]
            reason="interactive_selection",
            source="interactive",
        )
    finally:
        if close_streams:
            with suppress(Exception):
                stdin.close()
            with suppress(Exception):
                stdout.close()


def resolve_human_decision(*, payload: dict[str, Any], runtime: Any) -> HITLDecision:
    workdir = str(payload.get("working_directory") or os.getcwd())
    paths = escalation_paths(workdir)
    policy = normalize_hitl_policy(getattr(runtime, "hitl_policy", None))
    default_action = str(payload.get("default_timeout_action") or "skip_material")

    timeout_seconds = getattr(getattr(runtime, "agent_runtime", None), "human_review_timeout_seconds", 300)
    if timeout_seconds is None:
        timeout_seconds = 300
    if policy == "interactive" and can_interactively_prompt():
        return _interactive_decision(payload, timeout_seconds=int(timeout_seconds), default_action=default_action)

    response = wait_for_response_file(
        response_path=paths["response_path"],
        timeout_s=int(timeout_seconds),
    )
    if response:
        return normalize_hitl_decision(response, source="response_file")
    return timeout_decision(policy=policy, default_action=default_action)
