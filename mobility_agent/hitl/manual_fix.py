from __future__ import annotations

import json
import sys

from ..agents.schemas import ManualFixInstruction, ManualFixPreview
from ..utils import dedupe_keep_order
from .cleanup import preview_cleanup
from .resume_rules import (
    ModificationType,
    ResumeStrategyName,
    build_custom_resume_rule,
    compute_default_resume_rule,
)
from ..graph.stage_contracts import CleanupPolicyName, find_previous_stage


def _prompt_choice(prompt: str, mapping: dict[str, str], *, default: str) -> str:
    print(prompt)
    for key, value in mapping.items():
        print(f"[{key}] {value}")
    raw = input(f"select [{default}]: ").strip() or default
    return raw if raw in mapping else default


def build_manual_fix_instruction(
    *,
    current_stage: str,
    workdir: str,
    modification_type: ModificationType,
    requested_resume_strategy: ResumeStrategyName | None = None,
    selected_resume_stage: str | None = None,
    selected_cleanup_policy: CleanupPolicyName | None = None,
) -> ManualFixInstruction:
    base_rule = compute_default_resume_rule(
        current_stage=current_stage,
        modification_type=modification_type,
    )
    resolved_strategy = requested_resume_strategy or base_rule.requested_resume_strategy
    resolved_stage = selected_resume_stage or base_rule.resume_stage
    resolved_policy = selected_cleanup_policy or base_rule.cleanup_policy
    if (
        resolved_strategy != base_rule.requested_resume_strategy
        or resolved_stage != base_rule.resume_stage
        or resolved_policy != base_rule.cleanup_policy
    ):
        base_rule = build_custom_resume_rule(
            resume_stage=resolved_stage,
            cleanup_policy=resolved_policy,
            modified_files=list(base_rule.modified_files),
            current_stage=current_stage,
            modification_type=modification_type,
            requested_resume_strategy=resolved_strategy,
        )
    cleanup = preview_cleanup(
        workdir=workdir,
        resume_stage=base_rule.resume_stage,
        cleanup_policy=base_rule.cleanup_policy,
    )
    preview = ManualFixPreview(
        modified_files=dedupe_keep_order(
            list(base_rule.modified_files or [])
            + ([modification_type] if modification_type not in list(base_rule.modified_files or []) else [])
        ),
        requested_resume_strategy=base_rule.requested_resume_strategy,
        computed_resume_stage=base_rule.resume_stage,
        cleanup_policy=base_rule.cleanup_policy,
        invalidated_stages=cleanup.invalidated_stages,
        invalidated_artifacts=cleanup.invalidated_artifacts,
        warnings=dedupe_keep_order(list(cleanup.warnings or []) + list(base_rule.warnings or [])),
    )
    return ManualFixInstruction(
        modified_files=preview.modified_files,
        modification_type=modification_type,
        requested_resume_strategy=preview.requested_resume_strategy,
        resume_stage=base_rule.resume_stage,
        cleanup_policy=base_rule.cleanup_policy,
        invalidated_stages=preview.invalidated_stages,
        invalidated_artifacts=preview.invalidated_artifacts,
        preview=preview,
    )


def _print_preview(preview: ManualFixPreview, instruction: ManualFixInstruction) -> None:
    print("\nManual-fix resume preview:")
    print(f"modified files: {preview.modified_files}")
    print(f"requested resume strategy: {preview.requested_resume_strategy}")
    print(f"computed resume stage: {preview.computed_resume_stage}")
    print(f"cleanup policy: {preview.cleanup_policy}")
    print(f"invalidated downstream stages: {preview.invalidated_stages}")
    print("invalidated downstream artifacts:")
    for path in preview.invalidated_artifacts:
        print(f"  - {path}")
    if preview.warnings:
        print("warnings:")
        for warning in preview.warnings:
            print(f"  - {warning}")
    print("resulting resume instruction payload:")
    print(json.dumps(instruction.model_dump(mode="json"), ensure_ascii=False, indent=2))


def collect_manual_fix_instruction(*, current_stage: str, workdir: str) -> ManualFixInstruction:
    while True:
        print("You selected: retry with manual fix")
        print(f"Working directory:\n{workdir}\n")
        print("Please edit files, then confirm.\n")
        mod_map = {
            "1": "INCAR",
            "2": "KPOINTS",
            "3": "POSCAR",
            "4": "multiple",
            "5": "custom",
        }
        modification_key = _prompt_choice(
            "Select modification type:",
            mod_map,
            default="1",
        )
        modification_type = mod_map[modification_key]
        base_rule = compute_default_resume_rule(
            current_stage=current_stage,
            modification_type=modification_type,  # type: ignore[arg-type]
        )
        previous_stage = find_previous_stage(current_stage) or current_stage
        selected_resume_stage: str | None = None
        selected_cleanup_policy: CleanupPolicyName | None = None
        selected_strategy: ResumeStrategyName | None = None
        try:
            if modification_type in {"multiple", "custom"}:
                selected_strategy = "custom_stage"
                selected_resume_stage = input(
                    f"Select custom resume stage [{base_rule.resume_stage}]: "
                ).strip() or base_rule.resume_stage
                raw_cleanup = input(
                    f"Select cleanup policy [{base_rule.cleanup_policy}] (retry_current_stage_only/invalidate_downstream/restart_from_stage): "
                ).strip() or base_rule.cleanup_policy
                selected_cleanup_policy = raw_cleanup  # type: ignore[assignment]
            else:
                strategy_labels = {
                    "1": f"retry current stage ({current_stage})",
                    "2": f"rerun previous stage ({previous_stage})",
                    "3": "rerun from relax",
                    "4": "custom stage",
                }
                strategy_key = _prompt_choice(
                    "Select resume strategy:",
                    strategy_labels,
                    default="1",
                )
                strategy_map: dict[str, ResumeStrategyName] = {
                    "1": "retry_current_stage",
                    "2": "rerun_previous_stage",
                    "3": "rerun_from_relax",
                    "4": "custom_stage",
                }
                selected_strategy = strategy_map[strategy_key]
                if selected_strategy == "custom_stage":
                    selected_resume_stage = input(
                        f"Select custom resume stage [{base_rule.resume_stage}]: "
                    ).strip() or base_rule.resume_stage
                    raw_cleanup = input(
                        f"Select cleanup policy [{base_rule.cleanup_policy}] (retry_current_stage_only/invalidate_downstream/restart_from_stage): "
                    ).strip() or base_rule.cleanup_policy
                    selected_cleanup_policy = raw_cleanup  # type: ignore[assignment]
            instruction = build_manual_fix_instruction(
                current_stage=current_stage,
                workdir=workdir,
                modification_type=modification_type,  # type: ignore[arg-type]
                requested_resume_strategy=selected_strategy,
                selected_resume_stage=selected_resume_stage,
                selected_cleanup_policy=selected_cleanup_policy,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"manual-fix validation error: {exc}")
            retry = input("Try again? [Y/n]: ").strip().lower()
            if retry in {"", "y", "yes", "n", "no"}:
                print("Returning to selection menu.\n")
                continue
            raise
        _print_preview(
            instruction.preview
            or ManualFixPreview(
                modified_files=instruction.modified_files,
                requested_resume_strategy=instruction.requested_resume_strategy,
                computed_resume_stage=instruction.resume_stage,
                cleanup_policy=instruction.cleanup_policy,
                invalidated_stages=instruction.invalidated_stages,
                invalidated_artifacts=instruction.invalidated_artifacts,
            ),
            instruction,
        )
        confirm = input("Proceed with this resume instruction? [Y/n]: ").strip().lower()
        if confirm not in {"", "y", "yes"}:
            print("Manual-fix instruction not applied. Returning to selection menu.\n")
            continue
        return instruction


def can_interactively_prompt() -> bool:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return True
    try:
        with open("/dev/tty", "r", encoding="utf-8"), open("/dev/tty", "w", encoding="utf-8"):
            return True
    except OSError:
        return False
