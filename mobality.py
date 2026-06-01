#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from mobility_agent.env import ensure_project_env_loaded
from mobility_agent.skills import default_skills_root, list_skill_packages
from mobility_agent.runtime import RuntimeContext, default_material_workdir, run_single_material
from mobility_agent.runtime.context import normalize_hitl_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-material shared-memory multi-agent mobility runtime",
    )
    parser.add_argument("--root-path", default=os.getcwd(), help="Material root directory. Defaults to the current directory.")
    parser.add_argument("--workdir", default=None, help="Runtime workdir. Defaults to <root-path>/mobility_calculation.")
    parser.add_argument("--material-id", default=None, help="Material identifier. Defaults to the folder name.")
    parser.add_argument("--poscar", default=None, help="POSCAR path. Defaults to <root-path>/POSCAR.")
    parser.add_argument("--potcar", default=None, help="POTCAR path. Defaults to <root-path>/POTCAR.")
    parser.add_argument("--goal", default="calculate_2d_mobility", help="User goal recorded in shared state.")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved runtime checkpoints and restart from scratch.")
    parser.add_argument(
        "--no-relax-retry",
        action="store_true",
        help="Disable relax-stage retry behavior via RELAX_RETRY=false for this run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run orchestration in dry-run mode without launching VASP.")
    parser.add_argument(
        "--dry-run-fail-stages",
        default="",
        help="Comma-separated stages to inject failures into during dry-run mode.",
    )
    parser.add_argument(
        "--hitl-policy",
        choices=[
            "interactive",
            "non_interactive_skip_on_timeout",
            "non_interactive_abort_on_timeout",
            "non_interactive_wait",
            "non_interactive_skip",
        ],
        default=None,
        help="Human-in-the-loop runtime policy override.",
    )
    parser.add_argument(
        "--no-compatibility-checkpoint",
        action="store_true",
        help="Disable compatibility checkpoint.pkl exports.",
    )
    parser.add_argument(
        "--skills-root",
        default=None,
        help="Override the disk-backed skill package root. Defaults to MOBILITY_SKILLS_ROOT or <repo>/skills.",
    )
    parser.add_argument(
        "--skill-auto-resolve-limit",
        type=int,
        default=None,
        help="Maximum number of skills auto-selected per agent call.",
    )
    parser.add_argument(
        "--skill-inline-body-limit",
        type=int,
        default=None,
        help="Maximum inline SKILL.md body characters when a skill opts into summary_and_body loading.",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List installed disk-backed skill packages and exit without requiring LLM configuration.",
    )
    parser.add_argument("--json", action="store_true", help="Print the canonical material outcome as JSON.")
    return parser


def _apply_env_overrides(args: argparse.Namespace) -> None:
    if args.no_relax_retry:
        os.environ["RELAX_RETRY"] = "false"
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.dry_run_fail_stages:
        os.environ["DRY_RUN_FAIL_STAGES"] = args.dry_run_fail_stages
    if args.hitl_policy:
        os.environ["HITL_POLICY"] = args.hitl_policy
    if args.no_compatibility_checkpoint:
        os.environ["COMPATIBILITY_CHECKPOINT_EXPORT"] = "false"
    if args.skills_root:
        os.environ["MOBILITY_SKILLS_ROOT"] = os.path.abspath(args.skills_root)
    if args.skill_auto_resolve_limit is not None:
        os.environ["SKILL_AUTO_RESOLVE_LIMIT"] = str(int(args.skill_auto_resolve_limit))
    if args.skill_inline_body_limit is not None:
        os.environ["SKILL_INLINE_BODY_LIMIT"] = str(int(args.skill_inline_body_limit))


def _skills_root_for_listing(args: argparse.Namespace) -> str:
    if args.skills_root:
        return os.path.abspath(args.skills_root)
    env_root = str(os.environ.get("MOBILITY_SKILLS_ROOT") or "").strip()
    if env_root:
        return os.path.abspath(env_root)
    return default_skills_root()


def _print_skill_listing(args: argparse.Namespace) -> int:
    skills_root = _skills_root_for_listing(args)
    payload = {
        "skills_root": skills_root,
        "skills": list_skill_packages(skills_root),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"skills_root: {skills_root}")
    if not payload["skills"]:
        print("skills: 0")
        return 0
    print(f"skills: {len(payload['skills'])}")
    for item in payload["skills"]:
        roles = ", ".join(item["roles"]) if item["roles"] else "-"
        task_types = ", ".join(item["task_types"]) if item["task_types"] else "-"
        print(
            f"- {item['name']}: load_strategy={item['load_strategy']}, roles={roles}, "
            f"task_types={task_types}, resources={item['resource_count']}"
        )
        if item["description"]:
            print(f"  description: {item['description']}")
    return 0


def main() -> int:
    ensure_project_env_loaded()
    parser = build_parser()
    args = parser.parse_args()
    if args.list_skills:
        return _print_skill_listing(args)
    _apply_env_overrides(args)

    try:
        runtime = RuntimeContext.from_env()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        runtime = replace(runtime, dry_run=True, dry_run_fail_stages=tuple(stage.strip() for stage in args.dry_run_fail_stages.split(",") if stage.strip()))
    if args.hitl_policy:
        runtime = replace(runtime, hitl_policy=normalize_hitl_policy(args.hitl_policy))
    if args.no_compatibility_checkpoint:
        runtime = replace(runtime, compatibility_export_enabled=False)
    if args.skills_root:
        runtime = replace(runtime, skills_root=os.path.abspath(args.skills_root))
    if args.skill_auto_resolve_limit is not None:
        runtime = replace(runtime, skill_auto_resolve_limit=max(1, int(args.skill_auto_resolve_limit)))
    if args.skill_inline_body_limit is not None:
        runtime = replace(runtime, skill_inline_body_limit=max(400, int(args.skill_inline_body_limit)))

    root_path = os.path.abspath(args.root_path)
    workdir = os.path.abspath(args.workdir or default_material_workdir(root_path))
    material_id = args.material_id or os.path.basename(root_path) or "2D_Material"
    poscar_path = os.path.abspath(args.poscar or os.path.join(root_path, "POSCAR"))
    potcar_path = os.path.abspath(args.potcar or os.path.join(root_path, "POTCAR"))
    os.environ["MOBILITY_ACTIVE_ROOT_PATH"] = root_path
    os.environ["MOBILITY_ACTIVE_WORKDIR"] = workdir
    os.environ["MOBILITY_ACTIVE_MATERIAL_ID"] = material_id

    outcome = run_single_material(
        runtime=runtime,
        material_id=material_id,
        root_path=root_path,
        workdir=workdir,
        poscar_path=poscar_path,
        potcar_path=potcar_path,
        user_goal=args.goal,
        fresh=bool(args.fresh),
    )

    payload = outcome.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 70)
        print("2D mobility calculation")
        print("=" * 70)
        print(f"material_id: {outcome.material_id}")
        print(f"status: {outcome.status}")
        if outcome.final_acceptance:
            print(f"final_acceptance: {outcome.final_acceptance}")
        if outcome.termination_reason:
            print(f"termination_reason: {outcome.termination_reason}")
        print(f"workdir: {outcome.workdir}")
        if outcome.warnings:
            print(f"warnings: {len(outcome.warnings)}")
        if outcome.errors:
            print(f"errors: {len(outcome.errors)}")
        final_summary_path = outcome.artifact_paths.get("final_summary_path")
        material_outcome_path = outcome.artifact_paths.get("material_outcome_path")
        if final_summary_path:
            print(f"final_summary: {final_summary_path}")
        if material_outcome_path:
            print(f"material_outcome: {material_outcome_path}")
        print("=" * 70)

    return 0 if outcome.status in {"completed", "skipped", "running", "waiting_external", "needs_human"} else 1


if __name__ == "__main__":
    sys.exit(main())
