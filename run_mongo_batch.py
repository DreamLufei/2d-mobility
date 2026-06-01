#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from mobility_agent.env import ensure_project_env_loaded
from mobility_agent.skills import default_skills_root, list_skill_packages
from mobility_agent.runtime import RuntimeContext, run_mongo_batch
from mobility_agent.runtime.checkpointing import build_batch_thread_id
from mobility_agent.runtime.batch_config import load_config
from mobility_agent.runtime.context import normalize_hitl_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch/database mobility screening runtime")
    parser.add_argument("--dry-run", action="store_true", help="Run batch orchestration without launching VASP.")
    parser.add_argument(
        "--dry-run-fail-stages",
        default="",
        help="Comma-separated dry-run stage failures to inject into every per-material run.",
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
        help="Human-in-the-loop policy override for per-material runs.",
    )
    parser.add_argument(
        "--fresh-materials",
        action="store_true",
        help="Force every per-material run to ignore its saved checkpoints.",
    )
    parser.add_argument(
        "--skills-root",
        default=None,
        help="Override the disk-backed skill package root for batch orchestration and per-material runs.",
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
    parser.add_argument("--json", action="store_true", help="Print the final batch state as JSON.")
    return parser


def _apply_env_overrides(args: argparse.Namespace) -> None:
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    if args.dry_run_fail_stages:
        os.environ["DRY_RUN_FAIL_STAGES"] = args.dry_run_fail_stages
    if args.hitl_policy:
        os.environ["HITL_POLICY"] = args.hitl_policy
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

    cfg = load_config()
    try:
        runtime = RuntimeContext.from_env()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        runtime = replace(runtime, dry_run=True, dry_run_fail_stages=tuple(stage.strip() for stage in args.dry_run_fail_stages.split(",") if stage.strip()))
    if args.hitl_policy:
        runtime = replace(runtime, hitl_policy=normalize_hitl_policy(args.hitl_policy))
    if args.skills_root:
        runtime = replace(runtime, skills_root=os.path.abspath(args.skills_root))
    if args.skill_auto_resolve_limit is not None:
        runtime = replace(runtime, skill_auto_resolve_limit=max(1, int(args.skill_auto_resolve_limit)))
    if args.skill_inline_body_limit is not None:
        runtime = replace(runtime, skill_inline_body_limit=max(400, int(args.skill_inline_body_limit)))

    final_state = run_mongo_batch(
        cfg=cfg,
        runtime=runtime,
        thread_id=build_batch_thread_id(batch_id=cfg.batch_tag),
        fresh_materials=bool(args.fresh_materials),
    )

    if args.json:
        print(json.dumps(final_state, ensure_ascii=False, indent=2))
    else:
        stats = dict(final_state.get("batch", {}).get("global_statistics", {}) or {})
        print("\n=== Batch Done ===")
        print(f"processed={stats.get('processed', 0)}")
        print(f"succeeded={stats.get('succeeded', 0)}")
        print(f"failed={stats.get('failed', 0)}")
        print(f"skipped={stats.get('skipped', 0)}")
        summary_path = final_state.get("batch", {}).get("summary_path")
        if summary_path:
            print(f"summary={summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
