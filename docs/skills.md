# Skills

## Purpose
Skills are disk-backed context packages for bounded workflow decisions. They are not physics executors and they do not replace deterministic tools.

This repository now treats skills as Anthropic-style skill packages:
- each package is rooted at `skills/<skill_name>/`
- `SKILL.md` is the canonical manifest plus human-readable guide
- optional resources such as JSON rules, templates, and references live beside `SKILL.md` or in nested folders
- runtime resolution is on-demand rather than hard-wiring every skill into every prompt

## Core Workflow Skill Packages
- `single_material_mobility`
- `batch_mobility_screening`
- `recovery`
- `strain_refinement`
- `physics_validation`
- `reporting`

## Role Skill Packages
- `admission`
- `planning`
- `critique`
- `orchestration`
- `cost_guardian`
- `execution_feasibility`
- `validation`

## Loader Behavior
- `mobility_agent/skills/registry.py` discovers installed skill packages.
- `mobility_agent/skills/loader.py` loads `SKILL.md` plus optional resource files.
- `mobility_agent/skills/resolver.py` ranks skills by role, task type, stage, run status, errors, and anomaly flags.
- `mobility_agent/skills/context.py` keeps compatibility defaults and delegates to resolver-backed selection when registry metadata is available.
- Runtime startup records discovered skill metadata into the LangGraph store under the `skill_registry` namespace.
- `python mobality.py --list-skills` and `python run_mongo_batch.py --list-skills` let you inspect installed packages without requiring LLM credentials.

## Runtime Semantics
- Prompts receive skill summaries by default.
- Each agent automatically carries its role package plus the matching workflow package for the current task type before any extra explicit skills are added.
- Agents can call runtime tools such as `resolve_skills`, `load_skill`, `list_skill_resources`, and `read_skill_resource` to pull more detail on demand.
- Skill resolution and loading are written into runtime `skill_trace` artifacts for auditability.
- Skill packages may guide orchestration, recovery, refinement, validation, and reporting, but must not bypass deterministic VASP-native execution.

## Authoring Rules
- Declare manifest metadata near the top of `SKILL.md` using `+++ ... +++` TOML frontmatter when possible.
- Declare purpose, when to use, required inputs, relevant state fields, allowed tools, decision rules, stop conditions, expected output schema, and caveats.
- Keep skill content stage-scoped and deterministic-friendly.
- Use resource files for detailed rules, warning catalogs, templates, or examples instead of bloating the main prompt.
