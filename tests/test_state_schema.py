from __future__ import annotations

import unittest

from mobility_agent.graph.state import (
    MaterialTaskState,
    apply_state_patch,
    apply_state_updates,
    build_state_patch,
    build_material_outcome,
    make_initial_material_state,
)


class StateSchemaTests(unittest.TestCase):
    def test_append_friendly_updates_preserve_history(self) -> None:
        state = make_initial_material_state(
            material_id="mat-1",
            root_path="/tmp/material",
            workdir="/tmp/material/mobility_calculation",
            poscar_path="/tmp/material/POSCAR",
            potcar_path="/tmp/material/POTCAR",
            user_goal="test",
            decision_engine="llm_required",
            llm_required=True,
            llm_provider="openai",
            max_refinement_rounds=1,
            dry_run=True,
        ).to_dict()
        updated = apply_state_updates(
            state,
            {
                "material": {"warnings": ["warn-a"]},
                "diagnostics": {"errors": ["err-a"]},
                "execution": {"tool_trace": [{"stage": "prepare"}]},
            },
        )
        updated = apply_state_updates(
            updated,
            {
                "material": {"warnings": ["warn-b"]},
                "diagnostics": {"errors": ["err-b"]},
                "execution": {"tool_trace": [{"stage": "relax"}]},
            },
        )
        updated = apply_state_updates(
            updated,
            {
                "execution": {"skill_trace": [{"phase": "observe_state", "selected_skills": ["recovery"]}]},
            },
        )
        updated = apply_state_updates(
            updated,
            {
                "execution": {"skill_trace": [{"phase": "proposal_phase", "selected_skills": ["reporting"]}]},
            },
        )
        normalized = MaterialTaskState.from_dict(updated)
        self.assertEqual(normalized.material.warnings, ["warn-a", "warn-b"])
        self.assertEqual(normalized.diagnostics.errors, ["err-a", "err-b"])
        self.assertEqual([item["stage"] for item in normalized.execution.tool_trace], ["prepare", "relax"])
        self.assertEqual(len(normalized.execution.skill_trace), 2)

    def test_material_outcome_contains_results(self) -> None:
        state = make_initial_material_state(
            material_id="mat-2",
            root_path="/tmp/material",
            workdir="/tmp/material/mobility_calculation",
            poscar_path="/tmp/material/POSCAR",
            potcar_path="/tmp/material/POTCAR",
            user_goal="test",
            decision_engine="llm_required",
            llm_required=True,
            llm_provider="openai",
            max_refinement_rounds=1,
            dry_run=True,
        ).to_dict()
        state["workflow"]["run_status"] = "completed"
        state["physics_results"]["results"] = {"results_by_direction": {"x": {"electron": {"mobility_cm2_Vs": 1.0}}}}
        outcome = build_material_outcome(state)
        self.assertIn("results_by_direction", outcome.results)

    def test_top_level_state_patch_round_trips_with_typed_sections(self) -> None:
        before = make_initial_material_state(
            material_id="mat-3",
            root_path="/tmp/material",
            workdir="/tmp/material/mobility_calculation",
            poscar_path="/tmp/material/POSCAR",
            potcar_path="/tmp/material/POTCAR",
            user_goal="test",
            decision_engine="llm_required",
            llm_required=True,
            llm_provider="openai",
            max_refinement_rounds=1,
            dry_run=True,
        ).to_dict()
        after = MaterialTaskState.from_dict(before).to_dict()
        after["workflow"]["run_status"] = "waiting_external"
        after["workflow"]["wait_reason"] = "awaiting_external_event:scf"
        after["execution"]["pending_events"] = [{"event_type": "job_completed", "event_id": "evt-1"}]
        after["services"]["framework_diagnostics"] = [{"code": "test", "detail": {}}]
        patch = build_state_patch(before, after)
        self.assertEqual(set(patch), {"workflow", "execution", "services"})
        merged = apply_state_patch(before, patch)
        self.assertEqual(MaterialTaskState.from_dict(merged).workflow.run_status, "waiting_external")
        self.assertEqual(MaterialTaskState.from_dict(merged).execution.pending_events[0]["event_id"], "evt-1")


if __name__ == "__main__":
    unittest.main()
