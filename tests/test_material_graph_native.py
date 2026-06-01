from __future__ import annotations

import os
import tempfile
import unittest

from langgraph.types import Command

from mobility_agent.graph.material_graph import build_material_graph
from mobility_agent.graph.state import MaterialTaskState, build_state_patch, make_initial_material_state


def _initial_state(tmpdir: str) -> dict[str, object]:
    poscar = os.path.join(tmpdir, "POSCAR")
    potcar = os.path.join(tmpdir, "POTCAR")
    with open(poscar, "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(potcar, "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")
    return make_initial_material_state(
        material_id="graph-native",
        root_path=tmpdir,
        workdir=os.path.join(tmpdir, "mobility_calculation"),
        poscar_path=poscar,
        potcar_path=potcar,
        user_goal="calculate_2d_mobility",
        decision_engine="llm_required",
        llm_required=True,
        llm_provider="openai",
        max_refinement_rounds=1,
        dry_run=True,
    ).to_dict()


class MaterialGraphNativeTests(unittest.TestCase):
    def test_material_graph_accepts_typed_boundary_and_command_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            executed = {"execute": False}

            def observe(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "observe_state"
                after["workflow"]["run_status"] = "running"
                return build_state_patch(state, after)

            def propose(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "proposal_phase"
                after["deliberation"]["round_index"] = 1
                return build_state_patch(state, after)

            def critique(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "critique_phase"
                return build_state_patch(state, after)

            def arbitration(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "arbitration_phase"
                after["services"]["selected_action_requires_execution"] = False
                return Command(update=build_state_patch(state, after), goto="check_termination")

            def execute(state: MaterialTaskState):
                executed["execute"] = True
                return {}

            def reflect(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "reflect_round"
                return build_state_patch(state, after)

            def check_termination(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "check_termination"
                after["workflow"]["run_status"] = "completed"
                after["services"]["termination_requested"] = True
                return Command(update=build_state_patch(state, after), goto="final_report")

            def final_report(state: MaterialTaskState):
                after = state.to_dict()
                after["workflow"]["current_stage"] = "final_report"
                after["workflow"]["stage_status"]["final_report"] = "success"
                after["services"]["final_report"] = {"ok": True}
                return build_state_patch(state, after)

            graph = build_material_graph(
                {
                    "observe_state": observe,
                    "proposal_phase": propose,
                    "critique_phase": critique,
                    "arbitration_phase": arbitration,
                    "execute_selected_action": execute,
                    "reflect_round": reflect,
                    "check_termination": check_termination,
                    "final_report": final_report,
                }
            )
            app = graph.compile()
            output = app.invoke(_initial_state(tmpdir))
            normalized = MaterialTaskState.from_dict(output)
            self.assertFalse(executed["execute"])
            self.assertEqual(normalized.workflow.run_status, "completed")
            self.assertEqual(normalized.workflow.current_stage, "final_report")
            self.assertEqual(normalized.services.final_report["ok"], True)


if __name__ == "__main__":
    unittest.main()
