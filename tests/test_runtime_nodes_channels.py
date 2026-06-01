from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.graph.runtime_nodes import make_execute_selected_action_node
from mobility_agent.graph.state import apply_state_patch, make_initial_material_state
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(*, store_path: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
    )


def _state(tmpdir: str) -> dict[str, object]:
    poscar = os.path.join(tmpdir, "POSCAR")
    potcar = os.path.join(tmpdir, "POTCAR")
    with open(poscar, "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(potcar, "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")
    return make_initial_material_state(
        material_id="runtime-node-channel-test",
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


class RuntimeNodeChannelTests(unittest.TestCase):
    def test_execute_invalidate_channel_accepts_channels_to_invalidate_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["blackboard"]["anomaly_flags"] = ["negative_mobility"]
            state["physics_results"]["accepted_channels"] = ["x", "y"]
            state["physics_results"]["rejected_channels"] = []
            state["execution"]["current_action"] = {
                "action_family": "invalidate_channel",
                "target_capability": "mobility",
                "selected_skill": "single_material_mobility",
                "parameters": {"channels_to_invalidate": ["y"]},
            }
            state["deliberation"]["round_index"] = 2
            runtime = _runtime(store_path=os.path.join(tmpdir, "store.sqlite"))
            node = make_execute_selected_action_node(
                runtime,
                skills_root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                tools={},
            )

            with patch_test_llm_clients():
                patch_payload = node(state)
            updated = apply_state_patch(state, patch_payload)

            self.assertEqual(updated["execution"]["latest_execution_observation"]["status"], "success")
            self.assertEqual(updated["physics_results"]["accepted_channels"], ["x"])
            self.assertIn("y", updated["physics_results"]["rejected_channels"])


if __name__ == "__main__":
    unittest.main()
