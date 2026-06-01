from __future__ import annotations

import unittest

from mobility_agent.graph.recovery_registry import allowed_actions_for_stage


class RecoveryRegistryTests(unittest.TestCase):
    def test_stage_actions_are_scoped(self) -> None:
        relax_actions = allowed_actions_for_stage("relax")
        self.assertIn("modify_params_and_retry", relax_actions)
        self.assertNotIn("skip_point", relax_actions)

        strain_actions = allowed_actions_for_stage("strain_loop")
        self.assertIn("skip_point", strain_actions)

        unknown_actions = allowed_actions_for_stage("unknown")
        self.assertEqual(unknown_actions, ["skip_material", "abort_task"])


if __name__ == "__main__":
    unittest.main()
