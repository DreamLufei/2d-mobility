from __future__ import annotations

import unittest

from mobility_agent.graph.state import MaterialTaskState


class BackwardCompatTests(unittest.TestCase):
    def test_legacy_state_payload_maps_to_shared_state(self) -> None:
        legacy = {
            "material_id": "legacy-1",
            "base_dir": "/tmp/legacy-material",
            "poscar_path": "/tmp/legacy-material/POSCAR",
            "potcar_path": "/tmp/legacy-material/POTCAR",
            "current_stage": "scf",
            "run_status": "running",
            "warnings": ["legacy-warning"],
            "errors": ["legacy-error"],
            "relax_completed": True,
            "scf_completed": False,
            "band_completed": False,
            "effmass_completed": False,
            "strain_completed": False,
            "fermi_energy": 0.42,
            "structure_summary": {"atom_count": 4},
        }
        state = MaterialTaskState.from_dict(legacy)
        self.assertEqual(state.material.material_id, "legacy-1")
        self.assertEqual(state.execution.workdir, "/tmp/legacy-material")
        self.assertEqual(state.workflow.current_stage, "scf")
        self.assertEqual(state.workflow.stage_status["relax"], "success")
        self.assertEqual(state.physics_results.fermi_energy, 0.42)
        self.assertEqual(state.material.atom_count, 4)
        self.assertIn("legacy-warning", state.material.warnings)
        self.assertIn("legacy-error", state.diagnostics.errors)


if __name__ == "__main__":
    unittest.main()
