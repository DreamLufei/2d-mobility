from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from pymatgen.io.vasp.inputs import Incar

from mobility_agent.tools.strain_tool import StrainTool, StrainToolInput


class _FakePlan:
    def __init__(self, incar_overrides: dict[str, object] | None = None):
        self.incar_overrides = dict(incar_overrides or {})
        self.kpoints_policy: dict[str, object] = {}
        self.source = "test"
        self.confidence = 1.0
        self.evidence_items: list[object] = []
        self.rationale = "test"

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        return {
            "incar_overrides": self.incar_overrides,
            "kpoints_policy": self.kpoints_policy,
            "source": self.source,
            "confidence": self.confidence,
            "evidence_items": [],
            "rationale": self.rationale,
        }


class _FakePolicyEngine:
    def plan_stage(self, *, stage: str, extra_context: dict[str, object] | None = None, **_: object) -> _FakePlan:
        substage = str((extra_context or {}).get("substage") or "")
        if stage == "scf" and substage == "scf":
            return _FakePlan({"PREC": "Accurate", "LREAL": False, "LASPH": True, "ADDGRID": True})
        if stage == "band" and substage == "band":
            return _FakePlan({"PREC": "High"})
        return _FakePlan()


class StrainToolTests(unittest.TestCase):
    @staticmethod
    def _write_valid_poscar(path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
            )

    def test_completed_points_from_nested_state_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write("fake poscar\n")
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            rows = [
                {"direction": "x", "strain": -0.02, "completed": True, "folder": "x/-0.02"},
                {"direction": "x", "strain": -0.01, "completed": True, "folder": "x/-0.01"},
                {"direction": "x", "strain": 0.0, "completed": True, "folder": "x/0.0"},
                {"direction": "x", "strain": 0.01, "completed": True, "folder": "x/0.01"},
                {"direction": "x", "strain": 0.02, "completed": True, "folder": "x/0.02"},
                {"direction": "y", "strain": -0.02, "completed": True, "folder": "y/-0.02"},
                {"direction": "y", "strain": -0.01, "completed": True, "folder": "y/-0.01"},
                {"direction": "y", "strain": 0.0, "completed": True, "folder": "y/0.0"},
                {"direction": "y", "strain": 0.01, "completed": True, "folder": "y/0.01"},
                {"direction": "y", "strain": 0.02, "completed": True, "folder": "y/0.02"},
            ]
            state_payload = {
                "physics_results": {
                    "strain_data": rows,
                }
            }
            tool = StrainTool()
            inputs = StrainToolInput(
                material_id="strain-skip-test",
                base_dir=tmpdir,
                state_payload=state_payload,
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [-0.02, -0.01, 0.0, 0.01, 0.02], "y": [-0.02, -0.01, 0.0, 0.01, 0.02]},
            )

            with patch("mobility_agent.tools.strain_tool.run_relax_vasp_with_retry", side_effect=AssertionError("should not rerun completed strain points")):
                with patch("mobility_agent.tools.strain_tool.run_vasp", side_effect=AssertionError("should not rerun completed strain points")):
                    output = tool.run(inputs)

            self.assertTrue(output.success)
            self.assertEqual(len(output.strain_data), len(rows))
            self.assertEqual(output.strain_summary["completed_points"], len(rows))
            self.assertEqual(output.strain_summary["failed_points"], 0)
            self.assertTrue(output.strain_summary["strain_completed"])
            self.assertEqual(output.strain_summary["per_direction_summary"]["x"]["planned_points"], 5)
            self.assertEqual(output.strain_summary["per_direction_summary"]["y"]["planned_points"], 5)

    def test_completed_points_can_be_recovered_from_status_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write("fake poscar\n")
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            strains = [-0.02, -0.01, 0.0, 0.01, 0.02]
            with open(os.path.join(tmpdir, "strain_status.csv"), "w", encoding="utf-8") as handle:
                handle.write("direction,strain,completed,error,folder\n")
                for direction in ("x", "y"):
                    for strain in strains:
                        folder = os.path.join(tmpdir, "05_strain", direction, f"strain_{strain:+.4f}")
                        os.makedirs(os.path.join(folder, "02_scf"), exist_ok=True)
                        band_dir = os.path.join(folder, "03_band")
                        os.makedirs(band_dir, exist_ok=True)
                        with open(os.path.join(band_dir, "EIGENVAL"), "w", encoding="utf-8") as eigenval:
                            eigenval.write("fake eigenval\n")
                        handle.write(f"{direction},{strain},True,,{folder}\n")

            inputs = StrainToolInput(
                material_id="strain-disk-recover-test",
                base_dir=tmpdir,
                state_payload={},
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": strains, "y": strains},
            )
            fake_ref = {
                "vbm_energy": -1.0,
                "vbm_kpoint": [0.0, 0.0, 0.0],
                "vbm_spin": 0,
                "cbm_energy": 1.0,
                "cbm_kpoint": [0.0, 0.0, 0.0],
                "cbm_spin": 0,
            }

            with patch("mobility_agent.tools.strain_tool.load_strain_reference_from_band", return_value=fake_ref):
                with patch("mobility_agent.tools.strain_tool.read_final_total_energy_eV", return_value=-10.0):
                    with patch("mobility_agent.tools.strain_tool.read_vacuum_level_from_locpot", return_value=4.0):
                        with patch(
                            "mobility_agent.tools.strain_tool.find_band_edges_from_eigenval_occupancy",
                            return_value=(-1.0, [0.0, 0.0, 0.0], 4, 0, 1.0, [0.0, 0.0, 0.0], 5, 0),
                        ):
                            with patch("mobility_agent.tools.strain_tool.extract_edge_energy_at_fixed_kpoint", side_effect=[-1.0, 1.0] * 10):
                                with patch("mobility_agent.tools.strain_tool.run_relax_vasp_with_retry", side_effect=AssertionError("should not rerun recovered strain points")):
                                    with patch("mobility_agent.tools.strain_tool.run_vasp", side_effect=AssertionError("should not rerun recovered strain points")):
                                        output = StrainTool().run(inputs)

            self.assertTrue(output.success)
            self.assertEqual(len(output.strain_data), 10)
            self.assertTrue(all(row.get("recovered_from_disk") for row in output.strain_data))
            self.assertTrue(output.strain_summary["strain_completed"])

    def test_zero_byte_completed_eigenval_does_not_crash_disk_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            self._write_valid_poscar(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            with open(os.path.join(tmpdir, "strain_status.csv"), "w", encoding="utf-8") as handle:
                handle.write("direction,strain,completed,error,folder\n")
                for direction in ("x", "y"):
                    folder = os.path.join(tmpdir, "05_strain", direction, "strain_+0.0000")
                    band_dir = os.path.join(folder, "03_band")
                    os.makedirs(band_dir, exist_ok=True)
                    open(os.path.join(band_dir, "EIGENVAL"), "w", encoding="utf-8").close()
                    handle.write(f"{direction},0.0,True,,{folder}\n")

            inputs = StrainToolInput(
                material_id="strain-disk-recover-zero-byte-test",
                base_dir=tmpdir,
                state_payload={},
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [0.0], "y": [0.0]},
            )
            with patch(
                "mobility_agent.tools.strain_tool.run_relax_vasp_with_retry",
                return_value=(False, [], {"error_type": "test_relax_failed", "applied_action": "skip_point"}),
            ) as run_relax:
                output = StrainTool().run(inputs)

            self.assertFalse(output.success)
            self.assertEqual(run_relax.call_count, 2)
            self.assertIn("strain_campaign_incomplete:2_failed_points", output.error_summary or "")
            self.assertNotIn("list index out of range", output.error_summary or "")

    def test_failed_points_mark_stage_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write("fake poscar\n")
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            tool = StrainTool(
                executor=lambda _: {
                    "strain_data": [
                        {"direction": "x", "strain": 0.0, "completed": True, "folder": "x/0.0"},
                        {"direction": "x", "strain": 0.02, "completed": False, "folder": "x/0.02", "error": "SCF_FAILED"},
                    ],
                    "strain_summary": {
                        "completed_points": 1,
                        "failed_points": 1,
                        "missing_points": 0,
                        "strain_completed": False,
                    },
                }
            )
            inputs = StrainToolInput(
                material_id="strain-failure-test",
                base_dir=tmpdir,
                state_payload={},
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [0.02], "y": []},
            )

            output = tool.run(inputs)

            self.assertFalse(output.success)
            self.assertIn("strain_campaign_incomplete:1_failed_points", output.warnings)
            self.assertIn("strain_campaign_incomplete:1_failed_points", output.error_summary or "")
            self.assertFalse(output.key_summary["strain_completed"])
            self.assertEqual(output.strain_summary["failed_points"], 1)

    def test_fit_ready_campaign_with_extra_failed_points_is_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write("fake poscar\n")
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            tool = StrainTool(
                executor=lambda _: {
                    "strain_data": [
                        {"direction": "x", "strain": -0.02, "completed": True, "folder": "x/-0.02"},
                        {"direction": "x", "strain": -0.01, "completed": True, "folder": "x/-0.01"},
                        {"direction": "x", "strain": 0.0, "completed": True, "folder": "x/0.0"},
                        {"direction": "x", "strain": 0.01, "completed": True, "folder": "x/0.01"},
                        {"direction": "x", "strain": 0.02, "completed": True, "folder": "x/0.02"},
                        {"direction": "x", "strain": 0.03, "completed": False, "folder": "x/0.03", "error": "SCF_FAILED"},
                        {"direction": "y", "strain": -0.02, "completed": True, "folder": "y/-0.02"},
                        {"direction": "y", "strain": -0.01, "completed": True, "folder": "y/-0.01"},
                        {"direction": "y", "strain": 0.0, "completed": True, "folder": "y/0.0"},
                        {"direction": "y", "strain": 0.01, "completed": True, "folder": "y/0.01"},
                        {"direction": "y", "strain": 0.02, "completed": True, "folder": "y/0.02"},
                    ],
                    "strain_summary": {
                        "completed_points": 10,
                        "failed_points": 1,
                        "missing_points": 0,
                        "strain_completed": True,
                    },
                }
            )
            inputs = StrainToolInput(
                material_id="strain-fit-ready-warning-test",
                base_dir=tmpdir,
                state_payload={},
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], "y": [-0.02, -0.01, 0.0, 0.01, 0.02]},
            )

            output = tool.run(inputs)

            self.assertFalse(output.success)
            self.assertIn("strain_campaign_incomplete:1_failed_points", output.warnings)
            self.assertIn("strain_campaign_incomplete:1_failed_points", output.error_summary or "")
            self.assertTrue(output.key_summary["strain_completed"])

    def test_historical_failed_points_outside_active_plan_do_not_fail_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write("fake poscar\n")
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("fake potcar\n")

            rows = [
                {"direction": "x", "strain": -0.02, "completed": True, "folder": "x/-0.02"},
                {"direction": "x", "strain": -0.01, "completed": True, "folder": "x/-0.01"},
                {"direction": "x", "strain": 0.0, "completed": True, "folder": "x/0.0"},
                {"direction": "x", "strain": 0.01, "completed": True, "folder": "x/0.01"},
                {"direction": "x", "strain": 0.02, "completed": True, "folder": "x/0.02"},
                {"direction": "x", "strain": 0.03, "completed": False, "folder": "x/0.03", "error": "SCF_FAILED"},
                {"direction": "y", "strain": -0.02, "completed": True, "folder": "y/-0.02"},
                {"direction": "y", "strain": -0.01, "completed": True, "folder": "y/-0.01"},
                {"direction": "y", "strain": 0.0, "completed": True, "folder": "y/0.0"},
                {"direction": "y", "strain": 0.01, "completed": True, "folder": "y/0.01"},
                {"direction": "y", "strain": 0.02, "completed": True, "folder": "y/0.02"},
            ]
            state_payload = {"physics_results": {"strain_data": rows}}
            tool = StrainTool()
            inputs = StrainToolInput(
                material_id="strain-historical-failure-test",
                base_dir=tmpdir,
                state_payload=state_payload,
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [-0.02, -0.01, 0.0, 0.01, 0.02], "y": [-0.02, -0.01, 0.0, 0.01, 0.02]},
            )

            with patch("mobility_agent.tools.strain_tool.run_relax_vasp_with_retry", side_effect=AssertionError("should not rerun completed strain points")):
                with patch("mobility_agent.tools.strain_tool.run_vasp", side_effect=AssertionError("should not rerun completed strain points")):
                    output = tool.run(inputs)

            self.assertTrue(output.success)
            self.assertEqual(output.strain_summary["failed_points"], 0)
            self.assertEqual(output.strain_summary["historical_failed_points"], 1)
            self.assertEqual(output.strain_summary["per_direction_summary"]["x"]["failed_points"], 0)
            self.assertEqual(output.strain_summary["per_direction_summary"]["x"]["historical_failed_points"], 1)

    def test_zero_reference_point_runs_before_negative_strains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            self._write_valid_poscar(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")

            tool = StrainTool()
            inputs = StrainToolInput(
                material_id="strain-zero-reference-test",
                base_dir=tmpdir,
                state_payload={},
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [-0.02, -0.01, 0.0, 0.01, 0.02], "y": [-0.02, -0.01, 0.0, 0.01, 0.02]},
            )

            def fake_relax(*, workdir: str, **_: object):
                contcar = os.path.join(workdir, "CONTCAR")
                self._write_valid_poscar(contcar)
                return True, [], {}

            with patch("mobility_agent.tools.strain_tool.run_relax_vasp_with_retry", side_effect=fake_relax):
                with patch("mobility_agent.tools.strain_tool.run_vasp", return_value=True):
                    with patch("mobility_agent.tools.strain_tool.read_final_total_energy_eV", return_value=-1.0):
                        with patch("mobility_agent.tools.strain_tool.read_vacuum_level_from_locpot", return_value=0.0):
                            with patch(
                                "mobility_agent.tools.strain_tool.find_band_edges_from_eigenval_occupancy",
                                return_value=(0.1, [0.0, 0.0, 0.0], 0, 0, 0.2, [0.0, 0.0, 0.0], 1, 0),
                            ):
                                with patch("mobility_agent.tools.strain_tool.extract_edge_energy_at_fixed_kpoint", return_value=0.1):
                                    with patch("mobility_agent.tools.strain_tool.load_strain_reference_from_band", return_value=None):
                                        with patch("mobility_agent.tools.strain_tool.prune_dir_keep_files", return_value=None):
                                            output = tool.run(inputs)

            self.assertTrue(output.success)
            self.assertEqual(output.strain_summary["completed_points"], 10)
            self.assertEqual(output.strain_summary["failed_points"], 0)
            self.assertTrue(output.strain_summary["per_direction_summary"]["x"]["fit_ready"])
            self.assertTrue(output.strain_summary["per_direction_summary"]["y"]["fit_ready"])

    def test_strain_band_inherits_scf_chgcar_compatible_incar_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            poscar = os.path.join(tmpdir, "POSCAR")
            potcar = os.path.join(tmpdir, "POTCAR")
            self._write_valid_poscar(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")

            tool = StrainTool(policy_engine=_FakePolicyEngine())
            inputs = StrainToolInput(
                material_id="strain-band-incar-test",
                base_dir=tmpdir,
                state_payload={
                    "execution": {
                        "current_action": {
                            "action_family": "retry_capability",
                            "target_capability": "strain_loop",
                        }
                    },
                    "workflow": {"stage_status": {"strain_loop": "failed"}},
                },
                relaxed_poscar=poscar,
                potcar_path=potcar,
                strain_plan_by_direction={"x": [0.0], "y": [0.0]},
            )

            def fake_relax(*, workdir: str, **_: object):
                self._write_valid_poscar(os.path.join(workdir, "CONTCAR"))
                return True, [], {}

            def fake_run_vasp(*, cwd: str, **_: object) -> bool:
                if cwd.endswith(os.path.join("strain_+0.0000", "02_scf")):
                    with open(os.path.join(cwd, "CHGCAR"), "w", encoding="utf-8") as handle:
                        handle.write("FAKE CHGCAR\n")
                return True

            with patch("mobility_agent.tools.strain_tool.run_relax_vasp_with_retry", side_effect=fake_relax):
                with patch("mobility_agent.tools.strain_tool.run_vasp", side_effect=fake_run_vasp):
                    with patch("mobility_agent.tools.strain_tool.read_final_total_energy_eV", return_value=-1.0):
                        with patch("mobility_agent.tools.strain_tool.read_vacuum_level_from_locpot", return_value=0.0):
                            with patch(
                                "mobility_agent.tools.strain_tool.find_band_edges_from_eigenval_occupancy",
                                return_value=(0.1, [0.0, 0.0, 0.0], 0, 0, 0.2, [0.0, 0.0, 0.0], 1, 0),
                            ):
                                with patch("mobility_agent.tools.strain_tool.extract_edge_energy_at_fixed_kpoint", return_value=0.1):
                                    with patch("mobility_agent.tools.strain_tool.load_strain_reference_from_band", return_value=None):
                                        with patch("mobility_agent.tools.strain_tool.prune_dir_keep_files", return_value=None):
                                            output = tool.run(inputs)

            self.assertEqual(output.strain_summary["completed_points"], 2)
            band_incar_path = os.path.join(tmpdir, "05_strain", "x", "strain_+0.0000", "03_band", "INCAR")
            incar = Incar.from_file(band_incar_path)
            self.assertEqual(incar["ICHARG"], 11)
            self.assertEqual(incar["PREC"], "Accurate")
            self.assertFalse(incar["LREAL"])
            self.assertTrue(incar["LASPH"])
            self.assertTrue(incar["ADDGRID"])
            self.assertFalse(incar["LCHARG"])
            self.assertFalse(incar["LWAVE"])


if __name__ == "__main__":
    unittest.main()
