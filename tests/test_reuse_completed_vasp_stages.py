from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar

from mobility_agent.tools.band_tool import BandTool, BandToolInput
from mobility_agent.tools.relax_tool import RelaxTool, RelaxToolInput
from mobility_agent.tools.scf_tool import ScfTool, ScfToolInput


def _write_structure(path: str) -> None:
    structure = Structure(
        Lattice.from_parameters(3.0, 3.0, 20.0, 90.0, 90.0, 90.0),
        ["Si"],
        [[0.0, 0.0, 0.5]],
    )
    structure.to(filename=path, fmt="poscar")


class ReuseCompletedVaspStagesTests(unittest.TestCase):
    def test_relax_reuses_existing_contcar_without_vasp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            relax_dir = os.path.join(base_dir, "01_relax")
            os.makedirs(relax_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            contcar = os.path.join(relax_dir, "CONTCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            _write_structure(contcar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")

            inputs = RelaxToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
            )
            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.relax_tool.run_relax_vasp_with_retry") as run_vasp:
                    result = RelaxTool()._execute(inputs)

            run_vasp.assert_not_called()
            self.assertTrue(result["relax_completed"])
            self.assertEqual(result["relaxed_poscar"], contcar)
            self.assertIn("reused_completed_stage:relax", result["warnings"])

    def test_relax_ignores_empty_reused_contcar_and_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            relax_dir = os.path.join(base_dir, "01_relax")
            os.makedirs(relax_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            contcar = os.path.join(relax_dir, "CONTCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            open(contcar, "w", encoding="utf-8").close()

            inputs = RelaxToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
            )

            def fake_relax(*, workdir: str, **_kwargs):
                _write_structure(os.path.join(workdir, "CONTCAR"))
                return True, [], {"stage": "relax", "final_outcome": "success"}

            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.relax_tool.run_relax_vasp_with_retry", side_effect=fake_relax) as run_vasp:
                    result = RelaxTool()._execute(inputs)

            run_vasp.assert_called_once()
            self.assertTrue(result["relax_completed"])
            self.assertEqual(result["relaxed_poscar"], contcar)
            self.assertIn("ignored_invalid_reused_relax_contcar:empty CONTCAR", result["warnings"])

    def test_scf_reuses_existing_chgcar_and_fermi_without_vasp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            scf_dir = os.path.join(base_dir, "02_scf")
            os.makedirs(scf_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            chgcar = os.path.join(scf_dir, "CHGCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(chgcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE CHGCAR\n")
            with open(os.path.join(scf_dir, "OUTCAR"), "w", encoding="utf-8") as handle:
                handle.write(" E-fermi :   1.2345     XC(G=0): 0.0000\n")

            inputs = ScfToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
                material_name="reuse-test",
            )
            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.scf_tool.run_vasp") as run_vasp:
                    result = ScfTool()._execute(inputs)

            run_vasp.assert_not_called()
            self.assertTrue(result["scf_completed"])
            self.assertEqual(result["chgcar_path"], chgcar)
            self.assertEqual(result["fermi_energy"], 1.2345)
            self.assertIn("reused_completed_stage:scf", result["warnings"])

    def test_scf_reruns_existing_chgcar_when_fermi_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            scf_dir = os.path.join(base_dir, "02_scf")
            os.makedirs(scf_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            chgcar = os.path.join(scf_dir, "CHGCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(chgcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE CHGCAR\n")

            inputs = ScfToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
                material_name="reuse-test",
            )
            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.scf_tool.run_vasp", return_value=True) as run_vasp:
                    with patch(
                        "mobility_agent.tools.scf_tool.read_fermi_energy_eV",
                        side_effect=[FileNotFoundError("missing fermi"), 4.56],
                    ):
                        with patch("mobility_agent.tools.scf_tool.prune_dir_keep_files") as prune:
                            result = ScfTool()._execute(inputs)

            run_vasp.assert_called_once()
            prune.assert_called_once()
            self.assertIn("OUTCAR", prune.call_args.args[1])
            self.assertIn("OSZICAR", prune.call_args.args[1])
            self.assertTrue(result["scf_completed"])
            self.assertEqual(result["fermi_energy"], 4.56)
            self.assertIn("reused_scf_without_fermi_energy", result["warnings"])
            self.assertIn("reran_scf_to_restore_fermi_energy", result["warnings"])

    def test_band_reuses_existing_eigenval_without_vasp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            band_dir = os.path.join(base_dir, "03_band")
            os.makedirs(band_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            eigenval = os.path.join(band_dir, "EIGENVAL")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(eigenval, "w", encoding="utf-8") as handle:
                handle.write("FAKE EIGENVAL\n")

            inputs = BandToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
            )
            parsed_edges = (
                -1.0,
                [0.0, 0.0, 0.0],
                0,
                0,
                1.0,
                [0.5, 0.0, 0.0],
                1,
                0,
            )
            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.band_tool.run_vasp") as run_vasp:
                    with patch(
                        "mobility_agent.tools.band_tool.find_band_edges_from_eigenval_occupancy",
                        return_value=parsed_edges,
                    ):
                        result = BandTool()._execute(inputs)

            run_vasp.assert_not_called()
            self.assertTrue(result["band_completed"])
            self.assertEqual(result["vbm_energy"], -1.0)
            self.assertEqual(result["cbm_energy"], 1.0)
            self.assertIn("reused_completed_stage:band", result["warnings"])

    def test_band_reruns_when_existing_eigenval_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            band_dir = os.path.join(base_dir, "03_band")
            os.makedirs(band_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(os.path.join(band_dir, "EIGENVAL"), "w", encoding="utf-8") as handle:
                handle.write("BROKEN EIGENVAL\n")

            inputs = BandToolInput(
                material_id="reuse-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
            )
            with patch.dict(os.environ, {"MOBILITY_REUSE_COMPLETED_VASP_STAGES": "true"}):
                with patch("mobility_agent.tools.band_tool.run_vasp", return_value=False) as run_vasp:
                    with patch(
                        "mobility_agent.tools.band_tool.find_band_edges_from_eigenval_occupancy",
                        side_effect=ValueError("bad eigenval"),
                    ):
                        result = BandTool()._execute(inputs)

            run_vasp.assert_called_once()
            self.assertEqual(result["errors"], ["BAND 失败"])
            self.assertIn("ignored_invalid_reused_band_eigenval:bad eigenval", result["warnings"])
            self.assertFalse(os.path.exists(os.path.join(band_dir, "EIGENVAL")))

    def test_band_inherits_scf_chgcar_compatible_incar_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            scf_dir = os.path.join(base_dir, "02_scf")
            os.makedirs(scf_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            os.makedirs(base_dir, exist_ok=True)
            _write_structure(poscar)
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(os.path.join(scf_dir, "CHGCAR"), "w", encoding="utf-8") as handle:
                handle.write("FAKE CHGCAR\n")
            Incar(
                {
                    "SYSTEM": "SCF Calculation",
                    "ICHARG": 2,
                    "ENCUT": 600,
                    "PREC": "Accurate",
                    "LREAL": False,
                    "LASPH": True,
                    "ADDGRID": True,
                    "LCHARG": True,
                    "LWAVE": True,
                }
            ).write_file(os.path.join(scf_dir, "INCAR"))

            inputs = BandToolInput(
                material_id="band-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
            )
            with patch("mobility_agent.tools.band_tool.run_vasp", return_value=False):
                result = BandTool()._execute(inputs)

            self.assertEqual(result["errors"], ["BAND 失败"])
            incar = Incar.from_file(os.path.join(base_dir, "03_band", "INCAR"))
            self.assertEqual(incar["ICHARG"], 11)
            self.assertEqual(incar["PREC"], "Accurate")
            self.assertFalse(incar["LREAL"])
            self.assertTrue(incar["LASPH"])
            self.assertTrue(incar["ADDGRID"])
            self.assertFalse(incar["LCHARG"])
            self.assertFalse(incar["LWAVE"])


if __name__ == "__main__":
    unittest.main()
