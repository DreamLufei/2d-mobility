from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.io.vasp.inputs import Incar

from mobility_agent.tools.mass_tool import MassTool, MassToolInput
from mobility_agent.tools.physics_common import read_fermi_energy_eV


class MassToolIncarTests(unittest.TestCase):
    def test_effective_mass_inherits_scf_chgcar_compatible_incar_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "mobility_calculation")
            scf_dir = os.path.join(base_dir, "02_scf")
            os.makedirs(scf_dir, exist_ok=True)
            poscar = os.path.join(base_dir, "POSCAR")
            potcar = os.path.join(base_dir, "POTCAR")
            os.makedirs(base_dir, exist_ok=True)
            with open(poscar, "w", encoding="utf-8") as handle:
                handle.write(
                    "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\n"
                    "Si\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
                )
            with open(potcar, "w", encoding="utf-8") as handle:
                handle.write("FAKE POTCAR\n")
            with open(os.path.join(scf_dir, "CHGCAR"), "w", encoding="utf-8") as handle:
                handle.write("fake charge density\n")
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

            inputs = MassToolInput(
                material_id="mass-test",
                base_dir=base_dir,
                poscar_path=poscar,
                potcar_path=potcar,
                reciprocal_lattice=[
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.3],
                ],
                vbm_kpoint=[0.0, 0.0, 0.0],
                cbm_kpoint=[0.1, 0.0, 0.0],
                vbm_band_index=0,
                cbm_band_index=1,
            )
            with patch("mobility_agent.tools.mass_tool.run_vasp", return_value=False):
                result = MassTool()._execute(inputs)

            self.assertEqual(result["errors"], ["electron x 有效质量计算失败"])
            incar = Incar.from_file(os.path.join(base_dir, "04_effmass_electron_x", "INCAR"))
            self.assertEqual(incar["ICHARG"], 11)
            self.assertEqual(incar["PREC"], "Accurate")
            self.assertFalse(incar["LREAL"])
            self.assertTrue(incar["LASPH"])
            self.assertTrue(incar["ADDGRID"])
            self.assertFalse(incar["LCHARG"])
            self.assertFalse(incar["LWAVE"])

    def test_read_fermi_energy_prefers_outcar_without_vasprun(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "OUTCAR"), "w", encoding="utf-8") as handle:
                handle.write(" E-fermi :  -4.3210     XC(G=0):  0.0000\n")

            self.assertAlmostEqual(read_fermi_energy_eV(tmpdir), -4.3210)

    def test_dynamic_band_edge_uses_fermi_energy_when_available(self) -> None:
        energies = np.asarray(
            [
                [
                    [-5.0, -4.0, 1.0],
                    [-5.1, -3.9, 1.1],
                ]
            ],
            dtype=float,
        )
        occupations = np.full_like(energies, 0.25)

        tool = MassTool()
        hole_edge, hole_band, source = tool._dynamic_band_edge(
            energies,
            occupations,
            spin_idx=0,
            carrier_type="hole",
            fermi_energy=-4.5,
        )
        electron_edge, electron_band, electron_source = tool._dynamic_band_edge(
            energies,
            occupations,
            spin_idx=0,
            carrier_type="electron",
            fermi_energy=-4.5,
        )

        self.assertEqual(source, "fermi_energy")
        self.assertEqual(electron_source, "fermi_energy")
        self.assertEqual(hole_band.tolist(), [0, 0])
        self.assertEqual(electron_band.tolist(), [1, 1])
        self.assertTrue(np.allclose(hole_edge, [-5.0, -5.1]))
        self.assertTrue(np.allclose(electron_edge, [-4.0, -3.9]))

    def test_dynamic_selector_switch_is_warning_not_rejection(self) -> None:
        k_array = np.linspace(-0.04, 0.04, 9)
        branch = -5.0 - 3.81 * np.square(k_array)
        dynamic_bands = np.asarray([4, 4, 4, 5, 5, 5, 5, 5, 5], dtype=int)

        mass, diag = MassTool()._fit_mass_branch(
            carrier_type="hole",
            direction="x",
            k_array=k_array,
            branch_energy=branch,
            fixed_band_index=4,
            dynamic_band_indices=dynamic_bands,
            dynamic_edge_energy=branch,
            fixed_branch_occupations=np.ones_like(branch),
            fermi_energy=-4.5,
            fermi_source="scf_state",
            dynamic_selector_source="fermi_energy",
        )

        self.assertIsNotNone(mass)
        self.assertEqual(diag["status"], "accepted")
        self.assertTrue(diag["dynamic_band_switch"])
        self.assertIn("fermi_energy_edge_selector_switch", diag["warnings"])
        self.assertNotIn("band_edge_selector_switch", diag["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
