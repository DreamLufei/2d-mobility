from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.graph.state import make_initial_material_state
from mobility_agent.runtime.validation_policy import build_validation_report


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
        material_id="validation-policy-test",
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


class ValidationPolicyTests(unittest.TestCase):
    def test_validation_rejects_catastrophic_single_subchannel_without_rejecting_whole_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["physics_results"]["results"] = {
                "results_by_direction": {
                    "x": {
                        "n_points": 5,
                        "electron": {
                            "mobility_cm2_Vs": 3186.0,
                            "E1_eV": 1.1,
                            "E1_eV_sigma": 0.05,
                            "E1_fit_R2": 0.997,
                            "C2D_J_m2": 55.0,
                            "C2D_sigma_J_m2": 2.0,
                            "C2D_fit_R2": 0.96,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 100.0,
                            "E1_eV": 1.4,
                            "E1_eV_sigma": 0.06,
                            "E1_fit_R2": 0.998,
                            "C2D_J_m2": 55.0,
                            "C2D_sigma_J_m2": 2.0,
                            "C2D_fit_R2": 0.96,
                        },
                    },
                    "y": {
                        "n_points": 5,
                        "electron": {
                            "mobility_cm2_Vs": 902.0,
                            "E1_eV": 3.4,
                            "E1_eV_sigma": 0.05,
                            "E1_fit_R2": 0.995,
                            "C2D_J_m2": 60.0,
                            "C2D_sigma_J_m2": 1.0,
                            "C2D_fit_R2": 0.997,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 1.26653648e8,
                            "E1_eV": 0.001,
                            "E1_eV_sigma": 0.02,
                            "E1_fit_R2": 0.0017,
                            "C2D_J_m2": 60.0,
                            "C2D_sigma_J_m2": 1.0,
                            "C2D_fit_R2": 0.997,
                        },
                    },
                }
            }
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.0017,
                "effective_fit_quality": 0.0017,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.95, "n_points": 5},
                    "y": {"effective_fit_quality": 0.0017, "n_points": 5},
                },
            }
            report = build_validation_report(state)
            self.assertEqual(report["recommended_action"], "finalize")
            self.assertIn("hole_y", report["rejected_subchannels"])
            self.assertIn("electron_y", report["retained_subchannels"])
            self.assertEqual(report["accepted_channels"], ["x", "y"])
            self.assertEqual(report["rejected_channels"], [])

    def test_validation_requests_refinement_for_recoverable_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["physics_results"]["results"] = {
                "results_by_direction": {
                    "x": {
                        "n_points": 5,
                        "electron": {
                            "mobility_cm2_Vs": 1800.0,
                            "E1_eV": 1.1,
                            "E1_eV_sigma": 0.10,
                            "E1_fit_R2": 0.78,
                            "C2D_J_m2": 52.0,
                            "C2D_sigma_J_m2": 2.1,
                            "C2D_fit_R2": 0.95,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 350.0,
                            "E1_eV": 1.4,
                            "E1_eV_sigma": 0.12,
                            "E1_fit_R2": 0.81,
                            "C2D_J_m2": 52.0,
                            "C2D_sigma_J_m2": 2.1,
                            "C2D_fit_R2": 0.95,
                        },
                    },
                    "y": {
                        "n_points": 5,
                        "electron": {
                            "mobility_cm2_Vs": 900.0,
                            "E1_eV": 1.6,
                            "E1_eV_sigma": 0.08,
                            "E1_fit_R2": 0.97,
                            "C2D_J_m2": 57.0,
                            "C2D_sigma_J_m2": 1.8,
                            "C2D_fit_R2": 0.98,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 420.0,
                            "E1_eV": 1.5,
                            "E1_eV_sigma": 0.08,
                            "E1_fit_R2": 0.96,
                            "C2D_J_m2": 57.0,
                            "C2D_sigma_J_m2": 1.8,
                            "C2D_fit_R2": 0.98,
                        },
                    },
                }
            }
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.78,
                "effective_fit_quality": 0.78,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.78, "n_points": 5},
                    "y": {"effective_fit_quality": 0.96, "n_points": 5},
                },
            }
            report = build_validation_report(state)
            self.assertEqual(report["recommended_action"], "refine_sampling")
            self.assertEqual(report["refinement_targets"], ["x"])
            self.assertTrue(report["refinement_preview"]["applied_points"]["x"])
            self.assertIn("electron_x", report["retained_subchannels"])

    def test_validation_keeps_unresolved_refine_candidates_after_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["refinement_rounds"] = 1
            state["workflow"]["max_refinement_rounds"] = 1
            state["physics_results"]["results"] = {
                "results_by_direction": {
                    "x": {
                        "n_points": 9,
                        "electron": {
                            "mobility_cm2_Vs": 1800.0,
                            "E1_eV": 1.1,
                            "E1_eV_sigma": 0.10,
                            "E1_fit_R2": 0.78,
                            "C2D_J_m2": 52.0,
                            "C2D_sigma_J_m2": 2.1,
                            "C2D_fit_R2": 0.95,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 350.0,
                            "E1_eV": 1.4,
                            "E1_eV_sigma": 0.12,
                            "E1_fit_R2": 0.81,
                            "C2D_J_m2": 52.0,
                            "C2D_sigma_J_m2": 2.1,
                            "C2D_fit_R2": 0.95,
                        },
                    },
                    "y": {
                        "n_points": 9,
                        "electron": {
                            "mobility_cm2_Vs": 900.0,
                            "E1_eV": 1.6,
                            "E1_eV_sigma": 0.08,
                            "E1_fit_R2": 0.97,
                            "C2D_J_m2": 57.0,
                            "C2D_sigma_J_m2": 1.8,
                            "C2D_fit_R2": 0.98,
                        },
                        "hole": {
                            "mobility_cm2_Vs": 420.0,
                            "E1_eV": 1.5,
                            "E1_eV_sigma": 0.08,
                            "E1_fit_R2": 0.96,
                            "C2D_J_m2": 57.0,
                            "C2D_sigma_J_m2": 1.8,
                            "C2D_fit_R2": 0.98,
                        },
                    },
                }
            }
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.78,
                "effective_fit_quality": 0.78,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.78, "n_points": 9},
                    "y": {"effective_fit_quality": 0.96, "n_points": 9},
                },
            }
            state["physics_results"]["strain_plan_by_direction"] = {
                "x": [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02],
                "y": [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02],
            }

            report = build_validation_report(state)

            self.assertEqual(report["recommended_action"], "finalize")
            self.assertEqual(report["accepted_channels"], ["x", "y"])
            self.assertEqual(report["rejected_channels"], [])
            self.assertIn("electron_x", report["retained_subchannels"])
            self.assertIn("hole_x", report["retained_subchannels"])
            self.assertEqual(report["channel_reviews"]["electron_x"]["status"], "accepted_with_warning")
            self.assertEqual(report["channel_reviews"]["electron_x"]["reason"], "refinement_budget_exhausted")
            self.assertIn("refinement_budget_exhausted", report["warnings"])

    def test_validation_keeps_subchannel_with_warning_when_effective_mass_qc_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["physics_results"]["results"] = {
                "results_by_direction": {
                    "x": {
                        "n_points": 5,
                        "electron": {
                            "mobility_cm2_Vs": None,
                            "raw_mobility_cm2_Vs": 475956.0,
                            "E1_eV": 3.2,
                            "E1_eV_sigma": 0.04,
                            "E1_fit_R2": 0.97,
                            "C2D_J_m2": 90.0,
                            "C2D_sigma_J_m2": 1.0,
                            "C2D_fit_R2": 0.99,
                            "mass_status": "rejected",
                            "mass_valid_for_mobility": False,
                            "mass_fit_R2": 0.77,
                            "mass_dynamic_band_switch": True,
                            "mass_rejection_reasons": ["band_edge_selector_switch"],
                        },
                        "hole": {
                            "mobility_cm2_Vs": 410.0,
                            "E1_eV": 1.4,
                            "E1_eV_sigma": 0.05,
                            "E1_fit_R2": 0.97,
                            "C2D_J_m2": 90.0,
                            "C2D_sigma_J_m2": 1.0,
                            "C2D_fit_R2": 0.99,
                        },
                    }
                }
            }

            report = build_validation_report(state)

            self.assertIn("electron_x", report["retained_subchannels"])
            self.assertIn("hole_x", report["retained_subchannels"])
            review = report["channel_reviews"]["electron_x"]
            self.assertEqual(review["status"], "accepted_with_warning")
            self.assertEqual(review["reason"], "effective_mass_quality_warning")
            self.assertEqual(review["mobility_cm2_Vs"], 475956.0)
            self.assertEqual(review["mass_rejection_reasons"], ["band_edge_selector_switch"])

    def test_validation_keeps_signed_mobility_with_warning_for_diagnostic_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["refinement_rounds"] = 1
            state["workflow"]["max_refinement_rounds"] = 1
            state["physics_results"]["strain_plan_by_direction"] = {
                "x": [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02],
            }
            state["physics_results"]["results"] = {
                "results_by_direction": {
                    "x": {
                        "n_points": 9,
                        "electron": {
                            "mobility_cm2_Vs": None,
                            "raw_mobility_cm2_Vs": -148245.5,
                            "E1_eV": 0.045,
                            "E1_eV_sigma": 0.01,
                            "E1_fit_R2": 0.0015,
                            "C2D_J_m2": 20.0,
                            "C2D_sigma_J_m2": 1.0,
                            "C2D_fit_R2": 0.53,
                            "mass_status": "rejected",
                            "mass_valid_for_mobility": False,
                            "mass_rejection_reasons": ["wrong_curvature_sign"],
                        },
                    }
                }
            }

            report = build_validation_report(state)

            review = report["channel_reviews"]["electron_x"]
            self.assertIn("electron_x", report["retained_subchannels"])
            self.assertEqual(review["status"], "accepted_with_warning")
            self.assertEqual(review["mobility_cm2_Vs"], -148245.5)
            self.assertIn("non_positive_signed_mobility", review["warning_reasons"])
            self.assertIn("severe_e1_fit_quality_warning", review["warning_reasons"])


if __name__ == "__main__":
    unittest.main()
