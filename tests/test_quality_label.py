from __future__ import annotations

import unittest

from mobility_agent.runtime.quality_label import (
    HIGH_QUALITY_LABEL,
    MODERATE_QUALITY_LABEL,
    NOT_RETAINED_LABEL,
    classify_material_quality,
)


def _carrier(
    *,
    mobility: float,
    e1: float,
    e1_sigma: float,
    e1_fit_r2: float,
    c2d: float,
    c2d_sigma: float,
    c2d_fit_r2: float,
) -> dict[str, float]:
    return {
        "mobility_cm2_Vs": mobility,
        "E1_eV": e1,
        "E1_eV_sigma": e1_sigma,
        "E1_fit_R2": e1_fit_r2,
        "C2D_J_m2": c2d,
        "C2D_sigma_J_m2": c2d_sigma,
        "C2D_fit_R2": c2d_fit_r2,
    }


class QualityLabelTests(unittest.TestCase):
    def test_classify_material_quality_marks_filtered_channels_as_high_quality(self) -> None:
        results = {
            "results_by_direction": {
                "x": {
                    "n_points": 8,
                    "electron": _carrier(
                        mobility=1800.0,
                        e1=1.2,
                        e1_sigma=0.08,
                        e1_fit_r2=0.992,
                        c2d=55.0,
                        c2d_sigma=2.0,
                        c2d_fit_r2=0.991,
                    )
                }
            }
        }

        self.assertEqual(classify_material_quality(results), HIGH_QUALITY_LABEL)

    def test_classify_material_quality_marks_caution_only_channels_as_moderate_quality(self) -> None:
        results = {
            "results_by_direction": {
                "y": {
                    "n_points": 5,
                    "hole": _carrier(
                        mobility=120.0,
                        e1=0.4,
                        e1_sigma=0.12,
                        e1_fit_r2=0.96,
                        c2d=30.0,
                        c2d_sigma=4.5,
                        c2d_fit_r2=0.97,
                    )
                }
            }
        }

        self.assertEqual(classify_material_quality(results), MODERATE_QUALITY_LABEL)

    def test_classify_material_quality_marks_weak_channels_as_not_retained(self) -> None:
        results = {
            "results_by_direction": {
                "x": {
                    "n_points": 8,
                    "electron": _carrier(
                        mobility=1500.0,
                        e1=0.001,
                        e1_sigma=0.1,
                        e1_fit_r2=0.2,
                        c2d=55.0,
                        c2d_sigma=10.0,
                        c2d_fit_r2=0.9,
                    )
                }
            }
        }

        self.assertEqual(classify_material_quality(results), NOT_RETAINED_LABEL)


if __name__ == "__main__":
    unittest.main()
