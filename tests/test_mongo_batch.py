from __future__ import annotations

import unittest

from mobility_agent.runtime.mongo_batch import mark_completed, mark_failed


class _FakeCollection:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def update_one(self, query, update):  # type: ignore[no-untyped-def]
        self.calls.append((dict(query), dict(update)))


class MongoBatchTests(unittest.TestCase):
    def test_mark_completed_projects_science_metadata_and_clears_failed_fields(self) -> None:
        collection = _FakeCollection()
        validation = {
            "decision": "fail",
            "quality_grade": "low_confidence",
            "channel_reviews": {
                "electron_x": {
                    "status": "rejected",
                    "reason": "catastrophic_fit_quality",
                    "direction": "x",
                    "carrier": "electron",
                    "n_points": 9,
                    "mobility_cm2_Vs": -2.0,
                    "E1_fit_R2": 0.23,
                    "C2D_fit_R2": 0.47,
                },
                "hole_x": {
                    "status": "accepted_with_warning",
                    "reason": "low_fit_margin",
                    "direction": "x",
                    "carrier": "hole",
                    "n_points": 9,
                    "mobility_cm2_Vs": 120.0,
                    "E1_fit_R2": 0.81,
                    "C2D_fit_R2": 0.92,
                },
            },
        }

        mark_completed(
            collection,
            doc_id="doc-1",
            results={"results_by_direction": {"x": {"electron": {"mobility_cm2_Vs": 1.0}}}},
            run_dir="/tmp/run",
            quality_label="not retained",
            validation=validation,
            final_status="skipped",
            final_acceptance="fail",
            quality_grade="low_confidence",
            accepted_channels=["x"],
            rejected_channels=["y"],
        )

        self.assertEqual(len(collection.calls), 1)
        _, update = collection.calls[0]
        payload = update["$set"]
        self.assertEqual(payload["mobility_calc.status"], "completed")
        self.assertEqual(payload["mobility_calc.scientific_decision"], "fail")
        self.assertEqual(payload["mobility_calc.quality_grade"], "low_confidence")
        self.assertEqual(payload["mobility_agent.status"], "completed")
        self.assertEqual(payload["mobility_agent.final_status"], "skipped")
        self.assertEqual(payload["mobility_calc.channel_labels"]["electron_x"]["status"], "rejected")
        self.assertEqual(payload["mobility_calc.channel_labels"]["hole_x"]["status"], "accepted_with_warning")
        self.assertEqual(payload["mobility_calc.channel_labels"]["electron_y"]["status"], "unknown")
        self.assertIn("mobility_calc.failed_at", update["$unset"])
        self.assertIn("mobility_calc.error", update["$unset"])

    def test_mark_failed_clears_success_residue(self) -> None:
        collection = _FakeCollection()

        mark_failed(
            collection,
            doc_id="doc-2",
            error="runner_exception",
            run_dir="/tmp/run",
            final_status="failed",
        )

        self.assertEqual(len(collection.calls), 1)
        _, update = collection.calls[0]
        self.assertEqual(update["$set"]["mobility_calc.status"], "failed")
        self.assertIn("mobility_calc.results", update["$unset"])
        self.assertIn("mobility_calc.completed_at", update["$unset"])
        self.assertIn("mobility_calc.channel_labels", update["$unset"])
        self.assertIn("mobility_agent.completed_at", update["$unset"])


if __name__ == "__main__":
    unittest.main()
