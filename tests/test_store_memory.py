from __future__ import annotations

import tempfile
import unittest

from mobility_agent.memory import (
    find_recovery_cases,
    find_validation_heuristics,
    list_batch_statistics,
    list_skill_metadata,
    open_memory_store,
    record_batch_statistics,
    record_recovery_case,
    record_skill_metadata,
    record_validation_heuristic,
)


class StoreMemoryTests(unittest.TestCase):
    def test_langgraph_store_helpers_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with open_memory_store(tmpdir) as store:
                record_recovery_case(
                    store,
                    task_id="task-1",
                    payload={
                        "stage": "relax",
                        "error_signature": "zbrent_fatal",
                        "chosen_action": "modify_params_and_retry",
                        "success_or_failure": "completed",
                    },
                )
                record_validation_heuristic(
                    store,
                    heuristic_name="negative_mobility",
                    payload={
                        "heuristic_name": "negative_mobility",
                        "description": "negative mobility observed historically",
                        "trigger_pattern": "negative_mobility",
                        "recommendation": "fail",
                        "severity": "error",
                    },
                )
                record_batch_statistics(
                    store,
                    collection_name="demo_collection",
                    payload={"collection_name": "demo_collection", "processed": 2, "failed": 1},
                )
                record_skill_metadata(
                    store,
                    skill_name="single_material_mobility",
                    payload={"skill_name": "single_material_mobility", "description": "demo"},
                )

                self.assertEqual(len(find_recovery_cases(store, stage="relax", error_signature="zbrent")), 1)
                self.assertEqual(len(find_validation_heuristics(store, anomaly_flags=["negative_mobility"])), 1)
                self.assertEqual(len(list_batch_statistics(store, collection_name="demo_collection")), 1)
                self.assertEqual(len(list_skill_metadata(store)), 1)


if __name__ == "__main__":
    unittest.main()
