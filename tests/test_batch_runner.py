from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from mobility_agent.graph.state import MaterialRunOutcome
from mobility_agent.runtime.batch_config import BatchConfig
from mobility_agent.runtime.batch_runner import run_mongo_batch
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


class BatchRunnerTests(unittest.TestCase):
    def test_batch_runner_reuses_single_material_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="test-batch",
                runs_root=tmpdir,
                potcar_method="concat",
                vaspkit_cmd="vaspkit",
                vaspkit_task=103,
                potcar_root=tmpdir,
                potcar_map_path=None,
                retry_failed=False,
                running_stale_s=3600,
            )
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                store_path=os.path.join(tmpdir, "batch_store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            structure = Structure(
                lattice=Lattice.hexagonal(3.0, 20.0),
                species=["Si", "Si"],
                coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
            )
            doc = {"_id": "doc-1", "material_id": "mat-1", "structure": structure.as_dict()}
            claims = [doc, None]

            def fake_claim(*args, **kwargs):
                return claims.pop(0)

            def fake_build_potcar(struct, *, potcar_root, dest_path, potcar_map_path=None):
                with open(dest_path, "w", encoding="utf-8") as handle:
                    handle.write("FAKE POTCAR\n")
                return ["Si"]

            handles = SimpleNamespace(client=SimpleNamespace(close=lambda: None), collection=object())
            completed_calls: list[dict] = []
            failed_calls: list[dict] = []

            with patch_test_llm_clients(), patch("mobility_agent.runtime.batch_runner.connect", return_value=handles), patch(
                "mobility_agent.runtime.batch_runner.claim_next_material", side_effect=fake_claim
            ), patch(
                "mobility_agent.runtime.batch_runner.build_potcar", side_effect=fake_build_potcar
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_completed",
                side_effect=lambda *args, **kwargs: completed_calls.append(kwargs),
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_failed",
                side_effect=lambda *args, **kwargs: failed_calls.append(kwargs),
            ):
                final_state = run_mongo_batch(cfg=cfg, runtime=runtime, thread_id="batch-test", fresh_materials=True)

            stats = final_state["batch"]["global_statistics"]
            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(
                stats["scientifically_passed"]
                + stats["scientifically_warning"]
                + stats["scientifically_failed"]
                + stats["scientifically_unknown"],
                1,
            )
            self.assertEqual(len(completed_calls), 1)
            self.assertEqual(len(failed_calls), 0)
            self.assertEqual(completed_calls[0]["final_status"], "completed")
            self.assertEqual(completed_calls[0]["quality_grade"], "high_confidence")
            self.assertEqual(completed_calls[0]["accepted_channels"], ["x", "y"])

    def test_batch_runner_writes_material_quality_label_for_completed_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="test-batch",
                runs_root=tmpdir,
                potcar_method="concat",
                vaspkit_cmd="vaspkit",
                vaspkit_task=103,
                potcar_root=tmpdir,
                potcar_map_path=None,
                retry_failed=False,
                running_stale_s=3600,
            )
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                store_path=os.path.join(tmpdir, "batch_store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            structure = Structure(
                lattice=Lattice.hexagonal(3.0, 20.0),
                species=["Si", "Si"],
                coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
            )
            doc = {"_id": "doc-1", "material_id": "mat-1", "structure": structure.as_dict()}
            claims = [doc, None]

            def fake_claim(*args, **kwargs):
                return claims.pop(0)

            def fake_build_potcar(struct, *, potcar_root, dest_path, potcar_map_path=None):
                with open(dest_path, "w", encoding="utf-8") as handle:
                    handle.write("FAKE POTCAR\n")
                return ["Si"]

            def fake_material_runner(**kwargs):
                workdir = kwargs["workdir"]
                return MaterialRunOutcome(
                    task_id="quality-task",
                    material_id=kwargs["material_id"],
                    final_status="completed",
                    workdir=workdir,
                    results={
                        "material_id": kwargs["material_id"],
                        "temperature_K": 300.0,
                        "results_by_direction": {
                            "x": {
                                "n_points": 8,
                                "electron": {
                                    "mobility_cm2_Vs": 1800.0,
                                    "E1_eV": 1.2,
                                    "E1_eV_sigma": 0.08,
                                    "E1_fit_R2": 0.992,
                                    "C2D_J_m2": 55.0,
                                    "C2D_sigma_J_m2": 2.0,
                                    "C2D_fit_R2": 0.991,
                                },
                            }
                        },
                    },
                    warnings=[],
                    errors=[],
                    validation_report={},
                    final_summary={"material_id": kwargs["material_id"], "run_status": "completed"},
                )

            handles = SimpleNamespace(client=SimpleNamespace(close=lambda: None), collection=object())
            completed_calls: list[dict] = []

            with patch_test_llm_clients(), patch("mobility_agent.runtime.batch_runner.connect", return_value=handles), patch(
                "mobility_agent.runtime.batch_runner.claim_next_material", side_effect=fake_claim
            ), patch(
                "mobility_agent.runtime.batch_runner.build_potcar", side_effect=fake_build_potcar
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_completed",
                side_effect=lambda *args, **kwargs: completed_calls.append(kwargs),
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_failed",
                side_effect=AssertionError("unexpected failure"),
            ):
                final_state = run_mongo_batch(
                    cfg=cfg,
                    runtime=runtime,
                    thread_id="batch-test",
                    fresh_materials=True,
                    material_runner=fake_material_runner,
                )

            self.assertEqual(final_state["batch"]["global_statistics"]["processed"], 1)
            self.assertEqual(len(completed_calls), 1)
            self.assertEqual(completed_calls[0]["quality_label"], "high-quality")

    def test_batch_runner_promotes_skipped_with_results_to_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="test-batch",
                runs_root=tmpdir,
                potcar_method="concat",
                vaspkit_cmd="vaspkit",
                vaspkit_task=103,
                potcar_root=tmpdir,
                potcar_map_path=None,
                retry_failed=False,
                running_stale_s=3600,
            )
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                store_path=os.path.join(tmpdir, "batch_store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            structure = Structure(
                lattice=Lattice.hexagonal(3.0, 20.0),
                species=["Si", "Si"],
                coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
            )
            doc = {"_id": "doc-1", "material_id": "mat-1", "structure": structure.as_dict()}
            claims = [doc, None]

            def fake_claim(*args, **kwargs):
                return claims.pop(0)

            def fake_build_potcar(struct, *, potcar_root, dest_path, potcar_map_path=None):
                with open(dest_path, "w", encoding="utf-8") as handle:
                    handle.write("FAKE POTCAR\n")
                return ["Si"]

            def fake_material_runner(**kwargs):
                workdir = kwargs["workdir"]
                return MaterialRunOutcome(
                    task_id="promote-skip-task",
                    material_id=kwargs["material_id"],
                    final_status="completed",
                    final_acceptance="fail",
                    termination_reason="validation_finalized_without_followup_action",
                    workdir=workdir,
                    results={
                        "material_id": kwargs["material_id"],
                        "temperature_K": 300.0,
                        "results_by_direction": {
                            "x": {
                                "n_points": 9,
                                "electron": {
                                    "mobility_cm2_Vs": 1500.0,
                                    "E1_eV": 1.2,
                                    "E1_eV_sigma": 0.12,
                                    "E1_fit_R2": 0.76,
                                    "C2D_J_m2": 55.0,
                                    "C2D_sigma_J_m2": 2.1,
                                    "C2D_fit_R2": 0.95,
                                },
                            }
                        },
                    },
                    warnings=[],
                    errors=[],
                    validation_report={"decision": "fail", "quality_grade": "low_confidence"},
                    final_summary={"material_id": kwargs["material_id"], "run_status": "completed"},
                    stage_status={
                        "prepare": "success",
                        "relax": "success",
                        "scf": "success",
                        "band": "success",
                        "effective_mass": "success",
                        "strain_loop": "success",
                        "mobility": "success",
                        "validation": "success",
                    },
                )

            handles = SimpleNamespace(client=SimpleNamespace(close=lambda: None), collection=object())
            completed_calls: list[dict] = []

            with patch_test_llm_clients(), patch("mobility_agent.runtime.batch_runner.connect", return_value=handles), patch(
                "mobility_agent.runtime.batch_runner.claim_next_material", side_effect=fake_claim
            ), patch(
                "mobility_agent.runtime.batch_runner.build_potcar", side_effect=fake_build_potcar
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_completed",
                side_effect=lambda *args, **kwargs: completed_calls.append(kwargs),
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_failed",
                side_effect=AssertionError("unexpected failure"),
            ), patch(
                "mobility_agent.runtime.batch_runner._mark_skipped",
                side_effect=AssertionError("unexpected skipped write"),
            ):
                final_state = run_mongo_batch(
                    cfg=cfg,
                    runtime=runtime,
                    thread_id="batch-test",
                    fresh_materials=True,
                    material_runner=fake_material_runner,
                )

            stats = final_state["batch"]["global_statistics"]
            self.assertEqual(stats["processed"], 1)
            self.assertEqual(stats["succeeded"], 1)
            self.assertEqual(stats["failed"], 0)
            self.assertEqual(stats["skipped"], 0)
            self.assertEqual(stats["scientifically_failed"], 1)
            self.assertEqual(len(completed_calls), 1)
            self.assertEqual(completed_calls[0]["final_status"], "completed")
            self.assertEqual(completed_calls[0]["final_acceptance"], "fail")

    def test_batch_runner_forwards_loop_metadata_to_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = BatchConfig(
                mongo_uri="mongodb://example",
                mongo_db="db",
                mongo_collection="collection",
                batch_tag="test-batch",
                runs_root=tmpdir,
                potcar_method="concat",
                vaspkit_cmd="vaspkit",
                vaspkit_task=103,
                potcar_root=tmpdir,
                potcar_map_path=None,
                retry_failed=False,
                running_stale_s=3600,
            )
            runtime = RuntimeContext(
                agent_runtime=build_test_agent_runtime(),
                hitl_policy="non_interactive_skip_on_timeout",
                dry_run=True,
                store_path=os.path.join(tmpdir, "batch_store.sqlite"),
                compatibility_export_enabled=False,
                compatibility_export_pickle=False,
            )
            structure = Structure(
                lattice=Lattice.hexagonal(3.0, 20.0),
                species=["Si", "Si"],
                coords=[[0.0, 0.0, 0.5], [1 / 3, 2 / 3, 0.5]],
            )
            doc = {
                "_id": "doc-1",
                "material_id": "loop_01__mat-1",
                "structure": structure.as_dict(),
                "loop_metadata": {
                    "round_index": 1,
                    "round_id": "loop_01",
                    "pipeline_run_id": "loop_01_pipeline",
                },
            }
            claims = [doc, None]

            def fake_claim(*args, **kwargs):
                return claims.pop(0)

            def fake_build_potcar(struct, *, potcar_root, dest_path, potcar_map_path=None):
                with open(dest_path, "w", encoding="utf-8") as handle:
                    handle.write("FAKE POTCAR\n")
                return ["Si"]

            handles = SimpleNamespace(client=SimpleNamespace(close=lambda: None), collection=object())
            completed_calls: list[dict] = []

            with patch_test_llm_clients(), patch("mobility_agent.runtime.batch_runner.connect", return_value=handles), patch(
                "mobility_agent.runtime.batch_runner.claim_next_material", side_effect=fake_claim
            ), patch(
                "mobility_agent.runtime.batch_runner.build_potcar", side_effect=fake_build_potcar
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_completed",
                side_effect=lambda *args, **kwargs: completed_calls.append(kwargs),
            ), patch(
                "mobility_agent.runtime.batch_runner.mark_failed",
                side_effect=AssertionError("unexpected failure"),
            ), patch(
                "mobility_agent.runtime.batch_runner._mark_skipped",
                side_effect=AssertionError("unexpected skipped write"),
            ):
                final_state = run_mongo_batch(
                    cfg=cfg,
                    runtime=runtime,
                    thread_id="batch-test",
                    fresh_materials=True,
                )

            self.assertEqual(final_state["batch"]["global_statistics"]["processed"], 1)
            self.assertEqual(len(completed_calls), 1)
            self.assertEqual(completed_calls[0]["round_index"], 1)
            self.assertEqual(completed_calls[0]["round_id"], "loop_01")
            self.assertEqual(completed_calls[0]["pipeline_run_id"], "loop_01_pipeline")


if __name__ == "__main__":
    unittest.main()
