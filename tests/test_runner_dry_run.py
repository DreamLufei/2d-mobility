from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.schemas import ArbitrationRecord, HITLDecision
from mobility_agent.graph.state import apply_state_patch, build_material_outcome, make_initial_material_state
from mobility_agent.runtime.agentic_controller import (
    AgenticMaterialController,
    _ensure_selected_retry_accounted_after_execution,
    _ensure_validation_finalize_ready,
)
from mobility_agent.runtime.contracts import WorkflowContract
from mobility_agent.runtime.context import RuntimeContext
from mobility_agent.runtime.runner import run_single_material
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(
    *,
    store_path: str,
    fail_stages: tuple[str, ...] = (),
    hitl_policy: str = "non_interactive_skip_on_timeout",
) -> RuntimeContext:
    agent_runtime = build_test_agent_runtime(human_review_timeout_seconds=0)
    return RuntimeContext(
        agent_runtime=agent_runtime,
        hitl_policy=hitl_policy,
        dry_run=True,
        dry_run_fail_stages=fail_stages,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
    )


def _prepare_material_root(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "POSCAR"), "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(os.path.join(tmpdir, "POTCAR"), "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")


class RunnerDryRunTests(unittest.TestCase):
    def test_single_material_dry_run_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=runtime,
                    material_id="dry-run-ok",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.status, "completed")
            self.assertTrue(os.path.exists(outcome.artifact_paths["material_outcome_path"]))
            with open(outcome.artifact_paths["material_outcome_path"], "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "completed")
            workflow_contract_path = outcome.artifact_paths.get("workflow_contract_path", "")
            decision_ledger_path = outcome.artifact_paths.get("decision_ledger_path", "")
            execution_checkpoint_path = outcome.artifact_paths.get("execution_checkpoint_path", "")
            self.assertTrue(os.path.exists(workflow_contract_path))
            self.assertTrue(os.path.exists(decision_ledger_path))
            self.assertTrue(os.path.exists(execution_checkpoint_path))
            with open(workflow_contract_path, "r", encoding="utf-8") as handle:
                contract = json.load(handle)
            self.assertGreaterEqual(int(contract["version"]), 1)
            self.assertEqual(contract["council_mode"], "validation_followup_council")
            self.assertEqual(contract["planned_capabilities"], [])
            with open(decision_ledger_path, "r", encoding="utf-8") as handle:
                decision_ledger = json.load(handle)
            authored_segments = [
                list((item.get("summary") or {}).get("planned_capabilities", []) or [])
                for item in decision_ledger
                if str(item.get("entry_type") or "") == "workflow_contract_updated"
            ]
            self.assertIn(["prepare", "relax", "scf", "band", "effective_mass", "strain_loop"], authored_segments)
            self.assertIn(["mobility", "validation"], authored_segments)
            snapshot_path = runtime.state_snapshot_path_for(outcome.workdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            round_metrics = list((snapshot.get("services", {}) or {}).get("council_round_metrics", []) or [])
            self.assertGreaterEqual(len(round_metrics), 3)
            self.assertEqual(round_metrics[0]["council_mode"], "segment_council")
            self.assertEqual(round_metrics[-1]["council_mode"], "validation_followup_council")

    def test_planner_failure_aborts_in_strict_agentic_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with (
                patch_test_llm_clients(),
                patch("mobility_agent.runtime.agentic_controller.PlannerAgent.propose", side_effect=RuntimeError("planner_structured_output_invalid")),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_single_material(
                        runtime=_runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite")),
                        material_id="planner-fallback",
                        root_path=tmpdir,
                        fresh=True,
                    )
            self.assertIn("agentic_council_role_failed:planner", str(ctx.exception))

    def test_single_material_human_gate_skip_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            with patch_test_llm_clients():
                outcome = run_single_material(
                    runtime=_runtime(
                        store_path=os.path.join(tmpdir, "runner_store.sqlite"),
                        fail_stages=("scf",),
                    ),
                    material_id="dry-run-skip",
                    root_path=tmpdir,
                    fresh=True,
                )
            self.assertEqual(outcome.status, "failed")
            self.assertEqual(outcome.final_status, "skipped")
            payload_path = os.path.join(outcome.workdir, "human_escalation_payload.json")
            self.assertTrue(os.path.exists(payload_path))

    def test_noop_arbitration_becomes_controlled_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))

            def _always_noop_arbitrate(self, state, proposals, critiques, preferences, round_id):  # type: ignore[no-untyped-def]
                del proposals, critiques, preferences
                return ArbitrationRecord(
                    agent_name="orchestrator",
                    round_id=round_id,
                    target_task_id=str(state.get("task", {}).get("task_id") or ""),
                    selected_proposal_id=None,
                    selected_action=None,
                    rationale="no_legal_proposals_survived_deliberation_without_auto_fallback",
                    guardrail_notes=["no_legal_proposals"],
                    whether_noop=True,
                )

            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _always_noop_arbitrate):
                outcome = run_single_material(
                    runtime=runtime,
                    material_id="dry-run-noop",
                    root_path=tmpdir,
                    fresh=True,
                )

            self.assertEqual(outcome.status, "failed")
            snapshot_path = runtime.state_snapshot_path_for(outcome.workdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["workflow"]["termination_reason"], "agentic_no_viable_action_after_arbitration")
            diagnostic_codes = [
                str(item.get("code") or "")
                for item in list((snapshot.get("services", {}) or {}).get("framework_diagnostics", []) or [])
                if isinstance(item, dict)
            ]
            self.assertIn("agentic_no_selected_action_after_arbitration", diagnostic_codes)

    def test_agentic_retry_execution_reconciles_missing_retry_count_patch(self) -> None:
        state = {
            "execution": {
                "retry_counts": {"effective_mass": 1},
                "latest_execution_observation": {
                    "action_family": "retry_capability",
                    "target_capability": "effective_mass",
                    "status": "failed",
                    "error_summary": "electron x effective mass failed",
                },
            },
            "workflow": {
                "retry_counts": {"effective_mass": 1},
                "retry_budget": 3,
            },
        }
        selected_action = {
            "action_family": "retry_capability",
            "target_capability": "effective_mass",
        }

        updated = _ensure_selected_retry_accounted_after_execution(
            state,
            selected_action=selected_action,
            before_retry_counts={"effective_mass": 1},
        )

        self.assertEqual(updated["execution"]["retry_counts"]["effective_mass"], 2)
        self.assertEqual(updated["workflow"]["retry_counts"]["effective_mass"], 2)

    def test_low_fit_mobility_completion_reopens_validation_followup_council(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            controller = AgenticMaterialController(runtime)
            state = make_initial_material_state(
                material_id="dry-run-low-fit",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            state["execution"]["latest_execution_observation"] = {
                "target_capability": "mobility",
                "status": "success",
                "risk_flags": ["fit_quality_below_threshold:0.76"],
            }
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.7665,
                "effective_fit_quality": 0.7665,
            }
            contract = WorkflowContract(planned_capabilities=["mobility", "validation"])

            reason = controller._deliberation_reason(state, contract)

            self.assertEqual(reason, "post_mobility_quality_review")
            self.assertEqual(controller._council_mode(state, reason=reason), "validation_followup_council")

    def test_validation_followup_noop_is_skipped_instead_of_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))

            def _always_noop_arbitrate(self, state, proposals, critiques, preferences, round_id):  # type: ignore[no-untyped-def]
                del proposals, critiques, preferences
                return ArbitrationRecord(
                    agent_name="orchestrator",
                    round_id=round_id,
                    target_task_id=str(state.get("task", {}).get("task_id") or ""),
                    selected_proposal_id=None,
                    selected_action=None,
                    rationale="no_legal_proposals_survived_deliberation_without_auto_fallback",
                    guardrail_notes=["no_legal_proposals"],
                    whether_noop=True,
                )

            with (
                patch_test_llm_clients(),
                patch.object(OrchestratorAgent, "arbitrate", _always_noop_arbitrate),
                patch.object(AgenticMaterialController, "_deliberation_reason", return_value="validation_followup"),
            ):
                outcome = run_single_material(
                    runtime=runtime,
                    material_id="dry-run-validation-noop",
                    root_path=tmpdir,
                    fresh=True,
                )

            self.assertEqual(outcome.status, "skipped")
            snapshot_path = runtime.state_snapshot_path_for(outcome.workdir)
            with open(snapshot_path, "r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
            self.assertEqual(snapshot["workflow"]["termination_reason"], "validation_followup_no_viable_action")
            self.assertEqual(snapshot["workflow"]["run_status"], "skipped")

    def test_validation_followup_noop_with_mobility_results_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            original_arbitrate = OrchestratorAgent.arbitrate

            def _conditional_noop_arbitrate(self, state, proposals, critiques, preferences, round_id):  # type: ignore[no-untyped-def]
                reason = str(
                    (((state.get("services", {}) or {}).get("runtime_strategy", {}) or {}).get("deliberation_reason") or "")
                )
                if reason == "validation_followup":
                    return ArbitrationRecord(
                        agent_name="orchestrator",
                        round_id=round_id,
                        target_task_id=str(state.get("task", {}).get("task_id") or ""),
                        selected_proposal_id=None,
                        selected_action=None,
                        rationale="no_legal_proposals_survived_validation_followup",
                        guardrail_notes=["no_legal_proposals"],
                        whether_noop=True,
                    )
                return original_arbitrate(
                    self,
                    state=state,
                    proposals=proposals,
                    critiques=critiques,
                    preferences=preferences,
                    round_id=round_id,
                )

            with patch_test_llm_clients(), patch.object(OrchestratorAgent, "arbitrate", _conditional_noop_arbitrate):
                outcome = run_single_material(
                    runtime=runtime,
                    material_id="dry-run-validation-noop-completed",
                    root_path=tmpdir,
                    fresh=True,
                )

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.final_status, "completed")
            self.assertEqual(outcome.termination_reason, "validation_finalized_without_followup_action")

    def test_validation_followup_noop_synthesizes_validation_report_for_completed_compute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            state = make_initial_material_state(
                material_id="dry-run-validation-synth",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            os.makedirs(state["execution"]["workdir"], exist_ok=True)
            state["workflow"]["run_status"] = "running"
            state["workflow"]["completed_stages"] = [
                "prepare",
                "relax",
                "scf",
                "band",
                "effective_mass",
                "strain_loop",
                "mobility",
            ]
            state["workflow"]["stage_status"] = {
                "prepare": "success",
                "relax": "success",
                "scf": "success",
                "band": "success",
                "effective_mass": "success",
                "strain_loop": "success",
                "mobility": "success",
            }
            state["diagnostics"]["fit_diagnostics"] = {"fit_r2_min": 0.277, "effective_fit_quality": 0.277}
            state["physics_results"]["results"] = {
                "material_id": "dry-run-validation-synth",
                "temperature_K": 300.0,
                "results_by_direction": {
                    "x": {
                        "electron": {"mobility_cm2_Vs": -0.68},
                        "hole": {"mobility_cm2_Vs": -508.5},
                        "n_points": 9,
                    },
                    "y": {
                        "electron": {"mobility_cm2_Vs": -72.3},
                        "hole": {"mobility_cm2_Vs": -1142.0},
                        "n_points": 9,
                    },
                },
            }

            ready = _ensure_validation_finalize_ready(state)

            self.assertTrue(ready)
            self.assertEqual(state["workflow"]["stage_status"]["validation"], "success")
            self.assertIn("validation", state["workflow"]["completed_stages"])
            self.assertEqual(state["diagnostics"]["validation_report"].get("decision"), "fail")
            self.assertEqual(state["diagnostics"]["validation_report"].get("recommended_action"), "finalize")
            self.assertEqual(state["physics_results"]["accepted_channels"], [])
            self.assertEqual(sorted(state["physics_results"]["rejected_channels"]), ["x", "y"])

    def test_escalation_skip_material_with_mobility_results_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            controller = AgenticMaterialController(runtime)
            state = make_initial_material_state(
                material_id="dry-run-escalation-skip-promote",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            os.makedirs(state["execution"]["workdir"], exist_ok=True)
            state["execution"]["current_action"] = {
                "action_family": "escalate_human",
                "target_capability": "validation",
                "risk_class": "medium",
                "parameters": {"recommended_options": ["skip_material", "abort_task"]},
            }
            state["deliberation"]["round_index"] = 5
            state["workflow"]["completed_stages"] = [
                "prepare",
                "relax",
                "scf",
                "band",
                "effective_mass",
                "strain_loop",
                "mobility",
            ]
            state["workflow"]["stage_status"] = {
                "prepare": "success",
                "relax": "success",
                "scf": "success",
                "band": "success",
                "effective_mass": "success",
                "strain_loop": "success",
                "mobility": "success",
            }
            state["physics_results"]["results"] = {
                "material_id": "dry-run-escalation-skip-promote",
                "temperature_K": 300.0,
                "results_by_direction": {
                    "x": {
                        "electron": {"mobility_cm2_Vs": 1200.0},
                        "hole": {"mobility_cm2_Vs": 800.0},
                    }
                },
            }

            with patch(
                "mobility_agent.runtime.agentic_controller.resolve_human_decision",
                return_value=HITLDecision(action="skip_material", reason="test", source="precomputed"),
            ), patch(
                "mobility_agent.runtime.agentic_controller.notify_escalation",
                return_value={"sent": False},
            ):
                updated = controller._execute_escalation_action(state)

            self.assertEqual(updated["workflow"]["run_status"], "skipped")
            self.assertEqual(updated["workflow"]["termination_reason"], "skip_material_with_computed_results")

    def test_timeout_second_attempt_with_mobility_results_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            controller = AgenticMaterialController(runtime)
            state = make_initial_material_state(
                material_id="dry-run-timeout-promote",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            os.makedirs(state["execution"]["workdir"], exist_ok=True)
            state["execution"]["current_action"] = {
                "action_family": "escalate_human",
                "target_capability": "validation",
                "risk_class": "medium",
                "parameters": {"recommended_options": ["rerun_previous_stage", "skip_material", "abort_task"]},
            }
            state["deliberation"]["round_index"] = 6
            state["diagnostics"]["recovery_history"] = [
                {"origin": "timeout_auto", "target_stage": "validation", "reason": "previous_timeout"}
            ]
            state["workflow"]["completed_stages"] = [
                "prepare",
                "relax",
                "scf",
                "band",
                "effective_mass",
                "strain_loop",
                "mobility",
            ]
            state["workflow"]["stage_status"] = {
                "prepare": "success",
                "relax": "success",
                "scf": "success",
                "band": "success",
                "effective_mass": "success",
                "strain_loop": "success",
                "mobility": "success",
            }
            state["physics_results"]["results"] = {
                "material_id": "dry-run-timeout-promote",
                "temperature_K": 300.0,
                "results_by_direction": {
                    "x": {
                        "electron": {"mobility_cm2_Vs": 1200.0},
                        "hole": {"mobility_cm2_Vs": 800.0},
                    }
                },
            }

            with patch(
                "mobility_agent.runtime.agentic_controller.resolve_human_decision",
                return_value=HITLDecision(action="rerun_previous_stage", reason="timeout", source="timeout_default"),
            ), patch(
                "mobility_agent.runtime.agentic_controller.notify_escalation",
                return_value={"sent": False},
            ):
                updated = controller._execute_escalation_action(state)

            self.assertEqual(updated["workflow"]["run_status"], "skipped")
            self.assertEqual(updated["workflow"]["termination_reason"], "skip_material_with_computed_results")
            latest_decision = dict(updated.get("services", {}).get("latest_human_decision", {}) or {})
            self.assertEqual(latest_decision.get("effective_action"), "skip_material")

    def test_validation_rejection_keeps_material_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)

            def _forced_validation_report(*args, **kwargs):  # type: ignore[no-untyped-def]
                del args, kwargs
                return {
                    "decision": "fail",
                    "reason": "forced_validation_rejection_for_test",
                    "warnings": ["fit_quality_low"],
                    "failed_checks": ["fit_quality_below_threshold"],
                    "fit_metrics": {"effective_fit_quality": 0.62, "fit_r2_min": 0.62},
                    "anomaly_flags": ["fit_quality_below_threshold"],
                    "channel_reviews": {},
                    "retained_subchannels": ["electron_x", "hole_x", "electron_y", "hole_y"],
                    "rejected_subchannels": [],
                    "accepted_channels": ["x", "y"],
                    "rejected_channels": [],
                    "recommended_action": "finalize",
                    "refinement_targets": [],
                    "refinement_preview": {
                        "target_channels": [],
                        "suggested_points": {},
                        "applied_points": {},
                        "full_plan_by_direction": {},
                        "refinement_strategy": "midpoint_enrichment",
                        "max_points_per_direction": 9,
                    },
                    "all_subchannels": ["electron_x", "hole_x", "electron_y", "hole_y"],
                }

            with (
                patch_test_llm_clients(),
                patch("mobility_agent.graph.runtime_nodes.build_validation_report", side_effect=_forced_validation_report),
            ):
                outcome = run_single_material(
                    runtime=_runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite")),
                    material_id="dry-run-validation-fail-completed",
                    root_path=tmpdir,
                    fresh=True,
                )

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.final_acceptance, "fail")
            self.assertEqual(outcome.validation_report.get("decision"), "fail")
            self.assertEqual(outcome.validation_report.get("quality_grade"), "low_confidence")
            workflow_contract_path = outcome.artifact_paths.get("workflow_contract_path", "")
            self.assertTrue(os.path.exists(workflow_contract_path))
            with open(workflow_contract_path, "r", encoding="utf-8") as handle:
                contract = json.load(handle)
            self.assertEqual(contract.get("plan_status"), "completed")

    def test_final_report_marks_validation_followup_finalize_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            controller = AgenticMaterialController(runtime)
            state = make_initial_material_state(
                material_id="dry-run-promote-skip",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            os.makedirs(state["execution"]["workdir"], exist_ok=True)
            state["workflow"]["run_status"] = "ready_to_finalize"
            state["workflow"]["termination_reason"] = "validation_finalized_without_followup_action"
            state["workflow"]["completed_stages"] = [
                "prepare",
                "relax",
                "scf",
                "band",
                "effective_mass",
                "strain_loop",
                "mobility",
                "validation",
            ]
            state["workflow"]["stage_status"] = {
                "prepare": "success",
                "relax": "success",
                "scf": "success",
                "band": "success",
                "effective_mass": "success",
                "strain_loop": "success",
                "mobility": "success",
                "validation": "success",
            }
            state["physics_results"]["results"] = {
                "material_id": "dry-run-promote-skip",
                "temperature_K": 300.0,
                "results_by_direction": {
                    "x": {
                        "electron": {"mobility_cm2_Vs": 1000.0},
                        "hole": {"mobility_cm2_Vs": 900.0},
                    }
                },
            }
            state["diagnostics"]["validation_report"] = {
                "decision": "fail",
                "reason": "forced_validation_rejection_for_test",
            }

            with patch(
                "mobility_agent.agents.reporter.ReporterAgent.summarize_material",
                return_value={"material_id": "dry-run-promote-skip", "run_status": "ready_to_finalize"},
            ):
                updated = apply_state_patch(state, controller.final_report_node(state))

            outcome = build_material_outcome(updated)
            self.assertEqual(updated["workflow"]["run_status"], "completed")
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.final_status, "completed")
            self.assertEqual(updated["diagnostics"]["quality_grade"], "low_confidence")

    def test_non_failure_escalation_does_not_send_email(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _prepare_material_root(tmpdir)
            runtime = _runtime(store_path=os.path.join(tmpdir, "runner_store.sqlite"))
            controller = AgenticMaterialController(runtime)
            state = make_initial_material_state(
                material_id="dry-run-nonfailure-escalation",
                root_path=tmpdir,
                workdir=os.path.join(tmpdir, "mobility_calculation"),
                poscar_path=os.path.join(tmpdir, "POSCAR"),
                potcar_path=os.path.join(tmpdir, "POTCAR"),
                user_goal="calculate_2d_mobility",
                decision_engine="llm_required",
                llm_required=True,
                llm_provider="openai",
                max_refinement_rounds=1,
                dry_run=True,
            ).to_dict()
            os.makedirs(state["execution"]["workdir"], exist_ok=True)
            state["workflow"]["run_status"] = "completed"
            state["workflow"]["stage_status"] = {
                "prepare": "success",
                "relax": "success",
                "scf": "success",
                "band": "success",
                "effective_mass": "success",
                "strain_loop": "success",
                "mobility": "success",
                "validation": "success",
            }
            state["execution"]["current_action"] = {
                "action_family": "escalate_human",
                "target_capability": "validation",
                "risk_class": "medium",
                "parameters": {"recommended_options": ["skip_material", "abort_task"]},
            }
            state["execution"]["latest_execution_observation"] = {
                "target_capability": "validation",
                "status": "success",
            }
            state["physics_results"]["results"] = {
                "material_id": "dry-run-nonfailure-escalation",
                "results_by_direction": {"x": {"electron": {"mobility_cm2_Vs": 1000.0}}},
            }
            state["diagnostics"]["validation_report"] = {
                "decision": "fail",
                "recommended_action": "finalize",
            }

            with patch(
                "mobility_agent.runtime.agentic_controller.resolve_human_decision",
                return_value=HITLDecision(action="skip_material", reason="test", source="precomputed"),
            ), patch(
                "mobility_agent.runtime.agentic_controller.notify_escalation",
                return_value={"sent": True},
            ) as notify_mock:
                updated = controller._execute_escalation_action(state)

            notify_mock.assert_not_called()
            trace = list(updated.get("diagnostics", {}).get("consultation_trace", []) or [])
            self.assertTrue(trace)
            self.assertEqual(trace[-1]["notify_result"].get("reason"), "non_failure_escalation_suppressed")


if __name__ == "__main__":
    unittest.main()
