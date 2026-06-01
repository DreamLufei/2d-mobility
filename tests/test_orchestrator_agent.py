from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.refinement import RefinementAgent
from mobility_agent.agents.schemas import Proposal
from mobility_agent.graph.state import make_initial_material_state
from mobility_agent.runtime.context import RuntimeContext
from tests.llm_test_utils import build_test_agent_runtime, patch_test_llm_clients


def _runtime(store_path: str) -> RuntimeContext:
    return RuntimeContext(
        agent_runtime=build_test_agent_runtime(human_review_timeout_seconds=0),
        hitl_policy="non_interactive_skip_on_timeout",
        dry_run=True,
        store_path=store_path,
        compatibility_export_enabled=False,
        compatibility_export_pickle=False,
    )


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
        material_id="orch-test",
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


class OrchestratorAgentTests(unittest.TestCase):
    def test_no_proposals_waiting_external_becomes_noop_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["run_status"] = "waiting_external"
            state["workflow"]["wait_reason"] = "awaiting_external_event:scf"
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                record = orchestrator.arbitrate(state=state, proposals=[], critiques=[], preferences=[], round_id=1)
            self.assertTrue(record.whether_noop)
            self.assertTrue(record.whether_waiting_external)
            self.assertIsNone(record.selected_action)

    def test_no_proposals_after_resolved_tasks_marks_ready_to_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["task_board"]["pending_tasks"] = []
            state["task_board"]["active_tasks"] = []
            state["task_board"]["blocked_tasks"] = []
            state["task_board"]["abandoned_tasks"] = []
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                record = orchestrator.arbitrate(state=state, proposals=[], critiques=[], preferences=[], round_id=2)
            self.assertTrue(record.whether_noop)
            self.assertTrue(record.whether_ready_to_finalize)
            self.assertIsNone(record.selected_action)

    def test_failed_state_with_no_legal_proposals_stays_noop_without_auto_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["blackboard"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "scf",
                "error_summary": "dry_run_injected_failure:scf",
            }
            proposal = Proposal(
                agent_name="planner",
                round_id=3,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="planner::illegal::3::scf",
                action_family="run_capability",
                target_capability="scf",
                rationale="try_scf_anyway",
                confidence=0.6,
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=3)
            self.assertTrue(record.whether_noop)
            self.assertIsNone(record.selected_action)
            self.assertIn("no_legal_proposals", record.rationale)
            self.assertTrue(any(":illegal:" in note for note in record.guardrail_notes))

    def test_orchestrator_canonicalizes_selected_action_to_legal_proposal_execution_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.78,
                "effective_fit_quality": 0.78,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.78},
                    "y": {"effective_fit_quality": 0.94},
                },
            }
            state["physics_results"]["strain_plan_by_direction"] = {
                "x": [-0.02, -0.01, 0.0, 0.01, 0.02],
                "y": [-0.02, -0.01, 0.0, 0.01, 0.02],
            }
            proposal = Proposal.model_validate(
                {
                    "agent_name": "refinement",
                    "round_id": 10,
                    "target_task_id": "capability::strain_loop",
                    "proposal_id": "refinement::10::refine_sampling::strain_loop",
                    "action_family": "refine_sampling",
                    "target_capability": "strain_loop",
                    "selected_skill": "strain_refinement",
                    "parameters": {
                        "target_channels": ["x"],
                        "suggested_points": {"x": [-0.015]},
                        "verify_non_redundancy": True,
                    },
                    "content": {"cost_class": "medium", "risk_class": "medium"},
                    "rationale": "inject midpoint for x",
                    "expected_observation": "strain_loop reruns locally",
                    "success_criteria": ["strain_loop recomputes only missing points"],
                    "fallback_if_failed": ["skip_channel", "escalate_human"],
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                with patch.object(
                    OrchestratorAgent,
                    "_call_llm_structured_with_tools",
                    return_value={
                        "selected_proposal_id": proposal.proposal_id,
                        "selected_action": {
                            "action_family": "refine_sampling",
                            "target_capability": "strain_loop",
                            "selected_skill": "recovery",
                            "source_proposal_id": proposal.proposal_id,
                            "parameters": {"job_id": "fake-job", "target_channels": ["x"]},
                            "rationale": "use async refinement execution",
                            "expected_observation": "wait for external event",
                            "success_criteria": ["external job submitted"],
                            "fallback_if_failed": ["abort_material"],
                            "submit_external_job": True,
                            "wait_for_event_after_submission": True,
                        },
                        "rationale": "chosen refinement proposal",
                    },
                ):
                    record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=10)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_action.source_proposal_id, proposal.proposal_id)
            self.assertEqual(record.selected_action.selected_skill, "strain_refinement")
            self.assertEqual(
                record.selected_action.parameters,
                {
                    "target_channels": ["x"],
                    "suggested_points": {"x": [-0.015]},
                    "verify_non_redundancy": True,
                },
            )
            self.assertFalse(record.selected_action.submit_external_job)
            self.assertFalse(record.selected_action.wait_for_event_after_submission)
            self.assertFalse(record.whether_waiting_external)
            self.assertIn("canonicalized_selected_action:selected_skill", record.guardrail_notes)
            self.assertIn("canonicalized_selected_action:parameters", record.guardrail_notes)
            self.assertIn("canonicalized_selected_action:submit_external_job", record.guardrail_notes)
            self.assertIn("canonicalized_selected_action:wait_for_event_after_submission", record.guardrail_notes)

    def test_orchestrator_falls_back_to_guardrail_preferred_legal_proposal_when_llm_override_is_illegal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            proposal = Proposal.model_validate(
                {
                    "agent_name": "planner",
                    "round_id": 12,
                    "target_task_id": "capability::prepare",
                    "proposal_id": "planner::12::validation",
                    "action_family": "run_capability",
                    "target_capability": "prepare",
                    "selected_skill": "single_material_mobility",
                    "rationale": "default_prepare_followup",
                    "content": {"cost_class": "low", "risk_class": "low"},
                    "expected_observation": "prepare stage is executed",
                    "success_criteria": ["prepare stage succeeds"],
                    "fallback_if_failed": ["retry_capability", "abort_material"],
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                with patch.object(
                    OrchestratorAgent,
                    "_call_llm_structured_with_tools",
                    return_value={
                        "selected_proposal_id": None,
                        "selected_action": {
                            "action_family": "run_capability",
                            "target_capability": "prepare",
                            "selected_skill": "single_material_mobility",
                            "source_proposal_id": "orchestrator_override",
                        },
                        "rationale": "invalid override should fall back to the legal prepare proposal",
                    },
                ):
                    record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=12)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_action.source_proposal_id, proposal.proposal_id)
            self.assertEqual(record.selected_action.target_capability, "prepare")
            self.assertTrue(
                any("illegal_or_unknown_override_fell_back_to" in note for note in record.guardrail_notes)
            )

    def test_orchestrator_falls_back_to_guardrail_preferred_legal_proposal_when_llm_returns_no_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            proposal = Proposal.model_validate(
                {
                    "agent_name": "planner",
                    "round_id": 12,
                    "target_task_id": "capability::prepare",
                    "proposal_id": "planner::12::prepare",
                    "action_family": "run_capability",
                    "target_capability": "prepare",
                    "selected_skill": "single_material_mobility",
                    "rationale": "default_prepare_followup",
                    "content": {"cost_class": "low", "risk_class": "low"},
                    "expected_observation": "prepare stage is executed",
                    "success_criteria": ["prepare stage succeeds"],
                    "fallback_if_failed": ["retry_capability", "abort_material"],
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                with patch.object(
                    OrchestratorAgent,
                    "_call_llm_structured_with_tools",
                    return_value={
                        "selected_proposal_id": None,
                        "selected_action": None,
                        "rationale": "llm returned no concrete selection",
                    },
                ):
                    record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=12)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_action.source_proposal_id, proposal.proposal_id)
            self.assertEqual(record.selected_action.target_capability, "prepare")
            self.assertTrue(
                any("missing_selection_fell_back_to_guardrail_preferred" in note for note in record.guardrail_notes)
            )

    def test_orchestrator_normalizes_invalidate_channel_parameter_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["blackboard"]["anomaly_flags"] = ["negative_mobility"]
            proposal = Proposal.model_validate(
                {
                    "agent_name": "planner",
                    "round_id": 21,
                    "target_task_id": str(state["task"]["task_id"]),
                    "proposal_id": "planner::21::invalidate-y",
                    "action_family": "invalidate_channel",
                    "target_capability": "mobility",
                    "selected_skill": "single_material_mobility",
                    "parameters": {"channels_to_invalidate": ["y"]},
                    "rationale": "reject unphysical y-channel",
                    "content": {"cost_class": "low", "risk_class": "low"},
                    "expected_observation": "y moved into rejected channels",
                    "success_criteria": ["y in rejected_channels"],
                    "fallback_if_failed": ["skip_channel", "escalate_human"],
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                with patch.object(
                    OrchestratorAgent,
                    "_call_llm_structured_with_tools",
                    return_value={
                        "selected_proposal_id": proposal.proposal_id,
                        "selected_action": {
                            "action_family": "invalidate_channel",
                            "target_capability": "mobility",
                            "selected_skill": "single_material_mobility",
                            "source_proposal_id": proposal.proposal_id,
                            "parameters": {"channels_to_invalidate": ["y"]},
                            "rationale": "normalize alias and invalidate y",
                        },
                        "rationale": "choose invalidate_channel",
                    },
                ):
                    record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=21)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_action.parameters.get("target_channels"), ["y"])
            self.assertNotIn("channels_to_invalidate", record.selected_action.parameters)

    def test_orchestrator_accepts_explicit_legal_override_to_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["task_board"]["completed_tasks"] = [
                {"capability": capability}
                for capability in ["prepare", "relax", "scf", "band", "effective_mass", "strain_loop", "mobility"]
            ]
            state["workflow"]["completed_stages"] = ["prepare", "relax", "scf", "band", "effective_mass", "strain_loop", "mobility"]
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.15,
                "effective_fit_quality": 0.15,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.15, "n_points": 5},
                    "y": {"effective_fit_quality": 0.22, "n_points": 5},
                },
            }
            state["physics_results"]["strain_plan_by_direction"] = {
                "x": [-0.02, -0.01, 0.0, 0.01, 0.02],
                "y": [-0.02, -0.01, 0.0, 0.01, 0.02],
            }
            proposal = Proposal.model_validate(
                {
                    "agent_name": "refinement",
                    "round_id": 13,
                    "target_task_id": "capability::strain_loop",
                    "proposal_id": "refinement::13::refine_sampling::strain_loop",
                    "action_family": "refine_sampling",
                    "target_capability": "strain_loop",
                    "selected_skill": "strain_refinement",
                    "parameters": {
                        "target_channels": ["x", "y"],
                        "suggested_points": {"x": [-0.015], "y": [-0.015]},
                        "verify_non_redundancy": True,
                    },
                    "content": {"cost_class": "medium", "risk_class": "medium"},
                    "rationale": "refine before validation",
                    "expected_observation": "strain_loop reruns with more points",
                    "success_criteria": ["new strain points are injected"],
                    "fallback_if_failed": ["abort_material"],
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                with patch.object(
                    OrchestratorAgent,
                    "_call_llm_structured_with_tools",
                    return_value={
                        "selected_proposal_id": "ORCHESTRATOR_OVERRIDE:proceed_to_validation",
                        "selected_action": {
                            "action_family": "run_capability",
                            "target_capability": "validation",
                            "selected_skill": "physics_validation",
                            "parameters": {"capability": "validation"},
                            "rationale": "validation is the correct next gate",
                        },
                        "rationale": "reject refinement and advance to validation",
                    },
                ):
                    record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=13)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_proposal_id, "ORCHESTRATOR_OVERRIDE:proceed_to_validation")
            self.assertEqual(record.selected_action.target_capability, "validation")
            self.assertEqual(record.selected_action.selected_skill, "physics_validation")
            self.assertTrue(any("explicit_legal_override" in note for note in record.guardrail_notes))

    def test_orchestrator_legalizes_local_prepare_proposal_with_bogus_wait_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            proposal = Proposal.model_validate(
                {
                    "agent_name": "planner",
                    "round_id": 11,
                    "target_task_id": str(state["task"]["task_id"]),
                    "proposal_id": "planner::11::prepare",
                    "action_family": "run_capability",
                    "target_capability": "prepare",
                    "selected_skill": "single_material_mobility",
                    "parameters": {"material_id": "orch-test"},
                    "rationale": "prepare the local workspace first",
                    "submit_external_job": False,
                    "wait_for_event_after_submission": True,
                    "confidence": 0.82,
                }
            )
            with patch_test_llm_clients():
                orchestrator = OrchestratorAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                record = orchestrator.arbitrate(state=state, proposals=[proposal], critiques=[], preferences=[], round_id=11)
            self.assertIsNotNone(record.selected_action)
            self.assertEqual(record.selected_action.target_capability, "prepare")
            self.assertFalse(record.selected_action.submit_external_job)
            self.assertFalse(record.selected_action.wait_for_event_after_submission)
            self.assertIn(
                "planner::11::prepare:canonicalized_proposal:wait_for_event_after_submission",
                record.guardrail_notes,
            )

    def test_refinement_agent_emits_local_strain_refinement_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.78,
                "effective_fit_quality": 0.78,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.78, "edge_fit_r2": 0.78, "energy_fit_r2": 0.79},
                    "y": {"effective_fit_quality": 0.95, "edge_fit_r2": 0.95, "energy_fit_r2": 0.96},
                },
            }
            state["physics_results"]["strain_data_summary"] = {"x": [-0.02, -0.01, 0.0, 0.01, 0.02]}
            with patch_test_llm_clients():
                refinement = RefinementAgent(
                    _runtime(os.path.join(tmpdir, "store.sqlite")),
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
                )
                proposals = refinement.propose(state=state, round_id=4)
            self.assertEqual(len(proposals), 1)
            proposal = proposals[0]
            self.assertEqual(proposal.action_family, "refine_sampling")
            self.assertEqual(proposal.selected_skill, "strain_refinement")
            self.assertTrue(proposal.parameters.get("verify_non_redundancy"))
            self.assertFalse(proposal.submit_external_job)
            self.assertFalse(proposal.wait_for_event_after_submission)


if __name__ == "__main__":
    unittest.main()
