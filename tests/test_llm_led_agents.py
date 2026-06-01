from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.agents.critic import CriticAgent
from mobility_agent.agents.orchestrator import OrchestratorAgent
from mobility_agent.agents.planner import PlannerAgent
from mobility_agent.agents.recovery import RecoveryAgent
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
        material_id="llm-agent-test",
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


class LLMDrivenAgentTests(unittest.TestCase):
    def test_planner_uses_hints_and_returns_llm_generated_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            agent = PlannerAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            captured: dict[str, object] = {}

            def _fake_call(**kwargs):
                captured.update(kwargs)
                return {
                    "proposals": [
                        {
                            "agent_name": "planner",
                            "round_id": 1,
                            "target_task_id": str(state["task"]["task_id"]),
                            "proposal_id": "planner::llm::1::revalidate",
                            "action_family": "revalidate_result",
                            "target_capability": "mobility",
                            "selected_skill": "single_material_mobility",
                            "rationale": "llm_prefers_validation_before_direct_progression",
                            "expected_benefit": "improve confidence before advancing",
                            "expected_risk": "moderate additional delay",
                            "expected_observation": "validation diagnostics",
                            "success_criteria": ["validation diagnostics captured"],
                            "fallback_if_failed": ["run_capability"],
                            "confidence": 0.73,
                        }
                    ]
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                proposals = agent.propose(state=state, round_id=1)
            self.assertIn("planning_hints", captured["payload"])
            self.assertNotIn("baseline_proposals", captured["payload"])
            self.assertEqual(proposals[0].action_family, "revalidate_result")

    def test_planner_injects_default_validation_proposal_when_llm_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["workflow"]["run_status"] = "running"
            state["task_board"]["completed_tasks"] = [
                {"capability": "prepare"},
                {"capability": "relax"},
                {"capability": "scf"},
                {"capability": "band"},
                {"capability": "effective_mass"},
                {"capability": "strain_loop"},
                {"capability": "mobility"},
            ]
            state["task_board"]["pending_tasks"] = [{"capability": "validation"}]
            state["task_board"]["active_tasks"] = []
            state["diagnostics"]["fit_diagnostics"] = {"fit_r2_min": 0.78, "effective_fit_quality": 0.78}
            agent = PlannerAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))

            with patch.object(agent, "_call_llm_structured_with_tools", return_value={"proposals": []}):
                proposals = agent.propose(state=state, round_id=5)

            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].action_family, "run_capability")
            self.assertEqual(proposals[0].target_capability, "validation")
            self.assertEqual(proposals[0].selected_skill, "physics_validation")

    def test_recovery_uses_failure_hints_and_returns_llm_generated_repair_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["blackboard"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "scf",
                "error_summary": "dry_run_injected_failure:scf",
                "error_category": "nonconverged",
            }
            agent = RecoveryAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            captured: dict[str, object] = {}

            def _fake_call(**kwargs):
                captured.update(kwargs)
                return {
                    "proposals": [
                        {
                            "agent_name": "recovery",
                            "round_id": 2,
                            "target_task_id": str(state["task"]["task_id"]),
                            "proposal_id": "recovery::llm::2::repair",
                            "action_family": "repair_execution_context",
                            "target_capability": "scf",
                            "rationale": "llm_diagnosis_prefers_context_repair_before_retry",
                            "expected_benefit": "repair broken execution context",
                            "expected_risk": "low",
                            "confidence": 0.78,
                        }
                    ]
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                proposals = agent.propose(state=state, round_id=2)
            self.assertIn("recovery_hints", captured["payload"])
            self.assertIn("agentic_diagnosis", captured["payload"])
            self.assertIn("failure_evidence", captured["payload"]["state_summary"])
            self.assertNotIn("baseline_proposals", captured["payload"])
            self.assertEqual(proposals[0].action_family, "repair_execution_context")

    def test_recovery_forces_human_escalation_for_relax_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["blackboard"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "strain_loop",
                "error_summary": "strain_campaign_incomplete:1_failed_points",
                "raw_evidence": {
                    "raw_payload": {
                        "strain_data": [
                            {
                                "direction": "x",
                                "strain": 0.015,
                                "completed": False,
                                "error": "RELAX_FAILED",
                                "error_type": "relax_failed",
                            }
                        ]
                    }
                },
            }
            agent = RecoveryAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=AssertionError("relax failures should not depend on LLM recovery proposals")):
                proposals = agent.propose(state=state, round_id=4)

            self.assertEqual(len(proposals), 1)
            self.assertEqual(proposals[0].action_family, "escalate_human")
            self.assertIsNone(proposals[0].target_capability)
            self.assertIn("manual_fix_resume", proposals[0].parameters["recommended_options"])
            self.assertEqual(agent.last_failure_diagnosis["source"], "deterministic_relax_failure_policy")
            self.assertTrue(agent.last_failure_diagnosis["needs_human"])

    def test_recovery_does_not_force_human_escalation_for_non_relax_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["blackboard"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "strain_loop",
                "error_summary": "strain_campaign_incomplete:1_failed_points",
                "raw_evidence": {
                    "raw_payload": {
                        "strain_data": [
                            {
                                "direction": "x",
                                "strain": 0.015,
                                "completed": False,
                                "error": "SCF_FAILED",
                                "error_type": "scf_failed",
                            }
                        ]
                    }
                },
            }
            agent = RecoveryAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            captured: dict[str, object] = {}

            def _fake_call(**kwargs):
                captured.update(kwargs)
                return {
                    "proposals": [
                        {
                            "agent_name": "recovery",
                            "round_id": 5,
                            "target_task_id": str(state["task"]["task_id"]),
                            "proposal_id": "recovery::llm::5::repair",
                            "action_family": "repair_execution_context",
                            "target_capability": "strain_loop",
                            "rationale": "non_relax_failure_remains_under_standard_recovery",
                            "confidence": 0.77,
                        }
                    ]
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                proposals = agent.propose(state=state, round_id=5)

            self.assertIn("recovery_hints", captured["payload"])
            self.assertEqual(proposals[0].action_family, "repair_execution_context")
            self.assertNotEqual(agent.last_failure_diagnosis["source"], "deterministic_relax_failure_policy")

    def test_recovery_uses_needs_recovery_context_without_failed_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["workflow"]["run_status"] = "needs_recovery"
            state["workflow"]["current_stage"] = "strain_loop"
            state["diagnostics"]["last_error"] = "post_execution_wait_boundary:waiting_external"
            state["blackboard"]["latest_execution_observation"] = {
                "status": "running",
                "target_capability": "strain_loop",
            }
            agent = RecoveryAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            captured: dict[str, object] = {}

            def _fake_call(**kwargs):
                captured.update(kwargs)
                return {
                    "proposals": [
                        {
                            "agent_name": "recovery",
                            "round_id": 6,
                            "target_task_id": str(state["task"]["task_id"]),
                            "proposal_id": "recovery::llm::6::abort",
                            "action_family": "abort_material",
                            "target_capability": "strain_loop",
                            "rationale": "llm_detects_unrecoverable_wait_boundary",
                            "expected_benefit": "terminate cleanly when no legal continuation exists",
                            "expected_risk": "material aborted",
                            "confidence": 0.74,
                        }
                    ]
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                proposals = agent.propose(state=state, round_id=6)
            self.assertIn("recovery_hints", captured["payload"])
            self.assertEqual(captured["payload"]["recovery_hints"]["failure_context"]["status"], "failed")
            self.assertEqual(captured["payload"]["recovery_hints"]["failure_context"]["stage"], "strain_loop")
            self.assertTrue(proposals)
            self.assertEqual(proposals[0].action_family, "abort_material")

    def test_critic_uses_review_hints_not_prebuilt_review_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            agent = CriticAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            proposal = Proposal(
                agent_name="planner",
                round_id=3,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="planner::3::prepare",
                action_family="run_capability",
                target_capability="prepare",
                rationale="advance_to_prepare",
                confidence=0.8,
            )
            captured: dict[str, object] = {}

            def _fake_call(**kwargs):
                captured.update(kwargs)
                return {
                    "critiques": [
                        {
                            "agent_name": "critic",
                            "message_type": "critique",
                            "round_id": 3,
                            "target_task_id": str(state["task"]["task_id"]),
                            "proposal_id": proposal.proposal_id,
                            "concerns": ["need_explicit_workspace_check"],
                            "recommendation": "run_if_workspace_checks_pass",
                            "confidence": 0.77,
                            "risk_estimate": 0.35,
                        }
                    ],
                    "preferences": [],
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                critiques, preferences = agent.review(state=state, proposals=[proposal], round_id=3)
            self.assertIn("review_hints", captured["payload"])
            self.assertNotIn("baseline_critiques", captured["payload"])
            self.assertEqual(len(preferences), 0)
            self.assertEqual(critiques[0].concerns, ["need_explicit_workspace_check"])

    def test_orchestrator_lets_llm_override_guardrail_preferred_legal_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            agent = OrchestratorAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            planner_proposal = Proposal(
                agent_name="planner",
                round_id=4,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="planner::4::prepare",
                action_family="run_capability",
                target_capability="prepare",
                rationale="follow_default_mainline",
                confidence=0.8,
            )
            recovery_proposal = Proposal(
                agent_name="recovery",
                round_id=4,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="recovery::4::human",
                action_family="escalate_human",
                target_capability="prepare",
                rationale="seek_manual_confirmation",
                confidence=0.7,
            )

            def _fake_call(**kwargs):
                guardrail_context = kwargs["payload"]["guardrail_context"]
                self.assertEqual(
                    guardrail_context["content"]["guardrail_preferred_proposal_id"],
                    planner_proposal.proposal_id,
                )
                return {
                    "selected_proposal_id": recovery_proposal.proposal_id,
                    "selected_action": {
                        "action_family": "escalate_human",
                        "target_capability": "prepare",
                        "source_proposal_id": recovery_proposal.proposal_id,
                        "rationale": "llm_selects_human_escalation_despite_default_progression",
                        "cost_class": "low",
                        "risk_class": "medium",
                    },
                    "rejected_proposal_ids": [planner_proposal.proposal_id],
                    "guardrail_notes": ["legal_but_nonpreferred_choice"],
                    "rationale": "llm_chief_agent_prioritizes_human_review",
                    "disagreement_summary": [],
                    "whether_noop": False,
                    "whether_waiting_external": False,
                    "whether_ready_to_finalize": False,
                }

            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=_fake_call):
                record = agent.arbitrate(
                    state=state,
                    proposals=[planner_proposal, recovery_proposal],
                    critiques=[],
                    preferences=[],
                    round_id=4,
                )
            self.assertEqual(record.selected_proposal_id, recovery_proposal.proposal_id)
            self.assertEqual(record.selected_action.action_family, "escalate_human")

    def test_orchestrator_raises_strict_failure_after_llm_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            state = _state(tmpdir)
            state["blackboard"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "prepare",
                "error_summary": "synthetic_failure",
            }
            agent = OrchestratorAgent(_runtime(os.path.join(tmpdir, "store.sqlite")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")))
            legal_proposal = Proposal(
                agent_name="recovery",
                round_id=5,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="recovery::5::human",
                action_family="escalate_human",
                target_capability="prepare",
                rationale="human_help_needed",
                confidence=0.8,
            )
            illegal_proposal = Proposal(
                agent_name="planner",
                round_id=5,
                target_task_id=str(state["task"]["task_id"]),
                proposal_id="planner::5::illegal_finalize",
                action_family="finalize_material",
                rationale="illegal_finalize_during_failure",
                confidence=0.8,
            )
            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=RuntimeError("synthetic_llm_failure")):
                with self.assertRaises(RuntimeError) as ctx:
                    agent.arbitrate(
                        state=state,
                        proposals=[illegal_proposal, legal_proposal],
                        critiques=[],
                        preferences=[],
                        round_id=5,
                    )
            self.assertIn("orchestrator_strict_agentic_failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
