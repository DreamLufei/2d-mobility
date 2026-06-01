from __future__ import annotations

import os
import tempfile
import unittest

from mobility_agent.graph.state import make_initial_material_state
from mobility_agent.runtime.agent_tools import AgentToolGateway
from mobility_agent.memory import open_memory_store


def _prepare_material_root(tmpdir: str) -> tuple[str, str]:
    poscar = os.path.join(tmpdir, "POSCAR")
    potcar = os.path.join(tmpdir, "POTCAR")
    with open(poscar, "w", encoding="utf-8") as handle:
        handle.write(
            "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 20.0\nSi\n2\nDirect\n0.0 0.0 0.5\n0.333333 0.666667 0.5\n"
        )
    with open(potcar, "w", encoding="utf-8") as handle:
        handle.write("FAKE POTCAR\n")
    return poscar, potcar


def _state(tmpdir: str) -> dict[str, object]:
    poscar, potcar = _prepare_material_root(tmpdir)
    return make_initial_material_state(
        material_id="tool-test",
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


class AgentToolTests(unittest.TestCase):
    def test_workspace_inspection_and_metadata_registry(self) -> None:
        gateway = AgentToolGateway()
        metadata = gateway.metadata()
        self.assertTrue(any(item["name"] == "inspect_workspace" for item in metadata))
        self.assertTrue(any(item["name"] == "check_action_legality" for item in metadata))
        self.assertTrue(any(item["name"] == "retrieve_policy_evidence" for item in metadata))
        self.assertTrue(any(item["name"] == "resolve_skills" for item in metadata))
        self.assertTrue(any(item["name"] == "load_skill" for item in metadata))
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            result = gateway.call(
                "inspect_workspace",
                {
                    "workdir": state["execution"]["workdir"],
                    "poscar_path": state["material"]["poscar_path"],
                    "potcar_path": state["material"]["potcar_path"],
                    "checkpoint_path": os.path.join(state["execution"]["workdir"], ".runtime", "langgraph.sqlite"),
                    "artifact_registry": {},
                },
            )
            self.assertTrue(result["available_inputs"]["poscar"])
            self.assertTrue(result["available_inputs"]["potcar"])
            self.assertIn("structure_summary", result["facts"])
            evidence = gateway.call(
                "retrieve_policy_evidence",
                {
                    "state": state,
                    "stage": "relax",
                    "top_k": 3,
                },
            )
            self.assertTrue(evidence["evidence"])
            self.assertTrue(any(item["corpus"] == "house_policy" for item in evidence["evidence"]))

    def test_skill_resolution_and_loading_are_agent_callable(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["current_stage"] = "proposal_phase"
            state["workflow"]["run_status"] = "needs_recovery"
            state["diagnostics"]["last_error"] = "zbrent_fatal"
            resolved = gateway.call(
                "resolve_skills",
                {
                    "state": state,
                    "role": "recovery",
                    "explicit_skills": ["recovery"],
                    "limit": 4,
                },
            )
            self.assertIn("recovery", resolved["selected_skills"])
            loaded = gateway.call(
                "load_skill",
                {
                    "skill_name": "recovery",
                    "include_body": True,
                    "include_resources": True,
                    "resource_limit": 5,
                },
            )
            self.assertEqual(loaded["name"], "recovery")
            self.assertIn("manifest", loaded)
            resources = gateway.call("list_skill_resources", {"skill_name": "recovery"})
            self.assertTrue(resources["resources"])
            first_resource = resources["resources"][0]["path"]
            resource_body = gateway.call(
                "read_skill_resource",
                {"skill_name": "recovery", "resource_path": first_resource},
            )
            self.assertEqual(resource_body["resource_path"], first_resource)

    def test_legality_guardrail_blocks_invalid_finalize_and_missing_dependencies(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            finalize = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "finalize_material",
                    "target_capability": None,
                    "parameters": {},
                },
            )
            self.assertFalse(finalize["allowed"])
            self.assertIn("finalize_requires_all_tasks_resolved", finalize["refusal_reasons"])

            illegal_run = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "run_capability",
                    "target_capability": "scf",
                    "parameters": {},
                },
            )
            self.assertFalse(illegal_run["allowed"])
            self.assertTrue(any(reason.startswith("missing_dependencies:") for reason in illegal_run["refusal_reasons"]))

    def test_legality_guardrail_allows_finalize_after_validation_review(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
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
            state["diagnostics"]["validation_report"] = {
                "decision": "fail",
                "recommended_action": "finalize",
            }
            state["physics_results"]["results"] = {
                "material_id": "tool-test",
                "results_by_direction": {"x": {"electron": {"mobility_cm2_Vs": 1000.0}}},
            }
            finalize = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "finalize_material",
                    "target_capability": None,
                    "parameters": {},
                },
            )
            self.assertTrue(finalize["allowed"])

    def test_legality_guardrail_allows_finalize_with_structured_validation_report(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
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
            state["diagnostics"]["validation_report"] = {
                "decision": "fail",
                "recommended_action": "finalize",
                "accepted_channels": [],
                "rejected_channels": ["x", "y"],
            }
            state["physics_results"]["results"] = {
                "material_id": "tool-test",
                "results_by_direction": {"x": {"electron": {"mobility_cm2_Vs": -1.0}}},
            }
            finalize = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "finalize_material",
                    "target_capability": None,
                    "parameters": {},
                },
            )
            self.assertTrue(finalize["allowed"])

    def test_legality_guardrail_respects_waiting_external_and_duplicate_submission(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["run_status"] = "waiting_external"
            state["workflow"]["wait_reason"] = "awaiting_external_event:scf"
            state["execution"]["external_jobs"] = [
                {
                    "job_id": "job-1",
                    "target_capability": "scf",
                    "status": "submitted",
                }
            ]
            waiting_finalize = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "finalize_material",
                    "target_capability": None,
                    "parameters": {},
                },
            )
            self.assertFalse(waiting_finalize["allowed"])
            self.assertIn("waiting_external_event", waiting_finalize["refusal_reasons"])
            duplicate_run = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "run_capability",
                    "target_capability": "scf",
                    "parameters": {},
                },
            )
            self.assertFalse(duplicate_run["allowed"])
            self.assertIn("external_job_already_pending:scf", duplicate_run["refusal_reasons"])

    def test_legality_guardrail_blocks_external_wait_in_full_autonomy(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["mission"]["runtime_constraints"]["full_autonomy"] = True
            state["mission"]["runtime_constraints"]["allow_external_wait"] = False
            result = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "run_capability",
                    "target_capability": "prepare",
                    "parameters": {},
                    "submit_external_job": True,
                    "wait_for_event_after_submission": True,
                },
            )
            self.assertFalse(result["allowed"])
            self.assertIn("external_wait_not_allowed_in_full_autonomy", result["refusal_reasons"])

    def test_legality_guardrail_allows_validation_even_when_refinement_is_available(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["task_board"]["completed_tasks"] = [{"capability": "mobility"}]
            state["diagnostics"]["fit_diagnostics"] = {
                "fit_r2_min": 0.775,
                "effective_fit_quality": 0.775,
                "per_direction": {
                    "x": {"effective_fit_quality": 0.775, "n_points": 5},
                    "y": {"effective_fit_quality": 0.791, "n_points": 5},
                },
            }
            state["physics_results"]["strain_plan_by_direction"] = {
                "x": [-0.02, -0.01, 0.0, 0.01, 0.02],
                "y": [-0.02, -0.01, 0.0, 0.01, 0.02],
            }
            result = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "run_capability",
                    "target_capability": "validation",
                    "parameters": {},
                },
            )
            self.assertTrue(result["allowed"])
            self.assertIn("validation_has_available_refinement_followup", result["warnings"])

    def test_legality_guardrail_rejects_refinement_without_fresh_points(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            nine_points = [-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02]
            state["physics_results"]["strain_plan_by_direction"] = {"x": list(nine_points), "y": list(nine_points)}
            result = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "refine_sampling",
                    "target_capability": "strain_loop",
                    "parameters": {"refinement_strategy": "midpoint_enrichment", "target_directions": ["x", "y"]},
                },
            )
            self.assertFalse(result["allowed"])
            self.assertIn("no_fresh_refinement_points", result["refusal_reasons"])

    def test_legality_guardrail_handles_channel_alias_and_missing_target_channels(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["blackboard"]["anomaly_flags"] = ["negative_mobility"]
            normalized = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "invalidate_channel",
                    "target_capability": "mobility",
                    "parameters": {"channels_to_invalidate": ["y"]},
                },
            )
            self.assertTrue(normalized["allowed"])

            missing = gateway.call(
                "check_action_legality",
                {
                    "state": state,
                    "action_family": "invalidate_channel",
                    "target_capability": "mobility",
                    "parameters": {},
                },
            )
            self.assertFalse(missing["allowed"])
            self.assertIn("missing_required_parameter:target_channels", missing["refusal_reasons"])

    def test_synthesize_observation_does_not_assume_perfect_fit_without_mobility_results(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            result = gateway.call("synthesize_observation", {"state": state})
            self.assertIsNone(result["fit_quality"])
            self.assertFalse(result["mobility_window_summary"]["results_present"])
            self.assertIsNone(result["mobility_window_summary"]["effective_fit_quality"])

    def test_memory_write_and_query_are_agent_callable(self) -> None:
        gateway = AgentToolGateway()
        with tempfile.TemporaryDirectory() as tmpdir:
            state = _state(tmpdir)
            state["workflow"]["current_stage"] = "scf"
            state["diagnostics"]["last_error"] = "dry_run_injected_failure:scf"
            state["execution"]["latest_execution_observation"] = {
                "status": "failed",
                "target_capability": "scf",
                "error_summary": "dry_run_injected_failure:scf",
            }
            state["execution"]["current_action"] = {
                "action_family": "retry_capability",
                "target_capability": "scf",
            }
            with open_memory_store(os.path.join(tmpdir, "memory.sqlite")) as store:
                write_result = gateway.call(
                    "write_memory_reflection",
                    {"state": state, "round_id": 2},
                    store=store,
                )
                self.assertIn("recovery_cases", write_result["recorded_categories"])
                query_result = gateway.call(
                    "query_memory_hits",
                    {"state": state, "limit": 5},
                    store=store,
                )
                self.assertTrue(query_result["recovered_case_patterns"])

    def test_batch_aggregation_tool_summarizes_statuses(self) -> None:
        gateway = AgentToolGateway()
        result = gateway.call(
            "summarize_batch_outcomes",
            {
                "outcomes": [
                    {"status": "completed", "final_acceptance": "pass", "stage_status": {}},
                    {"status": "failed", "final_acceptance": "fail", "stage_status": {"scf": "failed"}},
                    {"status": "skipped", "termination_reason": "skip_material", "stage_status": {}},
                ]
            },
        )
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["scientifically_passed"], 1)
        self.assertEqual(result["scientifically_failed"], 1)
        self.assertEqual(result["scientifically_unknown"], 1)
        self.assertIn("scf", result["common_failure_stages"])


if __name__ == "__main__":
    unittest.main()
