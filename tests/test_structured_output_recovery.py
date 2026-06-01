from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from mobility_agent.agents.base import _coerce_structured_payload
from mobility_agent.agents.reporter import ReporterAgent
from mobility_agent.agents.schemas import ArbitrationDecisionPayload, ProposalBundle, ReportSummary
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
    state = make_initial_material_state(
        material_id="reporter-test",
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
    state["workflow"]["run_status"] = "completed"
    state["workflow"]["termination_reason"] = "validation_finalized_without_followup_action"
    state["diagnostics"]["validation_report"] = {"decision": "fail"}
    state["diagnostics"]["confidence_score"] = 0.91
    state["physics_results"]["accepted_channels"] = []
    state["physics_results"]["rejected_channels"] = ["x", "y"]
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
    state["execution"]["artifact_paths"] = {"material_outcome": os.path.join(tmpdir, "material_outcome.json")}
    return state


class StructuredOutputRecoveryTests(unittest.TestCase):
    def test_coerce_structured_payload_accepts_list_candidate_for_proposal_bundle(self) -> None:
        payload = _coerce_structured_payload(
            response={
                "parsed": [
                    {
                        "action": "run_capability",
                        "capability": "prepare",
                        "selected_skill": "single_material_mobility",
                        "rationale": ["Align with mainline", "No blockers present"],
                        "expected_artifacts": ["stage_outputs"],
                    }
                ],
                "raw": AIMessage(content=""),
                "parsing_error": None,
            },
            schema=ProposalBundle,
            agent_name="planner",
        )
        parsed = ProposalBundle.model_validate(payload)
        self.assertEqual(len(parsed.proposals), 1)
        self.assertEqual(parsed.proposals[0].action_family, "run_capability")
        self.assertEqual(parsed.proposals[0].target_capability, "prepare")
        self.assertEqual(parsed.proposals[0].success_criteria, ["stage_outputs"])
        self.assertIn("Align with mainline", parsed.proposals[0].rationale)

    def test_coerce_structured_payload_recovers_arbitration_markdown(self) -> None:
        payload = _coerce_structured_payload(
            response={
                "parsed": None,
                "raw": AIMessage(
                    content=(
                        "**ARBITRATION DECISION**\n\n"
                        "**Selected Action**: `run_capability` → `mobility`\n"
                        "**Source Proposal**: `planner::2::single_material::mobility`\n"
                        "**Rationale**: proceed to the canonical next capability.\n"
                    )
                ),
                "parsing_error": "synthetic_invalid_json",
            },
            schema=ArbitrationDecisionPayload,
            agent_name="orchestrator",
        )
        parsed = ArbitrationDecisionPayload.model_validate(payload)
        self.assertEqual(parsed.selected_proposal_id, "planner::2::single_material::mobility")
        self.assertIsNotNone(parsed.selected_action)
        self.assertEqual(parsed.selected_action.action_family, "run_capability")
        self.assertEqual(parsed.selected_action.target_capability, "mobility")

    def test_coerce_structured_payload_recovers_report_markdown(self) -> None:
        payload = _coerce_structured_payload(
            response={
                "parsed": None,
                "raw": AIMessage(
                    content=(
                        "## WHAT WAS DONE\n"
                        "- prepare succeeded\n"
                        "- relax succeeded\n"
                        "- validation rejected the result\n"
                    )
                ),
                "parsing_error": "synthetic_invalid_json",
            },
            schema=ReportSummary,
            agent_name="reporter",
        )
        parsed = ReportSummary.model_validate(payload)
        self.assertIn("WHAT WAS DONE", parsed.final_summary.get("narrative_report", ""))

    def test_reporter_material_summary_falls_back_when_llm_call_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = ReporterAgent(
                _runtime(os.path.join(tmpdir, "store.sqlite")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
            )
            state = _state(tmpdir)
            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=RuntimeError("synthetic_llm_failure")):
                summary = agent.summarize_material(state=state)
        self.assertEqual(summary["report_generation_status"], "fallback")
        self.assertIn("narrative_report", summary)
        self.assertEqual(summary["run_status"], "completed")

    def test_reporter_batch_summary_falls_back_to_baseline_when_llm_call_raises(self) -> None:
        outcomes = [
            {
                "status": "completed",
                "final_acceptance": "pass",
                "stage_status": {"validation": "success"},
                "results": {"mobility_results": {"ok": True}},
            },
            {
                "status": "failed",
                "final_acceptance": "fail",
                "stage_status": {"validation": "failed"},
                "termination_reason": "validation_failed",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir, patch_test_llm_clients():
            agent = ReporterAgent(
                _runtime(os.path.join(tmpdir, "store.sqlite")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skills")),
            )
            with patch.object(agent, "_call_llm_structured_with_tools", side_effect=RuntimeError("synthetic_llm_failure")):
                summary = agent.summarize_batch(outcomes=outcomes)
        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.succeeded, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.scientifically_passed, 1)
        self.assertEqual(summary.scientifically_failed, 1)


if __name__ == "__main__":
    unittest.main()
