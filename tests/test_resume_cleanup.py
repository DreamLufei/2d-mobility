from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from mobility_agent.hitl.escalation import _interactive_decision
from mobility_agent.hitl.manual_fix import build_manual_fix_instruction
from mobility_agent.hitl.resume import build_resume_command
from mobility_agent.hitl.cleanup import preview_cleanup
from mobility_agent.hitl.resume_rules import build_custom_resume_rule, compute_default_resume_rule
from mobility_agent.tools.errors import ManualFixValidationError


class ResumeCleanupTests(unittest.TestCase):
    def test_default_resume_rules(self) -> None:
        incar = compute_default_resume_rule(current_stage="scf", modification_type="INCAR")
        self.assertEqual(incar.resume_stage, "scf")
        self.assertEqual(incar.cleanup_policy, "retry_current_stage_only")
        self.assertEqual(incar.invalidated_stages, ["scf"])
        self.assertEqual(incar.requested_resume_strategy, "default_rule")

        kpoints = compute_default_resume_rule(current_stage="band", modification_type="KPOINTS")
        self.assertEqual(kpoints.resume_stage, "scf")
        self.assertEqual(kpoints.cleanup_policy, "invalidate_downstream")
        self.assertIn("band", kpoints.invalidated_stages)

        poscar = compute_default_resume_rule(current_stage="band", modification_type="POSCAR")
        self.assertEqual(poscar.resume_stage, "relax")
        self.assertEqual(poscar.cleanup_policy, "restart_from_stage")

    def test_invalid_custom_stage_raises(self) -> None:
        with self.assertRaises(ManualFixValidationError):
            build_custom_resume_rule(
                resume_stage="not_a_stage",
                cleanup_policy="restart_from_stage",
                modified_files=["POSCAR"],
            )

    def test_incompatible_retry_current_stage_only_raises(self) -> None:
        with self.assertRaises(ManualFixValidationError):
            build_custom_resume_rule(
                resume_stage="scf",
                cleanup_policy="retry_current_stage_only",
                modified_files=["INCAR"],
                current_stage="band",
            )

    def test_incompatible_invalidate_downstream_on_current_stage_raises(self) -> None:
        with self.assertRaises(ManualFixValidationError):
            build_custom_resume_rule(
                resume_stage="band",
                cleanup_policy="invalidate_downstream",
                modified_files=["KPOINTS"],
                current_stage="band",
            )

    def test_cleanup_preview_lists_invalidated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "02_scf"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "03_band"), exist_ok=True)
            with open(os.path.join(tmpdir, "02_scf", "CHGCAR"), "w", encoding="utf-8") as handle:
                handle.write("x")
            with open(os.path.join(tmpdir, "validation_report.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            preview = preview_cleanup(
                workdir=tmpdir,
                resume_stage="scf",
                cleanup_policy="invalidate_downstream",
            )
            self.assertIn("band", preview.invalidated_stages)
            self.assertTrue(any(path.endswith("validation_report.json") for path in preview.invalidated_artifacts))

    def test_manual_fix_preview_schema_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "02_scf"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "03_band"), exist_ok=True)
            with open(os.path.join(tmpdir, "03_band", "EIGENVAL"), "w", encoding="utf-8") as handle:
                handle.write("x")
            with open(os.path.join(tmpdir, "validation_report.json"), "w", encoding="utf-8") as handle:
                handle.write("{}")
            instruction = build_manual_fix_instruction(
                current_stage="band",
                workdir=tmpdir,
                modification_type="KPOINTS",
            )
            self.assertIsNotNone(instruction.preview)
            self.assertEqual(instruction.preview.modified_files, ["KPOINTS"])
            self.assertEqual(instruction.preview.requested_resume_strategy, "default_rule")
            self.assertEqual(instruction.preview.computed_resume_stage, "scf")
            self.assertEqual(instruction.preview.cleanup_policy, "invalidate_downstream")
            self.assertIn("band", instruction.preview.invalidated_stages)
            self.assertTrue(any(path.endswith("validation_report.json") for path in instruction.preview.invalidated_artifacts))

    def test_manual_fix_resume_command_preserves_preview_fields(self) -> None:
        command = build_resume_command(
            {
                "action": "manual_fix_resume",
                "modified_files": ["KPOINTS"],
                "modification_type": "KPOINTS",
                "requested_resume_strategy": "default_rule",
                "resume_stage": "scf",
                "cleanup_policy": "invalidate_downstream",
                "invalidated_stages": ["band", "effective_mass", "mobility"],
                "invalidated_artifacts": ["/tmp/x/03_band"],
                "preview": {
                    "modified_files": ["KPOINTS"],
                    "requested_resume_strategy": "default_rule",
                    "computed_resume_stage": "scf",
                    "cleanup_policy": "invalidate_downstream",
                    "invalidated_stages": ["band", "effective_mass", "mobility"],
                    "invalidated_artifacts": ["/tmp/x/03_band"],
                    "warnings": [],
                },
                "reason": "user_manual_fix",
            }
        )
        self.assertEqual(command["action"], "manual_fix_resume")
        self.assertEqual(command["instruction"]["resume_stage"], "scf")
        self.assertEqual(command["instruction"]["cleanup_policy"], "invalidate_downstream")
        self.assertEqual(command["instruction"]["preview"]["requested_resume_strategy"], "default_rule")
        self.assertIn("band", command["instruction"]["invalidated_stages"])
        self.assertEqual(command["instruction"]["preview"]["computed_resume_stage"], "scf")

    def test_interactive_manual_fix_waits_for_continue_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "mobility_agent.hitl.escalation._open_terminal_streams",
                return_value=(object(), object(), False),
            ), patch(
                "mobility_agent.hitl.escalation._write_line",
            ), patch(
                "mobility_agent.hitl.escalation._timed_input",
                side_effect=[("1", False), ("continue", False)],
            ):
                decision = _interactive_decision(
                    {
                        "material_id": "mat-1",
                        "current_stage": "strain",
                        "working_directory": tmpdir,
                        "recommended_options": ["manual_fix_resume", "skip_material", "abort_task"],
                    },
                    timeout_seconds=300,
                    default_action="skip_material",
                )
        self.assertEqual(decision.action, "manual_fix_resume")
        self.assertIsNotNone(decision.instruction)
        self.assertEqual(decision.instruction.resume_stage, "strain_loop")
        self.assertEqual(decision.instruction.cleanup_policy, "retry_current_stage_only")

    def test_interactive_manual_fix_accepts_stage_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "mobility_agent.hitl.escalation._open_terminal_streams",
                return_value=(object(), object(), False),
            ), patch(
                "mobility_agent.hitl.escalation._write_line",
            ), patch(
                "mobility_agent.hitl.escalation._timed_input",
                side_effect=[("1", False), ("continue scf", False)],
            ):
                decision = _interactive_decision(
                    {
                        "material_id": "mat-2",
                        "current_stage": "band",
                        "working_directory": tmpdir,
                        "recommended_options": ["manual_fix_resume", "skip_material", "abort_task"],
                    },
                    timeout_seconds=300,
                    default_action="skip_material",
                )
        self.assertEqual(decision.action, "manual_fix_resume")
        self.assertIsNotNone(decision.instruction)
        self.assertEqual(decision.instruction.resume_stage, "scf")
        self.assertEqual(decision.instruction.cleanup_policy, "invalidate_downstream")


if __name__ == "__main__":
    unittest.main()
