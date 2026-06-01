from __future__ import annotations

import os
import shutil

from mobility_agent.graph.runtime_nodes import _classify_failure
from mobility_agent.tools.relax_tool import RelaxTool, RelaxToolInput
from mobility_agent.tools.vasp_common import classify_vasp_failure_text, policy_stage_planning_allowed


POSCAR_TEXT = """Si
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 20.0
Si
2
Direct
0.0 0.0 0.5
0.333333 0.666667 0.5
"""


class _FakePlan:
    incar_overrides = {"NSW": 7}
    kpoints_policy = {"gamma_centered": True}
    source = "test"
    confidence = 1.0
    evidence_items = []
    rationale = "test plan"

    def model_dump(self, mode: str = "json"):
        return {
            "incar_overrides": dict(self.incar_overrides),
            "kpoints_policy": dict(self.kpoints_policy),
            "source": self.source,
            "confidence": self.confidence,
            "evidence_items": [],
            "rationale": self.rationale,
        }


class _FakePolicy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def plan_stage(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _FakePlan()


def _write_inputs(tmp_path):
    poscar = tmp_path / "POSCAR"
    potcar = tmp_path / "POTCAR"
    poscar.write_text(POSCAR_TEXT, encoding="utf-8")
    potcar.write_text("FAKE POTCAR\n", encoding="utf-8")
    return str(poscar), str(potcar)


def _state(*, action_family: str, target: str = "relax", stage_status: dict[str, str] | None = None):
    return {
        "execution": {
            "current_action": {
                "action_family": action_family,
                "target_capability": target,
            }
        },
        "workflow": {
            "stage_status": dict(stage_status or {}),
            "retry_counts": {},
        },
    }


def _fake_relax_retry(*, workdir: str, **_kwargs):
    shutil.copy(os.path.join(workdir, "POSCAR"), os.path.join(workdir, "CONTCAR"))
    return True, [], {"stage": "relax", "error_type": "none", "final_outcome": "success"}


def test_policy_stage_planning_requires_recovery_action_and_failed_stage():
    assert not policy_stage_planning_allowed(
        _state(action_family="run_capability", stage_status={"relax": "failed"}),
        "relax",
    )
    assert not policy_stage_planning_allowed(
        _state(action_family="retry_capability", stage_status={}),
        "relax",
    )
    assert policy_stage_planning_allowed(
        _state(action_family="retry_capability", stage_status={"relax": "failed"}),
        "relax",
    )


def test_relax_first_run_does_not_call_policy(monkeypatch, tmp_path):
    import mobility_agent.tools.relax_tool as relax_tool_module

    monkeypatch.setattr(relax_tool_module, "run_relax_vasp_with_retry", _fake_relax_retry)
    poscar, potcar = _write_inputs(tmp_path)
    policy = _FakePolicy()
    tool = RelaxTool(policy_engine=policy)

    result = tool.run(
        RelaxToolInput(
            material_id="deterministic-first",
            base_dir=str(tmp_path / "work"),
            poscar_path=poscar,
            potcar_path=potcar,
            state_payload=_state(action_family="run_capability"),
        )
    )

    assert result.success, result.error_summary
    assert policy.calls == []
    incar_text = (tmp_path / "work" / "01_relax" / "INCAR").read_text(encoding="utf-8")
    assert "NSW = 7" not in incar_text


def test_relax_retry_after_failure_allows_policy(monkeypatch, tmp_path):
    import mobility_agent.tools.relax_tool as relax_tool_module

    monkeypatch.setattr(relax_tool_module, "run_relax_vasp_with_retry", _fake_relax_retry)
    poscar, potcar = _write_inputs(tmp_path)
    policy = _FakePolicy()
    tool = RelaxTool(policy_engine=policy)

    result = tool.run(
        RelaxToolInput(
            material_id="deterministic-retry",
            base_dir=str(tmp_path / "work"),
            poscar_path=poscar,
            potcar_path=potcar,
            state_payload=_state(action_family="retry_capability", stage_status={"relax": "failed"}),
        )
    )

    assert result.success, result.error_summary
    assert [call["stage"] for call in policy.calls] == ["relax"]
    incar_text = (tmp_path / "work" / "01_relax" / "INCAR").read_text(encoding="utf-8")
    assert "NSW = 7" in incar_text


def test_vasp_environment_and_chgcar_failures_are_classified_as_code_context():
    assert classify_vasp_failure_text("/bin/sh: 1: mpirun: not found")[0] == "runner_environment_failure"
    assert classify_vasp_failure_text("vasp_std: error while loading shared libraries: libmkl_rt.so")[0] == "runner_environment_failure"
    assert classify_vasp_failure_text("ERROR: CHGCAR has incompatible FFT grid dimensions")[0] == "chgcar_compatibility_failure"

    assert _classify_failure("SCF runner/environment failure: mpirun: not found") == "runner_environment_failure"
    assert _classify_failure("BAND CHGCAR compatibility failure: chgcar_grid_or_dimension_mismatch") == "chgcar_compatibility_failure"
