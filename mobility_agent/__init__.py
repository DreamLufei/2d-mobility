from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "AgentRuntimeConfig",
    "DecisionToggles",
    "MaterialTaskState",
    "BatchTaskState",
    "MaterialRunOutcome",
    "AdmissionDecision",
    "RecoveryDecision",
    "RefinementDecision",
    "ValidationDecision",
    "HumanEscalationDecision",
    "ReportSummary",
    "ManualFixInstruction",
    "BatchSummary",
    "RuntimeContext",
    "default_material_workdir",
    "run_single_material",
    "run_mongo_batch",
]


_EXPORT_MAP = {
    "AgentRuntimeConfig": ("mobility_agent.config_runtime", "AgentRuntimeConfig"),
    "DecisionToggles": ("mobility_agent.config_runtime", "DecisionToggles"),
    "MaterialTaskState": ("mobility_agent.graph.state", "MaterialTaskState"),
    "BatchTaskState": ("mobility_agent.graph.state", "BatchTaskState"),
    "MaterialRunOutcome": ("mobility_agent.graph.state", "MaterialRunOutcome"),
    "AdmissionDecision": ("mobility_agent.agents.schemas", "AdmissionDecision"),
    "RecoveryDecision": ("mobility_agent.agents.schemas", "RecoveryDecision"),
    "RefinementDecision": ("mobility_agent.agents.schemas", "RefinementDecision"),
    "ValidationDecision": ("mobility_agent.agents.schemas", "ValidationDecision"),
    "HumanEscalationDecision": ("mobility_agent.agents.schemas", "HumanEscalationDecision"),
    "ReportSummary": ("mobility_agent.agents.schemas", "ReportSummary"),
    "ManualFixInstruction": ("mobility_agent.agents.schemas", "ManualFixInstruction"),
    "BatchSummary": ("mobility_agent.agents.schemas", "BatchSummary"),
    "RuntimeContext": ("mobility_agent.runtime.context", "RuntimeContext"),
    "default_material_workdir": ("mobility_agent.runtime.runner", "default_material_workdir"),
    "run_single_material": ("mobility_agent.runtime.runner", "run_single_material"),
    "run_mongo_batch": ("mobility_agent.runtime.batch_runner", "run_mongo_batch"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(name)
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
