from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "BatchConfig",
    "RuntimeContext",
    "AgentToolGateway",
    "AgenticMaterialController",
    "list_agent_tool_metadata",
    "load_config",
    "normalize_hitl_policy",
    "default_material_workdir",
    "run_single_material",
    "run_single_material_external_event",
    "run_mongo_batch",
]


_EXPORT_MAP = {
    "BatchConfig": ("mobility_agent.runtime.batch_config", "BatchConfig"),
    "RuntimeContext": ("mobility_agent.runtime.context", "RuntimeContext"),
    "AgentToolGateway": ("mobility_agent.runtime.agent_tools", "AgentToolGateway"),
    "AgenticMaterialController": ("mobility_agent.runtime.agentic_controller", "AgenticMaterialController"),
    "list_agent_tool_metadata": ("mobility_agent.runtime.agent_tools", "list_agent_tool_metadata"),
    "load_config": ("mobility_agent.runtime.batch_config", "load_config"),
    "normalize_hitl_policy": ("mobility_agent.runtime.context", "normalize_hitl_policy"),
    "default_material_workdir": ("mobility_agent.runtime.runner", "default_material_workdir"),
    "run_single_material": ("mobility_agent.runtime.runner", "run_single_material"),
    "run_single_material_external_event": ("mobility_agent.runtime.runner", "run_single_material_external_event"),
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
