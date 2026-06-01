from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .state import MaterialTaskState


def build_material_graph(nodes: dict[str, object]) -> StateGraph:
    graph = StateGraph(MaterialTaskState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_edge(START, "observe_state")
    graph.add_edge("observe_state", "proposal_phase")
    graph.add_edge("proposal_phase", "critique_phase")
    graph.add_edge("critique_phase", "arbitration_phase")
    graph.add_edge("execute_selected_action", "reflect_round")
    graph.add_edge("reflect_round", "check_termination")
    graph.add_edge("final_report", END)
    return graph
