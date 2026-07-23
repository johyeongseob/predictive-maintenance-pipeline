"""LangGraph builder for the agent orchestration workflow."""

from langgraph.graph import StateGraph, END

from .analysis_agent import analysis_agent
from .evidence_agent import evidence_agent
from .meta_agent import collect_results, meta_collect_policy, meta_entry
from .policy_agent import policy_agent
from .utility.state import AgentState


def build_graph(execution_mode="sequential"):
    """
    Build LangGraph workflow with hub-and-spoke architecture.

    Args:
        execution_mode: 'sequential' or 'parallel'

    Workflow:
        START
          ↓
        meta_entry (hub: initialize)
          ↓
        policy_agent
          ↓
        meta_collect_policy (hub: redistribute policy data)
          ↓
        [Sequential: analysis → evidence]
        [Parallel: analysis + evidence simultaneously]
          ↓
        collect_results (hub: finalize)
          ↓
         END

    Returns:
        Compiled LangGraph StateGraph
    """
    builder = StateGraph(AgentState)

    # Add meta nodes (hubs)
    builder.add_node("meta_entry", meta_entry)
    builder.add_node("meta_collect_policy", meta_collect_policy)
    builder.add_node("collect_results", collect_results)

    # Add agent nodes (spokes)
    builder.add_node("policy", policy_agent)
    builder.add_node("analysis", analysis_agent)
    builder.add_node("evidence", evidence_agent)

    # Phase 1: Entry → Policy → Collect
    builder.set_entry_point("meta_entry")
    builder.add_edge("meta_entry", "policy")
    builder.add_edge("policy", "meta_collect_policy")

    # Phase 2: Mode-dependent execution
    if execution_mode == "parallel":
        # Parallel: Both agents run simultaneously
        builder.add_edge("meta_collect_policy", "analysis")
        builder.add_edge("meta_collect_policy", "evidence")
        builder.add_edge("analysis", "collect_results")
        builder.add_edge("evidence", "collect_results")
    else:
        # Sequential: analysis → evidence
        builder.add_edge("meta_collect_policy", "analysis")
        builder.add_edge("analysis", "evidence")
        builder.add_edge("evidence", "collect_results")

    # End
    builder.add_edge("collect_results", END)

    return builder.compile()
