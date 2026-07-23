"""LangGraph builder for the agent orchestration workflow."""

from langgraph.graph import StateGraph, END

from .analysis_agent import analysis_agent
from .evidence_agent import evidence_agent
from .meta_agent import collect_results, meta_collect_policy, meta_entry
from .policy_agent import policy_agent
from .utility.state import AgentState
 
DEFAULT_ACTIVE_AGENTS = ["policy", "analysis", "evidence"]

AGENT_REGISTRY = {
    "policy": {
        "module": "src.agents.policy_agent",
        "node_fn": policy_agent,
        "requires": {"resources"},
        "provides": {"policy"},
    },
    "analysis": {
        "module": "src.agents.analysis_agent",
        "node_fn": analysis_agent,
        "requires": {"resources", "policy", "for_analysis"},
        "provides": {"analysis"},
    },
    "evidence": {
        "module": "src.agents.evidence_agent",
        "node_fn": evidence_agent,
        "requires": {"resources", "policy", "for_evidence"},
        "provides": {"evidence"},
    },
}

META_PROVIDES = {"resources", "for_analysis", "for_evidence"}


class ConfigurationError(ValueError):
    """Raised when the configured agent graph is invalid."""


def _get_active_agents(config=None):
    agents_cfg = (config or {}).get("agents", {})
    active_agents = agents_cfg.get("active", DEFAULT_ACTIVE_AGENTS)
    if isinstance(active_agents, str):
        active_agents = [active_agents]
    return active_agents


def _validate_active_agents(active_agents):
    unknown = [name for name in active_agents if name not in AGENT_REGISTRY]
    if unknown:
        available = ", ".join(sorted(AGENT_REGISTRY))
        raise ConfigurationError(
            f"Unknown active agent(s): {', '.join(unknown)}. "
            f"Available agents: {available}"
        )

    provided = set(META_PROVIDES)
    for name in active_agents:
        spec = AGENT_REGISTRY[name]
        missing = spec["requires"] - provided
        if missing:
            raise ConfigurationError(
                f"Agent '{name}' requires missing resource(s): "
                f"{', '.join(sorted(missing))}"
            )
        provided.update(spec["provides"])


def build_graph(config=None, execution_mode="sequential"):
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

    active_agents = _get_active_agents(config)
    _validate_active_agents(active_agents)

    builder = StateGraph(AgentState)

    # Add meta nodes (hubs)
    builder.add_node("meta_entry", meta_entry)
    builder.add_node("meta_collect_policy", meta_collect_policy)
    builder.add_node("collect_results", collect_results)

    # Add agent nodes (spokes)
    for agent_name in active_agents:
        builder.add_node(agent_name, AGENT_REGISTRY[agent_name]["node_fn"])

    # Phase 1: Entry → Policy → Collect
    builder.set_entry_point("meta_entry")
    if "policy" in active_agents:
        builder.add_edge("meta_entry", "policy")
        builder.add_edge("policy", "meta_collect_policy")
    else:
        builder.add_edge("meta_entry", "meta_collect_policy")

    # Phase 2: Mode-dependent execution
    downstream_agents = [name for name in active_agents if name != "policy"]
    if execution_mode == "parallel":
        if downstream_agents:
            for agent_name in downstream_agents:
                builder.add_edge("meta_collect_policy", agent_name)
                builder.add_edge(agent_name, "collect_results")
        else:
            builder.add_edge("meta_collect_policy", "collect_results")
    else:
        previous_node = "meta_collect_policy"
        for agent_name in downstream_agents:
            builder.add_edge(previous_node, agent_name)
            previous_node = agent_name
        builder.add_edge(previous_node, "collect_results")

    # End
    builder.add_edge("collect_results", END)

    return builder.compile()
