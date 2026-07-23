import pytest

from src.agents.graph_builder import ConfigurationError, build_graph
from src.utility.chat_intent import resolve_active_chat_mode


def test_build_graph_with_default_agents():
    graph = build_graph(
        config={"agents": {"active": ["policy", "analysis", "evidence"]}},
        execution_mode="sequential",
    )

    assert graph is not None


def test_build_graph_without_evidence_agent():
    graph = build_graph(
        config={"agents": {"active": ["policy", "analysis"]}},
        execution_mode="sequential",
    )

    assert graph is not None


def test_build_graph_rejects_missing_policy_dependency():
    with pytest.raises(ConfigurationError):
        build_graph(config={"agents": {"active": ["analysis"]}})


def test_chat_routing_respects_active_agents():
    config = {"agents": {"active": ["policy", "analysis"]}}

    assert resolve_active_chat_mode("evidence", config) == "analysis"
    assert resolve_active_chat_mode("sql", config) == "sql"
