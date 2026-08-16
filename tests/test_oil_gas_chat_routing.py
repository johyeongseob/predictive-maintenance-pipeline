from src.utility.chat_intent import (
    classify_intent_keyword,
    get_active_chat_modes,
    is_valid_chat_mode,
    parse_intent_response,
    resolve_active_chat_mode,
)


def test_corrosion_keyword_routing():
    questions = [
        "Which material has the highest corrosion risk?",
        "Which pipeline samples should be prioritized for corrosion maintenance?",
        "What are the high-pressure corrosion risks?",
        "Show material risk from predicted thickness loss.",
    ]

    for question in questions:
        assert classify_intent_keyword(question) == "corrosion"


def test_corrosion_llm_response_parsing():
    assert parse_intent_response("corrosion") == "corrosion"
    assert parse_intent_response("The correct mode is corrosion.") == "corrosion"


def test_corrosion_valid_chat_mode():
    assert is_valid_chat_mode("corrosion")


def test_corrosion_respects_active_agents():
    active_config = {
        "agents": {
            "active": ["policy", "analysis", "evidence", "corrosion"],
        }
    }
    inactive_config = {
        "agents": {
            "active": ["policy", "analysis", "evidence"],
        }
    }

    assert "corrosion" in get_active_chat_modes(active_config)
    assert resolve_active_chat_mode("corrosion", active_config) == "corrosion"
    assert resolve_active_chat_mode("corrosion", inactive_config) == "analysis"
