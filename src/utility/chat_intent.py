"""Chat intent classification utilities.

Includes:
- Strategy A: keyword/regex-based routing
- Strategy B: LLM-based routing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern


VALID_CHAT_MODES = {"analysis", "evidence", "sql"}


@dataclass(frozen=True)
class IntentRule:
    mode: str
    pattern: Pattern[str]


def _compile(pattern: str) -> Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Strategy A: keyword/regex rules.
# Priority matters: first match wins.
_KEYWORD_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        "sql",
        _compile(
            r"\b("
            r"select|where|count|show\s+me|list\s+all|how\s+many|"
            r"detections?|samples?|frames?|images?|confidence"
            r")\b"
        ),
    ),
    IntentRule(
        "evidence",
        _compile(
            r"\b("
            r"evidence|trail|audit|log|history|policy"
            r")\b"
        ),
    ),
    IntentRule(
        "analysis",
        _compile(
            r"\b("
            r"analysis|summary|summarize|findings?|report"
            r")\b"
        ),
    ),
)


def classify_intent_keyword(question: str) -> str:
    """Classify a user question using keyword/regex rules."""
    text = (question or "").strip()
    if not text:
        return "analysis"

    for rule in _KEYWORD_RULES:
        if rule.pattern.search(text):
            return rule.mode

    return "analysis"


def build_llm_intent_prompt(question: str) -> str:
    """Build the prompt used for LLM-based intent classification."""
    return f"""Classify the user question into exactly one mode.

Use the examples below to decide the mode.

Examples:
What policy was used for this run? -> evidence
Review the policy used for this run. -> evidence
Inspect the per-class policy thresholds. -> evidence
Verify whether database storage was enabled. -> evidence
Trace where the evidence artifacts were stored. -> evidence
Calculate the average confidence for each gas class. -> sql
Select detections where the label is Mixture. -> sql
Summarize the analysis in 3 bullet points. -> analysis

Return only one word: analysis, evidence, or sql.

Question: {question}
"""


def parse_intent_response(response: str) -> str:
    """Parse an LLM response into one of the supported chat modes."""
    text = (response or "").strip().lower()
    if not text:
        return "analysis"

    if text in VALID_CHAT_MODES:
        return text

    first_token = re.split(r"[\s,.:;`\"'\n]+", text)[0]
    if first_token in VALID_CHAT_MODES:
        return first_token

    for mode in ("sql", "evidence", "analysis"):
        if re.search(rf"\b{mode}\b", text):
            return mode

    return "analysis"


def classify_intent_with_llm(question: str, llm) -> str:
    """Classify a user question using an LLM."""
    prompt = build_llm_intent_prompt(question)
    response = llm.invoke(prompt, temperature=0.0, max_new_tokens=20)
    return parse_intent_response(response)


def is_valid_chat_mode(mode: str | None) -> bool:
    """Return whether a mode string maps to a supported chat handler."""
    return mode in VALID_CHAT_MODES


def classify_many_keyword(questions: Iterable[str]) -> list[str]:
    """Classify multiple questions with keyword/regex rules."""
    return [classify_intent_keyword(question) for question in questions]


def classify_many_with_llm(questions: Iterable[str], llm) -> list[str]:
    """Classify multiple questions with an LLM."""
    return [classify_intent_with_llm(question, llm) for question in questions]