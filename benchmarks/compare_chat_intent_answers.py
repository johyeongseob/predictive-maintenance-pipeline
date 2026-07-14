"""Compare chatbot answer previews from keyword and LLM router benchmark results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_single_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"{path} contains {len(data)} strategies; expected exactly one.")
        return data[0]
    return data


def normalize_answer(answer: str | None) -> str:
    text = answer or ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def short_text(text: str | None, limit: int = 180) -> str:
    normalized = normalize_answer(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def compare_results(keyword_result: dict, llm_result: dict) -> dict:
    keyword_rows = {row["id"]: row for row in keyword_result.get("rows", [])}
    llm_rows = {row["id"]: row for row in llm_result.get("rows", [])}
    common_ids = sorted(set(keyword_rows) & set(llm_rows))

    rows = []
    for row_id in common_ids:
        keyword_row = keyword_rows[row_id]
        llm_row = llm_rows[row_id]
        keyword_answer = keyword_row.get("answer")
        llm_answer = llm_row.get("answer")

        rows.append(
            {
                "id": row_id,
                "question": keyword_row.get("question"),
                "expected_mode": keyword_row.get("expected_mode"),
                "keyword_predicted_mode": keyword_row.get("predicted_mode"),
                "llm_predicted_mode": llm_row.get("predicted_mode"),
                "keyword_answer_preview": short_text(keyword_answer),
                "llm_answer_preview": short_text(llm_answer),
            }
        )

    return {
        "rows": rows,
    }


def print_summary(result: dict) -> None:
    print(f"Compared {len(result['rows'])} rows.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyword-results",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "chat_intent_keyword_results.json",
    )
    parser.add_argument(
        "--llm-results",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "chat_intent_llm_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "chat_intent_answer_comparison.json",
    )
    args = parser.parse_args()

    keyword_result = load_single_result(args.keyword_results)
    llm_result = load_single_result(args.llm_results)
    comparison = compare_results(keyword_result, llm_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print_summary(comparison)
    print(f"\nSaved comparison to {args.output}")


if __name__ == "__main__":
    main()
