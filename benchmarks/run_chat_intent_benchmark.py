"""Benchmark keyword and LLM chat intent routers."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.chat_intent import classify_intent_keyword, classify_intent_with_llm


def load_queries(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_default_config() -> dict:
    with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
        main_config = json.load(f)
    use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
    config_path = PROJECT_ROOT / "config" / use_case_id / "config.yaml"
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        config = {"glue": parse_simple_yaml_section(config_path, "glue")}
    return {"use_case_id": use_case_id, "config": config}


def parse_simple_yaml_section(path: Path, section_name: str) -> dict:
    section = {}
    in_section = False
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            if not line.startswith(" ") and line.endswith(":"):
                in_section = line[:-1] == section_name
                continue
            if not in_section or not line.startswith("  ") or ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            value = value.strip().strip("'\"")
            if value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            else:
                parsed_value = value
            section[key] = parsed_value
    return section


def load_openvino_llm_classes():
    from src.agents.utility.openvino_llm import OpenVINOLLM, RemoteLLM

    return OpenVINOLLM, RemoteLLM


def default_output_path(strategy: str) -> Path:
    output_names = {
        "keyword": "chat_intent_keyword_results.json",
        "llm": "chat_intent_llm_results.json",
        "both": "chat_intent_benchmark_results.json",
    }
    return PROJECT_ROOT / "benchmarks" / output_names[strategy]


def load_router_llm():
    OpenVINOLLM, RemoteLLM = load_openvino_llm_classes()

    loaded = get_default_config()
    glue_cfg = loaded["config"].get("glue", {})
    mode = glue_cfg.get("mode", "model")
    enable_cache = glue_cfg.get("enable_cache", False)

    if mode == "server":
        return RemoteLLM(
            server_url=glue_cfg.get("server_url", "http://localhost:8000"),
            verbose=False,
            enable_cache=enable_cache,
        )

    if mode == "model":
        return OpenVINOLLM(
            model_path=glue_cfg.get("model_id", "models/ov_models/llms/Phi-4-mini-instruct-int4gq"),
            device=glue_cfg.get("device", "CPU"),
            verbose=False,
            enable_cache=enable_cache,
            suppress_thinking=glue_cfg.get("suppress_thinking", True),
        )

    raise ValueError(f"Unsupported glue.mode: {mode}")


def load_chat_engine():
    from interactive_chat import InteractiveChat

    chat = InteractiveChat()
    if not chat.load_config():
        raise RuntimeError("Failed to load chat configuration")
    if not chat.load_artifacts():
        raise RuntimeError("Failed to load chat artifacts")
    return chat


def answer_question_with_chat(chat, question: str, mode: str) -> str:
    if mode == "analysis":
        prompt = f"Context:\n{chat.analysis_summary}\n\nQuestion: {question}\n\n{chat.QA_INSTRUCTION}"
        return chat.ask_question(prompt, show_thinking=False)

    if mode == "evidence":
        evidence_trail = ""
        with open(PROJECT_ROOT / "config.json", "r", encoding="utf-8") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        trail_path = PROJECT_ROOT / "out" / use_case_id / "agent" / "evidence_trail.txt"
        if trail_path.exists():
            with open(trail_path, "r", encoding="utf-8") as f:
                evidence_trail = f.read()

        prompt = f"Context:\n{evidence_trail}\n\nQuestion: {question}\n\n{chat.QA_INSTRUCTION}"
        return chat.ask_question(prompt, show_thinking=False)

    if mode == "sql":
        sql_query = chat.generate_sql_query(question)
        results = chat.execute_and_format_query(sql_query)
        return f"Generated SQL:\n{sql_query}\n\n{results}"

    return f"Unknown mode: {mode}"


def benchmark_strategy(name: str, queries: list[dict], llm=None, chat=None) -> dict:
    rows = []
    for item in queries:
        t0 = time.perf_counter()
        if name == "keyword":
            predicted = classify_intent_keyword(item["question"])
        elif name == "llm":
            predicted = classify_intent_with_llm(item["question"], llm)
        else:
            raise ValueError(f"Unsupported strategy: {name}")
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        expected = item["expected_mode"]
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_mode": expected,
            "predicted_mode": predicted,
            "correct": predicted == expected,
            "latency_ms": latency_ms,
        }
        if chat is not None:
            answer_t0 = time.perf_counter()
            row["answer"] = answer_question_with_chat(chat, item["question"], predicted)
            row["answer_latency_ms"] = round((time.perf_counter() - answer_t0) * 1000, 2)
        rows.append(row)

    correct = sum(row["correct"] for row in rows)
    latencies = [row["latency_ms"] for row in rows]
    answer_latencies = [
        row["answer_latency_ms"]
        for row in rows
        if "answer_latency_ms" in row
    ]
    by_mode = {}
    for row in rows:
        mode = row["expected_mode"]
        by_mode.setdefault(mode, {"correct": 0, "total": 0})
        by_mode[mode]["correct"] += int(row["correct"])
        by_mode[mode]["total"] += 1

    result = {
        "strategy": name,
        "correct": correct,
        "total": len(rows),
        "accuracy": round(correct / len(rows), 2) if rows else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
    }
    if answer_latencies:
        result["avg_answer_latency_ms"] = round(statistics.mean(answer_latencies), 2)
    result["by_mode"] = by_mode
    result["rows"] = rows
    return result


def print_summary(result: dict) -> None:
    print(
        f"{result['strategy']}: "
        f"{result['correct']}/{result['total']} "
        f"({result['accuracy']:.2%}), "
        f"avg latency {result['avg_latency_ms']:.2f} ms"
    )
    if "avg_answer_latency_ms" in result:
        print(f"  avg answer latency: {result['avg_answer_latency_ms']:.2f} ms")
    for mode, stats in result["by_mode"].items():
        print(f"  {mode}: {stats['correct']}/{stats['total']}")
    wrong = [row for row in result["rows"] if not row["correct"]]
    if wrong:
        print("  Wrong predictions:")
        for row in wrong:
            print(
                f"    {row['id']}: expected={row['expected_mode']} "
                f"predicted={row['predicted_mode']} | {row['question']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "chat_intent_queries.json",
    )
    parser.add_argument(
        "--strategy",
        choices=["keyword", "llm", "both"],
        default="both",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to a strategy-specific results file.",
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Also run the selected chatbot backend and store its answer for each query.",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    strategies = ["keyword", "llm"] if args.strategy == "both" else [args.strategy]
    output_path = args.output or default_output_path(args.strategy)

    chat = None
    llm = None
    if args.include_answers:
        chat = load_chat_engine()
        llm = chat.llm
    elif "llm" in strategies:
        llm = load_router_llm()

    results = [
        benchmark_strategy(strategy, queries, llm=llm, chat=chat)
        for strategy in strategies
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for result in results:
        print_summary(result)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
