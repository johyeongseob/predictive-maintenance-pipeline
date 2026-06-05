#!/usr/bin/env python3
"""
Test SQL-based query executor with the same test cases.
Compare performance with VDMS template-based approach.
"""

import yaml
from src.agents.utility.openvino_llm import OpenVINOLLM
from src.utility.sqlite_client import SQLiteClient
from src.agents.sql_query_executor import SQLQueryExecutor

# Reuse same test queries (natural language only)
TEST_QUERIES = [
    "show all detections",
    "list everything",
    "show detections from frame 5",
    "what's in frame 10",
    "show me all obstacles",
    "find deformations",
    "how many ruptures",
    "show high confidence detections above 0.8",
    "detections with confidence over 0.9",
    "low confidence detections below 0.3",
    "show detections below 0.4 confidence",
    "show detections between 0.5 and 0.7 confidence",
    "confidence from 0.6 to 0.8",
    "show large detections bigger than 50x50",
    "big detections over 100 pixels wide",
    "find small detections under 20x20",
    "tiny detections less than 15 pixels",
    "show detections between frame 10 and 20",
    "detections from frame 5 to 15",
    "detections in top-left corner x<100 y<100",
    "show region x from 50 to 200 and y from 100 to 300",
    "show obstacles and deformations",
    "find ruptures or depositions",
    "show obstacles in frame 5",
    "deformations in frame 10",
    "high confidence obstacles above 0.8",
    "deformations with confidence over 0.9",
    "show critical detections with high confidence and large size",
    "critical issues above 0.8 confidence and 50x50 size",
    "which frames have detections",
    "list all frames with issues",
    "count all detections",
    "how many total detections",
    "show detection distribution by class",
    "what's the breakdown by class",
    "which frames have the most detections",
    "rank frames by detection count",
    "show detections on the edges",
    "find detections near image border within 50 pixels",
    "show detections in center region",
    "confidence statistics for obstacles",
    "what's the average confidence of deformations",
    "overall confidence statistics",
    "average confidence across all detections",
    "average size of deformations",
    "size statistics for obstacles",
    "stats for each class",
    "comprehensive statistics by class",
    "show top 10 highest confidence detections",
    "give me 5 best detections by confidence",
    "show 10 lowest confidence detections",
    "worst 5 detections by confidence",
]


def run_sql_tests():
    """Run all test queries with SQL backend."""
    
    # Load config from config.json -> config/<use_case_id>.yaml
    import json
    print("Loading configuration...", flush=True)
    with open('config.json', 'r') as f:
        main_config = json.load(f)
    use_case_id = main_config.get('default-use-case', 'pipeline_defects_detection')
    config_path = f'config/{use_case_id}/config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"  Config: {config_path}", flush=True)
    
    sql_cfg = config.get('sql', {})
    sqlite_cfg = config.get('sqlite', {})
    
    # Connect to SQLite
    print("Connecting to SQLite...", flush=True)
    sqlite_client = SQLiteClient(db_path=sqlite_cfg.get('db_path', 'out/sql_data/detections.db'))
    
    # Load SQL model
    model_id = sql_cfg.get('model_id', 'models/ov_models/llms/sqlcoder-7b-2-int4cw')
    device = sql_cfg.get('device', 'GPU')
    print(f"Loading SQL model: {model_id} on {device}...", flush=True)
    print("⏳ This may take 30-60 seconds...", flush=True)
    llm = OpenVINOLLM(model_path=model_id, device=device, verbose=False)
    print("✅ Model loaded successfully!", flush=True)
    
    executor = SQLQueryExecutor(sqlite_client, llm)
    
    # Verify data
    count = sqlite_client.count_detections()
    print(f"Database has {count} detections", flush=True)
    print()
    
    # Run tests
    results = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "failures": []
    }
    
    print("=" * 100, flush=True)
    print("TESTING SQL-BASED QUERY EXECUTOR", flush=True)
    print("=" * 100, flush=True)
    print(flush=True)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"Running test {i}/{len(TEST_QUERIES)}: '{query}'...", flush=True)
        
        try:
            # Execute query
            formatted_output, sql, raw_results = executor.execute_natural_language_query(
                query, format_output=True
            )
            
            status = "✅ PASS"
            results["passed"] += 1
            
            print(f"Test {i:2d}: {status}", flush=True)
            print(f"  Query: '{query}'", flush=True)
            print(f"  SQL:   {sql[:100]}...", flush=True)
            print(f"  Results: {len(raw_results)} rows", flush=True)
            
            # Show output preview (first 150 chars)
            output_preview = formatted_output.replace('\n', ' ')[:150]
            print(f"  Output: {output_preview}...", flush=True)
            print(flush=True)
            
        except Exception as e:
            status = "⚠️ ERROR"
            results["errors"] += 1
            results["failures"].append({
                "query": query,
                "error": str(e)
            })
            
            print(f"Test {i:2d}: {status}", flush=True)
            print(f"  Query: '{query}'", flush=True)
            print(f"  Error: {e}", flush=True)
            print(flush=True)
    
    # Summary
    print("=" * 100, flush=True)
    print("TEST SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"Total Tests: {len(TEST_QUERIES)}", flush=True)
    print(f"Passed:      {results['passed']} ({results['passed']/len(TEST_QUERIES)*100:.1f}%)", flush=True)
    print(f"Errors:      {results['errors']} ({results['errors']/len(TEST_QUERIES)*100:.1f}%)", flush=True)
    print(flush=True)
    
    # Show failures
    if results["failures"]:
        print("=" * 100, flush=True)
        print("FAILED TESTS", flush=True)
        print("=" * 100, flush=True)
        for failure in results["failures"]:
            print(f"Query: '{failure['query']}'", flush=True)
            print(f"  Error: {failure['error']}", flush=True)
            print(flush=True)
    
    sqlite_client.close()
    return results


if __name__ == "__main__":
    results = run_sql_tests()
    exit(0 if results["errors"] == 0 else 1)
