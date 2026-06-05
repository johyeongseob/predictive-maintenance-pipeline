#!/usr/bin/env python3
"""
Benchmark PACE pipeline comparing CPU vs GPU performance across all stages.
Captures detailed timing breakdown for each component including SQL database operations.
"""

import subprocess
import json
import time
import sys
import yaml
from pathlib import Path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.clean_slate import clean_database, clean_outputs

def update_config_device(device: str, config_path: str = "config/pipeline_defects_detection.yaml"):
    """Update inference, glue, and sql device in config.yaml."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['inference']['device'] = device
    config['glue']['device'] = device
    config['sql']['device'] = device
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✏️  Updated config: inference.device={device}, glue.device={device}, sql.device={device}")

def run_pipeline_with_timing(device: str, num_images: int = 1000):
    """Run pipeline and capture timing for each stage."""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: {device} with {num_images} images")
    print(f"{'='*80}\n")
    
    timings = {
        "device": device,
        "num_images": num_images,
        "stages": {}
    }
    
    # Update config.yaml for this device
    update_config_device(device)
    
    # Clean outputs directory based on config
    with open('config.json', 'r') as f:
        main_config = json.load(f)
    use_case_id = main_config.get('default-use-case', 'pipeline_defects_detection')
    with open(f'config/{use_case_id}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir = f'out/{use_case_id}'
    sqlite_cfg = config.get('sqlite', {})
    if sqlite_cfg.get('clear_outputs', True):
        clean_outputs(out_dir=out_dir)
    if sqlite_cfg.get('clear_on_run', True):
        clean_database(db_path=sqlite_cfg.get('db_path', f'{out_dir}/sql_data/detections.db'))
    
    # Start total timer
    total_start = time.time()
    
    # Stage 1: YOLO Inference
    print(f"\n[Stage 1] Running YOLO inference on {device}...")
    stage_start = time.time()
    result = subprocess.run(
        ["python", "run_inference_oep.py", "--num-images", str(num_images),
         "--device", device, "--output", f"{out_dir}/detections.jsonl"],
        capture_output=True,
        text=True
    )
    timings["stages"]["yolo_inference"] = time.time() - stage_start
    
    if result.returncode != 0:
        print(f"❌ YOLO inference failed: {result.stderr}")
        return None
    
    # Extract FPS from output
    for line in result.stdout.split('\n'):
        if 'it/s' in line and '%' in line:
            # Extract fps from progress bar
            parts = line.split('[')[-1].split(',')
            for part in parts:
                if 'it/s' in part:
                    fps = float(part.split('it/s')[0].strip())
                    timings["yolo_fps"] = fps
                    break
    
    print(f"✅ YOLO Inference: {timings['stages']['yolo_inference']:.2f}s")
    
    # Stage 2: SQL Database Verification (quick check)
    print(f"\n[Stage 2] Verifying SQL database...")
    stage_start = time.time()
    subprocess.run(
        ["python", "-m", "scripts.read_sqlite_samples", "--limit", "5"],
        check=True,
        capture_output=True,
        text=True
    )
    timings["stages"]["sql_verify"] = time.time() - stage_start
    print(f"✅ SQL Verify: {timings['stages']['sql_verify']:.2f}s")
    
    # Stage 3: SQL Query Execution (5 queries)
    print(f"\n[Stage 3] Benchmarking SQL query execution on {device}...")
    stage_start = time.time()
    
    # Load config to get SQL model
    with open('config/pipeline_defects_detection.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize SQL components
    from src.agents.utility.openvino_llm import OpenVINOLLM
    from src.utility.sqlite_client import SQLiteClient
    from src.agents.sql_query_executor import SQLQueryExecutor
    
    sql_model = OpenVINOLLM(
        model_path=config['sql']['model_id'],
        device=device
    )
    db_client = SQLiteClient(db_path=config['sqlite']['db_path'])
    sql_executor = SQLQueryExecutor(
        sqlite_client=db_client,
        llm=sql_model
    )
    
    # Test queries
    test_queries = [
        "show all detections",
        "show high confidence detections above 0.8",
        "count all detections",
        "which frames have the most detections",
        "show detection distribution by class"
    ]
    
    query_times = []
    for i, query in enumerate(test_queries, 1):
        query_start = time.time()
        try:
            sql_executor.execute_natural_language_query(query)
            query_time = time.time() - query_start
            query_times.append(query_time)
            print(f"   Query {i}/5: {query_time:.2f}s")
        except Exception as e:
            print(f"   Query {i}/5 failed: {e}")
            query_times.append(0)
    
    timings["stages"]["sql_queries"] = time.time() - stage_start
    timings["sql_query_times"] = {
        "individual": query_times,
        "average": sum(query_times) / len(query_times) if query_times else 0,
        "total": sum(query_times)
    }
    print(f"✅ SQL Queries: {timings['stages']['sql_queries']:.2f}s (avg: {timings['sql_query_times']['average']:.2f}s/query)")
    
    # Stage 4: Agent Orchestration (LLM)
    print(f"\n[Stage 4] Running LLM Agent Orchestration on {device}...")
    stage_start = time.time()
    
    # Set PYTHONPATH for agent imports
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    result = subprocess.run(
        ["python", "-m", "scripts.run_agent_orchestration"],
        capture_output=True,
        text=True,
        env=env
    )
    timings["stages"]["agent_orchestration"] = time.time() - stage_start
    
    # Extract individual agent timings
    agent_times = {}
    for line in result.stdout.split('\n'):
        if '⏱️  Completed in' in line:
            if 'Policy Agent' in line:
                agent_times['policy'] = float(line.split('in ')[1].split('s')[0])
            elif 'Analysis Agent' in line:
                agent_times['analysis'] = float(line.split('in ')[1].split('s')[0])
            elif 'Evidence Agent' in line:
                agent_times['evidence'] = float(line.split('in ')[1].split('s')[0])
    
    if agent_times:
        timings["agent_times"] = agent_times
    
    print(f"✅ Agent Orchestration: {timings['stages']['agent_orchestration']:.2f}s")
    if agent_times:
        print(f"   - Policy Agent: {agent_times.get('policy', 0):.2f}s")
        print(f"   - Analysis Agent: {agent_times.get('analysis', 0):.2f}s")
        print(f"   - Evidence Agent: {agent_times.get('evidence', 0):.2f}s")
    
    # Calculate total
    timings["total_time"] = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"TOTAL TIME ({device}): {timings['total_time']:.2f}s")
    print(f"{'='*80}\n")
    
    return timings


def print_comparison(cpu_timings, gpu_timings):
    """Print detailed comparison table."""
    
    print("\n" + "="*100)
    print("PERFORMANCE COMPARISON: CPU vs GPU (1000 images)")
    print("="*100)
    
    # Overall comparison
    print(f"\n{'Stage':<30} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<15} {'Winner':<10}")
    print("-"*100)
    
    stages = [
        ("YOLO Inference", "yolo_inference"),
        ("SQL Database Verification", "sql_verify"),
        ("SQL Query Execution (5 queries)", "sql_queries"),
        ("Agent Orchestration (LLM)", "agent_orchestration"),
    ]
    
    for stage_name, stage_key in stages:
        cpu_time = cpu_timings["stages"].get(stage_key, 0)
        gpu_time = gpu_timings["stages"].get(stage_key, 0)
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            winner = "GPU" if speedup > 1 else "CPU"
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
            winner = "N/A"
        
        print(f"{stage_name:<30} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup_str:<15} {winner:<10}")
    
    print("-"*100)
    total_speedup = cpu_timings["total_time"] / gpu_timings["total_time"]
    print(f"{'TOTAL PIPELINE':<30} {cpu_timings['total_time']:<15.2f} {gpu_timings['total_time']:<15.2f} {total_speedup:.2f}x{'GPU' if total_speedup > 1 else 'CPU':<15}")
    
    # SQL Query breakdown
    if "sql_query_times" in cpu_timings and "sql_query_times" in gpu_timings:
        print("\n" + "="*100)
        print("SQL QUERY PERFORMANCE (5 queries)")
        print("="*100)
        print(f"\n{'Metric':<30} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<15} {'Winner':<10}")
        print("-"*100)
        
        cpu_avg = cpu_timings["sql_query_times"]["average"]
        gpu_avg = gpu_timings["sql_query_times"]["average"]
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        winner = "GPU" if speedup > 1 else "CPU"
        
        print(f"{'Average Query Time':<30} {cpu_avg:<15.2f} {gpu_avg:<15.2f} {speedup:.2f}x{winner:<15}")
        print(f"{'Total (5 queries)':<30} {cpu_timings['sql_query_times']['total']:<15.2f} {gpu_timings['sql_query_times']['total']:<15.2f} {'':>15} {'':>10}")
    
    # Agent breakdown
    if "agent_times" in cpu_timings and "agent_times" in gpu_timings:
        print("\n" + "="*100)
        print("AGENT BREAKDOWN (LLM Performance)")
        print("="*100)
        print(f"\n{'Agent':<30} {'CPU (s)':<15} {'GPU (s)':<15} {'Speedup':<15} {'Winner':<10}")
        print("-"*100)
        
        for agent_name in ['policy', 'analysis', 'evidence']:
            cpu_time = cpu_timings["agent_times"].get(agent_name, 0)
            gpu_time = gpu_timings["agent_times"].get(agent_name, 0)
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                winner = "GPU" if speedup > 1 else "CPU"
                print(f"{agent_name.capitalize() + ' Agent':<30} {cpu_time:<15.2f} {gpu_time:<15.2f} {speedup:.2f}x{winner:<15}")
    
    # YOLO FPS comparison
    if "yolo_fps" in cpu_timings and "yolo_fps" in gpu_timings:
        print("\n" + "="*100)
        print("YOLO THROUGHPUT")
        print("="*100)
        print(f"CPU: {cpu_timings['yolo_fps']:.2f} images/sec")
        print(f"GPU: {gpu_timings['yolo_fps']:.2f} images/sec")
        print(f"Speedup: {gpu_timings['yolo_fps'] / cpu_timings['yolo_fps']:.2f}x")
    
    print("\n" + "="*100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark PACE pipeline CPU vs GPU")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU benchmark")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args()
    
    results = {}
    
    # Run CPU benchmark
    if not args.skip_cpu:
        cpu_timings = run_pipeline_with_timing("CPU", args.num_images)
        if cpu_timings:
            results["cpu"] = cpu_timings
            # Save intermediate results
            with open("out/benchmark_cpu.json", "w") as f:
                json.dump(cpu_timings, f, indent=2)
    else:
        # Load from file if skipping
        try:
            with open("out/benchmark_cpu.json", "r") as f:
                results["cpu"] = json.load(f)
        except FileNotFoundError:
            print("❌ No CPU benchmark results found. Run without --skip-cpu first.")
            sys.exit(1)
    
    # Run GPU benchmark
    if not args.skip_gpu:
        gpu_timings = run_pipeline_with_timing("GPU", args.num_images)
        if gpu_timings:
            results["gpu"] = gpu_timings
            # Save intermediate results
            with open("out/benchmark_gpu.json", "w") as f:
                json.dump(gpu_timings, f, indent=2)
    else:
        # Load from file if skipping
        try:
            with open("out/benchmark_gpu.json", "r") as f:
                results["gpu"] = json.load(f)
        except FileNotFoundError:
            print("❌ No GPU benchmark results found. Run without --skip-gpu first.")
            sys.exit(1)
    
    # Print comparison
    if "cpu" in results and "gpu" in results:
        print_comparison(results["cpu"], results["gpu"])
        
        # Save complete results
        with open("out/benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Benchmark complete! Results saved to out/benchmark_results.json")


if __name__ == "__main__":
    main()
