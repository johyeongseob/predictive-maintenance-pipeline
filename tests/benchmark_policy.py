#!/usr/bin/env python3
"""
Benchmark Policy Agent across all available LLM models.

Tests each model in models/ov_models/llms on both GPU and NPU,
measures execution time, and saves policy outputs.
"""

import sys
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.agents.temp_meta_agent import run_policy_only
from src.agents.utility import DatabaseBackend, OpenVINOLLM
from src.utility import load_prompts

# Paths
MODELS_DIR = Path("models/ov_models/llms")
OUTPUT_DIR = Path("tests/out")
POLICY_OUTPUT = Path("out/agent/policy.json")


def get_available_models() -> List[str]:
    """Get list of available model directories."""
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for item in sorted(MODELS_DIR.iterdir()):
        if item.is_dir():
            # Check if it has openvino_model.xml (valid OV model)
            if (item / "openvino_model.xml").exists():
                models.append(item.name)
    
    return models


def load_model_and_setup(model_name: str, device: str):
    """
    Load model and create meta structure (one-time setup).
    
    Returns:
        tuple of (policy_llm, meta, cfg) or (None, None, None) on error
    """
    model_path = MODELS_DIR / model_name
    
    if not model_path.exists():
        print(f"❌ Model path not found: {model_path}")
        return None, None, None
    
    try:
        # Load config
        with open("config.json", "r", encoding="utf-8") as f:
            main_config = json.load(f)
        use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
        
        config_path = f"config/{use_case_id}/config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Initialize database backend
        db_backend = DatabaseBackend(cfg)
        
        # Load model
        print(f"[Setup] Loading model from {model_path}...")
        policy_llm = OpenVINOLLM(
            model_path=str(model_path), 
            device=device, 
            verbose=False, 
            enable_cache=False,
            suppress_thinking=True
        )
        
        print(f"[Setup] 🔥 Warming up LLM...")
        if hasattr(policy_llm, 'warmup'):
            policy_llm.warmup()
        print(f"[Setup] ✅ Warmup complete")

        # Load prompts
        prompt_file = f"prompts/pipeline_defects_detection.txt"
        prompts = load_prompts(prompt_file)
        
        # Build meta namespace
        meta = {
            "resources": {
                "db_backend": db_backend,
                "llms": {
                    "policy": policy_llm
                },
                "config": cfg,
                "prompts": {
                    "policy_prompt": prompts.get("policy", "")
                }
            }
        }
        
        return policy_llm, meta, cfg
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None


def run_policy_agent(meta: dict) -> Dict:
    """
    Run policy agent with already-loaded model (measures inference only).
    
    Returns:
        dict with 'success', 'time', 'error' keys
    """
    # Remove old policy output if exists
    if POLICY_OUTPUT.exists():
        POLICY_OUTPUT.unlink()
    
    start_time = time.time()
    
    try:
        # Run policy agent directly (no subprocess, no reload)
        run_policy_only(meta)
        
        elapsed = time.time() - start_time
        
        # Check if policy output was created
        if POLICY_OUTPUT.exists():
            return {
                "success": True,
                "time": elapsed,
                "error": None
            }
        else:
            return {
                "success": False,
                "time": elapsed,
                "error": "Policy output not created"
            }
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "time": elapsed,
            "error": str(e)
        }


def save_policy_output(model_name: str, device: str):
    """Save policy output to benchmark directory."""
    if not POLICY_OUTPUT.exists():
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy policy output with model name and device
    output_name = f"policy_{model_name}_{device}.json"
    output_path = OUTPUT_DIR / output_name
    
    shutil.copy2(POLICY_OUTPUT, output_path)
    print(f"✅ Saved: {output_path}")


def main():
    print("\n" + "="*80)
    print("Policy Agent Benchmark")
    print("="*80)
    
    # Get available models
    models = get_available_models()
    
    if not models:
        print("❌ No models found in models/ov_models/llms/")
        return
    
    print(f"\n📋 Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        print(f"   {i:2d}. {model}")
    
    # Test devices
    devices = ["GPU"]
    
    # Results storage
    results = []
    
    # Number of runs per model
    NUM_RUNS = 1
    
    # Run benchmarks
    total_tests = len(models) * len(devices)
    current_test = 0
    
    for model_name in models:
        for device in devices:
            current_test += 1
            
            print(f"\n{'#'*80}")
            print(f"# Test {current_test}/{total_tests}: {model_name} on {device}")
            print(f"{'#'*80}")
            
            # Load model ONCE (this includes compilation time)
            load_start = time.time()
            policy_llm, meta, cfg = load_model_and_setup(model_name, device)
            load_time = time.time() - load_start
            
            if policy_llm is None:
                # Failed to load
                for run_num in range(1, NUM_RUNS + 1):
                    test_result = {
                        "model": model_name,
                        "device": device,
                        "run": run_num,
                        "success": False,
                        "time_seconds": 0.0,
                        "load_time_seconds": round(load_time, 3),
                        "error": "Failed to load model"
                    }
                    results.append(test_result)
                print(f"❌ Failed to load model - skipping all runs")
                continue
            
            print(f"\n⏱️  Model load + warmup time: {load_time:.3f}s")
            print(f"{'='*80}")
            
            # Run benchmark 3 times with SAME loaded model
            run_results = []
            for run_num in range(1, NUM_RUNS + 1):
                print(f"\n--- Run {run_num}/{NUM_RUNS} (inference only) ---")
                
                result = run_policy_agent(meta)
                
                # Save result for this run
                test_result = {
                    "model": model_name,
                    "device": device,
                    "run": run_num,
                    "success": result["success"],
                    "time_seconds": round(result["time"], 3),
                    "load_time_seconds": round(load_time, 3) if run_num == 1 else 0.0,
                    "error": result["error"]
                }
                results.append(test_result)
                run_results.append(result)
                
                # Print result
                if result["success"]:
                    print(f"✅ Run {run_num} SUCCESS - Inference time: {result['time']:.3f}s")
                    save_policy_output(f"{model_name}_run{run_num}", device)
                else:
                    print(f"❌ Run {run_num} FAILED - Time: {result['time']:.3f}s")
                    print(f"   Error: {result['error']}")
            
            # Print summary for this model
            successful_runs = [r for r in run_results if r["success"]]
            if successful_runs:
                times = [r["time"] for r in successful_runs]
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"\n{'='*80}")
                print(f"📊 Summary for {model_name} on {device}:")
                print(f"   Load + warmup time: {load_time:.3f}s (one-time)")
                print(f"   Successful runs: {len(successful_runs)}/{NUM_RUNS}")
                print(f"   Avg inference time: {avg_time:.3f}s")
                print(f"   Min inference time: {min_time:.3f}s")
                print(f"   Max inference time: {max_time:.3f}s")
                print(f"{'='*80}")
    
    # Save benchmark results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_file = OUTPUT_DIR / "policy_benchmark_results.json"
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("📊 BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    # Summary statistics
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    print(f"Total Tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    # Print timing table
    print(f"\n{'Model':<40} {'Device':<8} {'Run':<5} {'Infer (s)':<12} {'Load (s)':<12} {'Status'}")
    print("-" * 100)
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        time_str = f"{r['time_seconds']:.3f}" if r["success"] else "FAILED"
        load_str = f"{r['load_time_seconds']:.3f}" if r['load_time_seconds'] > 0 else "-"
        print(f"{r['model']:<40} {r['device']:<8} {r['run']:<5} {time_str:<12} {load_str:<12} {status}")
    
    print(f"\n📁 Results saved to:")
    print(f"   • Benchmark data: {benchmark_file}")
    print(f"   • Policy outputs: {OUTPUT_DIR}/policy_*.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
