
#!/usr/bin/env python3
"""
Agent Orchestration Runner - Hub-and-Spoke Architecture.

Runs the multi-agent pipeline using Meta Agent as central hub.
All agents communicate ONLY through Meta Agent.

Usage:
    python -m scripts.run_agent_orchestration
    python -m scripts.run_agent_orchestration --use-case my_use_case
"""

import sys
from pathlib import Path

# Add project root to path for imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import json
import os
from src.agents.graph_builder import build_graph
from src.agents.utility import DatabaseBackend, OpenVINOLLM, RemoteLLM
from src.utility import load_prompts


def _create_llm(cfg: dict, agent_name: str = None):
    """
    Create LLM instance based on agent-specific or global config.
    
    Args:
        cfg: Full configuration dict
        agent_name: Agent name (policy, analysis, evidence) for per-agent config
    """
    # Try agent-specific config first
    if agent_name:
        agent_cfg = cfg.get("agents", {}).get(agent_name, {})
        mode = agent_cfg.get("llm_mode")
        if mode:
            if mode == "server":
                server_url = agent_cfg.get("server_url", "http://localhost:8000")
                try:
                    llm = RemoteLLM(server_url=server_url, verbose=False, enable_cache=False)
                    return llm
                except Exception:
                    return None
            
            elif mode == "model":
                model_id = agent_cfg.get("model_id", "models/ov_models/llms/phi-3.5-mini")
                device = agent_cfg.get("device", "GPU")
                suppress_thinking = agent_cfg.get("suppress_thinking", True)
                if os.path.exists(model_id):
                    try:
                        llm = OpenVINOLLM(model_path=model_id, device=device, verbose=False, enable_cache=False, suppress_thinking=suppress_thinking)
                        return llm
                    except Exception:
                        return None
                return None
            
            elif mode == "fallback":
                return None
    
    # Fall back to global glue config
    glue_cfg = cfg.get("glue", {})
    mode = glue_cfg.get("mode", "fallback")
    enable_cache = glue_cfg.get("enable_cache", False)
    
    if mode == "server":
        server_url = glue_cfg.get("server_url", "http://localhost:8000")
        try:
            return RemoteLLM(server_url=server_url, verbose=False, enable_cache=enable_cache)
        except Exception:
            return None
            
    elif mode == "model":
        model_id = glue_cfg.get("model_id", "models/ov_models/llms/phi-3.5-mini")
        device = glue_cfg.get("device", "CPU")
        suppress_thinking = glue_cfg.get("suppress_thinking", True)
        
        if not os.path.exists(model_id):
            # List available models to help user
            models_dir = "models/ov_models/llms"
            available_models = []
            if os.path.exists(models_dir):
                available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            
            error_msg = f"\n{'='*80}\nERROR: Model not found: {model_id}\n{'='*80}\n"
            if available_models:
                error_msg += f"\nAvailable models in {models_dir}:\n"
                for model in sorted(available_models):
                    error_msg += f"  - {model}\n"
                error_msg += f"\nUpdate your config file to use one of these models.\n"
            else:
                error_msg += f"\nNo models found in {models_dir}. Please download models first.\n"
            error_msg += f"{'='*80}\n"
            
            raise FileNotFoundError(error_msg)
        
        try:
            return OpenVINOLLM(model_path=model_id, device=device, verbose=False, enable_cache=enable_cache, suppress_thinking=suppress_thinking)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")
    
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-case", type=str, default=None,
                    help="Use case prompt file name (default: from config.json)")
    args = ap.parse_args()

    # Load default use case from config.json
    with open("config.json", "r", encoding="utf-8") as f:
        main_config = json.load(f)
    use_case_id = main_config.get("default-use-case", "pipeline_defects_detection")
    
    # Override use_case_id if user provided --use-case
    if args.use_case:
        use_case_id = args.use_case
    
    # Load the use-case specific config
    config_path = f"config/{use_case_id}/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize database backend
    db_backend = DatabaseBackend(cfg)
    
    # Determine execution mode
    agents_cfg = cfg.get("agents", {})
    execution_mode = agents_cfg.get("execution_mode", "sequential")
    
    if execution_mode == "sequential":
        # Sequential: All agents share one LLM instance on same device
        shared_device = agents_cfg.get("shared_device", "GPU")
        model_id = agents_cfg.get("policy", {}).get("model_id", "models/ov_models/llms/phi-3.5-mini")
        suppress_thinking = agents_cfg.get("suppress_thinking", cfg.get("glue", {}).get("suppress_thinking", True))
        print(f"[Setup] Mode: Sequential (shared LLM on {shared_device})")
        
        if os.path.exists(model_id):
            shared_llm = OpenVINOLLM(model_path=model_id, device=shared_device, verbose=False, enable_cache=False, suppress_thinking=suppress_thinking)
            print(f"[Setup] 🔥 Warming up shared LLM...")
            if hasattr(shared_llm, 'warmup'):
                shared_llm.warmup()
            print(f"[Setup] ✅ Warmup complete")
        else:
            shared_llm = None
        
        policy_llm = shared_llm
        analysis_llm = shared_llm
        evidence_llm = shared_llm
    
    else:  # parallel mode
        # Parallel: Each agent gets separate LLM instance (can use different devices)
        print(f"[Setup] Mode: Parallel (per-agent LLMs)")
        policy_llm = _create_llm(cfg, "policy")
        analysis_llm = _create_llm(cfg, "analysis")
        evidence_llm = _create_llm(cfg, "evidence")
        
        # Warm up all LLMs SEQUENTIALLY (one at a time) to avoid NPU contention
        print(f"[Setup] 🔥 Warming up LLMs (sequential)...")
        if policy_llm and hasattr(policy_llm, 'warmup'):
            print(f"[Setup]   - Warming up policy LLM...")
            policy_llm.warmup()
        if analysis_llm and hasattr(analysis_llm, 'warmup'):
            print(f"[Setup]   - Warming up analysis LLM...")
            analysis_llm.warmup()
        if evidence_llm and hasattr(evidence_llm, 'warmup'):
            print(f"[Setup]   - Warming up evidence LLM...")
            evidence_llm.warmup()
        print(f"[Setup] ✅ Warmup complete")

    # Load prompts
    prompt_file = f"prompts/{use_case_id}.txt"
    prompts = load_prompts(prompt_file)
    
    # Compute output directory
    out_dir = f"out/{use_case_id}"
    
    # ✅ Build meta namespace with per-agent resources
    meta = {
        "resources": {
            "db_backend": db_backend,
            "llms": {
                "policy": policy_llm,
                "analysis": analysis_llm,
                "evidence": evidence_llm
            },
            "config": cfg,
            "out_dir": out_dir,
            "prompts": {
                "policy_prompt": prompts.get("policy", ""),
                "analysis_prompt": prompts.get("analysis", ""),
                "evidence_prompt": prompts.get("evidence", "")
            }
        }
    }
    
    # ✅ Build initial state (hub-and-spoke)
    initial_state = {"meta": meta}
    
    # Build and run LangGraph workflow with execution mode
    graph = build_graph(execution_mode=execution_mode)
    
    # Run workflow
    final_state = graph.invoke(initial_state)
    
    # Extract results from meta namespace
    meta_final = final_state.get("meta", {})
    analysis_data = meta_final.get("analysis", {})
    
    # Print results
    print("\n" + "="*70)
    print("[Results] ✅ Pipeline Complete")
    print("="*70)
    print(f"Total Defects Analyzed: {analysis_data.get('total_defects', 0)}")
    print(f"\n📁 Outputs saved to:")
    print(f"   • Policy: {out_dir}/agent/policy.json")
    print(f"   • Analysis: {out_dir}/agent/analysis_report.json")
    print(f"   • Analysis Summary: {out_dir}/agent/analysis_summary.txt")
    print(f"   • Evidence: {out_dir}/agent/evidence.json")
    print(f"   • Audit Trail: {out_dir}/agent/evidence_trail.txt")


if __name__ == "__main__":
    main()
