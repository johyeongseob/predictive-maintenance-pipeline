"""
Policy Agent - Generate policy using LLM.

Returns policy JSON with filtering rules for downstream agents.
"""

from typing import Dict, Any
import json
import re
import time


def policy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate policy using LLM."""
    start_time = time.time()
    
    meta = state.get("meta", {})
    resources = meta.get("resources", {})
    
    # Get policy-specific LLM
    llms = resources.get("llms", {})
    llm = llms.get("policy")
    config = resources.get("config", {})
    prompts = resources.get("prompts", {})
    policy_prompt = prompts.get("policy_prompt", "")
    
    # Get class names
    class_names = list(config.get("names", {}).values()) if "names" in config else []
    
    # Generate policy with LLM
    if llm and policy_prompt:
        try:
            policy = _generate_policy_with_llm(llm, policy_prompt, class_names, config)
        except Exception as e:
            print(f"[Policy Agent] ⚠️  LLM failed ({e}), using config fallback")
            policy = _fallback_policy(config)
    else:
        policy = _fallback_policy(config)
    
    # Save policy
    import os
    out_dir = resources.get("out_dir", "out")
    agent_dir = f"{out_dir}/agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/policy.json", "w") as f:
        json.dump(policy, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"[Policy Agent] ✅ Complete ({elapsed:.2f}s)")
    
    # Return policy in meta
    return {"meta": {**meta, "policy": policy}}


def _generate_policy_with_llm(llm, policy_prompt: str, class_names: list, config: Dict) -> Dict:
    """Generate policy using LLM with JSON schema."""
    
    prompt = f"""{policy_prompt}

Available defect classes: {', '.join(class_names)}

global minimum confidence threshold is {config.get("agents", {}).get("confidence_threshold", None)}.

Return JSON with keys: min_conf_global, per_class_thresholds, bbox_min_size"""
    
    # Use more tokens for reasoning models to allow for thinking + answer
    max_tokens = 2048 if 'reasoning' in str(llm).lower() or 'deepseek-r1' in str(llm).lower() else 1024
    result = llm.invoke(prompt, temperature=0.1, max_new_tokens=max_tokens)
    
    # Parse JSON from LLM output
    if isinstance(result, dict):
        return result
    
    # Extract JSON from code blocks if present
    if '```' in result:
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        else:
            result = result.split('```')[1].split('```')[0].strip()
    
    # Remove JavaScript-style comments (// and /* */)
    result = re.sub(r'//.*?(?=\n|$)', '', result)  # Remove single-line comments
    result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL)  # Remove multi-line comments
    
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return json.loads(result)


def _fallback_policy(config: Dict) -> Dict:
    """Load fallback policy from config/policy_fallback.json."""
    fallback_path = "config/policy_fallback.json"
    try:
        with open(fallback_path, "r") as f:
            policy = json.load(f)
        print(f"[Policy Agent] Loaded fallback policy from {fallback_path}")
        return policy
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[Policy Agent] ⚠️  Could not load {fallback_path} ({e}), using defaults")
        agents_cfg = config.get("agents", {})
        return {
            "min_conf_global": agents_cfg.get("confidence_threshold", 0.4),
            "per_class_thresholds": agents_cfg.get("per_class_thresholds", {}),
            "bbox_min_size": agents_cfg.get("bbox_min_size", {"width": 0, "height": 0})
        }
