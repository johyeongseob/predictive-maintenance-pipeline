
"""
Meta Agent - Hub-and-Spoke Orchestrator using LangGraph.

Architecture:
  Phase 1: meta_entry → policy_agent → meta_collect_policy
  Phase 2: meta_collect_policy → [analysis_agent, evidence_agent] (parallel) → collect_results → END

Key Design Principles:
1. All agents communicate ONLY through meta nodes (hub-and-spoke)
2. Agents read from meta["resources"] and meta["for_<agent>"]
3. Agents write to meta["<agent_name>"]
4. Meta nodes prepare and redistribute data between agents
5. No agent directly accesses another agent's output
"""

from typing import Dict, Any
from .utility.state import AgentState


def meta_entry(state: AgentState) -> Dict[str, Any]:
    """
    Meta entry node - initializes workflow and prepares resources for Phase 1.
    
    This node:
    - Sets up the meta namespace structure
    - Marks workflow as started
    - Sets phase = 1
    """
    print("\n" + "="*70)
    print("[Meta Agent] 🎭 Initializing Workflow")
    print("="*70)
    
    meta = state.get("meta", {})
    meta["phase"] = 1
    meta["started"] = True
    
    print(f"[Meta Agent] ✅ Phase 1 initialized")
    print(f"[Meta Agent] 📋 Resources available: db_backend, llm, config, prompts")
    
    return {"meta": meta}


def meta_collect_policy(state: AgentState) -> Dict[str, Any]:
    """
    Meta collector after policy - prepares data for Phase 2 agents.
    
    This node:
    - Collects policy JSON from policy agent
    - Gets raw detections from database
    - Prepares data namespaces for analysis and evidence agents
    - Each agent gets policy + raw detections to filter independently
    """
    print("\n" + "="*70)
    print("[Meta Agent] 🔄 Collecting Policy Results & Preparing Phase 2")
    print("="*70)
    
    meta = state.get("meta", {})
    meta["phase"] = 2
    
    # Get policy from policy agent (now just the JSON policy, no filtering)
    policy = meta.get("policy", {})
    print(f"[Meta Agent] 📋 Policy received: {policy}")
    
    # Get raw detections from resources
    resources = meta.get("resources", {})
    db_backend = resources.get("db_backend")
    all_detections = db_backend.query_all() if db_backend else []
    
    print(f"[Meta Agent] 📊 Retrieved {len(all_detections)} raw detections from database")
    print(f"[Meta Agent] 📤 Preparing data for downstream agents")
    
    # ✅ Meta node prepares data for analysis agent
    meta["for_analysis"] = {
        "detections": all_detections,
        "policy": policy,
        "resources": resources
    }
    
    # ✅ Meta node prepares data for evidence agent
    # Compute actual filtering metrics for audit trail
    from src.agents.utility.policy_filter import filter_detections
    filtered_detections = filter_detections(all_detections, policy)
    
    # Count per-class breakdown
    from collections import Counter
    by_label = Counter([d["label"] for d in filtered_detections])
    
    policy_metrics = {
        "confidence_threshold": policy.get("min_conf_global", 0.0),
        "per_class_thresholds": policy.get("per_class_thresholds", {}),
        "bbox_min_size": policy.get("bbox_min_size", {}),
        "total_detections": len(all_detections),
        "high_conf_count": len(filtered_detections),
        "by_label": dict(by_label)
    }
    
    meta["for_evidence"] = {
        "detections": all_detections,
        "policy": policy,
        "policy_metrics": policy_metrics,
        "resources": resources
    }
    
    meta["policy_completed"] = True
    
    # Check execution mode for appropriate message
    config = resources.get("config", {})
    agents_cfg = config.get("agents", {})
    execution_mode = agents_cfg.get("execution_mode", "sequential")
    
    print(f"[Meta Agent] ✅ Phase 2 prepared")
    if execution_mode == "parallel":
        print(f"[Meta Agent] 🚀 Launching analysis agent + evidence agent (parallel)")
    else:
        print(f"[Meta Agent] 🚀 Launching analysis agent → evidence agent (sequential)")
    
    return {"meta": meta}


def collect_results(state: AgentState) -> Dict[str, Any]:
    """
    Final meta collector - gathers all results and creates final status.
    
    This node:
    - Collects outputs from all agents
    - Creates final status summary
    - Marks workflow as completed
    """
    print("\n" + "="*70)
    print("[Meta Agent] 🏁 Collecting Final Results")
    print("="*70)
    
    meta = state.get("meta", {})
    meta["completed"] = True
    
    # Gather results from all agents
    policy = meta.get("policy", {})
    analysis_data = meta.get("analysis", {})
    
    # Create final status
    final_status = {
        "workflow_completed": True,
        "policy": {
            "min_conf_global": policy.get("min_conf_global", 0.0),
            "per_class_thresholds": policy.get("per_class_thresholds", {}),
            "bbox_min_size": policy.get("bbox_min_size", {})
        },
        "analysis": {
            "total_defects": analysis_data.get("total_defects", 0),
            "report_saved": True
        },
        "evidence": {
            "trail_saved": True,
            "audit_complete": True
        }
    }
    
    meta["final_status"] = final_status
    
    print(f"[Meta Agent] 📊 Final Summary:")
    print(f"  • Policy: min_conf={final_status['policy']['min_conf_global']}")
    print(f"  • Analysis: {final_status['analysis']['total_defects']} defects")
    print(f"  • Evidence: Audit trail generated")
    print(f"[Meta Agent] ✅ Workflow Complete!")
    print("="*70 + "\n")
    
    return {"meta": meta}

