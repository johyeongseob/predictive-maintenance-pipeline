"""
Analysis Agent - Generate statistics and insights.

Uses policy to filter detections and generates analysis report.
"""

from typing import Dict, Any
from collections import Counter
import json
import os
import statistics
import time
from src.agents.utility.policy_filter import filter_detections


def analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate statistics and insights on detections."""
    start_time = time.time()
    print(f"[Analysis Agent] ⏱️ Started at {time.strftime('%H:%M:%S')}")
    
    meta = state.get("meta", {})
    for_analysis = meta.get("for_analysis", {})
    
    detections = for_analysis.get("detections", [])
    policy = for_analysis.get("policy", {})
    resources = for_analysis.get("resources", {})
    
    # Get analysis-specific LLM
    llms = resources.get("llms", {})
    llm = llms.get("analysis")
    prompts = resources.get("prompts", {})
    analysis_prompt = prompts.get("analysis_prompt", "")
    
    # Filter using utility
    filtered = filter_detections(detections, policy)
    
    if not filtered:
        print("[Analysis Agent] ⚠️  No detections after filtering")
        return {"meta": meta}
    
    # Generate statistics
    stats = _generate_stats(filtered)
    
    # Generate text summary with LLM if available
    summary_text = ""
    if llm and analysis_prompt:
        try:
            summary_text = _generate_summary_with_llm(llm, analysis_prompt, stats)
        except Exception as e:
            print(f"[Analysis Agent] ⚠️  LLM failed ({e}), using fallback")
            summary_text = _generate_fallback_summary(stats)
    else:
        summary_text = _generate_fallback_summary(stats)
    
    # Create output
    analysis = {
        "total_defects": len(filtered),
        "stats": stats,
        "summary_text": summary_text
    }
    
    # Save outputs
    out_dir = resources.get("out_dir", "out")
    agent_dir = f"{out_dir}/agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2)
    with open(f"{agent_dir}/analysis_summary.txt", "w") as f:
        f.write(summary_text)
    
    elapsed = time.time() - start_time
    print(f"[Analysis Agent] ✅ Complete: {analysis['total_defects']} defects ({elapsed:.2f}s)")
    
    return {"meta": {**meta, "analysis": analysis}}


def _generate_stats(detections):
    """Generate statistics from filtered detections."""
    label_counts = Counter([d["label"] for d in detections])
    
    label_confidences = {}
    for label in label_counts:
        confs = [d["confidence"] for d in detections if d["label"] == label]
        label_confidences[label] = {
            "count": len(confs),
            "mean": round(statistics.mean(confs), 3),
            "median": round(statistics.median(confs), 3),
            "min": round(min(confs), 3),
            "max": round(max(confs), 3)
        }
    
    return {
        "labels_found": list(label_counts.keys()),
        "counts_per_label": dict(label_counts),
        "confidence_stats": label_confidences
    }


def _generate_summary_with_llm(llm, prompt: str, stats: Dict) -> str:
    """Generate summary using LLM."""
    enhanced_prompt = f"{prompt}\n\nStatistics:\n{json.dumps(stats, indent=2)}\n\nGenerate analysis report:"
    result = llm.invoke(enhanced_prompt, temperature=0.5, max_new_tokens=400)
    return result.strip()


def _generate_fallback_summary(stats: Dict) -> str:
    """Generate fallback summary without LLM."""
    counts = stats.get("counts_per_label", {})
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    
    lines = ["="*60, "DETECTION ANALYSIS REPORT", "="*60, ""]
    
    total = sum(counts.values())
    lines.append(f"Total Detections: {total}\n")
    
    lines.append("Top Detection Types:")
    for label, count in ranked:
        pct = (count / total) * 100 if total > 0 else 0
        lines.append(f"  • {label}: {count} ({pct:.1f}%)")
    
    lines.append("\nConfidence Statistics:")
    conf_stats = stats.get("confidence_stats", {})
    for label, stat in conf_stats.items():
        lines.append(f"\n  {label}:")
        lines.append(f"    Count: {stat['count']}")
        lines.append(f"    Mean: {stat['mean']:.3f}")
        lines.append(f"    Range: {stat['min']:.3f} - {stat['max']:.3f}")
    
    lines.append("\n" + "="*60)
    return "\n".join(lines)
