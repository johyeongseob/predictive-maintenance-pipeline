"""
Corrosion Agent - Generate oil/gas pipeline corrosion risk summary.

This specialist agent focuses on regression-style O&G pipeline outputs.
"""

from typing import Any, Dict
from collections import defaultdict
import json
import os
import time

from src.agents.utility.policy_filter import filter_detections


def corrosion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate corrosion-focused summary for O&G pipeline detections."""
    start_time = time.time()
    print(f"[Corrosion Agent] Started at {time.strftime('%H:%M:%S')}")

    meta = state.get("meta", {})
    for_corrosion = meta.get("for_corrosion", {})

    detections = for_corrosion.get("detections", [])
    policy = for_corrosion.get("policy", {})
    resources = for_corrosion.get("resources", {})

    llms = resources.get("llms", {})
    llm = llms.get("corrosion")
    prompts = resources.get("prompts", {})
    corrosion_prompt = prompts.get("corrosion_prompt", "")

    filtered = filter_detections(detections, policy)
    report = _generate_corrosion_report(filtered, policy)

    if llm and corrosion_prompt:
        try:
            summary = _generate_summary_with_llm(llm, corrosion_prompt, report)
        except Exception as exc:
            print(f"[Corrosion Agent] LLM failed ({exc}), using fallback")
            summary = _generate_fallback_summary(report)
    else:
        summary = _generate_fallback_summary(report)

    corrosion = {
        "report": report,
        "summary_text": summary,
    }

    out_dir = resources.get("out_dir", "out")
    agent_dir = f"{out_dir}/agent"
    os.makedirs(agent_dir, exist_ok=True)
    with open(f"{agent_dir}/corrosion_report.json", "w", encoding="utf-8") as f:
        json.dump(corrosion, f, indent=2)
    with open(f"{agent_dir}/corrosion_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    elapsed = time.time() - start_time
    print(
        f"[Corrosion Agent] Complete: {len(filtered)} high-degradation samples "
        f"({elapsed:.2f}s)"
    )

    return {"meta": {**meta, "corrosion": corrosion}}


def _safe_float(value):
    """Convert numeric-like values to float, returning None when unavailable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(value, digits=3):
    """Round numeric values while keeping unavailable values as None."""
    value = _safe_float(value)
    if value is None:
        return None
    return round(value, digits)


def _percentile(values, percentile):
    """Return a simple percentile value for a non-empty numeric list."""
    nums = sorted(v for v in values if v is not None)
    if not nums:
        return None
    index = round((len(nums) - 1) * percentile)
    return nums[index]


def _generate_corrosion_report(detections, policy):
    """Build corrosion-specific risk priorities from filtered O&G rows."""
    threshold = _safe_float(policy.get("degradation_threshold"))
    pressure_values = [
        _safe_float(row.get("max_pressure"))
        for row in detections
    ]
    pressure_values = [value for value in pressure_values if value is not None]
    pressure_threshold = (
        _safe_float(policy.get("pressure_alert_threshold"))
        or _percentile(pressure_values, 0.75)
    )

    material_risk = defaultdict(lambda: {
        "high_pressure_high_loss_count": 0,
        "critical_high_pressure_count": 0,
        "max_predicted_loss": 0.0,
        "max_pressure": 0.0,
        "priority_sources": [],
    })
    priority_rows = []

    for row in detections:
        material = row.get("material_type") or "unknown"
        label = row.get("label") or "unknown"
        source = row.get("source") or "unknown"
        loss = _safe_float(row.get("continuous_value"))
        pressure = _safe_float(row.get("max_pressure"))

        risk = material_risk[material]
        if loss is not None:
            risk["max_predicted_loss"] = max(
                risk["max_predicted_loss"],
                _round_float(loss),
            )
        if pressure is not None:
            risk["max_pressure"] = max(
                risk["max_pressure"],
                _round_float(pressure, digits=1),
            )

        high_pressure = (
            pressure is not None
            and pressure_threshold is not None
            and pressure >= pressure_threshold
        )
        if high_pressure:
            risk["high_pressure_high_loss_count"] += 1
        if label == "Critical" and high_pressure:
            risk["critical_high_pressure_count"] += 1

        if label == "Critical" and high_pressure and loss is not None:
            priority_rows.append({
                "source": source,
                "material_type": material,
                "label": label,
                "continuous_value": _round_float(loss),
                "max_pressure": _round_float(pressure, digits=1),
            })
            if len(risk["priority_sources"]) < 5:
                risk["priority_sources"].append(source)

    top_material_groups = sorted(
        material_risk.items(),
        key=lambda item: (
            item[1]["critical_high_pressure_count"],
            item[1]["high_pressure_high_loss_count"],
            item[1]["max_predicted_loss"],
        ),
        reverse=True,
    )[:3]

    sorted_priority_rows = sorted(
        priority_rows,
        key=lambda row: (row["continuous_value"], row["max_pressure"] or 0),
        reverse=True,
    )
    priority_rows = []
    seen_sources = set()
    for row in sorted_priority_rows:
        source = row["source"]
        if source in seen_sources:
            continue
        priority_rows.append(row)
        seen_sources.add(source)
        if len(priority_rows) >= 5:
            break

    return {
        "policy_type": policy.get("policy_type"),
        "degradation_threshold": _round_float(threshold),
        "corrosion_alert_level": policy.get("corrosion_alert_level"),
        "pressure_alert_threshold": _round_float(pressure_threshold, digits=1),
        "material_pressure_risk": dict(material_risk),
        "top_risk_material_groups": dict(top_material_groups),
        "maintenance_priority_samples": priority_rows,
        "corrosion_recommendation": (
            "Prioritize Critical rows where predicted thickness loss exceeds "
            "the degradation threshold and max_pressure is in the high-pressure "
            "range for this run."
        ),
    }


def _generate_summary_with_llm(llm, prompt: str, report: Dict) -> str:
    """Generate concise corrosion summary using LLM."""
    enhanced_prompt = (
        f"{prompt}\n\n"
        "Keep the corrosion summary concise. Use at most 3 bullets. "
        "Do not repeat the general analysis distribution. "
        "Focus on corrosion risk, material-pressure combinations, "
        "and maintenance priority. Clearly state the priority criterion "
        "(Critical condition + high pressure + predicted thickness loss above "
        "the degradation threshold). Round numeric values to 3 decimal places "
        "or fewer. If listing example rows, include only the first 5 rows from "
        "maintenance_priority_samples. Do not list more than 5 rows total.\n\n"
        f"Corrosion Report:\n{json.dumps(report, indent=2)}\n\n"
        "Generate corrosion specialist summary:"
    )
    result = llm.invoke(enhanced_prompt, temperature=0.3, max_new_tokens=300)
    return result.strip()


def _generate_fallback_summary(report: Dict) -> str:
    """Generate fallback corrosion summary without LLM."""
    lines = ["O&G Corrosion Risk Summary", ""]
    lines.append(
        f"- Degradation threshold: {report.get('degradation_threshold', 'N/A')}"
    )
    lines.append(
        f"- Corrosion alert level: {report.get('corrosion_alert_level', 'N/A')}"
    )
    lines.append(
        f"- Pressure alert threshold: {report.get('pressure_alert_threshold', 'N/A')}"
    )

    top_groups = report.get("top_risk_material_groups", {})
    if top_groups:
        lines.append("- Priority material-pressure groups:")
        for material, stats in top_groups.items():
            lines.append(
                f"  - {material}: {stats.get('critical_high_pressure_count', 0)} "
                "Critical high-pressure samples, "
                f"max loss {stats.get('max_predicted_loss', 0):.3f} mm, "
                f"max pressure {stats.get('max_pressure', 0):.1f}"
            )

    priority_samples = report.get("maintenance_priority_samples", [])
    if priority_samples:
        first = priority_samples[0]
        lines.append(
            "- First inspection candidate: "
            f"{first.get('source')} ({first.get('material_type')}, "
            f"loss {first.get('continuous_value'):.3f} mm, "
            f"pressure {first.get('max_pressure')})"
        )

    lines.append(f"- Recommendation: {report.get('corrosion_recommendation')}")
    return "\n".join(lines)
