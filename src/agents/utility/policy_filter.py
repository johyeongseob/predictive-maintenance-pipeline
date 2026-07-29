"""
Policy-based detection filtering utility.

Shared by Analysis and Evidence agents to apply policy rules consistently.
Schema-agnostic: only applies filters for fields that exist in each detection.
"""

from typing import List, Dict, Any


def filter_detections(detections: List[Dict[str, Any]], policy: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter detections based on policy rules.
    
    Works with any detection schema — only applies a rule if the relevant
    field exists in the detection dict. Detections are flat dicts straight
    from SQL (e.g. {'label': ..., 'confidence': ..., 'width': ..., ...}).
    
    Args:
        detections: List of flat detection dicts (columns depend on use-case schema)
        policy: Policy dict with optional keys:
            {
                "min_conf_global": float,
                "per_class_thresholds": {"label": float, ...},
                "bbox_min_size": {"width": float, "height": float},
                "min_sensor_confidence": float
            }
            
    Returns:
        Filtered list of detections that pass policy criteria
    """
    if not detections or not policy:
        return []
    
    min_conf_global = policy.get("min_conf_global", 0.0)
    per_class_thresholds = policy.get("per_class_thresholds", {})
    bbox_min_size = policy.get("bbox_min_size", {})
    min_sensor_confidence = policy.get("min_sensor_confidence")
    
    filtered = []
    
    for det in detections:
        # Confidence filter (only if confidence field exists)
        confidence = det.get("confidence")
        if confidence is not None:
            label = det.get("label", "")
            threshold = per_class_thresholds.get(label, min_conf_global)
            if confidence < threshold:
                continue
        
        # Sensor confidence filter (only if sensor_confidence field exists)
        sensor_confidence = det.get("sensor_confidence")
        if sensor_confidence is not None and min_sensor_confidence is not None:
            if sensor_confidence < min_sensor_confidence:
                continue

        # Bbox size filter (only if width/height fields exist AND policy specifies min size)
        if bbox_min_size:
            # Support both flat keys and nested bbox dict
            width = det.get("width") or (det.get("bbox", {}) or {}).get("width")
            height = det.get("height") or (det.get("bbox", {}) or {}).get("height")
            
            if width is not None and height is not None:
                if (width < bbox_min_size.get("width", 0) or
                        height < bbox_min_size.get("height", 0)):
                    continue
        
        filtered.append(det)
    
    return filtered
