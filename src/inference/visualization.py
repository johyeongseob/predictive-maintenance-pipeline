"""Visualization helpers for inference outputs."""

from pathlib import Path


def generate_classification_viz(images_dir, fused_results, image_probs, sensor_probs,
                                class_names, out_dir):
    """
    Generate visualization for classification results.
    Overlays predicted labels from image/sensor/overall on each image in white text.

    Args:
        images_dir: Directory containing source images
        fused_results: List of dicts with 'source', 'label', 'confidence'
        image_probs: Dict mapping image_name -> list of class probabilities
        sensor_probs: Dict mapping image_name -> list of class probabilities
        class_names: Dict mapping class_id -> class_name
        out_dir: Output directory (viz/ will be created inside)
    """
    import cv2
    import numpy as np

    viz_dir = Path(out_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names) if class_names else 0

    for result in fused_results:
        img_name = result['source']
        img_path = Path(images_dir) / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Get per-modality predictions
        overall_label = result['label']
        overall_conf = result['confidence']

        # Image prediction
        img_p = image_probs.get(img_name)
        if img_p and n_classes > 0:
            best_img_idx = int(np.argmax(img_p))
            image_label = class_names.get(best_img_idx, str(best_img_idx))
        else:
            image_label = "N/A"

        # Sensor prediction
        sen_p = sensor_probs.get(img_name)
        if sen_p and n_classes > 0:
            best_sen_idx = int(np.argmax(sen_p))
            sensor_label = class_names.get(best_sen_idx, str(best_sen_idx))
        else:
            sensor_label = "N/A"

        # Text styling
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(h, w) / 800)
        thickness = max(1, int(font_scale * 2))
        color_white = (255, 255, 255)
        color_shadow = (0, 0, 0)
        line_height = int(35 * font_scale)
        margin = int(15 * font_scale)

        lines = [
            f"Image: {image_label}",
            f"Sensors: {sensor_label}",
            f"Overall: {overall_label} ({overall_conf:.2f})",
        ]

        # Draw text with black outline for readability
        y_pos = margin + line_height
        for line in lines:
            # Shadow/outline
            cv2.putText(img, line, (margin, y_pos), font, font_scale, color_shadow, thickness + 2, cv2.LINE_AA)
            # White text
            cv2.putText(img, line, (margin, y_pos), font, font_scale, color_white, thickness, cv2.LINE_AA)
            y_pos += line_height

        # Save
        out_name = Path(img_name).stem + '.jpg'
        cv2.imwrite(str(viz_dir / out_name), img)

    print(f"✓ Classification visualizations saved: {viz_dir}/ ({len(fused_results)} images)")
