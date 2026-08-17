import numpy as np
import unittest

from src.inference.handlers.openvino_detect import OpenVINODetectHandler
from src.inference.output_writer import _apply_detection_mapping


CLASS_NAMES = {index: f"class_{index}" for index in range(12)}


def test_decodes_transposed_yolo_output_and_applies_class_aware_nms():
    handler = OpenVINODetectHandler()
    output = np.zeros((1, 16, 3), dtype=np.float32)
    output[0, :4, 0] = [100, 100, 40, 40]
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = [102, 102, 40, 40]
    output[0, 4, 1] = 0.8
    output[0, :4, 2] = [100, 100, 40, 40]
    output[0, 5, 2] = 0.7

    detections = handler._postprocess(
        output=output,
        class_names=CLASS_NAMES,
        conf_threshold=0.25,
        iou_threshold=0.45,
        ratio=1.0,
        pad_x=0.0,
        pad_y=0.0,
        orig_w=640,
        orig_h=640,
        source="cell.jpg",
        modality_source="electroluminescence",
    )

    assert len(detections) == 2
    assert [item["detection"]["label_id"] for item in detections] == [0, 1]
    assert detections[0]["detection"]["bounding_box"] == {
        "x_min": 80.0,
        "y_min": 80.0,
        "x_max": 120.0,
        "y_max": 120.0,
    }


def test_detection_object_matches_config_mapping():
    detection = {
        "source": "cell.jpg",
        "modality_source": "electroluminescence",
        "detection": {
            "label": "crack",
            "confidence": 0.9,
            "bounding_box": {"x_min": 1.0, "y_min": 2.0, "x_max": 3.0, "y_max": 4.0},
        },
    }
    mapping = {
        "image_id": "__frame_index__",
        "source": "source",
        "label": "detection.label",
        "confidence": "detection.confidence",
        "bbox_xmin": "detection.bounding_box.x_min",
        "bbox_ymin": "detection.bounding_box.y_min",
        "bbox_xmax": "detection.bounding_box.x_max",
        "bbox_ymax": "detection.bounding_box.y_max",
        "modality_source": "modality_source",
    }

    row = _apply_detection_mapping(detection, 7, 640, 640, mapping)
    assert row == {
        "image_id": 7,
        "source": "cell.jpg",
        "label": "crack",
        "confidence": 0.9,
        "bbox_xmin": 1.0,
        "bbox_ymin": 2.0,
        "bbox_xmax": 3.0,
        "bbox_ymax": 4.0,
        "modality_source": "electroluminescence",
    }


def test_rejects_output_that_does_not_match_class_count():
    handler = OpenVINODetectHandler()
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "does not match"):
        handler._postprocess(
            output=np.zeros((1, 15, 10), dtype=np.float32),
            class_names=CLASS_NAMES,
            conf_threshold=0.25,
            iou_threshold=0.45,
            ratio=1.0,
            pad_x=0.0,
            pad_y=0.0,
            orig_w=640,
            orig_h=640,
            source="cell.jpg",
            modality_source="electroluminescence",
        )
