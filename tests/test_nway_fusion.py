"""Tests for N-way late fusion over multiple probability branches."""

import pytest

from src.inference.handlers.sensor_flat import SensorFlatHandler


CLASS_NAMES = {
    0: "Mixture",
    1: "NoGas",
    2: "Perfume",
    3: "Smoke",
}


def test_two_branch_fusion():
    handler = SensorFlatHandler()

    results = handler.fuse(
        branch_probs={
            "image": {"sample.png": [0.1, 0.8, 0.05, 0.05]},
            "sensor": {"sample.png": [0.2, 0.6, 0.1, 0.1]},
        },
        fusion_weights={"image": 0.6, "sensor": 0.4},
        image_names=["sample.png"],
        class_names=CLASS_NAMES,
        metadata_by_image={"sample.png": {"sensor_raw_json": '{"MQ2": 786.0}'}},
    )

    result = results[0]
    assert result["label"] == "NoGas"
    assert result["label_id"] == 1
    assert result["confidence"] == pytest.approx(0.72)
    assert result["image_confidence"] == pytest.approx(0.8)
    assert result["sensor_confidence"] == pytest.approx(0.6)
    assert result["sensor_raw_json"] == '{"MQ2": 786.0}'


def test_three_branch_fusion():
    handler = SensorFlatHandler()

    results = handler.fuse(
        branch_probs={
            "image": {"sample.png": [0.2, 0.4, 0.3, 0.1]},
            "sensor": {"sample.png": [0.1, 0.3, 0.5, 0.1]},
            "imu": {"sample.png": [0.1, 0.2, 0.6, 0.1]},
        },
        fusion_weights={"image": 0.5, "sensor": 0.3, "imu": 0.2},
        image_names=["sample.png"],
        class_names=CLASS_NAMES,
    )

    result = results[0]
    assert result["label"] == "Perfume"
    assert result["label_id"] == 2
    assert result["confidence"] == pytest.approx(0.42)
    assert result["probabilities"]["NoGas"] == pytest.approx(0.33)


def test_five_branch_fusion():
    handler = SensorFlatHandler()

    results = handler.fuse(
        branch_probs={
            "image": {"sample.png": [0.1, 0.2, 0.1, 0.6]},
            "sensor": {"sample.png": [0.2, 0.1, 0.1, 0.6]},
            "lidar": {"sample.png": [0.1, 0.1, 0.2, 0.6]},
            "imu": {"sample.png": [0.1, 0.1, 0.1, 0.7]},
            "acoustic": {"sample.png": [0.05, 0.05, 0.1, 0.8]},
        },
        fusion_weights={
            "image": 0.2,
            "sensor": 0.2,
            "lidar": 0.2,
            "imu": 0.2,
            "acoustic": 0.2,
        },
        image_names=["sample.png"],
        class_names=CLASS_NAMES,
    )

    result = results[0]
    assert result["label"] == "Smoke"
    assert result["label_id"] == 3
    assert result["confidence"] == pytest.approx(0.66)
    assert result["probabilities"]["Mixture"] == pytest.approx(0.11)
