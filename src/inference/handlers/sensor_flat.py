"""Flat 7-channel gas sensor inference handler."""

import csv
import json
from pathlib import Path

from src.inference import InferenceHandler, register_handler


SENSOR_CHANNELS = ("MQ2", "MQ3", "MQ5", "MQ6", "MQ7", "MQ8", "MQ135")


@register_handler("sensor_flat")
class SensorFlatHandler(InferenceHandler):
    """Run inference for flat MQ gas sensor vectors."""

    def __init__(self):
        self.config = {}
        self.compiled = None
        self.output_layer = None
        self.sensor_lookup = {}
        self.sensor_readings_lookup = {}
        self.sensor_mean = None
        self.sensor_std = None
        self.class_names = {}

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        task = config.get("task", inference_cfg.get("task", "detect"))
        modality = config.get("modality", "image")
        sensor_cfg = config.get("sensor", {})
        return (
            task == "classify"
            and modality in ("sensor", "multi")
            and bool(sensor_cfg.get("model_path") or config.get("sensor_model_path"))
            and bool(sensor_cfg.get("data_path") or config.get("sensor_data_path"))
        )

    def load(self, config: dict) -> None:
        import numpy as np
        from openvino.runtime import Core

        self.config = config
        sensor_cfg = config.get("sensor", {})
        sensor_data_path = config.get("sensor_data_path", sensor_cfg.get("data_path"))
        sensor_model_path = config.get("sensor_model_path", sensor_cfg.get("model_path"))
        device = config.get("device", "CPU")
        self.class_names = config.get("class_names", config.get("names", {}))

        self.sensor_lookup = {}
        self.sensor_readings_lookup = {}
        all_sensor_vals = []
        with open(sensor_data_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_key = row["Corresponding Image Name"]
                sensor_vals = [float(row[channel]) for channel in SENSOR_CHANNELS]
                self.sensor_lookup[img_key] = sensor_vals
                self.sensor_readings_lookup[img_key] = {
                    "sensor_raw_json": json.dumps({
                        channel: sensor_vals[idx]
                        for idx, channel in enumerate(SENSOR_CHANNELS)
                    })
                }
                all_sensor_vals.append(sensor_vals)

        # Compute z-score normalization parameters from full dataset.
        all_sensor_vals = np.array(all_sensor_vals, dtype=np.float32)
        self.sensor_mean = all_sensor_vals.mean(axis=0)
        self.sensor_std = all_sensor_vals.std(axis=0)
        self.sensor_std[self.sensor_std == 0] = 1.0

        core = Core()
        model = core.read_model(sensor_model_path)
        self.compiled = core.compile_model(model, device)
        self.output_layer = self.compiled.output(0)

    def infer(self, inputs: list, config: dict) -> list[dict]:
        import numpy as np

        merged_config = {**self.config, **config}
        modality = merged_config.get("modality", "sensor")
        image_names = merged_config.get("image_names", inputs)
        class_names = merged_config.get("class_names", self.class_names)
        n_classes = len(class_names)

        results = []
        print(f"Running sensor MLP inference ({len(image_names)} samples)...")
        for img_name in image_names:
            # Strip extension to get lookup key (e.g., "586_Perfume.png" -> "586_Perfume")
            key = Path(img_name).stem
            if key not in self.sensor_lookup:
                # No sensor data for this image - return uniform distribution.
                results.append({
                    "source": img_name,
                    "probabilities": [1.0 / n_classes] * n_classes,
                })
                continue

            sensor_vals = self.sensor_lookup[key]
            input_data = np.array([sensor_vals], dtype=np.float32)
            # Z-score normalization (model was trained on standardized features)
            input_data = (input_data - self.sensor_mean) / self.sensor_std
            output = self.compiled([input_data])[self.output_layer]

            # Apply softmax if needed (raw logits)
            probs = output[0]
            if probs.min() < 0 or probs.sum() < 0.99 or probs.sum() > 1.01:
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()

            result = {
                "source": img_name,
                "probabilities": probs.tolist(),
            }
            result.update(self.sensor_readings_lookup[key])
            results.append(result)
        
        print("✓ Sensor inference completed")

        self.last_sensor_probs = {
            result["source"]: result["probabilities"]
            for result in results
        }
        self.last_sensor_readings = {
            result["source"]: {"sensor_raw_json": result["sensor_raw_json"]}
            for result in results
            if "sensor_raw_json" in result
        }

        if modality == "multi":
            from .openvino_classify import run_image_classification

            inference_cfg = merged_config.get("inference", {})
            fusion_weights = merged_config.get("fusion_weights") or {"image": 0.5, "sensor": 0.5}

            self.last_image_probs = run_image_classification(
                images_dir=merged_config.get("images_dir", inference_cfg.get("images_path")),
                model_xml=merged_config.get("model_path", inference_cfg.get("model_path")),
                class_names=class_names,
                device=merged_config.get("device", inference_cfg.get("device", "GPU")),
                num_images=merged_config.get("num_images"),
                img_size=merged_config.get("img_size", inference_cfg.get("imgsz", 640)),
            )

            print(
                f"Running late fusion (image_w={fusion_weights['image']:.2f}, "
                f"sensor_w={fusion_weights['sensor']:.2f})..."
            )
            fused_results = self.fuse(
                image_probs=self.last_image_probs,
                sensor_probs=self.last_sensor_probs,
                image_names=image_names,
                class_names=class_names,
                image_weight=fusion_weights.get("image", 0.5),
                sensor_weight=fusion_weights.get("sensor", 0.5),
                sensor_readings=self.last_sensor_readings,
            )
            print(f"✓ Late fusion completed: {len(fused_results)} classifications")
            return fused_results

        return results

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})

    def fuse(
        self,
        image_probs,
        sensor_probs,
        image_names,
        class_names,
        image_weight=0.5,
        sensor_weight=0.5,
        sensor_readings=None,
    ):
        """Late fusion: weighted average of image and sensor class probabilities."""
        import numpy as np

        results = []
        n_classes = len(class_names)
        sensor_readings = sensor_readings or {}

        for img_name in image_names:
            img_p = np.array(image_probs.get(img_name, [1.0/n_classes]*n_classes))
            sen_p = np.array(sensor_probs.get(img_name, [1.0/n_classes]*n_classes))

            fused = image_weight * img_p + sensor_weight * sen_p
            # Normalize
            fused = fused / fused.sum()

            best_idx = int(np.argmax(fused))
            result = {
                "source": img_name,
                "label": class_names.get(best_idx, str(best_idx)),
                "confidence": float(fused[best_idx]),
                "image_confidence": float(img_p[best_idx]),
                "sensor_confidence": float(sen_p[best_idx]),
                "label_id": best_idx,
                "probabilities": {class_names.get(i, str(i)): float(fused[i]) for i in range(n_classes)},
            }
            result.update(sensor_readings.get(img_name, {}))
            results.append(result)

        return results


def run_sensor_inference(sensor_data_path, sensor_model_path, image_names, class_names, device="CPU"):
    """Compatibility wrapper for the existing run_inference_oep.py call path."""
    handler = SensorFlatHandler()
    config = {
        "sensor_data_path": sensor_data_path,
        "sensor_model_path": sensor_model_path,
        "image_names": image_names,
        "class_names": class_names,
        "device": device,
        "task": "classify",
        "modality": "sensor",
    }
    handler.load(config)
    results = handler.infer(image_names, config)
    sensor_probs = {result["source"]: result["probabilities"] for result in results}
    sensor_readings = {
        result["source"]: {"sensor_raw_json": result["sensor_raw_json"]}
        for result in results
        if "sensor_raw_json" in result
    }
    return sensor_probs, sensor_readings


def run_late_fusion(image_probs, sensor_probs, image_names, class_names,
                    image_weight=0.5, sensor_weight=0.5, sensor_readings=None):
    """Compatibility wrapper for the existing run_inference_oep.py call path."""
    return SensorFlatHandler().fuse(
        image_probs=image_probs,
        sensor_probs=sensor_probs,
        image_names=image_names,
        class_names=class_names,
        image_weight=image_weight,
        sensor_weight=sensor_weight,
        sensor_readings=sensor_readings,
    )
