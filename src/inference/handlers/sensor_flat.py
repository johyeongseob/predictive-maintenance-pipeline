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
        self.output_layers = {}
        self.sensor_mode = "mq_gas"
        self.sensor_model_path = None
        self.sensor_device = None
        self.sensor_lookup = {}
        self.sensor_readings_lookup = {}
        self.sensor_mean = None
        self.sensor_std = None
        self.class_names = {}
        self.og_samples = []
        self.og_sample_lookup = {}
        self.last_image_probs = {}
        self.last_sensor_probs = {}
        self.last_sensor_readings = {}

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
        inference_cfg = config.get("inference", {})
        sensor_cfg = config.get("sensor", {})
        sensor_data_path = config.get("sensor_data_path", sensor_cfg.get("data_path"))
        sensor_model_path = config.get("sensor_model_path", sensor_cfg.get("model_path"))
        device = config.get("sensor_device", "CPU")
        self.sensor_model_path = sensor_model_path
        self.sensor_device = device
        self.class_names = config.get("class_names", config.get("names", {}))
        self.sensor_mode = "oil_gas" if sensor_cfg.get("scaler_path") else "mq_gas"

        if self.sensor_mode == "oil_gas":
            self._load_oil_gas_samples(sensor_data_path, sensor_cfg)
        else:
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
        self.output_layers = {
            output.get_any_name(): output
            for output in self.compiled.outputs
        }


    def infer(self, inputs: list, config: dict) -> list[dict]:
        import numpy as np

        merged_config = {**self.config, **config}
        modality = merged_config.get("modality", "sensor")
        image_names = merged_config.get("image_names", inputs)
        class_names = merged_config.get("class_names", self.class_names)
        n_classes = len(class_names)

        if self.sensor_mode == "oil_gas":
            return self._infer_oil_gas(image_names, class_names)

        fusion_weights = merged_config.get("fusion_weights") or {"image": 0.5, "sensor": 0.5}

        if modality == "multi":
            from .openvino_classify import run_image_classification

            inference_cfg = merged_config.get("inference", {})
            self.last_image_probs = run_image_classification(
                images_dir=merged_config.get("images_dir", inference_cfg.get("images_path")),
                model_xml=merged_config.get("model_path", inference_cfg.get("model_path")),
                class_names=class_names,
                device=merged_config.get("device", inference_cfg.get("device", "GPU")),
                num_images=merged_config.get("num_images"),
                img_size=merged_config.get("img_size", inference_cfg.get("imgsz", 640)),
            )
 
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

            print(
                f"Running late fusion (image_w={fusion_weights['image']:.2f}, "
                f"sensor_w={fusion_weights['sensor']:.2f})..."
            )
            fused_results = self.fuse(
                branch_probs={
                    "image": self.last_image_probs,
                    "sensor": self.last_sensor_probs,
                },
                fusion_weights=fusion_weights,
                image_names=image_names,
                class_names=class_names,
                metadata_by_image=self.last_sensor_readings,
            )
            print(f"✓ Late fusion completed: {len(fused_results)} classifications")
            return fused_results

        return results
    
    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})

    def _load_oil_gas_samples(self, sensor_data_path, sensor_cfg):
        """Load O&G tabular sensor rows and build 17-dim model features."""
        import numpy as np

        scaler_path = sensor_cfg.get("scaler_path")
        if not scaler_path:
            raise ValueError("Oil/gas sensor inference requires sensor.scaler_path")

        with open(scaler_path) as f:
            scaler = json.load(f)

        numeric_features = scaler["input_features"]
        material_values = scaler["material_values"]
        grade_values = scaler["grade_values"]
        numeric_mean = np.array(scaler["numeric_mean"], dtype=np.float32)
        numeric_std = np.array(scaler["numeric_std"], dtype=np.float32)
        numeric_std[numeric_std == 0] = 1.0

        target_value_column = sensor_cfg.get("target_value_column", "Thickness_Loss_mm")
        target_label_column = sensor_cfg.get("target_label_column", "Condition")

        self.og_samples = []
        self.og_sample_lookup = {}
        if not self.class_names and scaler.get("condition_classes"):
            self.class_names = {
                idx: name
                for idx, name in enumerate(scaler["condition_classes"])
            }

        with open(sensor_data_path) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                numeric = np.array(
                    [float(row[name]) for name in numeric_features],
                    dtype=np.float32,
                )
                numeric = (numeric - numeric_mean) / numeric_std
                material_onehot = [
                    1.0 if row.get("Material") == value else 0.0
                    for value in material_values
                ]
                grade_onehot = [
                    1.0 if row.get("Grade") == value else 0.0
                    for value in grade_values
                ]
                features = np.array(
                    numeric.tolist() + material_onehot + grade_onehot,
                    dtype=np.float32,
                )

                label_columns = {
                    target_value_column,
                    target_label_column,
                }
                sensor_raw = {
                    key: value
                    for key, value in row.items()
                    if key not in label_columns
                }

                source = f"row_{idx}"
                metadata = {
                    "sensor_raw_json": json.dumps(sensor_raw),
                    "material_type": row.get("Material"),
                    "max_pressure": float(row.get("Max_Pressure_psi")) if row.get("Max_Pressure_psi") else None,
                    "time_years": float(row.get("Time_Years")) if row.get("Time_Years") else None,
                }
                if target_label_column in row:
                    metadata["condition_true"] = row[target_label_column]

                sample = {
                    "source": source,
                    "features": features,
                    "metadata": metadata,
                }
                self.og_samples.append(sample)
                self.og_sample_lookup[source] = sample

        print(f"Loaded oil/gas sensor samples: {len(self.og_samples)}")

    def _infer_oil_gas(self, image_names, class_names):
        """Run O&G regression + condition classification inference."""
        import numpy as np

        if not class_names:
            class_names = self.class_names
        n_classes = len(class_names)
        
        # Use requested row ids only when they match O&G sources; otherwise run all CSV rows.
        if image_names and any(name in self.og_sample_lookup for name in image_names):
            requested = image_names
        else:
            requested = [sample["source"] for sample in self.og_samples]

        samples = [
            self.og_sample_lookup[name]
            for name in requested
            if name in self.og_sample_lookup
        ]

        print(f"Running oil/gas sensor MLP inference ({len(samples)} samples)...")
        results = []
        for sample in samples:
            output = self.compiled([sample["features"].reshape(1, -1)])
            thickness = self._get_named_output(output, "thickness_loss_pred")[0][0]
            logits = self._get_named_output(output, "condition_logits")[0]
            exp_probs = np.exp(logits - np.max(logits))
            probs = exp_probs / exp_probs.sum()

            best_idx = int(np.argmax(probs))
            label = class_names.get(best_idx, str(best_idx))
            metadata = sample["metadata"]
            result = {
                "source": sample["source"],
                "label": label,
                "confidence": float(probs[best_idx]),
                "label_id": best_idx,
                "probabilities": {
                    class_names.get(i, str(i)): float(probs[i])
                    for i in range(n_classes)
                },
                "degradation_score": float(probs[n_classes - 1]),
                "continuous_value": float(thickness),
            }
            result.update(metadata)
            results.append(result)

        print("✓ Oil/gas sensor inference completed")
        return results

    def _get_named_output(self, output, output_name):
        """Return an OpenVINO output tensor by friendly output name."""
        if output_name not in self.output_layers:
            available = ", ".join(sorted(self.output_layers)) or "none"
            raise ValueError(
                f"Model output '{output_name}' not found. "
                f"Available outputs: {available}"
            )
        return output[self.output_layers[output_name]]

    def fuse(
        self,
        branch_probs,
        fusion_weights,
        image_names,
        class_names,
        metadata_by_image=None,
    ):
        """Late fusion: weighted average over all configured probability branches."""
        import numpy as np

        results = []
        n_classes = len(class_names)
        metadata_by_image = metadata_by_image or {}
        uniform = [1.0 / n_classes] * n_classes

        for img_name in image_names:
            fused = np.zeros(n_classes)
            branch_arrays = {}

            for branch_name, weight in fusion_weights.items():
                probs_by_image = branch_probs.get(branch_name)
                if not probs_by_image:
                    continue
                branch_p = np.array(probs_by_image.get(img_name, uniform))
                branch_arrays[branch_name] = branch_p
                fused += float(weight) * branch_p

            if fused.sum() == 0:
                fused = np.array(uniform)
            # Normalize
            fused = fused / fused.sum()

            best_idx = int(np.argmax(fused))
            result = {
                "source": img_name,
                "label": class_names.get(best_idx, str(best_idx)),
                "confidence": float(fused[best_idx]),
                "label_id": best_idx,
                "probabilities": {class_names.get(i, str(i)): float(fused[i]) for i in range(n_classes)},
            }
            if "image" in branch_arrays:
                result["image_confidence"] = float(branch_arrays["image"][best_idx])
            if "sensor" in branch_arrays:
                result["sensor_confidence"] = float(branch_arrays["sensor"][best_idx])
            result.update(metadata_by_image.get(img_name, {}))
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
        branch_probs={
            "image": image_probs,
            "sensor": sensor_probs,
        },
        fusion_weights={
            "image": image_weight,
            "sensor": sensor_weight,
        },
        image_names=image_names,
        class_names=class_names,
        metadata_by_image=sensor_readings,
    )
