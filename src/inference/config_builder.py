"""Build runtime config dictionaries for inference handlers."""


def build_handler_config(config, schema, image_names, output_file=None, out_subdir=""):
    """Build the config shape consumed by inference handlers."""
    schema = schema or {}
    inference_cfg = config.get("inference", {})
    sensor_cfg = config.get("sensor", {})

    handler_name = inference_cfg.get("handler")
    task = config.get("task", inference_cfg.get("task", "detect"))
    modality = config.get("modality", "image")
    class_names = (
        config.get("names")
        or config.get("class_names")
        or schema.get("class_names", {})
    )

    return {
        "handler": handler_name,
        "modality": modality,
        "task": task,
        "model_path": inference_cfg.get("model_path"),
        "model_proc_path": inference_cfg.get("model_proc_path"),
        "images_dir": inference_cfg.get("images_path"),
        "output_file": output_file or inference_cfg.get("output_file"),
        "device": inference_cfg.get("device", "GPU"),
        "conf_threshold": inference_cfg.get("confidence_threshold", 0.25),
        "num_images": inference_cfg.get("num_images"),
        "nireq": inference_cfg.get("nireq", 4),
        "video_path": inference_cfg.get("video_path"),
        "inference_interval": inference_cfg.get("inference_interval", 1),
        "out_subdir": out_subdir or inference_cfg.get("out_subdir", ""),
        "class_names": class_names,
        "sensor_model_path": sensor_cfg.get("model_path"),
        "sensor_data_path": sensor_cfg.get("data_path"),
        "sensor_device": "CPU",
        "fusion_weights": config.get("fusion_weights"),
        "schema": schema,
        "image_names": image_names,
        "inference": {
            **inference_cfg,
            "handler": handler_name,
            "task": task,
        },
        "sensor": sensor_cfg,
    }