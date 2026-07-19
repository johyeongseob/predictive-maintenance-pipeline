"""OpenVINO image classification handler."""

from pathlib import Path

from src.inference import InferenceHandler, register_handler


@register_handler("openvino_classify")
class OpenVINOClassifyHandler(InferenceHandler):
    """Run image classification with direct OpenVINO inference."""

    def __init__(self):
        self.config = {}
        self.compiled = None
        self.input_layer = None
        self.output_layer = None
        self.class_names = {}

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        task = config.get("task", inference_cfg.get("task", "detect"))
        modality = config.get("modality", "image")
        return task == "classify" and modality in ("image", "multi")

    def load(self, config: dict) -> None:
        from openvino.runtime import Core

        self.config = config
        inference_cfg = config.get("inference", {})
        model_xml = config.get("model_xml", config.get("model_path", inference_cfg.get("model_path")))
        device = config.get("device", inference_cfg.get("device", "GPU"))
        self.class_names = config.get("class_names", config.get("names", {}))

        core = Core()
        model = core.read_model(str(model_xml))
        self.compiled = core.compile_model(model, device)
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)

    def infer(self, inputs: list, config: dict) -> list[dict]:
        import cv2
        import numpy as np

        merged_config = {**self.config, **config}
        inference_cfg = merged_config.get("inference", {})
        images_dir = merged_config.get("images_dir", inference_cfg.get("images_path"))
        device = merged_config.get("device", inference_cfg.get("device", "GPU"))
        img_size = merged_config.get("img_size", inference_cfg.get("imgsz", 640))
        num_images = merged_config.get("num_images")
        class_names = merged_config.get("class_names", self.class_names)

        images_path = Path(images_dir).resolve()
        image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        if not image_files:
            raise RuntimeError(f"No images found in {images_dir}")
        if num_images:
            image_files = image_files[:num_images]

        n_classes = len(class_names)
        results = []

        print(f"Running image classification on {len(image_files)} images (device: {device})...")
        for img_file in image_files:
            # Preprocess: resize, normalize, transpose to NCHW
            img = cv2.imread(str(img_file))
            if img is None:
                results.append({"source": img_file.name, "probabilities": [1.0 / n_classes] * n_classes})
                continue

            img_resized = cv2.resize(img, (img_size, img_size))
            img_float = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
            img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dim

            output = self.compiled([img_batch])[self.output_layer]
            probs = output[0]

            # Apply softmax if raw logits
            if probs.min() < 0 or probs.sum() < 0.99 or probs.sum() > 1.01:
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()

            results.append({"source": img_file.name, "probabilities": probs.tolist()})

        print(f"✓ Image classification completed ({len(results)} images)")
        return results

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})


def run_image_classification(images_dir, model_xml, class_names, device='GPU',
                             num_images=None, img_size=640):
    """Compatibility wrapper for the existing run_inference_oep.py call path."""
    handler = OpenVINOClassifyHandler()
    config = {
        "images_dir": images_dir,
        "model_xml": model_xml,
        "class_names": class_names,
        "device": device,
        "num_images": num_images,
        "img_size": img_size,
        "task": "classify",
        "modality": "image",
    }
    handler.load(config)
    results = handler.infer([], config)
    return {result["source"]: result["probabilities"] for result in results}
