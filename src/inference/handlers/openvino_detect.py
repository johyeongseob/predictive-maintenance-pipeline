"""OpenVINO YOLO-style object detection handler."""

from pathlib import Path

import numpy as np

from src.inference import InferenceHandler, register_handler


@register_handler("openvino_detect")
class OpenVINODetectHandler(InferenceHandler):
    """Run object detection with a direct OpenVINO YOLO model."""

    def __init__(self):
        self.config = {}
        self.compiled = None
        self.input_layer = None
        self.output_layer = None
        self.class_names = {}
        self.input_channels = 3

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        handler_name = inference_cfg.get("handler") or config.get("handler")
        return handler_name == "openvino_detect"

    def load(self, config: dict) -> None:
        from openvino import Core

        self.config = config
        inference_cfg = config.get("inference", {})
        model_xml = config.get("model_path") or inference_cfg.get("model_path")
        if not model_xml:
            raise ValueError("openvino_detect requires inference.model_path")
        device = config.get("device") or inference_cfg.get("device", "GPU")
        self.class_names = self._normalise_class_names(
            config.get("class_names") or config.get("names", {})
        )

        core = Core()
        model = core.read_model(str(model_xml))

        precision_hint = inference_cfg.get("inference_precision_hint")
        compile_config = {}
        if precision_hint:
            compile_config["INFERENCE_PRECISION_HINT"] = precision_hint

        self.compiled = core.compile_model(
            model,
            device,
            compile_config,
        )
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)

        try:
            self.input_channels = int(self.input_layer.shape[1])
        except (TypeError, ValueError, RuntimeError):
            self.input_channels = 3

    def infer(self, inputs: list, config: dict) -> list[dict]:
        import cv2

        merged_config = {**self.config, **config}
        inference_cfg = merged_config.get("inference", {})
        images_dir = merged_config.get("images_dir", inference_cfg.get("images_path"))
        device = merged_config.get("device", inference_cfg.get("device", "GPU"))
        img_size = merged_config.get("img_size") or inference_cfg.get("imgsz", 640)
        if isinstance(img_size, (list, tuple)):
            if len(img_size) != 2 or img_size[0] != img_size[1]:
                raise ValueError("openvino_detect currently requires a square imgsz")
            img_size = img_size[0]
        img_size = int(img_size)
        conf_threshold = float(
            merged_config.get(
                "conf_threshold",
                inference_cfg.get("confidence_threshold", 0.25),
            )
        )
        iou_threshold = float(inference_cfg.get("iou_threshold", 0.45))
        num_images = merged_config.get("num_images")
        modality_source = (
            merged_config.get("modality_source")
            or inference_cfg.get("modality_source")
        )
        class_names = self._normalise_class_names(
            merged_config.get("class_names") or self.class_names
        )

        image_files = self._collect_image_files(images_dir, inputs, num_images)
        print(
            f"Running OpenVINO detection on {len(image_files)} images "
            f"(device: {device})..."
        )

        frames = []
        for img_file in image_files:
            image = self._read_image(cv2, img_file)
            if image is None:
                frames.append({"source": img_file.name, "objects": []})
                continue

            orig_h, orig_w = image.shape[:2]
            blob, ratio, pad_x, pad_y = self._preprocess(cv2, image, img_size)
            output = self.compiled([blob])[self.output_layer]
            detections = self._postprocess(
                output=output,
                class_names=class_names,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                ratio=ratio,
                pad_x=pad_x,
                pad_y=pad_y,
                orig_w=orig_w,
                orig_h=orig_h,
                source=img_file.name,
                modality_source=modality_source,
            )
            frames.append({"source": img_file.name, "objects": detections})

        total = sum(len(frame["objects"]) for frame in frames)
        print(
            f"OpenVINO detection completed "
            f"({len(frames)} images, {total} detections)"
        )
        return frames

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})

    @staticmethod
    def _normalise_class_names(class_names):
        """Return metadata/YAML class names as an integer-keyed mapping."""
        if isinstance(class_names, (list, tuple)):
            return dict(enumerate(class_names))
        return {int(key): value for key, value in (class_names or {}).items()}

    @staticmethod
    def _collect_image_files(images_dir, inputs, num_images):
        images_path = Path(images_dir).resolve()

        if inputs:
            image_files = [
                Path(item) if Path(item).is_absolute() else images_path / item
                for item in inputs
            ]
        else:
            patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            image_files = []
            for pattern in patterns:
                image_files.extend(images_path.glob(pattern))
            image_files = sorted(image_files)

        if not image_files:
            raise RuntimeError(f"No images found in {images_dir}")
        if num_images:
            image_files = image_files[: int(num_images)]
        return image_files

    def _read_image(self, cv2, img_file):
        if self.input_channels == 1:
            return cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        return cv2.imread(str(img_file), cv2.IMREAD_COLOR)


    def _preprocess(self, cv2, image, img_size):
        if image.ndim == 2:
            orig_h, orig_w = image.shape
        else:
            orig_h, orig_w = image.shape[:2]

        ratio = min(img_size / orig_w, img_size / orig_h)
        resized_w = int(round(orig_w * ratio))
        resized_h = int(round(orig_h * ratio))
        resized = cv2.resize(image, (resized_w, resized_h))

        pad_x = (img_size - resized_w) / 2
        pad_y = (img_size - resized_h) / 2
        left = int(round(pad_x - 0.1))
        right = int(round(img_size - resized_w - left))
        top = int(round(pad_y - 0.1))
        bottom = int(round(img_size - resized_h - top))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=114,
        )

        if self.input_channels == 1:
            blob = padded.astype(np.float32) / 255.0
            blob = blob[None, None, :, :]
        else:
            blob = padded.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        return blob, ratio, pad_x, pad_y

    def _postprocess(
        self,
        *,
        output,
        class_names,
        conf_threshold,
        iou_threshold,
        ratio,
        pad_x,
        pad_y,
        orig_w,
        orig_h,
        source,
        modality_source,
    ):
        """Decode YOLO output, filter boxes, apply NMS, and restore original coordinates."""
        predictions = np.squeeze(output, axis=0)
        n_classes = len(class_names)

        if not n_classes:
            raise ValueError("openvino_detect requires at least one class name")

        if predictions.ndim != 2:
            raise RuntimeError(f"Unsupported YOLO output shape: {output.shape}")
        if predictions.shape[0] == n_classes + 4:
            predictions = predictions.T
        elif predictions.shape[1] != n_classes + 4:
            raise RuntimeError(
                f"YOLO output shape {output.shape} does not match "
                f"4 box values + {n_classes} class scores"
            )

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:4 + n_classes]
        if class_scores.size == 0:
            return []

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = confidences >= conf_threshold
        if not np.any(keep):
            return []

        boxes_xywh = boxes_xywh[keep]
        confidences = confidences[keep]
        class_ids = class_ids[keep]
        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / ratio
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        selected = self._nms(boxes_xyxy, confidences, class_ids, iou_threshold)
        detections = []
        for idx in selected:
            label_id = int(class_ids[idx])
            x_min, y_min, x_max, y_max = boxes_xyxy[idx]
            detection = {
                "source": source,
                "modality_source": modality_source,
                "detection": {
                    "label": class_names.get(label_id, str(label_id)),
                    "confidence": float(confidences[idx]),
                    "label_id": label_id,
                    "bounding_box": {
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                    },
                },
            }
            detections.append(detection)

        return detections

    @staticmethod
    def _xywh_to_xyxy(boxes):
        converted = boxes.copy()
        converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return converted

    @staticmethod
    def _nms(boxes, scores, class_ids, iou_threshold):
        selected = []
        for class_id in sorted(set(class_ids.tolist())):
            class_indices = np.where(class_ids == class_id)[0]
            order = class_indices[np.argsort(scores[class_indices])[::-1]]

            while len(order) > 0:
                current = order[0]
                selected.append(current)
                if len(order) == 1:
                    break

                ious = OpenVINODetectHandler._box_iou(
                    boxes[current],
                    boxes[order[1:]],
                )
                order = order[1:][ious <= iou_threshold]

        return selected

    @staticmethod
    def _box_iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        intersection = inter_w * inter_h

        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        boxes_area = (
            np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        )
        union = box_area + boxes_area - intersection
        return intersection / np.maximum(union, 1e-9)
