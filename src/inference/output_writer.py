"""Output helpers for inference results."""


import json
from pathlib import Path


def _write_jsonl_records(output_path, records):
    """Write dictionaries to a JSONL file."""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def _resolve_dotpath(obj, dotpath):
    """Resolve a dot-notation path against a nested dict."""
    parts = dotpath.split('.')
    val = obj
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p, 0.0)
        else:
            return 0.0
    return val


def _apply_detection_mapping(det_obj, frame_idx, orig_w, orig_h, detection_mapping):
    """
    Apply detection_mapping from config to convert a raw DLStreamer detection object
    into a flat dict suitable for SQLite insertion.

    Supported path_spec prefixes:
        __frame_index__          -> use frame_idx
        scale_w:<path>           -> resolve path, multiply by orig_w, cast to int
        scale_h:<path>           -> resolve path, multiply by orig_h, cast to int
        scale_w_diff:<pathA>-<pathB> -> (resolve A - resolve B) * orig_w, cast to int
        scale_h_diff:<pathA>-<pathB> -> (resolve A - resolve B) * orig_h, cast to int
        <path>                   -> resolve path directly

    Args:
        det_obj: Single raw DLStreamer object dict (native format)
        frame_idx: Current frame index
        orig_w: Original image width (for scale_w fields)
        orig_h: Original image height (for scale_h fields)
        detection_mapping: Dict from config mapping column names to path specs

    Returns:
        Dict with keys matching schema column names
    """
    row = {}
    for col_name, path_spec in detection_mapping.items():
        if path_spec == '__frame_index__':
            row[col_name] = frame_idx
        elif path_spec.startswith('scale_w_diff:'):
            expr = path_spec[len('scale_w_diff:'):]
            path_a, path_b = expr.split('-', 1)
            val = _resolve_dotpath(det_obj, path_a) - _resolve_dotpath(det_obj, path_b)
            row[col_name] = int(val * orig_w)
        elif path_spec.startswith('scale_h_diff:'):
            expr = path_spec[len('scale_h_diff:'):]
            path_a, path_b = expr.split('-', 1)
            val = _resolve_dotpath(det_obj, path_a) - _resolve_dotpath(det_obj, path_b)
            row[col_name] = int(val * orig_h)
        elif path_spec.startswith('scale_w:'):
            dotpath = path_spec[len('scale_w:'):]
            row[col_name] = int(_resolve_dotpath(det_obj, dotpath) * orig_w)
        elif path_spec.startswith('scale_h:'):
            dotpath = path_spec[len('scale_h:'):]
            row[col_name] = int(_resolve_dotpath(det_obj, dotpath) * orig_h)
        else:
            row[col_name] = _resolve_dotpath(det_obj, path_spec)
    return row


def _build_classification_row(result, frame_idx, detection_mapping):
    """Build a SQLite row from a classification result using detection_mapping."""
    row = {}
    for col_name, path_spec in detection_mapping.items():
        if path_spec == '__frame_index__':
            row[col_name] = frame_idx
        else:
            row[col_name] = _resolve_dotpath(result, path_spec)
    return row


def write_inference_outputs(
    *,
    results,
    handler,
    handler_config,
    sqlite_client,
    sqlite_db_path,
    clear_on_run,
    clean_database_fn,
    detection_mapping,
    image_files,
    visualization_fn=None,
    detection_visualization_fn=None,
):
    """Write handler results to JSONL, SQLite, and optional visualization."""
    task = handler_config["task"]
    modality = handler_config["modality"]
    output_file = handler_config["output_file"]
    image_names = handler_config.get("image_names", [])
    class_names = handler_config.get("class_names", {})
    images_dir = handler_config.get("images_dir")
    video_path = handler_config.get("video_path")
    out_subdir = handler_config.get("out_subdir", "")
    use_video = bool(video_path)

    image_probs = getattr(handler, "last_image_probs", {})
    sensor_probs = getattr(handler, "last_sensor_probs", {})

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    backend_detections = []
    n_frames = 0
    classification_results = []
    has_images = bool(images_dir) and bool(image_files)

    if task == 'classify' and modality == 'image':
        # Image-only classification (no fusion) - convert probability records to results
        import numpy as np

        image_probs = {
            result['source']: result['probabilities']
            for result in results
        }

        image_results = []
        n_frames = len(image_names)

        for frame_idx, img_name in enumerate(image_names):
            probs = image_probs.get(img_name, [1.0 / len(class_names)] * len(class_names))
            best_idx = int(np.argmax(probs))
            result = {
                'source': img_name,
                'label': class_names.get(best_idx, str(best_idx)),
                'confidence': float(probs[best_idx]),
                'label_id': best_idx,
            }
            image_results.append(result)
            total_detections += 1

            if sqlite_client and detection_mapping:
                row = _build_classification_row(result, frame_idx, detection_mapping)
                backend_detections.append(row)

        _write_jsonl_records(output_path, image_results)
        classification_results = image_results

    elif task == 'classify':
        # Classification output: one result per image
        fused_results = results
        classification_results = fused_results
        n_frames = len(fused_results)
        _write_jsonl_records(output_path, fused_results)
        for frame_idx, result in enumerate(fused_results):
            total_detections += 1
            # Build row for SQLite using detection_mapping
            if sqlite_client and detection_mapping:
                if result['source'] in image_names:
                    frame_idx = image_names.index(result['source'])
                row = _build_classification_row(result, frame_idx, detection_mapping)
                backend_detections.append(row)

    else:
        import cv2

        frames = results
        n_frames = len(frames)

        if use_video:
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            video_h, video_w = (first_frame.shape[:2] if ret else (640, 640))
            cap.release()
        else:
            video_h, video_w = 640, 640

        _write_jsonl_records(output_path, frames)
        for frame_idx, frame in enumerate(frames):
            total_detections += len(frame['objects'])

            if sqlite_client and frame['objects']:
                if use_video:
                    orig_h, orig_w = video_h, video_w
                elif frame_idx < len(image_files):
                    img = cv2.imread(str(image_files[frame_idx]))
                    orig_h, orig_w = (img.shape[:2] if img is not None else (640, 640))
                else:
                    orig_h, orig_w = 640, 640

                for det in frame['objects']:
                    row = _apply_detection_mapping(
                        det, frame_idx, orig_w, orig_h, detection_mapping
                    )
                    backend_detections.append(row)

    if sqlite_client and backend_detections:
        try:
            if clear_on_run:
                print()
                clean_database_fn(db_path=sqlite_db_path)
            else:
                print("\nAppending to existing SQLite data (clear_on_run: false)...")

            result_type = 'classifications' if task == 'classify' else 'detections'
            print(f"Writing {len(backend_detections)} {result_type} to SQLite...")
            sqlite_client.insert_detections_batch(backend_detections)
            print(f"✓ Successfully wrote to SQLite ({sqlite_db_path})")
        except Exception as e:
            print(f"⚠️  Failed to write to SQLite: {e}")
        finally:
            sqlite_client.close()

    if (
        visualization_fn
        and task == "classify"
        and has_images
        and classification_results
    ):
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        visualization_fn(
            images_dir=images_dir,
            fused_results=classification_results,
            image_probs=image_probs,
            sensor_probs=sensor_probs,
            class_names=class_names or {},
            out_dir=str(out_base),
        )

    if (
        detection_visualization_fn
        and task == "detect"
        and has_images
        and results
    ):
        out_base = Path("out") / out_subdir if out_subdir else Path("out")
        detection_visualization_fn(
            images_dir=images_dir,
            frames=results,
            out_dir=str(out_base),
        )

    return {
        "output_path": output_path,
        "total_detections": total_detections,
        "backend_detections": backend_detections,
        "n_frames": n_frames,
    }