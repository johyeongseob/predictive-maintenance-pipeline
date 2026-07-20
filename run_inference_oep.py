#!/usr/bin/env python3
"""
DLStreamer (OpenVINO Execution Provider) inference for YOLO.
Uses Intel DLStreamer via Docker for GPU-accelerated inference pipeline.
Writes results to JSONL file and SQLite database.
Uses gvawatermark for video visualization output.

Use-case-specific configuration (class names, SQL schema, detection mapping)
is loaded from config/<use_case_id>.yaml.
"""

import json
import yaml
from pathlib import Path
import argparse
from scripts.clean_slate import clean_database, clean_outputs
from src.inference import dispatch


# Import SQLite client
try:
    from src.utility.sqlite_client import SQLiteClient
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("⚠️  SQLite client not available.")


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


def apply_detection_mapping(det_obj, frame_idx, orig_w, orig_h, detection_mapping):
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


def run_inference(model_path, model_proc_path, images_dir, output_file,
                  device='GPU', conf_threshold=0.25, num_images=None,
                  nireq=4, sqlite_db_path=None, clear_on_run=True,
                  video_path=None, inference_interval=1,
                  schema=None, detection_mapping=None, out_subdir='',
                  task='detect', modality='image', class_names=None,
                  sensor_model_path=None, sensor_data_path=None,
                  fusion_weights=None, handler_name=None):
    """Run full inference pipeline: DLStreamer → parse → JSONL + SQLite.

    Supports modality-aware inference:
    - 'image': DLStreamer only (detect or classify)
    - 'sensor': Sensor MLP only
    - 'multi': DLStreamer + Sensor MLP + Late Fusion (weighted avg)

    Args:
        schema: Schema dict from config (passed to SQLiteClient).
        detection_mapping: Dict mapping column names to dot-paths in detection objects.
        out_subdir: Subdirectory under out/ for this use case.
        task: 'detect' or 'classify'
        modality: 'image', 'sensor', or 'multi'
        class_names: Dict mapping class_id -> class_name (for classification)
        sensor_model_path: Path to sensor MLP model XML (for sensor/multi modality)
        sensor_data_path: Path to sensor CSV data file
        fusion_weights: Dict with 'image' and 'sensor' weights for late fusion
    """

    # Initialize SQLite client
    sqlite_client = None
    if SQLITE_AVAILABLE and sqlite_db_path:
        try:
            sqlite_client = SQLiteClient(db_path=sqlite_db_path, schema=schema)
            print(f"✓ SQLite client connected to {sqlite_db_path}")
        except Exception as e:
            print(f"⚠️  Failed to connect to SQLite: {e}")
            print("   Continuing with JSONL output only...")

    # Resolve image list for sensor/multi modality matching
    use_video = video_path and Path(video_path).exists()
    images_path = Path(images_dir) if images_dir else None
    image_files = []
    image_names = []
    if images_path and not use_video:
        all_images = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        if num_images:
            all_images = all_images[:num_images]
        image_files = all_images
        image_names = [f.name for f in image_files]

    # Default fusion weights
    if not fusion_weights:
        fusion_weights = {'image': 0.5, 'sensor': 0.5}


    # Dispatch inference to the selected input-type handler.
    if not handler_name:
        if task == 'detect':
            handler_name = 'dlstreamer_detect'
        elif task == 'classify' and modality == 'image':
            handler_name = 'openvino_classify'
        elif task == 'classify' and modality in ('sensor', 'multi'):
            handler_name = 'sensor_flat'

    handler_config = {
        'handler': handler_name,
        'modality': modality,
        'task': task,
        'model_path': model_path,
        'model_proc_path': model_proc_path,
        'images_dir': images_dir,
        'output_file': output_file,
        'device': device,
        'conf_threshold': conf_threshold,
        'num_images': num_images,
        'nireq': nireq,
        'video_path': video_path,
        'inference_interval': inference_interval,
        'out_subdir': out_subdir,
        'class_names': class_names or {},
        'sensor_model_path': sensor_model_path,
        'sensor_data_path': sensor_data_path,
        'sensor_device': 'CPU',
        'fusion_weights': fusion_weights,
        'schema': schema,
        'image_names': image_names,
        'inference': {
            'handler': handler_name,
            'task': task,
            'model_path': model_path,
            'device': device,
            'confidence_threshold': conf_threshold,
            'imgsz': 640,
            'images_path': images_dir,
            'video_path': video_path,
            'inference_interval': inference_interval,
        },
        'sensor': {
            'model_path': sensor_model_path,
            'data_path': sensor_data_path,
        },
    }

    handler = dispatch(handler_config)
    handler.load(handler_config)
    results = handler.infer(image_names, handler_config)

    image_probs = getattr(handler, 'last_image_probs', {})
    sensor_probs = getattr(handler, 'last_sensor_probs', {})
    frames = []
    fused_results = []
    n_processed = len(results)

    if task == 'detect':
        frames = results
    elif task == 'classify' and modality == 'image':
        image_probs = {result['source']: result['probabilities'] for result in results}
    elif task == 'classify':
        fused_results = results


    # ── WRITE OUTPUTS ──
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    backend_detections = []

    if task == 'classify' and fused_results:
        # Classification output: one result per image
        with open(output_path, 'w') as f:
            for result in fused_results:
                f.write(json.dumps(result) + '\n')
                total_detections += 1
                # Build row for SQLite using detection_mapping
                if sqlite_client and detection_mapping:
                    row = {}
                    for col_name, path_spec in detection_mapping.items():
                        if path_spec == '__frame_index__':
                            row[col_name] = image_names.index(result['source']) if result['source'] in image_names else 0
                        else:
                            row[col_name] = _resolve_dotpath(result, path_spec)
                    backend_detections.append(row)
    elif task == 'classify' and modality == 'image' and image_probs:
        # Image-only classification (no fusion) - convert image_probs to results
        import numpy as np
        with open(output_path, 'w') as f:
            for frame_idx, img_name in enumerate(image_names):
                probs = image_probs.get(img_name, [1.0/len(class_names)]*len(class_names))
                best_idx = int(np.argmax(probs))
                result = {
                    'source': img_name,
                    'label': class_names.get(best_idx, str(best_idx)),
                    'confidence': float(probs[best_idx]),
                    'label_id': best_idx,
                }
                f.write(json.dumps(result) + '\n')
                total_detections += 1
                if sqlite_client and detection_mapping:
                    row = {}
                    for col_name, path_spec in detection_mapping.items():
                        if path_spec == '__frame_index__':
                            row[col_name] = frame_idx
                        else:
                            row[col_name] = _resolve_dotpath(result, path_spec)
                    backend_detections.append(row)
    else:
        # Detection task: existing behavior
        import cv2
        if use_video:
            cap = cv2.VideoCapture(str(video_path))
            ret, first_frame = cap.read()
            video_h, video_w = (first_frame.shape[:2] if ret else (640, 640))
            cap.release()
        else:
            video_h, video_w = 640, 640

        with open(output_path, 'w') as f:
            for frame_idx, frame in enumerate(frames):
                f.write(json.dumps(frame) + '\n')
                total_detections += len(frame['objects'])

                if sqlite_client and frame['objects']:
                    if use_video:
                        orig_h, orig_w = video_h, video_w
                    elif frame_idx < len(image_files):
                        img = __import__('cv2').imread(str(image_files[frame_idx]))
                        orig_h, orig_w = (img.shape[:2] if img is not None else (640, 640))
                    else:
                        orig_h, orig_w = 640, 640

                    for det in frame['objects']:
                        row = apply_detection_mapping(
                            det, frame_idx, orig_w, orig_h, detection_mapping
                        )
                        backend_detections.append(row)

    # Write to SQLite database
    if sqlite_client and backend_detections:
        try:
            if clear_on_run:
                print()
                clean_database(db_path=sqlite_db_path)
            else:
                print(f"\nAppending to existing SQLite data (clear_on_run: false)...")

            print(f"Writing {len(backend_detections)} {'classifications' if task == 'classify' else 'detections'} to SQLite...")
            sqlite_client.insert_detections_batch(backend_detections)
            print(f"✓ Successfully wrote to SQLite ({sqlite_db_path})")
        except Exception as e:
            print(f"⚠️  Failed to write to SQLite: {e}")
        finally:
            sqlite_client.close()

    # Generate classification visualizations
    if task == 'classify' and fused_results:
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        generate_classification_viz(
            images_dir=images_dir,
            fused_results=fused_results,
            image_probs=image_probs,
            sensor_probs=sensor_probs,
            class_names=class_names or {},
            out_dir=str(out_base)
        )

    n_frames = len(frames) if frames else len(fused_results)
    print(f"\n{'='*60}")
    print(f"Inference Complete ({modality} / {task})")
    print(f"{'='*60}")
    print(f"{'Frames' if task == 'detect' else 'Images'} processed: {n_frames}")
    print(f"Total {'detections' if task == 'detect' else 'classifications'}: {total_detections}")
    if task == 'detect':
        print(f"Average per frame: {total_detections/max(n_frames,1):.1f}")
    print(f"JSONL output: {output_path}")
    if sqlite_client:
        print(f"SQLite output: {sqlite_db_path}")
    if use_video:
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        annotated = out_base / 'annotated_output.mp4'
        if annotated.exists():
            print(f"Annotated video: {annotated}")
        viz_dir = out_base / 'viz'
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('frame_*.jpg')))
            if viz_count:
                print(f"Annotated frames: {viz_dir}/ ({viz_count} images, frame_NNNNNN.jpg matches SQL frame_id)")
    else:
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        viz_dir = out_base / 'viz'
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('*.jpg')))
            if viz_count:
                print(f"Annotated images: {viz_dir}/ ({viz_count} images)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLStreamer (OEP) inference for YOLO with SQLite output')
    parser.add_argument('--model', default=None,
                        help='Path to OpenVINO model XML (default: from config)')
    parser.add_argument('--model-proc', default=None,
                        help='Path to model_proc.json (default: alongside model)')
    parser.add_argument('--images', default=None,
                        help='Directory containing images (default: from config)')
    parser.add_argument('--video', default=None,
                        help='Path to input video file (overrides --images)')
    parser.add_argument('--output', default=None,
                        help='Output JSONL file (default: out/<use-case-id>/detections.jsonl)')
    parser.add_argument('--device', default='GPU', choices=['CPU', 'GPU'],
                        help='Inference device')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Limit number of images to process (image mode only)')
    parser.add_argument('--inference-interval', type=int, default=1,
                        help='Run inference every Nth frame (video mode, DLStreamer native, default: 1)')
    parser.add_argument('--nireq', type=int, default=4,
                        help='Number of inference requests (default: 4)')
    parser.add_argument('--config', default=None,
                        help='Path to config file (default: reads from config.json)')

    args = parser.parse_args()

    # Load config
    sqlite_db_path = None
    clear_on_run = True
    clear_outputs = True
    use_case_id = 'pipeline_defects_detection'
    schema = None
    detection_mapping = None
    task = 'detect'
    modality = 'image'
    class_names = None
    sensor_model_path = None
    sensor_data_path = None
    handler_name = None
    fusion_weights = None

    if args.config:
        config_path = args.config
    else:
        # Read default use case from config.json
        with open('config.json', 'r') as f:
            main_config = json.load(f)
        use_case_id = main_config.get('default-use-case', 'pipeline_defects_detection')
        config_path = f'config/{use_case_id}/config.yaml'

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load task and modality
            inference_cfg = config.get('inference', {})
            task = inference_cfg.get('task', 'detect')
            handler_name = inference_cfg.get('handler')
            modality = config.get('modality', 'image')

            # Load class names
            names = config.get('names', {})
            if names:
                class_names = names

            # Load sensor config (for sensor/multi modality)
            sensor_cfg = config.get('sensor', {})
            sensor_model_path = sensor_cfg.get('model_path')
            sensor_data_path = sensor_cfg.get('data_path')
            fusion_weights = config.get('fusion_weights')

            # Load SQLite config
            sqlite_cfg = config.get('sqlite', {})
            sqlite_db_path = sqlite_cfg.get('db_path', f'out/{use_case_id}/sql_data/detections.db')
            clear_on_run = sqlite_cfg.get('clear_on_run', True)
            clear_outputs = sqlite_cfg.get('clear_outputs', True)

            # Load schema and detection_mapping from config
            schema = config.get('schema')
            if schema:
                # Inject class names into schema for get_schema() LLM context
                if names:
                    schema['class_names'] = names

            detection_mapping = config.get('detection_mapping')

            # Load model path from config if not provided via --model
            if not args.model:
                args.model = inference_cfg.get('model_path', f'models/ov_models/{use_case_id}/image/best.xml')
            
            # Load images path from config if user didn't override via --images
            config_images_path = inference_cfg.get('images_path')
            if not args.images and config_images_path:
                args.images = config_images_path
            
            # Load video path from config if not provided via --video
            if not args.video:
                input_mode = inference_cfg.get('input_mode', 'images')
                if input_mode == 'video':
                    args.video = inference_cfg.get('video_path')
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            print("   Continuing with JSONL output only...")

    # Fallback detection_mapping if config doesn't provide one
    if not detection_mapping:
        detection_mapping = {
            'frame_id': '__frame_index__',
            'label': 'detection.label',
            'confidence': 'detection.confidence',
            'x': 'scale_w:detection.bounding_box.x_min',
            'y': 'scale_h:detection.bounding_box.y_min',
            'width': 'scale_w_diff:detection.bounding_box.x_max-detection.bounding_box.x_min',
            'height': 'scale_h_diff:detection.bounding_box.y_max-detection.bounding_box.y_min',
        }

    # Final fallback if model still not set
    if not args.model:
        args.model = f'models/ov_models/{use_case_id}/image/best.xml'

    # Final fallback if images path still not set
    if not args.images:
        args.images = f'datasets/{use_case_id}/images/val'

    # Default output path: out/<use_case_id>/detections.jsonl
    out_dir = f'out/{use_case_id}'
    if not args.output:
        args.output = f'{out_dir}/detections.jsonl'

    # Default sqlite path if not set from config
    if not sqlite_db_path:
        sqlite_db_path = f'{out_dir}/sql_data/detections.db'

    # Default model_proc path (alongside model XML)
    if not args.model_proc:
        args.model_proc = str(Path(args.model).parent / 'model_proc.json')

    # Clean outputs directory if specified in config
    if clear_outputs:
        clean_outputs(out_dir=out_dir)
        print()
    else:
        print(f"📌 Keeping existing outputs (clear_outputs: false)\n")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    run_inference(args.model, args.model_proc, args.images, args.output,
                  args.device, args.conf_threshold, args.num_images,
                  args.nireq, sqlite_db_path, clear_on_run, args.video,
                  args.inference_interval, schema=schema,
                  detection_mapping=detection_mapping, out_subdir=use_case_id,
                  task=task, modality=modality, class_names=class_names,
                  sensor_model_path=sensor_model_path,
                  sensor_data_path=sensor_data_path,
                  fusion_weights=fusion_weights, 
                  handler_name=handler_name)
