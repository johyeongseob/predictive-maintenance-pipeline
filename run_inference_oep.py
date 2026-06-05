#!/usr/bin/env python3
"""
DLStreamer (OpenVINO Execution Provider) inference for YOLO.
Uses Intel DLStreamer via Docker for GPU-accelerated inference pipeline.
Writes results to JSONL file and SQLite database.
Uses gvawatermark for video visualization output.

Use-case-specific configuration (class names, SQL schema, detection mapping)
is loaded from config/<use_case_id>.yaml.
"""

import subprocess
import json
import yaml
import os
import shutil
from pathlib import Path
import argparse
from scripts.clean_slate import clean_database, clean_outputs

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


def parse_dlstreamer_output(raw_jsonl_path, images_dir=None, video_mode=False):
    """
    Parse DLStreamer JSON array output into per-frame JSONL with source filenames.

    DLStreamer writes a JSON array of frame objects. In image mode we map sequential
    indices back to the original sorted image filenames. In video mode we generate
    sequential frame_NNNNNN.jpg names.

    Objects are passed through in their native DLStreamer format — no reformatting.
    The detection_mapping config handles field extraction and transformation.

    Returns:
        list of frame dicts with 'source' and 'objects' (raw DLStreamer objects)
    """
    with open(raw_jsonl_path) as f:
        content = f.read().strip().rstrip(',')
        if not content.startswith('['):
            content = '[' + content + ']'
        data = json.loads(content)

    # Get original sorted filenames (image mode only)
    images = sorted(list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))) if images_dir and not video_mode else []

    frames = []
    for idx, frame in enumerate(data):
        if video_mode:
            source = f'frame_{idx:06d}.jpg'
        else:
            source = images[idx].name if idx < len(images) else f'{idx:06d}.jpg'

        # Pass through raw DLStreamer objects without reformatting
        frames.append({
            'source': source,
            'objects': frame.get('objects', [])
        })

    return frames


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


def run_sensor_inference(sensor_data_path, sensor_model_path, image_names, class_names, device='CPU'):
    """
    Run sensor MLP inference on sensor readings corresponding to given images.

    Args:
        sensor_data_path: Path to CSV with sensor measurements
        sensor_model_path: Path to OpenVINO sensor MLP model XML
        image_names: List of image filenames (used to look up sensor rows)
        class_names: Dict mapping class_id -> class_name
        device: OpenVINO device for sensor inference

    Returns:
        Dict mapping image_name -> list of class probabilities (softmax output)
    """
    import csv
    import numpy as np

    # Load sensor CSV into a lookup by image name
    sensor_lookup = {}
    all_sensor_vals = []
    with open(sensor_data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_key = row['Corresponding Image Name']
            sensor_vals = [
                float(row['MQ2']), float(row['MQ3']), float(row['MQ5']),
                float(row['MQ6']), float(row['MQ7']), float(row['MQ8']),
                float(row['MQ135'])
            ]
            sensor_lookup[img_key] = sensor_vals
            all_sensor_vals.append(sensor_vals)

    # Compute z-score normalization parameters from full dataset
    all_sensor_vals = np.array(all_sensor_vals, dtype=np.float32)
    sensor_mean = all_sensor_vals.mean(axis=0)
    sensor_std = all_sensor_vals.std(axis=0)
    # Avoid division by zero
    sensor_std[sensor_std == 0] = 1.0

    # Load OpenVINO model
    from openvino.runtime import Core
    core = Core()
    model = core.read_model(sensor_model_path)
    compiled = core.compile_model(model, device)
    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    results = {}
    for img_name in image_names:
        # Strip extension to get lookup key (e.g., "586_Perfume.png" -> "586_Perfume")
        key = Path(img_name).stem
        if key not in sensor_lookup:
            # No sensor data for this image — return uniform distribution
            n_classes = len(class_names)
            results[img_name] = [1.0 / n_classes] * n_classes
            continue

        sensor_vals = sensor_lookup[key]
        input_data = np.array([sensor_vals], dtype=np.float32)
        # Z-score normalization (model was trained on standardized features)
        input_data = (input_data - sensor_mean) / sensor_std
        output = compiled([input_data])[output_layer]

        # Apply softmax if needed (raw logits)
        probs = output[0]
        if probs.min() < 0 or probs.sum() < 0.99 or probs.sum() > 1.01:
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / exp_probs.sum()

        results[img_name] = probs.tolist()

    return results


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


def run_late_fusion(image_probs, sensor_probs, image_names, class_names,
                    image_weight=0.5, sensor_weight=0.5):
    """
    Late fusion: weighted average of image and sensor class probabilities.

    Args:
        image_probs: Dict mapping image_name -> list of class probabilities
        sensor_probs: Dict mapping image_name -> list of class probabilities
        image_names: List of image filenames to process
        class_names: Dict mapping class_id -> class_name
        image_weight: Weight for image model probabilities
        sensor_weight: Weight for sensor model probabilities

    Returns:
        List of dicts with 'source', 'label', 'confidence', 'probabilities'
    """
    import numpy as np
    results = []
    n_classes = len(class_names)

    for img_name in image_names:
        img_p = np.array(image_probs.get(img_name, [1.0/n_classes]*n_classes))
        sen_p = np.array(sensor_probs.get(img_name, [1.0/n_classes]*n_classes))

        fused = image_weight * img_p + sensor_weight * sen_p
        # Normalize
        fused = fused / fused.sum()

        best_idx = int(np.argmax(fused))
        results.append({
            'source': img_name,
            'label': class_names.get(best_idx, str(best_idx)),
            'confidence': float(fused[best_idx]),
            'image_confidence': float(img_p[best_idx]),
            'sensor_confidence': float(sen_p[best_idx]),
            'label_id': best_idx,
            'probabilities': {class_names.get(i, str(i)): float(fused[i]) for i in range(n_classes)},
        })

    return results


def run_image_classification(images_dir, model_xml, class_names, device='GPU',
                             num_images=None, img_size=640):
    """
    Run image classification using direct OpenVINO inference (no DLStreamer Docker).

    Used for classification tasks where DLStreamer's gvainference has limitations
    with FP32-input models. Detection tasks still use DLStreamer.

    Args:
        images_dir: Directory containing input images (jpg/png)
        model_xml: Path to OpenVINO model XML
        class_names: Dict mapping class_id -> class_name
        device: OpenVINO device (GPU/CPU)
        num_images: Limit number of images
        img_size: Model input size

    Returns:
        Dict mapping image_name -> list of class probabilities
    """
    import numpy as np
    import cv2
    from openvino.runtime import Core

    images_path = Path(images_dir).resolve()
    image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")
    if num_images:
        image_files = image_files[:num_images]

    # Load model
    core = Core()
    model = core.read_model(str(model_xml))
    compiled = core.compile_model(model, device)
    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    n_classes = len(class_names)
    results = {}

    print(f"Running image classification on {len(image_files)} images (device: {device})...")
    for img_file in image_files:
        # Preprocess: resize, normalize, transpose to NCHW
        img = cv2.imread(str(img_file))
        if img is None:
            results[img_file.name] = [1.0 / n_classes] * n_classes
            continue

        img_resized = cv2.resize(img, (img_size, img_size))
        img_float = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dim

        output = compiled([img_batch])[output_layer]
        probs = output[0]

        # Apply softmax if raw logits
        if probs.min() < 0 or probs.sum() < 0.99 or probs.sum() > 1.01:
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / exp_probs.sum()

        results[img_file.name] = probs.tolist()

    print(f"✓ Image classification completed ({len(results)} images)")
    return results


def run_dlstreamer_inference(images_dir, output_file, model_xml, model_proc,
                             device='GPU', threshold=0.25, img_size=640,
                             nireq=4, num_images=None, video_path=None,
                             inference_interval=1, out_subdir='', task='detect'):
    """
    Run DLStreamer inference via Docker.

    Supports two modes:
    - Image directory: Creates sequential hard links for multifilesrc
    - Video file: Uses filesrc ! decodebin with inference-interval and gvawatermark

    Args:
        out_subdir: Subdirectory under out/ for this use case (e.g. 'pipeline_defects_detection')
        task: 'detect' for object detection (gvadetect), 'classify' for classification (gvainference)
    """
    workspace_root = Path(__file__).parent.resolve()
    output_path = Path(output_file).resolve()
    # Docker container path prefix for outputs
    docker_out_prefix = f'/workspace/out/{out_subdir}' if out_subdir else '/workspace/out'

    # Determine input mode
    use_video = video_path and Path(video_path).exists()
    seq_dir = None
    n_frames = 0

    if use_video:
        video_abs = Path(video_path).resolve()
        # Count frames for reporting
        import cv2
        cap = cv2.VideoCapture(str(video_abs))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_frames = n_total  # Process ALL frames in video mode
        print(f"Processing all {n_frames} frames from video: {video_path}")
        if inference_interval > 1:
            print(f"  inference-interval={inference_interval} (running inference every {inference_interval}th frame)")
    else:
        images_path = Path(images_dir).resolve()
        image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        if not image_files:
            raise RuntimeError(f"No .jpg or .png images found in {images_dir}")
        if num_images:
            image_files = image_files[:num_images]
        n_frames = len(image_files)

        # Create sequential JPEG images for multifilesrc compatibility
        # DLStreamer multifilesrc + jpegdec requires JPEG format
        seq_dir = workspace_root / '.tmp_seq_images'
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        seq_dir.mkdir(parents=True)

        print(f"Preparing {n_frames} sequential image links...")
        for idx, img_file in enumerate(image_files):
            link_path = seq_dir / f'{idx:06d}.jpg'
            if img_file.suffix.lower() == '.jpg':
                os.link(str(img_file), str(link_path))
            else:
                # Convert non-JPEG images (e.g., PNG) to JPEG for DLStreamer
                import cv2
                img = cv2.imread(str(img_file))
                cv2.imwrite(str(link_path), img)

    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure viz directory exists for gvawatermark output
        host_out_dir = workspace_root / 'out' / out_subdir if out_subdir else workspace_root / 'out'
        (host_out_dir / 'viz').mkdir(parents=True, exist_ok=True)

        # Build Docker command
        docker_output = f'{docker_out_prefix}/detections_raw.jsonl'

        gpu_args = []
        user_args = []
        if device == 'GPU' and Path('/dev/dri').exists():
            gpu_args = ['--device=/dev/dri:/dev/dri']
            try:
                import grp
                for group_name in ['render', 'video']:
                    try:
                        gid = grp.getgrnam(group_name).gr_gid
                        gpu_args.append(f'--group-add={gid}')
                    except KeyError:
                        pass  # Group does not exist on this system
            except ImportError:
                pass  # grp module not available on this platform
            user_args = ['--user', f'{os.getuid()}:{os.getgid()}']

        model_xml_abs = Path(model_xml).resolve()
        model_proc_abs = Path(model_proc).resolve()
        ov_models_root = (workspace_root / 'models' / 'ov_models').resolve()
        model_det_container = '/workspace/ov_models/' + str(model_xml_abs.relative_to(ov_models_root))
        model_proc_container = '/workspace/ov_models/' + str(model_proc_abs.relative_to(ov_models_root))

        # Build volume mounts and GStreamer pipeline based on input mode
        volumes = [
            '-v', f'{workspace_root}/datasets:/workspace/datasets:ro',
            '-v', f'{workspace_root}/models/ov_models:/workspace/ov_models:ro',
            '-v', f'{workspace_root}/out:/workspace/out',
        ]

        if use_video:
            # Video mode: mount video file, use filesrc ! decodebin
            video_abs = Path(video_path).resolve()
            video_dir = str(video_abs.parent)
            video_name = video_abs.name
            volumes.extend([
                '-v', f'{video_dir}:/workspace/video:ro',
            ])
            
            # inference-interval: run detection every Nth frame
            inference_interval_prop = f' inference-interval={inference_interval}' if inference_interval > 1 else ''
            
            # Output both annotated video AND individual frame images (for SQL frame_id lookup)
            annotated_video_output = f'{docker_out_prefix}/annotated_output.mp4'
            viz_frames_pattern = f'{docker_out_prefix}/viz/frame_%06d.jpg'
            
            # Select GStreamer inference element based on task
            if task == 'classify':
                infer_element = (
                    f'gvainference model={model_det_container} model-proc={model_proc_container} '
                    f'device={device} nireq={nireq}{inference_interval_prop}'
                )
            else:
                infer_element = (
                    f'gvadetect model={model_det_container} model-proc={model_proc_container} '
                    f'device={device} nireq={nireq} threshold={threshold}{inference_interval_prop}'
                )

            gst_pipeline = (
                f'gst-launch-1.0 -e '
                f'filesrc location=/workspace/video/{video_name} ! decodebin ! videoconvert ! '
                f'video/x-raw,format=BGRx ! '
                f'{infer_element} ! '
                f'gvametaconvert format=json add-empty-results=true ! '
                f'gvametapublish method=file file-format=json file-path={docker_output} ! '
                f'gvawatermark ! videoconvert ! video/x-raw,format=RGB ! '
                f'tee name=t '
                f't. ! queue ! jpegenc ! multifilesink location={viz_frames_pattern} '
                f't. ! queue ! videoconvert ! x264enc tune=zerolatency bitrate=4000 ! mp4mux ! '
                f'filesink location={annotated_video_output}'
            )
        else:
            # Image mode: mount seq dir and use multifilesrc
            image_pattern = '/workspace/.tmp_seq_images/%06d.jpg'
            viz_output_pattern = f'{docker_out_prefix}/viz/frame_%06d.jpg'
            volumes.extend(['-v', f'{seq_dir}:/workspace/.tmp_seq_images:ro'])
            
            # Select GStreamer inference element based on task
            if task == 'classify':
                infer_element = (
                    f'gvainference model=$MODEL_DET model-proc=$MODEL_PROC '
                    f'device=$OV_DEVICE nireq=$NIREQ'
                )
            else:
                infer_element = (
                    f'gvadetect model=$MODEL_DET model-proc=$MODEL_PROC '
                    f'device=$OV_DEVICE nireq=$NIREQ threshold=$THRESHOLD'
                )

            gst_pipeline = (
                f'gst-launch-1.0 -e '
                f'multifilesrc location={image_pattern} index=0 start-index=0 stop-index=$((N-1)) num-buffers=$N ! '
                f'jpegdec ! videoconvert ! videoscale ! '
                f'video/x-raw,format=BGRx,width=$IMG_SIZE,height=$IMG_SIZE ! '
                f'{infer_element} ! '
                f'gvametaconvert format=json add-empty-results=true ! '
                f'gvametapublish method=file file-format=json file-path={docker_output} ! '
                f'gvawatermark ! videoconvert ! video/x-raw,format=RGB ! '
                f'jpegenc ! multifilesink location={viz_output_pattern}'
            )

        # Environment variables (used by image mode's shell expansion)
        env_args = [
            '-e', f'IMAGE_PATTERN=/workspace/.tmp_seq_images/%06d.jpg',
            '-e', f'MODEL_DET={model_det_container}',
            '-e', f'MODEL_PROC={model_proc_container}',
            '-e', f'OV_DEVICE={device}',
            '-e', f'IMG_SIZE={img_size}',
            '-e', f'N={n_frames}',
            '-e', f'THRESHOLD={threshold}',
            '-e', f'NIREQ={nireq}',
        ]

        docker_cmd = [
            'docker', 'run', '--rm', '-i', '--network', 'host',
            *gpu_args,
            *user_args,
            *env_args,
            *volumes,
            'intel/dlstreamer:latest',
            'bash', '-lc', gst_pipeline
        ]

        print(f"Running DLStreamer inference on {n_frames} {'video frames' if use_video else 'images'} (device: {device})...")
        result = subprocess.run(docker_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            stderr = result.stderr
            # Filter out harmless GStreamer plugin warnings
            error_lines = [l for l in stderr.split('\n')
                           if l.strip() and 'GStreamer-WARNING' not in l and 'libva info' not in l]
            if error_lines:
                print(f"DLStreamer stderr:\n{''.join(error_lines[:10])}")
            raise RuntimeError(f"DLStreamer inference failed with exit code {result.returncode}")

        print("✓ DLStreamer inference completed")

        # The raw output is at out/<subdir>/detections_raw.jsonl
        raw_output = host_out_dir / 'detections_raw.jsonl'
        return str(raw_output), n_frames

    finally:
        # Cleanup temp sequential links
        if seq_dir and seq_dir.exists():
            shutil.rmtree(seq_dir)


def run_inference(model_path, model_proc_path, images_dir, output_file,
                  device='GPU', conf_threshold=0.25, num_images=None,
                  nireq=4, sqlite_db_path=None, clear_on_run=True,
                  video_path=None, inference_interval=1,
                  schema=None, detection_mapping=None, out_subdir='',
                  task='detect', modality='image', class_names=None,
                  sensor_model_path=None, sensor_data_path=None,
                  fusion_weights=None):
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

    # ── IMAGE MODALITY ──
    image_probs = {}
    frames = []
    n_processed = 0

    if modality in ('image', 'multi'):
        if task == 'classify':
            # Classification: direct OpenVINO inference (DLStreamer has limitations with FP32 cls models)
            image_probs = run_image_classification(
                images_dir=images_dir,
                model_xml=model_path,
                class_names=class_names or {},
                device=device,
                num_images=num_images,
                img_size=640
            )
            n_processed = len(image_probs)
        else:
            # Detection: DLStreamer Docker pipeline (gvadetect)
            raw_output, n_processed = run_dlstreamer_inference(
                images_dir=images_dir,
                output_file=output_file,
                model_xml=model_path,
                model_proc=model_proc_path,
                device=device,
                threshold=conf_threshold,
                num_images=num_images,
                nireq=nireq,
                video_path=video_path,
                inference_interval=inference_interval,
                out_subdir=out_subdir,
                task=task
            )

            print("Parsing DLStreamer output...")
            frames = parse_dlstreamer_output(raw_output, images_dir=images_dir, video_mode=use_video)

            # Clean up raw output
            raw_path = Path(raw_output)
            if raw_path.exists():
                raw_path.unlink()

    # ── SENSOR MODALITY ──
    sensor_probs = {}

    if modality in ('sensor', 'multi') and sensor_model_path and sensor_data_path:
        print(f"Running sensor MLP inference ({len(image_names)} samples)...")
        sensor_probs = run_sensor_inference(
            sensor_data_path=sensor_data_path,
            sensor_model_path=sensor_model_path,
            image_names=image_names,
            class_names=class_names or {},
            device='CPU'  # Sensor MLP is small, CPU is fine
        )
        print(f"✓ Sensor inference completed")

    # ── LATE FUSION (multi modality) ──
    fused_results = []

    if modality == 'multi' and class_names:
        print(f"Running late fusion (image_w={fusion_weights['image']:.2f}, sensor_w={fusion_weights['sensor']:.2f})...")
        # If image-only didn't run (sensor-only), image_probs will be empty
        fused_results = run_late_fusion(
            image_probs=image_probs,
            sensor_probs=sensor_probs,
            image_names=image_names,
            class_names=class_names,
            image_weight=fusion_weights.get('image', 0.5),
            sensor_weight=fusion_weights.get('sensor', 0.5)
        )
        print(f"✓ Late fusion completed: {len(fused_results)} classifications")
    elif modality == 'sensor' and class_names:
        # Sensor-only mode: convert sensor probs to classification results
        import numpy as np
        for img_name in image_names:
            probs = sensor_probs.get(img_name, [1.0/len(class_names)]*len(class_names))
            best_idx = int(np.argmax(probs))
            fused_results.append({
                'source': img_name,
                'label': class_names.get(best_idx, str(best_idx)),
                'confidence': float(probs[best_idx]),
                'label_id': best_idx,
                'probabilities': {class_names.get(i, str(i)): float(probs[i]) for i in range(len(class_names))},
            })

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
                  fusion_weights=fusion_weights)
