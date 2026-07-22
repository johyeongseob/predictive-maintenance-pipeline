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
from src.inference.visualization import generate_classification_viz
from src.inference.output_writer import write_inference_outputs
from src.inference.config_builder import build_handler_config


# Import SQLite client
try:
    from src.utility.sqlite_client import SQLiteClient
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("⚠️  SQLite client not available.")


def run_inference(model_path, model_proc_path, images_dir, output_file,
                  device='GPU', conf_threshold=0.25, num_images=None,
                  nireq=4, sqlite_db_path=None, clear_on_run=True,
                  video_path=None, inference_interval=1,
                  schema=None, detection_mapping=None, out_subdir='',
                  task='detect', modality='image', class_names=None,
                  sensor_model_path=None, sensor_data_path=None,
                  fusion_weights=None, handler_name=None, config=None):
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


    handler_config = build_handler_config(
        config,
        schema,
        image_names,
        output_file=output_file,
        out_subdir=out_subdir,
    )
    handler = dispatch(handler_config)
    handler.load(handler_config)
    results = handler.infer(image_names, handler_config)

    output_result = write_inference_outputs(
        results=results,
        handler=handler,
        handler_config=handler_config,
        sqlite_client=sqlite_client,
        sqlite_db_path=sqlite_db_path,
        clear_on_run=clear_on_run,
        clean_database_fn=clean_database,
        detection_mapping=detection_mapping,
        image_files=image_files,
        visualization_fn=generate_classification_viz,
    )

    output_path = output_result["output_path"]
    total_detections = output_result["total_detections"]
    n_frames = output_result["n_frames"]

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
                  handler_name=handler_name,
                  config=config)
