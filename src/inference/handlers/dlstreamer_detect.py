"""DLStreamer detection handler."""

import json
import os
import shutil
import subprocess
from pathlib import Path

from src.inference import InferenceHandler, register_handler


def parse_dlstreamer_output(raw_jsonl_path, images_dir=None, video_mode=False):
    """
    Parse DLStreamer JSON array output into per-frame JSONL with source filenames.

    DLStreamer writes a JSON array of frame objects. In image mode we map sequential
    indices back to the original sorted image filenames. In video mode we generate
    sequential frame_NNNNNN.jpg names.

    Objects are passed through in their native DLStreamer format - no reformatting.
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
    workspace_root = Path(__file__).resolve().parents[3]
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

        print("??DLStreamer inference completed")

        # The raw output is at out/<subdir>/detections_raw.jsonl
        raw_output = host_out_dir / 'detections_raw.jsonl'
        return str(raw_output), n_frames

    finally:
        # Cleanup temp sequential links
        if seq_dir and seq_dir.exists():
            shutil.rmtree(seq_dir)


@register_handler("dlstreamer_detect")
class DLStreamerDetectHandler(InferenceHandler):
    """Run image detection with the existing DLStreamer Docker pipeline."""

    def __init__(self):
        self.config = {}

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        task = config.get("task", inference_cfg.get("task", "detect"))
        modality = config.get("modality", "image")
        return task == "detect" and modality in ("image", "multi")

    def load(self, config: dict) -> None:
        self.config = config

    def infer(self, inputs: list, config: dict) -> list[dict]:
        merged_config = {**self.config, **config}
        inference_cfg = merged_config.get("inference", {})
        model_path = merged_config.get("model_path", inference_cfg.get("model_path"))
        model_proc_path = merged_config.get("model_proc_path")
        if not model_proc_path and model_path:
            model_proc_path = str(Path(model_path).parent / "model_proc.json")

        images_dir = merged_config.get("images_dir", inference_cfg.get("images_path"))
        output_file = merged_config.get("output_file", merged_config.get("output"))
        video_path = merged_config.get("video_path", inference_cfg.get("video_path"))
        use_video = video_path and Path(video_path).exists()

        raw_output, _ = run_dlstreamer_inference(
            images_dir=images_dir,
            output_file=output_file,
            model_xml=model_path,
            model_proc=model_proc_path,
            device=merged_config.get("device", inference_cfg.get("device", "GPU")),
            threshold=merged_config.get(
                "conf_threshold",
                inference_cfg.get("confidence_threshold", 0.25),
            ),
            img_size=merged_config.get("img_size", inference_cfg.get("imgsz", 640)),
            nireq=merged_config.get("nireq", 4),
            num_images=merged_config.get("num_images"),
            video_path=video_path,
            inference_interval=merged_config.get(
                "inference_interval",
                inference_cfg.get("inference_interval", 1),
            ),
            out_subdir=merged_config.get("out_subdir", ""),
            task="detect",
        )

        frames = parse_dlstreamer_output(raw_output, images_dir=images_dir, video_mode=use_video)
        raw_path = Path(raw_output)
        if raw_path.exists():
            raw_path.unlink()
        return frames

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})
