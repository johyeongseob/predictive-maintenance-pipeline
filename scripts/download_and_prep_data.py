#!/usr/bin/env python3
"""
Download a dataset from a given URL and prepare train/val splits.

The active use case is read from config.json ("default-use-case" key).
The preparation logic is use-case-specific:

  pipeline_defects_detection
      Splits the dataset randomly (default 90/10) into YOLO detection format:
        datasets/pipeline_defects_detection/images/{train,val}/
        datasets/pipeline_defects_detection/labels/{train,val}/
      Also creates a video from the val images.

  gas_detection
      Stratified 80/20 split per gas class (Mixture, NoGas, Perfume, Smoke):
        datasets/gas_detection/images/train/{class}/   (YOLO classification format)
        datasets/gas_detection/images/val/             (flat, for PACE inference)
      Sensor CSV (if present in the archive) is copied to:
        datasets/gas_detection/sensor_data/

IMPORTANT DISCLAIMER:
    By using this script, you acknowledge that YOU are solely responsible for
    ensuring you have the necessary rights, permissions, and licenses to
    download and use any dataset you provide. We take no responsibility for
    any misuse of data or violation of terms of service.

Usage:
    conda activate pace
    python scripts/download_and_prep_data.py <dataset_url>

    # Pipeline defect dataset (Kaggle)
    python scripts/download_and_prep_data.py "https://www.kaggle.com/api/v1/datasets/download/simplexitypipeline/pipeline-defect-dataset"

    # Gas detection dataset (Mendeley)
    python scripts/download_and_prep_data.py "https://data.mendeley.com/public-api/zip/zkwgkjkjn9/download/2"

    # Custom split ratio
    python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --train-ratio 0.8

    # Custom output directory
    python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --output datasets/my_dataset
"""

import argparse
import sys
import random
import shutil
import zipfile
import subprocess
import json
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


GAS_DETECTION_CLASSES = ["Mixture", "NoGas", "Perfume", "Smoke"]

DISCLAIMER = """
⚠️  DISCLAIMER: By using this script, you acknowledge that YOU are solely
    responsible for ensuring you have the necessary rights, permissions, and
    licenses to download and use the dataset at the provided URL. We take no
    responsibility for any misuse of data or violation of terms of service.
"""


def download_dataset(dataset_url: str, download_dir: Path, zip_name: str = "dataset.zip") -> Path:
    """Download the dataset from the given URL using curl."""
    print("📥 Downloading dataset...")
    print(f"   URL: {dataset_url}")
    print(f"   Download dir: {download_dir}")
    print()

    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / zip_name

    try:
        cmd = [
            "curl", "-L", "-o", str(zip_path),
            dataset_url
        ]
        print(f"   Running: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
        print("✅ Download complete\n")

        # Unzip
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(download_dir)
        zip_path.unlink()
        print("✅ Extraction complete\n")

        return download_dir

    except FileNotFoundError:
        print("❌ Error: 'curl' not found. Please install curl.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset (return code: {e.returncode})")
        sys.exit(1)
    except zipfile.BadZipFile:
        print("❌ Error: Downloaded file is not a valid zip archive.")
        sys.exit(1)


def find_images_and_labels(source_dir: Path):
    """Recursively find all image files and their corresponding label files.
    
    Matches images to labels by filename stem (e.g., img001.jpg -> img001.txt),
    regardless of directory structure. This handles various dataset layouts:
    - Parallel images/labels directories
    - Flat structure with mixed files
    - Pre-split train/val/test directories
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Step 1: Find all images
    images = []
    for f in source_dir.rglob('*'):
        if f.suffix.lower() in image_extensions and f.is_file():
            images.append(f)
    images.sort(key=lambda p: p.name)
    
    # Step 2: Find all .txt label files and index by stem
    label_map = {}
    for f in source_dir.rglob('*.txt'):
        if f.is_file():
            stem = f.stem
            # If multiple .txt files share the same stem, prefer the one
            # in a directory named 'labels'
            if stem not in label_map or 'labels' in str(f.parent).lower():
                label_map[stem] = f
    
    # Step 3: Match images to labels by stem
    pairs = []
    matched = 0
    for img_path in images:
        label_path = label_map.get(img_path.stem)
        if label_path:
            matched += 1
        pairs.append((img_path, label_path))
    
    print(f"   Matched {matched}/{len(images)} images with label files")
    
    return pairs


def split_dataset(pairs: list, train_ratio: float, seed: int):
    """Randomly split (image, label) pairs into train and val sets."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs = shuffled[split_idx:]

    return train_pairs, val_pairs


def copy_pairs(pairs: list, dest_images_dir: Path, dest_labels_dir: Path, split_name: str):
    """Copy image/label pairs to destination directories."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"   Copying {len(pairs)} samples to {split_name}...")

    for img_path, label_path in pairs:
        # Copy image
        shutil.copy2(img_path, dest_images_dir / img_path.name)
        # Copy label if it exists
        if label_path and label_path.exists():
            shutil.copy2(label_path, dest_labels_dir / label_path.name)

    print(f"   ✅ {split_name}: {len(pairs)} images copied")


def find_gas_images_by_class(source_dir: Path) -> dict:
    """Find all images in source_dir and group by gas class.

    Expects a flat directory (or nested) with files named {serial}_{ClassName}.png,
    e.g. 586_Perfume.png, 0_NoGas.png.  The class name is the last
    underscore-delimited token of the filename stem.

    Returns:
        Dict mapping class_name -> sorted list of Path objects.
    """
    class_files = {cls: [] for cls in GAS_DETECTION_CLASSES}

    for f in source_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            parts = f.stem.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in class_files:
                class_files[parts[1]].append(f)

    for cls in GAS_DETECTION_CLASSES:
        class_files[cls].sort(key=lambda p: p.name)

    return class_files


def prep_gas_detection(download_dir: Path, output_dir: Path,
                       train_ratio: float, seed: int) -> tuple:
    """Prepare the gas detection dataset.

    Stratified split per class, producing:
      images/train/{class}/  — YOLO classification training format
      images/val/            — flat directory for PACE inference
      sensor_data/           — sensor CSV(s) copied from the archive

    Args:
        download_dir: Root of the extracted archive.
        output_dir:   Destination dataset directory.
        train_ratio:  Fraction of each class used for training.
        seed:         Random seed for reproducibility.

    Returns:
        (n_train, n_val) counts.
    """
    print("🔍 Scanning for gas detection images...")
    class_files = find_gas_images_by_class(download_dir)

    total = sum(len(v) for v in class_files.values())
    if total == 0:
        print("❌ No gas detection images found in the downloaded archive.")
        print(f"   Expected files named {{serial}}_{{ClassName}}.png inside:")
        print(f"   {download_dir}")
        sys.exit(1)

    for cls in GAS_DETECTION_CLASSES:
        print(f"   {cls}: {len(class_files[cls])} images")
    print()

    # Stratified split
    print(f"🔀 Stratified split (seed={seed}, "
          f"train={train_ratio:.0%} / val={1 - train_ratio:.0%})...")
    random.seed(seed)
    train_files: dict = {}
    val_files: dict = {}
    for cls, files in class_files.items():
        shuffled = files.copy()
        random.shuffle(shuffled)
        idx = int(len(shuffled) * train_ratio)
        train_files[cls] = shuffled[:idx]
        val_files[cls] = shuffled[idx:]
        print(f"   {cls}: {len(train_files[cls])} train, {len(val_files[cls])} val")
    print()

    # Copy training images: images/train/{class}/ (YOLO classification format)
    print("📂 Organizing dataset...")
    for cls, files in train_files.items():
        dest = output_dir / "images" / "train" / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
    n_train = sum(len(v) for v in train_files.values())
    print(f"   ✅ train: {n_train} images  →  images/train/{{class}}/")

    # Copy validation images: flat images/val/ (for PACE inference)
    # Filenames encode the class (e.g. 586_Perfume.png) so class info is not lost.
    val_dir = output_dir / "images" / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    n_val = 0
    for cls, files in val_files.items():
        for f in files:
            shutil.copy2(f, val_dir / f.name)
            n_val += 1
    print(f"   ✅ val:   {n_val} images  →  images/val/ (flat, for inference)")
    print()

    # Copy sensor CSV(s) from the archive
    print("📊 Looking for sensor data CSV...")
    csv_files = list(download_dir.rglob('*.csv'))
    if csv_files:
        sensor_dir = output_dir / "sensor_data"
        sensor_dir.mkdir(parents=True, exist_ok=True)
        for csv_f in csv_files:
            shutil.copy2(csv_f, sensor_dir / csv_f.name)
            print(f"   ✅ {csv_f.name}  →  sensor_data/")
    else:
        print("   ⚠️  No CSV found in archive — place sensor data manually at:")
        print(f"        {output_dir}/sensor_data/Gas_Sensors_Measurements.csv")
    print()

    return n_train, n_val


def create_video_from_images(images_dir: Path, video_path: Path, fps: int = 30):
    """Create an MP4 video from all .jpg images in a directory.
    
    Args:
        images_dir: Directory containing .jpg images
        video_path: Output video file path
        fps: Frames per second (default: 30)
    """
    if cv2 is None:
        print("   ⚠️  opencv-python not installed, skipping video creation")
        print("   Install with: pip install opencv-python")
        return False
    
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"   ⚠️  No .jpg images found in {images_dir}, skipping video creation")
        return False
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"   ⚠️  Could not read {image_files[0]}, skipping video creation")
        return False
    
    h, w = first_frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    
    written = 0
    for img_file in image_files:
        frame = cv2.imread(str(img_file))
        if frame is not None:
            # Resize if dimensions don't match the first frame
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            written += 1
    
    writer.release()
    print(f"   ✅ Video created: {video_path} ({written} frames, {fps} fps, {written/fps:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download a dataset and prepare train/val splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 90/10 split
  python scripts/download_and_prep_data.py "https://www.kaggle.com/api/v1/datasets/download/simplexitypipeline/pipeline-defect-dataset"

  # Custom split ratio (80/20)
  python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --train-ratio 0.8

  # With a specific random seed for reproducibility
  python scripts/download_and_prep_data.py "https://example.com/dataset.zip" --seed 42
        """
    )
    parser.add_argument(
        "dataset_url",
        type=str,
        help="URL to download the dataset from (e.g. a Kaggle dataset URL)"
    )
    # Read default use case from config.json for default output path
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            use_case_id = json.load(f).get("default-use-case", "pipeline_defects_detection")
    else:
        use_case_id = "pipeline_defects_detection"
    default_output = f"datasets/{use_case_id}"

    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Output directory for the prepared dataset (default: {default_output})"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9 = 90%% train, 10%% val)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--keep-download",
        action="store_true",
        help="Keep the raw downloaded files after splitting"
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    download_dir = output_dir / "_raw_download"

    print(DISCLAIMER)
    print("=" * 70)
    print("Dataset — Download & Prepare")
    print("=" * 70)
    print(f"  Use case:      {use_case_id}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Train ratio:   {args.train_ratio:.0%}")
    print(f"  Val ratio:     {1 - args.train_ratio:.0%}")
    print(f"  Random seed:   {args.seed}")
    print()

    # ──────────────────────────────────────────────────────────────────────────
    # Gas detection use case
    # ──────────────────────────────────────────────────────────────────────────
    if use_case_id == "gas_detection":
        # Skip if already prepared
        if output_dir.exists() and (output_dir / "images" / "val").exists():
            print(f"✅ Dataset already available at {output_dir}, skipping download.")
            return

        # Step 1: Download
        download_dataset(args.dataset_url, download_dir, zip_name="gas-dataset.zip")

        # Step 2 + 3: Stratified split and copy
        n_train, n_val = prep_gas_detection(
            download_dir, output_dir, args.train_ratio, args.seed
        )

        # Step 4: Clean up raw download
        if not args.keep_download:
            print("🧹 Cleaning up raw download...")
            shutil.rmtree(download_dir, ignore_errors=True)
            print("   ✅ Raw download removed\n")

        # Summary
        print("=" * 70)
        print("✅ Dataset ready!")
        print("=" * 70)
        print(f"  📁 {output_dir}/images/train/{{class}}/  ({n_train} images, YOLO cls format)")
        print(f"  📁 {output_dir}/images/val/             ({n_val} images, flat for inference)")
        print(f"  📁 {output_dir}/sensor_data/            (sensor CSV)")
        print()
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline defects detection use case (default)
    # ──────────────────────────────────────────────────────────────────────────

    # Check if dataset already exists
    if output_dir.exists() and (output_dir / "images").exists():
        print(f"✅ Dataset already available at {output_dir}, skipping download.")
        return

    # Step 1: Download
    download_dataset(args.dataset_url, download_dir, zip_name="pipeline-defect-dataset.zip")

    # Step 2: Find all image/label pairs in the downloaded data
    print("🔍 Scanning for images and labels...")
    pairs = find_images_and_labels(download_dir)
    print(f"   Found {len(pairs)} images ({sum(1 for _, l in pairs if l)} with labels)\n")

    if not pairs:
        print("❌ No images found in downloaded data. Check the dataset structure.")
        print(f"   Downloaded to: {download_dir}")
        sys.exit(1)

    # Step 3: Split
    print(f"🔀 Splitting dataset (seed={args.seed})...")
    train_pairs, val_pairs = split_dataset(pairs, args.train_ratio, args.seed)
    print(f"   Train: {len(train_pairs)} samples ({len(train_pairs)/len(pairs):.0%})")
    print(f"   Val:   {len(val_pairs)} samples ({len(val_pairs)/len(pairs):.0%})\n")

    # Step 4: Copy to final directories
    print("📂 Organizing dataset...")
    copy_pairs(train_pairs, output_dir / "images" / "train", output_dir / "labels" / "train", "train")
    copy_pairs(val_pairs, output_dir / "images" / "val", output_dir / "labels" / "val", "val")
    print()

    # Step 4b: Create dataset.yaml config
    dataset_yaml_path = output_dir / "dataset.yaml"
    dataset_yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: [Deformation, Obstacle, Rupture, Disconnect, Misalignment, Deposition]\n"
    )
    with open(dataset_yaml_path, 'w') as f:
        f.write(dataset_yaml_content)
    print(f"📋 Created {dataset_yaml_path}")
    print()

    # Step 5: Create video from val images
    print("🎬 Creating video from validation images...")
    video_path = output_dir / "video" / "input.mp4"
    val_images_dir = output_dir / "images" / "val"
    create_video_from_images(val_images_dir, video_path)
    print()

    # Step 6: Clean up raw download
    if not args.keep_download:
        print("🧹 Cleaning up raw download...")
        shutil.rmtree(download_dir, ignore_errors=True)
        print("   ✅ Raw download removed\n")

    # Summary
    print("=" * 70)
    print("✅ Dataset ready!")
    print("=" * 70)
    print(f"  📁 {output_dir}/images/train/  ({len(train_pairs)} images)")
    print(f"  📁 {output_dir}/images/val/    ({len(val_pairs)} images)")
    print(f"  📁 {output_dir}/labels/train/  (YOLO annotations)")
    print(f"  📁 {output_dir}/labels/val/    (YOLO annotations)")
    print(f"  🎬 {output_dir}/video/input.mp4  (video from val images)")
    print()
    print(f"  Dataset config: {output_dir}/dataset.yaml")
    print()


if __name__ == "__main__":
    main()
