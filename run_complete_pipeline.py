#!/usr/bin/env python3
"""
Complete PACE Pipeline - Master Script
Runs: Inference → SQLite → Agent Orchestration → Visualization

This script executes the entire pipeline end-to-end:
1. Run YOLO inference on images
2. Push detections to SQLite
3. Query agents for analysis (Policy, Analysis, Evidence)
4. Generate visualizations with bounding boxes
"""

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
import yaml
from scripts.clean_slate import clean_database, clean_outputs


def run_command(cmd: list, description: str, check_output: bool = False):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        if check_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
        print(f"✅ {description} - SUCCESS\n")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        sys.exit(1)


def get_use_case_id_from_config():
    """Read the default use-case-id from config.json."""
    try:
        with open("config.json", "r") as f:
            main_config = json.load(f)
        return main_config.get("default-use-case", "pipeline_defects_detection")
    except Exception:
        return "pipeline_defects_detection"


def load_config(config_path: str = None):
    """Load configuration file."""
    try:
        if config_path is None:
            use_case_id = get_use_case_id_from_config()
            config_path = f"config/{use_case_id}/config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run complete PACE pipeline: Inference → SQLite → Agents → Visualization"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to process (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU"],
        help="OpenVINO device (default: GPU)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: reads from config.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: out/<use-case-id>)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (overrides image mode; omit for image-based inference)"
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="Run inference every Nth frame in video mode (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Load config (needed to determine use_case_id for output dir)
    config = load_config(args.config)
    
    # Determine use_case_id
    use_case_id = get_use_case_id_from_config()
    
    # Create output directory: out/<use-case-id>/
    out_dir = Path(args.output_dir) if args.output_dir else Path("out") / use_case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if config:
        print(f"✅ Loaded config: config/{use_case_id}/config.yaml")
        print(f"   Use case: {config.get('display_text', use_case_id)}")
        print(f"   Modality: {config.get('modality', 'image')}")
        print(f"   Backend: SQL")
        print(f"   Classes: {config.get('nc', 'unknown')}")
        print(f"   Device: {args.device}")
        print(f"   Inference engine: DLStreamer/OEP")
        if args.video:
            print(f"   Video input: {args.video}")
    
    # Clear output directory if enabled in config
    if config:
        sqlite_cfg = config.get('sqlite', {})
        clear_outputs_flag = sqlite_cfg.get('clear_outputs', True)
        
        if clear_outputs_flag and out_dir.exists():
            try:
                print()
                clean_outputs(out_dir=str(out_dir))
                print()
            except Exception as e:
                print(f"⚠️  Could not clear output directory: {e}\n")
        else:
            print(f"\n📌 Keeping existing outputs (clear_outputs: false)\n")
    
    # Clear SQLite database if enabled in config
    if config:
        sqlite_cfg = config.get('sqlite', {})
        clear_on_run = sqlite_cfg.get('clear_on_run', True)
        
        if clear_on_run:
            try:
                db_path = sqlite_cfg.get('db_path', 'out/sql_data/detections.db')
                clean_database(db_path=db_path)
                print()
            except Exception as e:
                print(f"⚠️  Could not clear SQL database: {e}\n")
        else:
            print(f"📌 Keeping existing SQL data (clear_on_run: false)\n")
    
    start_time = time.time()
    
    # ==========================================================================
    # STEP 1: Run Inference
    # ==========================================================================
    detections_file = out_dir / "detections.jsonl"
    
    inference_cmd = [
        "python", "run_inference_oep.py",
        "--device", args.device,
        "--output", str(detections_file)
    ]
    if args.video:
        inference_cmd.extend(["--video", args.video])
        if args.inference_interval > 1:
            inference_cmd.extend(["--inference-interval", str(args.inference_interval)])
        input_desc = f"video ({args.video})"
    else:
        inference_cmd.extend(["--num-images", str(args.num_images)])
        input_desc = f"{args.num_images} images"

    run_command(
        inference_cmd,
        f"Running YOLO inference on {input_desc} via DLStreamer/OEP (with SQL push)"
    )
    
    # ==========================================================================
    # STEP 2: Verify Database Data
    # ==========================================================================
    run_command(
        ["python", "-m", "scripts.read_sqlite_samples", "--limit", "5"],
        "Verifying SQLite data (showing 5 samples)"
    )
    
    # ==========================================================================
    # STEP 3: Run Agent Orchestration
    # ==========================================================================
    agent_cmd = [
        "python", "-m", "scripts.run_agent_orchestration"
    ]
    
    # Set PYTHONPATH for agent imports
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path.cwd())
    
    print(f"\n{'='*80}")
    print(f"STEP: Running Agent Orchestration (processing all detections)")
    print(f"{'='*80}")
    print(f"Command: {' '.join(agent_cmd)}\n")
    
    try:
        subprocess.run(agent_cmd, check=True, env=env)
        print(f"✅ Agent Orchestration - SUCCESS\n")
        
        # Show generated reports
        agent_dir = out_dir / "agent"
        if agent_dir.exists():
            reports = list(agent_dir.glob("*.txt"))
            if reports:
                print(f"📄 Generated {len(reports)} agent reports in {agent_dir}/:")
                for report in sorted(reports)[:5]:  # Show first 5
                    print(f"   - {report.name}")
                if len(reports) > 5:
                    print(f"   ... and {len(reports) - 5} more")
    except subprocess.CalledProcessError as e:
        print(f"❌ Agent Orchestration - FAILED")
        print("   Continuing to visualization...")
    
    # ==========================================================================
    # STEP 4: Summary (visualization done by gvawatermark in DLStreamer pipeline)
    # ==========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE ✅")
    print("="*80)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"\nGenerated files:")
    print(f"  • Detections (JSONL): {detections_file}")
    if config:
        sqlite_cfg = config.get('sqlite', {})
        print(f"  • Detections (SQLite): {sqlite_cfg.get('db_path', 'out/sql_data/detections.db')}")
    print(f"  • Agent reports: {out_dir}/agent/*.txt")
    viz_dir = out_dir / "viz"
    if viz_dir.exists():
        viz_count = len(list(viz_dir.glob("*.jpg")))
        if viz_count:
            print(f"  • Annotated frames: {viz_dir}/ ({viz_count} images)")
    if args.video:
        annotated = out_dir / "annotated_output.mp4"
        if annotated.exists():
            print(f"  • Annotated video: {annotated}")
    print("\nNext steps:")
    print(f"  • Review agent reports in {out_dir}/agent/")
    if viz_dir.exists() and any(viz_dir.glob("*.jpg")):
        print(f"  • Browse annotated frames in {viz_dir}/ (filenames match SQL frame_id)")
    if args.video:
        print(f"  • View annotated video: {out_dir}/annotated_output.mp4")
    print("  • Query SQLite: python -m scripts.read_sqlite_samples --limit 10")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
