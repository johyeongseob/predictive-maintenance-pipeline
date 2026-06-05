#!/usr/bin/env python3
"""
Convert YOLO PyTorch model (.pt) to OpenVINO format (.xml/.bin)

This script performs a two-stage conversion:
1. Export PyTorch model to ONNX format (with simplification)
2. Convert ONNX to OpenVINO IR format using ovc tool

The conversion maintains FP32 precision and produces a model with output shape [1,10,8400]
(raw YOLO format). Post-processing (NMS, confidence filtering) is handled by the inference code.

Important:
- GPU inference requires FP32 precision (automatically configured in run_inference_direct.py)
- The model outputs raw predictions without NMS baked in
- Ultralytics 8.3+ uses openvino format (not openvino_ovc)

Environment Setup:
- Use 'pace' conda environment for model conversion
- This environment includes: torch, ultralytics, openvino, onnx, onnxslim

Usage:
    conda activate pace
    python setup/convert_to_openvino.py
    # Or with custom paths:
    python setup/convert_to_openvino.py --input models/pt_models/custom.pt --output models/ov_models/custom

Requirements:
    - ultralytics >= 8.3.0
    - openvino >= 2025.3.0
    - torch >= 2.0.0
    - onnx >= 1.19.0
    - onnxslim >= 0.1.0
"""

import argparse
import os
import sys
import json
from pathlib import Path

try:
    from ultralytics import YOLO
    import openvino as ov
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


def convert_to_openvino(
    model_path: str,
    output_dir: str,
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True
):
    """
    Convert YOLO model to OpenVINO format
    
    Args:
        model_path: Path to the .pt model file
        output_dir: Directory to save the OpenVINO model
        imgsz: Input image size (default: 640)
        half: Use FP16 precision (default: False)
        simplify: Simplify ONNX model (default: True)
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    
    # Get model info
    print(f"Model type: {model.task}")
    print(f"Model input size: {imgsz}")
    print(f"⚠️  Note: Exporting with NMS included in the model")
    
    # Export to OpenVINO directly to the output directory
    print("\nExporting to OpenVINO format...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Export using ultralytics built-in export
        # The export will create files directly in the output directory
        export_path = model.export(
            format='openvino',
            imgsz=imgsz,
            half=half,
            simplify=simplify,
            dynamic=False,
            opset=13
        )
        
        print(f"\n✓ Model exported successfully!")
        print(f"Exported to: {export_path}")
        
        # Ultralytics exports to a folder next to the .pt file
        # We need to move it to our desired output location
        exported_folder = Path(export_path)
        
        # If exported folder is different from desired output, move the files
        if exported_folder.resolve() != output_path.resolve():
            import shutil
            
            # Copy all files to output directory and rename .xml/.bin to best.xml/best.bin
            for file in exported_folder.glob("*"):
                # Rename model files to best.xml/best.bin
                if file.suffix in ['.xml', '.bin']:
                    dest = output_path / f"best{file.suffix}"
                else:
                    dest = output_path / file.name
                shutil.copy(file, dest)
                print(f"Copied: {dest}")
            
            # Remove the temporary export folder
            shutil.rmtree(exported_folder)
            print(f"✓ Cleaned up temporary export folder")
        
        print(f"\n✓ OpenVINO model files ready in: {output_path}")
        print(f"  - best.xml (model architecture)")
        print(f"  - best.bin (model weights)")
        
        # Print model info
        print("\nVerifying OpenVINO model...")
        try:
            core = ov.Core()
            model_xml = str(output_path / "best.xml")
            ov_model = core.read_model(model_xml)
            
            # Get input info
            print(f"Input shape: {ov_model.inputs[0].shape}")
            
            # Try to get names, but don't fail if unavailable
            try:
                input_names = [inp.get_any_name() for inp in ov_model.inputs]
                print(f"Model inputs: {input_names}")
            except Exception:
                print(f"Model inputs: {len(ov_model.inputs)} input(s)")
            
            try:
                output_names = [out.get_any_name() for out in ov_model.outputs]
                print(f"Model outputs: {output_names}")
            except Exception:
                print(f"Model outputs: {len(ov_model.outputs)} output(s)")
                
            print(f"Output shape: {ov_model.outputs[0].shape}")
        except Exception as e:
            print(f"Note: Could not fully verify model details: {e}")
            print("However, model files were created successfully.")
        
        return str(output_path)
            
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO PyTorch model to OpenVINO IR")
    
    # Read use-case-id from config.json for defaults
    try:
        with open('config.json', 'r') as f:
            main_config = json.load(f)
        use_case_id = main_config.get('default-use-case', 'pipeline_defects_detection')
    except Exception as e:
        print(f"Warning: Could not read config.json ({e}), using default paths")
        use_case_id = 'pipeline_defects_detection'
    
    default_input = f"models/pt_models/{use_case_id}.pt"
    default_output = f"models/ov_models/{use_case_id}"
    
    parser.add_argument(
        "--input",
        type=str,
        default=default_input,
        help=f"Path to YOLO PyTorch model (default: {default_input})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help=f"Output directory for OpenVINO model (default: {default_output})"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 precision"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLO to OpenVINO Conversion")
    print("=" * 60)
    
    try:
        convert_to_openvino(
            model_path=args.input,
            output_dir=args.output,
            imgsz=args.imgsz,
            half=args.half
        )
        
        print("\n" + "=" * 60)
        print("✓ Conversion completed successfully!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Create model_proc.json in {args.output}")
        print(f"2. Run inference using run_inference_oep.py")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
