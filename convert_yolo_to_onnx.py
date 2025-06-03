#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics library not found. Please install it with: pip install ultralytics")
    sys.exit(1)

def convert_yolo_to_onnx(
    pytorch_model_path: Path,
    onnx_model_path: Path | None = None,
    imgsz_height: int | None = None,
    imgsz_width: int | None = None
) -> None:
    """
    Converts a YOLOv11 PyTorch model to ONNX format.

    Args:
        pytorch_model_path: Path to the input PyTorch model (.pt file).
        onnx_model_path: Path to save the output ONNX model.
                         If None, it defaults to the same name as the input model
                         but with a .onnx extension, saved in the same directory.
        imgsz_height: The height of the input image for the model.
        imgsz_width: The width of the input image for the model.
    """
    if not pytorch_model_path.is_file():
        print(f"Error: Input model file not found at {pytorch_model_path}")
        return

    if onnx_model_path is None:
        onnx_model_path = pytorch_model_path.with_suffix(".onnx")
    
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    export_params = {"format": "onnx", "simplify": True, "dynamic": False, "opset": 12}

    if imgsz_height and imgsz_width:
        export_params["imgsz"] = (imgsz_height, imgsz_width)
        print(f"Using specified input size: {imgsz_height}x{imgsz_width}")
    elif imgsz_height or imgsz_width:
        print("Warning: Both imgsz_height and imgsz_width must be provided. Using default export size.")
    else:
        print("Using default export input size (likely 640x640 or model's native).")


    try:
        # Load the YOLOv11 model
        model = YOLO(str(pytorch_model_path))

        # Export the model to ONNX format
        model.export(**export_params)

        print(f"Successfully converted {pytorch_model_path} to {onnx_model_path}")
        print(f"You can now use the ONNX model at: {onnx_model_path.resolve()}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a YOLOv11 PyTorch model to ONNX format.")
    parser.add_argument(
        "input_model",
        type=str,
        help="Path to the input YOLOv11 PyTorch model file (.pt)."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default=None,
        help="Path to save the output ONNX model. "
             "If not provided, it defaults to the input model name with a .onnx extension."
    )
    parser.add_argument(
        "--imgsz_height",
        type=int,
        default=None,
        help="Input image height for ONNX export (e.g., 192)."
    )
    parser.add_argument(
        "--imgsz_width",
        type=int,
        default=None,
        help="Input image width for ONNX export (e.g., 128)."
    )

    args = parser.parse_args()

    input_path = Path(args.input_model)
    output_path = Path(args.output_model) if args.output_model else None

    convert_yolo_to_onnx(input_path, output_path, args.imgsz_height, args.imgsz_width) 