#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics library not found. Please install it with: pip install ultralytics")
    sys.exit(1)

def convert_yolo_to_onnx(pytorch_model_path: Path, onnx_model_path: Path | None = None) -> None:
    """
    Converts a YOLOv11 PyTorch model to ONNX format.

    Args:
        pytorch_model_path: Path to the input PyTorch model (.pt file).
        onnx_model_path: Path to save the output ONNX model.
                         If None, it defaults to the same name as the input model
                         but with a .onnx extension, saved in the same directory.
    """
    if not pytorch_model_path.is_file():
        print(f"Error: Input model file not found at {pytorch_model_path}")
        return

    if onnx_model_path is None:
        onnx_model_path = pytorch_model_path.with_suffix(".onnx")
    
    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load the YOLOv11 model
        model = YOLO(str(pytorch_model_path))

        # Export the model to ONNX format
        # It's generally a good idea to use simplify=True for better compatibility and performance
        # and dynamic=True if you need to support variable input sizes.
        # opset can be specified if needed, otherwise, it uses the latest supported.
        model.export(format="onnx", simplify=True, dynamic=False, opset=12) # Using opset 12 for wider compatibility

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

    args = parser.parse_args()

    input_path = Path(args.input_model)
    output_path = Path(args.output_model) if args.output_model else None

    convert_yolo_to_onnx(input_path, output_path) 