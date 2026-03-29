"""
training/export_tflite.py
--------------------------
Convert trained .h5 Keras models to TensorFlow Lite format for
mobile/edge deployment.

Supports optional INT8 quantization for further size reduction.

HOW TO RUN:
    # Export all models (default float16):
    python training/export_tflite.py

    # Export with INT8 quantization:
    python training/export_tflite.py --quantize int8

    # Export a specific model:
    python training/export_tflite.py --model models/crop_model.h5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from utils.preprocessing import setup_gpu

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def export_to_tflite(
    h5_path: str,
    quantize: str = "float16",
) -> str:
    """
    Convert a Keras .h5 model to TFLite.

    Args:
        h5_path:  Path to the .h5 model file.
        quantize: Quantization mode — "none", "float16", or "int8".

    Returns:
        Path to the saved .tflite file.
    """
    print(f"\n  Loading: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  Quantization: float16")
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  Quantization: INT8 (dynamic range)")
    else:
        print("  Quantization: none (float32)")

    tflite_model = converter.convert()

    # Save .tflite alongside the .h5 file
    tflite_path = h5_path.replace(".h5", ".tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Print size comparison
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    tflite_size = len(tflite_model) / (1024 * 1024)
    reduction = (1 - tflite_size / h5_size) * 100

    print(f"  H5 size:     {h5_size:.1f} MB")
    print(f"  TFLite size: {tflite_size:.1f} MB")
    print(f"  Reduction:   {reduction:.1f}%")
    print(f"  Saved: {tflite_path}")

    return tflite_path


def main():
    setup_gpu()

    parser = argparse.ArgumentParser(description="Export Keras models to TFLite")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a specific .h5 model. If not set, exports all models in models/.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "float16", "int8"],
        default="float16",
        help="Quantization mode (default: float16).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  TFLite Model Export")
    print("=" * 60)

    if args.model:
        if not os.path.exists(args.model):
            print(f"[ERROR] Model not found: {args.model}")
            sys.exit(1)
        export_to_tflite(args.model, args.quantize)
    else:
        h5_files = sorted(MODELS_DIR.glob("*.h5"))
        if not h5_files:
            print(f"[ERROR] No .h5 files found in {MODELS_DIR}")
            sys.exit(1)

        print(f"  Found {len(h5_files)} model(s) to export.")
        for h5_path in h5_files:
            export_to_tflite(str(h5_path), args.quantize)

    print("\n[DONE] TFLite export complete!")


if __name__ == "__main__":
    main()
