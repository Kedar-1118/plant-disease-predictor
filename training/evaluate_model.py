"""
training/evaluate_model.py
---------------------------
Evaluate any trained model (.h5) and generate:
  - Accuracy & loss on test/validation data
  - Classification report (precision, recall, F1)
  - Confusion matrix heatmap (saved as PNG)

HOW TO RUN:
    # Evaluate crop classifier:
    python training/evaluate_model.py --model models/crop_model.h5 --data data/raw/val

    # Evaluate tomato disease classifier:
    python training/evaluate_model.py --model models/tomato_disease_model.h5 --data data/raw/val/tomato
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from utils.preprocessing import setup_gpu

# Fix for Keras mixed precision model loading bug
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
if not hasattr(mixed_precision.Policy, 'quantization_mode'):
    mixed_precision.Policy.quantization_mode = property(lambda self: None)


IMG_SIZE   = (224, 224)
BATCH_SIZE = 32


def evaluate(model_path: str, data_dir: str):
    """
    Load a model and evaluate it on the given data directory.

    Args:
        model_path (str): Path to the .h5 model file.
        data_dir   (str): Path to the evaluation data directory.
                          Expected structure: data_dir/class_name/image.jpg
    """
    print("=" * 60)
    print(f"  Model Evaluation")
    print("=" * 60)
    print(f"  Model : {model_path}")
    print(f"  Data  : {data_dir}")

    # ── Load model ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    model = load_model(model_path, compile=False)
    # Recompile for evaluation
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    print(f"      Model loaded. Input shape: {model.input_shape}")

    # ── Load evaluation data ───────────────────────────────────────────────
    print("\n[2/4] Loading evaluation data...")
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # IMPORTANT: Don't shuffle for correct label alignment
    )

    class_names = list(generator.class_indices.keys())
    num_classes = len(class_names)
    print(f"      Classes ({num_classes}): {class_names}")
    print(f"      Total samples: {generator.samples}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\n[3/4] Running evaluation...")
    loss, accuracy = model.evaluate(generator, verbose=1)
    print(f"\n  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  Overall Loss     : {loss:.4f}")

    # ── Generate predictions for detailed metrics ──────────────────────────
    print("\n[4/4] Generating detailed metrics...")

    # Reset generator to start from beginning
    generator.reset()
    predictions = model.predict(generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes

    # Classification report
    print("\n" + "─" * 60)
    print("  Classification Report")
    print("─" * 60)
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # ── Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(max(8, num_classes * 1.5), max(6, num_classes * 1.2)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix — {os.path.basename(model_path)}", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save confusion matrix
    save_path = model_path.replace(".h5", "_confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  Confusion matrix saved → {save_path}")
    plt.show()

    print("\n[DONE] Evaluation complete!")


def main():
    # Configure GPU
    setup_gpu()

    parser = argparse.ArgumentParser(description="Evaluate a trained plant disease model")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model file")
    parser.add_argument("--data",  type=str, required=True, help="Path to evaluation data directory")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[ERROR] Data directory not found: {args.data}")
        sys.exit(1)

    evaluate(args.model, args.data)


if __name__ == "__main__":
    main()
