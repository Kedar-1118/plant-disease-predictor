"""
training/train_disease_classifier.py
--------------------------------------
Stage 2: Train a crop-specific disease classifier.

One model is trained per crop. Each model classifies diseases
specific to that crop (e.g., tomato → early blight, late blight, etc.)

DATASET STRUCTURE EXPECTED:
    data/raw/train/<crop_name>/<disease_name>/<image.jpg>
    data/raw/val/<crop_name>/<disease_name>/<image.jpg>

Example for tomato:
    data/raw/train/tomato/early_blight/img001.jpg
    data/raw/train/tomato/late_blight/img002.jpg
    data/raw/train/tomato/healthy/img003.jpg

HOW TO RUN:
    # Train for a specific crop:
    python training/train_disease_classifier.py --crop tomato
    python training/train_disease_classifier.py --crop potato
    python training/train_disease_classifier.py --crop corn

    # Train for all crops:
    python training/train_disease_classifier.py --crop all

OUTPUT:
    models/tomato_disease_model.h5
    models/tomato_disease_classes.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path so we can import from utils/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from utils.preprocessing import create_data_generators, get_class_weights, setup_gpu


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_TRAIN_DIR  = "data/raw/train"
BASE_VAL_DIR    = "data/raw/val"
MODELS_DIR      = "models"
EPOCHS_FROZEN   = 10
EPOCHS_FINETUNE = 25
LEARNING_RATE   = 1e-4
FINETUNE_LR     = 1e-5

# Supported crops
SUPPORTED_CROPS = ["tomato", "potato", "corn"]


def build_disease_model(num_classes: int) -> tuple:
    """
    Build a MobileNetV2-based disease classification model.
    Same architecture as the crop classifier but with different output size.

    Args:
        num_classes (int): Number of disease classes for this crop.

    Returns:
        tuple: (model, base_model)
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False  # Freeze initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)   # Larger head for more disease classes
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_for_finetuning(model, base_model, num_layers: int = 40):
    """Unfreeze top layers of base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"  Unfroze top {num_layers} layers for fine-tuning")


def plot_history(history_frozen, history_finetune, crop: str):
    """Plot and save training history for a specific crop model."""
    acc = history_frozen.history["accuracy"] + history_finetune.history["accuracy"]
    val_acc = history_frozen.history["val_accuracy"] + history_finetune.history["val_accuracy"]
    loss = history_frozen.history["loss"] + history_finetune.history["loss"]
    val_loss = history_frozen.history["val_loss"] + history_finetune.history["val_loss"]

    split = len(history_frozen.history["accuracy"])
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc, "b-", label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-", label="Val Accuracy")
    ax1.axvline(x=split, color="green", linestyle="--", label="Fine-tune start")
    ax1.set_title(f"{crop.capitalize()} Disease Classifier — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, loss, "b-", label="Train Loss")
    ax2.plot(epochs, val_loss, "r-", label="Val Loss")
    ax2.axvline(x=split, color="green", linestyle="--", label="Fine-tune start")
    ax2.set_title(f"{crop.capitalize()} Disease Classifier — Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = f"{MODELS_DIR}/{crop}_training_history.png"
    plt.savefig(save_path)
    print(f"  Saved training plot → {save_path}")
    plt.close()


def train_for_crop(crop: str):
    """
    Train a disease classifier for a specific crop.

    Args:
        crop (str): Crop name (e.g., 'tomato').
    """
    print("\n" + "=" * 60)
    print(f"  Training Disease Classifier: {crop.upper()}")
    print("=" * 60)

    train_dir = os.path.join(BASE_TRAIN_DIR, crop)
    val_dir   = os.path.join(BASE_VAL_DIR, crop)

    # ── Validate directories ───────────────────────────────────────────────
    if not os.path.exists(train_dir):
        print(f"\n[ERROR] Training directory not found: {train_dir}")
        print(f"Please organize your {crop} disease images as:")
        print(f"  data/raw/train/{crop}/<disease_name>/<image.jpg>")
        return

    # ── Create data generators ─────────────────────────────────────────────
    print(f"\n[1/5] Loading {crop} disease dataset...")
    train_gen, val_gen = create_data_generators(train_dir, val_dir)

    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    print(f"      Diseases found ({num_classes}): {class_names}")
    print(f"      Training samples: {train_gen.samples}")
    print(f"      Validation samples: {val_gen.samples}")

    # Save class mapping
    os.makedirs(MODELS_DIR, exist_ok=True)
    class_map_path = f"{MODELS_DIR}/{crop}_disease_classes.json"
    with open(class_map_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"      Saved class mapping → {class_map_path}")

    # ── Class weights ──────────────────────────────────────────────────────
    print(f"\n[2/5] Computing class weights...")
    class_weights = get_class_weights(train_gen)

    # ── Build model ────────────────────────────────────────────────────────
    print(f"\n[3/5] Building model for {num_classes} disease classes...")
    model, base_model = build_disease_model(num_classes)

    model_save_path = f"{MODELS_DIR}/{crop}_disease_model.h5"

    # ── Phase 1: Train classification head ────────────────────────────────
    print(f"\n[4/5] Phase 1: Training head ({EPOCHS_FROZEN} epochs)...")
    callbacks_phase1 = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    history_frozen = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # ── Phase 2: Fine-tune ─────────────────────────────────────────────────
    print(f"\n[5/5] Phase 2: Fine-tuning ({EPOCHS_FINETUNE} epochs)...")
    unfreeze_for_finetuning(model, base_model, num_layers=40)

    callbacks_phase2 = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, verbose=1)
    ]

    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n  [{crop.upper()}] Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  [{crop.upper()}] Model saved → {model_save_path}")

    # ── Plot history ───────────────────────────────────────────────────────
    plot_history(history_frozen, history_finetune, crop)


def main():
    # Configure GPU
    setup_gpu()

    parser = argparse.ArgumentParser(
        description="Train crop-specific disease classifiers (Stage 2)"
    )
    parser.add_argument(
        "--crop",
        type=str,
        required=True,
        choices=SUPPORTED_CROPS + ["all"],
        help="Crop to train for. Use 'all' to train all crops."
    )
    args = parser.parse_args()

    if args.crop == "all":
        print("\n[INFO] Training disease classifiers for ALL crops...")
        for crop in SUPPORTED_CROPS:
            train_for_crop(crop)
    else:
        train_for_crop(args.crop)

    print("\n[DONE] Disease classifier training complete!")


if __name__ == "__main__":
    main()
