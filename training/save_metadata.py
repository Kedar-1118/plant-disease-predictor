"""
training/save_metadata.py
--------------------------
Saves a JSON metadata file alongside each trained .h5 model, recording
hyperparameters, accuracy, dataset info, and timestamps for reproducibility.

Example output:
    models/tomato_disease_model_metadata.json
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf


def save_model_metadata(
    model_path: str,
    val_accuracy: float,
    val_loss: float,
    epochs_run: int,
    num_classes: int,
    class_names: list[str],
    hyperparameters: dict,
    dataset_info: dict | None = None,
    notes: str = "",
) -> str:
    """
    Save a metadata JSON file next to the model .h5 file.

    Args:
        model_path:      Path to the saved .h5 model file.
        val_accuracy:    Final validation accuracy (0-1 scale).
        val_loss:        Final validation loss.
        epochs_run:      Total epochs actually run (after early stopping).
        num_classes:     Number of output classes.
        class_names:     List of class names.
        hyperparameters: Dict with lr, batch_size, etc.
        dataset_info:    Optional dict with train_samples, val_samples, etc.
        notes:           Optional free-text notes.

    Returns:
        Path to the saved metadata JSON file.
    """
    metadata = {
        "model_file": Path(model_path).name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "accuracy": round(val_accuracy, 4),
            "loss": round(val_loss, 4),
        },
        "training": {
            "epochs_run": epochs_run,
            "num_classes": num_classes,
            "class_names": class_names,
            "hyperparameters": hyperparameters,
        },
        "environment": {
            "tensorflow_version": tf.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0,
        },
    }

    if dataset_info:
        metadata["dataset"] = dataset_info

    if notes:
        metadata["notes"] = notes

    # Save alongside the model file: model.h5 → model_metadata.json
    meta_path = model_path.replace(".h5", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved metadata → {meta_path}")
    return meta_path
