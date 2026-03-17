"""Image preprocessing utilities for the Plant Disease Predictor."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LOGGER = logging.getLogger(__name__)
TESTED_TF_VERSION_PREFIXES = ("2.15",)
MIXED_PRECISION_PATCH_APPLIED = False


if not hasattr(mixed_precision.Policy, "quantization_mode"):
    mixed_precision.Policy.quantization_mode = property(lambda self: None)
    MIXED_PRECISION_PATCH_APPLIED = True


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_runtime_compatibility() -> dict:
    tensorflow_version = getattr(tf, "__version__", "unknown")
    is_tested_version = tensorflow_version.startswith(TESTED_TF_VERSION_PREFIXES)
    warnings = []

    if not is_tested_version:
        warnings.append(
            f"TensorFlow {tensorflow_version} is outside the tested range "
            f"({', '.join(TESTED_TF_VERSION_PREFIXES)}x)."
        )

    if MIXED_PRECISION_PATCH_APPLIED:
        warnings.append(
            "Applied a mixed-precision compatibility patch while loading TensorFlow policies."
        )

    return {
        "tensorflow_version": tensorflow_version,
        "is_tested_version": is_tested_version,
        "mixed_precision_patch_applied": MIXED_PRECISION_PATCH_APPLIED,
        "warnings": warnings,
    }


def setup_gpu(enable_mixed_precision: bool = False):
    compatibility = get_runtime_compatibility()
    LOGGER.info("[GPU] Configuring GPU settings")
    for warning in compatibility["warnings"]:
        LOGGER.warning(warning)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if enable_mixed_precision:
                mixed_precision.set_global_policy("mixed_float16")
                LOGGER.info("Mixed precision enabled (float16).")

            logical_gpus = tf.config.list_logical_devices("GPU")
            LOGGER.info(
                "Found %s physical GPU(s), %s logical GPU(s); memory growth enabled.",
                len(gpus),
                len(logical_gpus),
            )
        except RuntimeError as exc:
            LOGGER.warning("GPU configuration error: %s", exc)
    else:
        LOGGER.info("No GPU detected. Running on CPU.")


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_opencv(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def create_data_generators(train_dir: str, val_dir: str):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=False,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator


def get_class_weights(generator) -> dict:
    from sklearn.utils.class_weight import compute_class_weight

    labels = generator.classes
    class_indices = list(generator.class_indices.values())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_indices),
        y=labels,
    )

    return dict(enumerate(weights))
