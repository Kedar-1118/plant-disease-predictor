"""Two-stage prediction pipeline for the Plant Disease Predictor."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from utils.preprocessing import (
    get_runtime_compatibility,
    load_and_preprocess_image,
    setup_gpu,
)


setup_gpu()


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CROP_MODEL_PATH = MODELS_DIR / "crop_model.h5"
CROP_CLASSES_PATH = MODELS_DIR / "crop_classes.json"
TREATMENTS_PATH = BASE_DIR / "data" / "treatments.json"
CONFIDENCE_THRESHOLD = 0.60

_crop_model = None
_disease_models = {}
_crop_classes = None
_disease_classes = {}
_treatments = None


class PredictionError(RuntimeError):
    code = "prediction_error"


class InvalidImageError(PredictionError):
    code = "invalid_image"


class ModelNotReadyError(PredictionError):
    code = "model_not_ready"


class UnsupportedCropError(PredictionError):
    code = "unsupported_crop"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_supported_crops() -> list[str]:
    crops = []
    for classes_path in MODELS_DIR.glob("*_disease_classes.json"):
        crop = classes_path.name.replace("_disease_classes.json", "")
        model_path = MODELS_DIR / f"{crop}_disease_model.h5"
        if model_path.exists():
            crops.append(crop)
    return sorted(crops)


def startup_check() -> dict:
    missing_files = []
    warnings = []

    for required_path in (CROP_MODEL_PATH, CROP_CLASSES_PATH, TREATMENTS_PATH):
        if not required_path.exists():
            missing_files.append(str(required_path))

    supported_crops = get_supported_crops()
    if not supported_crops:
        warnings.append("No crop-specific disease models were found in the models directory.")

    compatibility = get_runtime_compatibility()
    warnings.extend(compatibility["warnings"])

    return {
        "ready": not missing_files,
        "missing_files": missing_files,
        "supported_crops": supported_crops,
        "compatibility": compatibility,
        "warnings": warnings,
    }


def _load_treatments() -> dict:
    global _treatments
    if _treatments is None:
        if not TREATMENTS_PATH.exists():
            raise ModelNotReadyError(f"Treatment database not found at '{TREATMENTS_PATH}'.")
        _treatments = _read_json(TREATMENTS_PATH)
    return _treatments


def _load_crop_model():
    global _crop_model, _crop_classes

    if _crop_model is None:
        if not CROP_MODEL_PATH.exists():
            raise ModelNotReadyError(
                f"Crop model not found at '{CROP_MODEL_PATH}'. "
                "Please train the model first: python training/train_crop_classifier.py"
            )
        _crop_model = tf.keras.models.load_model(CROP_MODEL_PATH, compile=False)

    if _crop_classes is None:
        if not CROP_CLASSES_PATH.exists():
            raise ModelNotReadyError(
                f"Crop class mapping not found at '{CROP_CLASSES_PATH}'."
            )
        _crop_classes = _read_json(CROP_CLASSES_PATH)

    return _crop_model, _crop_classes


def _load_disease_model(crop: str):
    global _disease_models, _disease_classes

    if crop not in _disease_models:
        model_path = MODELS_DIR / f"{crop}_disease_model.h5"
        classes_path = MODELS_DIR / f"{crop}_disease_classes.json"
        supported_crops = get_supported_crops()

        if crop not in supported_crops:
            raise UnsupportedCropError(
                f"No disease model is currently available for detected crop '{crop}'."
            )

        if not model_path.exists():
            raise ModelNotReadyError(
                f"Disease model for '{crop}' not found at '{model_path}'. "
                f"Please train it: python training/train_disease_classifier.py --crop {crop}"
            )

        if not classes_path.exists():
            raise ModelNotReadyError(
                f"Disease class mapping for '{crop}' not found at '{classes_path}'."
            )

        _disease_models[crop] = tf.keras.models.load_model(model_path, compile=False)
        _disease_classes[crop] = _read_json(classes_path)

    return _disease_models[crop], _disease_classes[crop]


def predict_with_tta(
    model: "tf.keras.Model",
    img_array: np.ndarray,
    n_augments: int = 5,
) -> np.ndarray:
    """
    Test-Time Augmentation: predict on multiple augmented versions of the
    image and average the results for more robust predictions.
    """
    import tensorflow as tf

    predictions = [model.predict(img_array, verbose=0)[0]]

    for _ in range(n_augments):
        aug = tf.image.random_flip_left_right(img_array)
        aug = tf.image.random_brightness(aug, max_delta=0.1)
        predictions.append(model.predict(aug.numpy(), verbose=0)[0])

    return np.mean(predictions, axis=0)


def predict_crop(image_path: str, use_tta: bool = False) -> dict:
    model, class_indices = _load_crop_model()
    img_array = load_and_preprocess_image(image_path)

    if use_tta:
        predictions = predict_with_tta(model, img_array)
    else:
        predictions = model.predict(img_array, verbose=0)[0]

    index_to_class = {v: k for k, v in class_indices.items()}

    predicted_index = int(np.argmax(predictions))
    predicted_crop = index_to_class[predicted_index]
    confidence = float(predictions[predicted_index])

    all_predictions = {
        index_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return {
        "crop": predicted_crop,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "_model": model,
        "_img_array": img_array,
        "_predicted_index": predicted_index,
    }


def predict_disease(crop: str, image_path: str, use_tta: bool = False) -> dict:
    model, class_indices = _load_disease_model(crop)
    img_array = load_and_preprocess_image(image_path)

    if use_tta:
        predictions = predict_with_tta(model, img_array)
    else:
        predictions = model.predict(img_array, verbose=0)[0]

    index_to_class = {v: k for k, v in class_indices.items()}

    predicted_index = int(np.argmax(predictions))
    predicted_disease = index_to_class[predicted_index]
    confidence = float(predictions[predicted_index])

    all_predictions = {
        index_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "all_predictions": all_predictions,
        "_model": model,
        "_img_array": img_array,
        "_predicted_index": predicted_index,
    }


def get_treatment(crop: str, disease: str) -> dict:
    treatments = _load_treatments()
    crop_key = crop.lower().replace(" ", "_")
    disease_key = disease.lower().replace(" ", "_")

    crop_data = treatments.get(crop_key, {})
    treatment_data = crop_data.get(disease_key)

    if treatment_data is None:
        return {
            "display_name": disease.replace("_", " ").title(),
            "description": "Disease information not available in the database.",
            "symptoms": [],
            "treatment": ["Consult a local agricultural expert for treatment advice."],
            "prevention": ["Practice good agricultural hygiene and crop rotation."],
            "severity": "Unknown",
        }

    return treatment_data


def full_pipeline(image_path: str, use_tta: bool = False) -> dict:
    """
    Full two-stage prediction with optional TTA and Grad-CAM explainability.

    Args:
        image_path: Path to the leaf image.
        use_tta:    Enable Test-Time Augmentation (slower but more robust).

    Returns:
        Dict with crop, disease, confidences, treatment, gradcam_image, and latency_ms.
    """
    if not os.path.exists(image_path):
        raise InvalidImageError(f"Image not found: {image_path}")

    pipeline_start = time.perf_counter()

    crop_result = predict_crop(image_path, use_tta=use_tta)
    result = {
        "success": True,
        "crop": crop_result["crop"],
        "crop_confidence": round(crop_result["confidence"] * 100, 2),
        "crop_all_predictions": {
            key: round(value * 100, 2)
            for key, value in crop_result["all_predictions"].items()
        },
        "low_confidence_warning": crop_result["confidence"] < CONFIDENCE_THRESHOLD,
    }

    if result["low_confidence_warning"]:
        result["warning_message"] = (
            f"Low confidence ({result['crop_confidence']}%) in crop detection. "
            "The image may not be a clear leaf photo."
        )

    disease_result = predict_disease(crop_result["crop"], image_path, use_tta=use_tta)
    result["disease"] = disease_result["disease"]
    result["disease_confidence"] = round(disease_result["confidence"] * 100, 2)
    result["disease_all_predictions"] = {
        key: round(value * 100, 2)
        for key, value in disease_result["all_predictions"].items()
    }
    result["treatment"] = get_treatment(crop_result["crop"], disease_result["disease"])

    # ── Grad-CAM Explainability ───────────────────────────────────────────
    try:
        from utils.gradcam import generate_gradcam_overlay_b64

        gradcam_b64 = generate_gradcam_overlay_b64(
            model=disease_result["_model"],
            img_array=disease_result["_img_array"],
            original_image_path=image_path,
            pred_index=disease_result["_predicted_index"],
        )
        result["gradcam_image"] = gradcam_b64
    except Exception as exc:
        LOGGER.warning("Grad-CAM generation failed: %s", exc)
        result["gradcam_image"] = None

    # ── Latency ───────────────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
    result["latency_ms"] = latency_ms
    LOGGER.info(
        "Prediction complete in %.1fms | crop=%s (%.1f%%) | disease=%s (%.1f%%)",
        latency_ms,
        result["crop"],
        result["crop_confidence"],
        result["disease"],
        result["disease_confidence"],
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-stage plant disease prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    try:
        result = full_pipeline(args.image)
        print("=" * 60)
        print("  PREDICTION RESULT")
        print("=" * 60)
        print(f"  Crop     : {result['crop'].capitalize()} ({result['crop_confidence']}%)")
        print(f"  Disease  : {result['treatment']['display_name']} ({result['disease_confidence']}%)")
        print(f"  Severity : {result['treatment']['severity']}")
        print()
        print(f"  TensorFlow: {get_runtime_compatibility()['tensorflow_version']}")
        print(f"  Description: {result['treatment']['description']}")
    except PredictionError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
