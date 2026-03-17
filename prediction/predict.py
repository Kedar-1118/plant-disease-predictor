"""
prediction/predict.py
----------------------
Two-stage prediction pipeline for the Plant Disease Predictor.

Stage 1: Predict crop type from image
Stage 2: Predict disease using crop-specific model
Stage 3: Fetch treatment recommendation from JSON database

This module is imported by the Flask app (app.py) and can also
be used standalone for testing.

STANDALONE USAGE:
    python prediction/predict.py --image path/to/leaf.jpg
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from utils.preprocessing import load_and_preprocess_image, setup_gpu

# Configure GPU memory growth
setup_gpu()


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR          = "models"
CROP_MODEL_PATH     = os.path.join(MODELS_DIR, "crop_model.h5")
CROP_CLASSES_PATH   = os.path.join(MODELS_DIR, "crop_classes.json")
TREATMENTS_PATH     = "data/treatments.json"

# Confidence threshold — if crop confidence is below this, warn the user
CONFIDENCE_THRESHOLD = 0.60


# ─────────────────────────────────────────────
# Model Cache (avoid reloading on every request)
# ─────────────────────────────────────────────
_crop_model = None
_disease_models = {}      # {crop_name: model}
_crop_classes = None      # {class_name: index}
_disease_classes = {}     # {crop_name: {class_name: index}}
_treatments = None        # Full treatments JSON


def _load_treatments() -> dict:
    """Load treatment database from JSON file (cached)."""
    global _treatments
    if _treatments is None:
        with open(TREATMENTS_PATH, "r") as f:
            _treatments = json.load(f)
    return _treatments


def _load_crop_model():
    """Load Stage 1 crop classifier model (cached)."""
    global _crop_model, _crop_classes

    if _crop_model is None:
        if not os.path.exists(CROP_MODEL_PATH):
            raise FileNotFoundError(
                f"Crop model not found at '{CROP_MODEL_PATH}'. "
                "Please train the model first: python training/train_crop_classifier.py"
            )
        print("[Loading] Crop classifier model...")
        _crop_model = tf.keras.models.load_model(CROP_MODEL_PATH, compile=False)

    if _crop_classes is None:
        with open(CROP_CLASSES_PATH, "r") as f:
            _crop_classes = json.load(f)

    return _crop_model, _crop_classes


def _load_disease_model(crop: str):
    """Load Stage 2 disease classifier model for a specific crop (cached)."""
    global _disease_models, _disease_classes

    if crop not in _disease_models:
        model_path   = os.path.join(MODELS_DIR, f"{crop}_disease_model.h5")
        classes_path = os.path.join(MODELS_DIR, f"{crop}_disease_classes.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Disease model for '{crop}' not found at '{model_path}'. "
                f"Please train it: python training/train_disease_classifier.py --crop {crop}"
            )

        print(f"[Loading] Disease model for {crop}...")
        _disease_models[crop] = tf.keras.models.load_model(model_path, compile=False)

        with open(classes_path, "r") as f:
            _disease_classes[crop] = json.load(f)

    return _disease_models[crop], _disease_classes[crop]


def predict_crop(image_path: str) -> dict:
    """
    Stage 1: Predict the crop type from a leaf image.

    Args:
        image_path (str): Path to the leaf image file.

    Returns:
        dict: {
            "crop": "tomato",
            "confidence": 0.97,
            "all_predictions": {"tomato": 0.97, "potato": 0.02, ...}
        }
    """
    model, class_indices = _load_crop_model()

    # Preprocess image
    img_array = load_and_preprocess_image(image_path)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)[0]  # Shape: (num_classes,)

    # Build index → class name mapping (reverse of class_indices)
    index_to_class = {v: k for k, v in class_indices.items()}

    # Get top prediction
    predicted_index = int(np.argmax(predictions))
    predicted_crop  = index_to_class[predicted_index]
    confidence      = float(predictions[predicted_index])

    # Build all predictions dict
    all_predictions = {
        index_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return {
        "crop": predicted_crop,
        "confidence": confidence,
        "all_predictions": all_predictions
    }


def predict_disease(crop: str, image_path: str) -> dict:
    """
    Stage 2: Predict the disease for a given crop from a leaf image.

    Args:
        crop       (str): Crop name (e.g., 'tomato').
        image_path (str): Path to the leaf image file.

    Returns:
        dict: {
            "disease": "early_blight",
            "confidence": 0.89,
            "all_predictions": {"early_blight": 0.89, "healthy": 0.08, ...}
        }
    """
    model, class_indices = _load_disease_model(crop)

    # Preprocess image
    img_array = load_and_preprocess_image(image_path)

    # Run prediction
    predictions = model.predict(img_array, verbose=0)[0]

    # Build index → class name mapping
    index_to_class = {v: k for k, v in class_indices.items()}

    predicted_index   = int(np.argmax(predictions))
    predicted_disease = index_to_class[predicted_index]
    confidence        = float(predictions[predicted_index])

    all_predictions = {
        index_to_class[i]: float(predictions[i])
        for i in range(len(predictions))
    }

    return {
        "disease": predicted_disease,
        "confidence": confidence,
        "all_predictions": all_predictions
    }


def get_treatment(crop: str, disease: str) -> dict:
    """
    Fetch treatment and prevention information from the JSON database.

    Args:
        crop    (str): Crop name (e.g., 'tomato').
        disease (str): Disease name (e.g., 'early_blight').

    Returns:
        dict: Treatment information, or a default dict if not found.
    """
    treatments = _load_treatments()

    # Normalize keys (lowercase, replace spaces with underscores)
    crop_key    = crop.lower().replace(" ", "_")
    disease_key = disease.lower().replace(" ", "_")

    # Look up treatment
    crop_data = treatments.get(crop_key, {})
    treatment_data = crop_data.get(disease_key)

    if treatment_data is None:
        # Return a generic response if not found in database
        return {
            "display_name": disease.replace("_", " ").title(),
            "description": "Disease information not available in the database.",
            "symptoms": [],
            "treatment": ["Consult a local agricultural expert for treatment advice."],
            "prevention": ["Practice good agricultural hygiene and crop rotation."],
            "severity": "Unknown"
        }

    return treatment_data


def full_pipeline(image_path: str) -> dict:
    """
    Run the complete two-stage prediction pipeline on a leaf image.

    Pipeline:
    1. Preprocess image
    2. Stage 1: Predict crop type
    3. Stage 2: Predict disease using crop-specific model
    4. Fetch treatment from JSON database
    5. Return combined result

    Args:
        image_path (str): Path to the uploaded leaf image.

    Returns:
        dict: Complete prediction result with crop, disease, and treatment info.
    """
    result = {
        "success": True,
        "image_path": image_path,
        "crop": None,
        "crop_confidence": None,
        "disease": None,
        "disease_confidence": None,
        "treatment": None,
        "low_confidence_warning": False,
        "error": None
    }

    try:
        # ── Stage 1: Crop prediction ───────────────────────────────────────
        crop_result = predict_crop(image_path)
        result["crop"]            = crop_result["crop"]
        result["crop_confidence"] = round(crop_result["confidence"] * 100, 2)
        result["crop_all_predictions"] = {
            k: round(v * 100, 2) for k, v in crop_result["all_predictions"].items()
        }

        # Warn if confidence is low
        if crop_result["confidence"] < CONFIDENCE_THRESHOLD:
            result["low_confidence_warning"] = True
            result["warning_message"] = (
                f"Low confidence ({result['crop_confidence']}%) in crop detection. "
                "The image may not be a clear leaf photo."
            )

        # ── Stage 2: Disease prediction ────────────────────────────────────
        disease_result = predict_disease(crop_result["crop"], image_path)
        result["disease"]            = disease_result["disease"]
        result["disease_confidence"] = round(disease_result["confidence"] * 100, 2)
        result["disease_all_predictions"] = {
            k: round(v * 100, 2) for k, v in disease_result["all_predictions"].items()
        }

        # ── Stage 3: Treatment recommendation ─────────────────────────────
        result["treatment"] = get_treatment(
            crop_result["crop"],
            disease_result["disease"]
        )

    except FileNotFoundError as e:
        result["success"] = False
        result["error"]   = str(e)

    except Exception as e:
        result["success"] = False
        result["error"]   = f"Prediction failed: {str(e)}"

    return result


# ─────────────────────────────────────────────
# Standalone CLI usage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-stage plant disease prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)

    print(f"\n[Running] Two-stage prediction on: {args.image}\n")
    result = full_pipeline(args.image)

    if result["success"]:
        print("=" * 60)
        print("  PREDICTION RESULT")
        print("=" * 60)
        print(f"  Crop     : {result['crop'].capitalize()} ({result['crop_confidence']}%)")
        print(f"  Disease  : {result['treatment']['display_name']} ({result['disease_confidence']}%)")
        print(f"  Severity : {result['treatment']['severity']}")
        print(f"\n  Description:")
        print(f"  {result['treatment']['description']}")
        print(f"\n  Treatment:")
        for t in result["treatment"]["treatment"]:
            print(f"    • {t}")
        print(f"\n  Prevention:")
        for p in result["treatment"]["prevention"]:
            print(f"    • {p}")
        if result.get("low_confidence_warning"):
            print(f"\n  ⚠ WARNING: {result['warning_message']}")
    else:
        print(f"\n[ERROR] {result['error']}")
