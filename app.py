"""
app.py
-------
Flask backend for the Plant Disease Predictor web application.

Routes:
  GET  /           → Serve the main UI (index.html)
  POST /predict    → Accept image upload, run two-stage prediction, return JSON
  GET  /health     → Health check endpoint

HOW TO RUN:
    python app.py

Then open: http://127.0.0.1:5000
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our prediction pipeline
from prediction.predict import full_pipeline


# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (useful during development)

# Upload settings
UPLOAD_FOLDER   = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename to avoid overwriting existing uploads.
    Uses UUID to ensure uniqueness.

    Example: 'leaf.jpg' → 'a3f2b1c4-leaf.jpg'
    """
    ext = original_filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex[:8]}_{secure_filename(original_filename)}"
    return unique_name


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint — useful for deployment monitoring."""
    return jsonify({"status": "ok", "message": "Plant Disease Predictor is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Image upload and prediction endpoint.

    Accepts: multipart/form-data with field 'image'
    Returns: JSON with crop, disease, confidence, and treatment info

    Example response:
    {
        "success": true,
        "crop": "tomato",
        "crop_confidence": 97.3,
        "disease": "early_blight",
        "disease_confidence": 89.1,
        "treatment": {
            "display_name": "Early Blight",
            "description": "...",
            "symptoms": [...],
            "treatment": [...],
            "prevention": [...],
            "severity": "Moderate"
        },
        "low_confidence_warning": false
    }
    """

    # ── Validate request ───────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Please upload an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose an image to upload."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Save uploaded file ─────────────────────────────────────────────────
    filename = generate_unique_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ── Run two-stage prediction ───────────────────────────────────────────
    try:
        result = full_pipeline(filepath)
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

    # Add image URL to result so frontend can display the uploaded image
    result["image_url"] = f"/static/uploads/{filename}"

    # Return prediction result as JSON
    return jsonify(result)


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────
@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeded error."""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum upload size is 16 MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Run App
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Plant Disease Predictor — Flask Server")
    print("=" * 60)
    print("  URL: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)

    # debug=True enables auto-reload during development
    # Set debug=False for production deployment
    app.run(host="0.0.0.0", port=5000, debug=True)
