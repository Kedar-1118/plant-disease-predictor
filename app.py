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

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import AppConfig


LOGGER = logging.getLogger(__name__)

'''


# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def error_response(status_code: int, code: str, message: str, details: dict | None = None):
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def allowed_file(filename: str, allowed_extensions: tuple[str, ...]) -> bool:
    """Check if the uploaded file has an allowed image extension."""
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in allowed_extensions
    )


def get_prediction_module():
    from prediction import predict as prediction_module
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
'''


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def error_response(status_code: int, code: str, message: str, details: dict | None = None):
    payload = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def allowed_file(filename: str, allowed_extensions: tuple[str, ...]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def get_prediction_module():
    from prediction import predict as prediction_module

    return prediction_module


def create_app(config: AppConfig | None = None) -> Flask:
    configure_logging()
    config = config or AppConfig.from_env()

    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = str(config.upload_folder)
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    app.config["ALLOWED_EXTENSIONS"] = config.allowed_extensions
    app.config["APP_SETTINGS"] = config

    config.upload_folder.mkdir(parents=True, exist_ok=True)

    startup_status = get_prediction_module().startup_check()
    app.config["STARTUP_STATUS"] = startup_status

    if startup_status["missing_files"]:
        LOGGER.warning("Startup validation missing files: %s", startup_status["missing_files"])
    for warning in startup_status["warnings"]:
        LOGGER.warning("Startup warning: %s", warning)

    @app.before_request
    def log_request():
        LOGGER.info("Request %s %s", request.method, request.path)

    @app.route("/")
    def index():
        return render_template("index.html", supported_crops=startup_status["supported_crops"])

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok" if startup_status["ready"] else "degraded",
                "message": "Plant Disease Predictor is running",
                "startup": {
                    "ready": startup_status["ready"],
                    "supported_crops": startup_status["supported_crops"],
                    "warnings": startup_status["warnings"],
                    "missing_files": startup_status["missing_files"],
                },
            }
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return error_response(
                400,
                "missing_image",
                "No image file provided. Please upload an image.",
            )

        file = request.files["image"]

        if file.filename == "":
            return error_response(
                400,
                "empty_filename",
                "No file selected. Please choose an image to upload.",
            )

        if not allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
            return error_response(
                400,
                "invalid_file_type",
                "Invalid file type. Allowed types: "
                + ", ".join(app.config["ALLOWED_EXTENSIONS"]),
            )

        temp_path = None
        original_name = secure_filename(file.filename) or "upload"

        try:
            suffix = Path(original_name).suffix.lower()
            fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix="leaf_",
            )
            os.close(fd)
            file.save(temp_path)

            result = get_prediction_module().full_pipeline(temp_path)
            return jsonify(result)
        except get_prediction_module().PredictionError as exc:
            LOGGER.warning("Prediction request failed: %s", exc)
            details = None
            if getattr(exc, "code", "") == "unsupported_crop":
                details = {"supported_crops": startup_status["supported_crops"]}
            return error_response(422, exc.code, str(exc), details)
        except Exception:
            LOGGER.exception("Unexpected prediction failure")
            return error_response(
                500,
                "internal_error",
                "Prediction failed due to an internal server error.",
            )
        finally:
            file.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    @app.errorhandler(413)
    def file_too_large(_error):
        max_mb = round(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024))
        return error_response(
            413,
            "file_too_large",
            f"File too large. Maximum upload size is {max_mb} MB.",
        )

    @app.errorhandler(404)
    def not_found(_error):
        return error_response(404, "not_found", "Endpoint not found.")

    @app.errorhandler(500)
    def server_error(_error):
        return error_response(500, "internal_error", "Internal server error.")

    return app


app = create_app()


if __name__ == "__main__":
    settings: AppConfig = app.config["APP_SETTINGS"]

    print("=" * 60)
    print("  Plant Disease Predictor - Flask Server")
    print("=" * 60)
    print(f"  URL: http://127.0.0.1:{settings.port}")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)

    app.run(host=settings.host, port=settings.port, debug=settings.debug)
