import importlib
import io
import sys
import types
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import AppConfig


@pytest.fixture
def client():
    startup_status = {
        "ready": True,
        "missing_files": [],
        "supported_crops": ["corn", "potato", "tomato"],
        "warnings": [],
    }

    prediction_module = types.ModuleType("prediction.predict")

    class PredictionError(RuntimeError):
        code = "prediction_error"

    class UnsupportedCropError(PredictionError):
        code = "unsupported_crop"

    def startup_check():
        return startup_status

    def full_pipeline(_image_path, use_tta=False):
        return {
            "success": True,
            "crop": "tomato",
            "crop_confidence": 97.3,
            "crop_all_predictions": {"tomato": 97.3, "potato": 2.7},
            "disease": "early_blight",
            "disease_confidence": 89.1,
            "disease_all_predictions": {"early_blight": 89.1, "healthy": 10.9},
            "treatment": {
                "display_name": "Early Blight",
                "description": "A fungal disease.",
                "symptoms": ["Dark brown spots"],
                "treatment": ["Use fungicide"],
                "prevention": ["Rotate crops"],
                "severity": "Moderate",
            },
            "low_confidence_warning": False,
        }

    prediction_module.PredictionError = PredictionError
    prediction_module.UnsupportedCropError = UnsupportedCropError
    prediction_module.startup_check = startup_check
    prediction_module.full_pipeline = full_pipeline

    prediction_package = types.ModuleType("prediction")
    prediction_package.predict = prediction_module

    sys.modules["prediction"] = prediction_package
    sys.modules["prediction.predict"] = prediction_module
    sys.modules.pop("app", None)

    app_module = importlib.import_module("app")

    upload_folder = ROOT_DIR / "tests_runtime_uploads_case"
    upload_folder.mkdir(exist_ok=True)
    for path in upload_folder.glob("*"):
        if path.is_file():
            path.unlink()

    config = AppConfig(
        upload_folder=upload_folder,
        max_content_length=16 * 1024 * 1024,
        allowed_extensions=("png", "jpg", "jpeg", "webp", "bmp"),
        debug=False,
        host="127.0.0.1",
        port=5000,
    )
    flask_app = app_module.create_app(config)
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as test_client:
        yield test_client, config.upload_folder


def test_health_endpoint_reports_startup_status(client):
    test_client, _upload_folder = client

    response = test_client.get("/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["startup"]["supported_crops"] == ["corn", "potato", "tomato"]


def test_predict_requires_image_field(client):
    test_client, _upload_folder = client

    response = test_client.post("/predict", data={}, content_type="multipart/form-data")
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["error"]["code"] == "missing_image"


def test_predict_rejects_invalid_extension(client):
    test_client, _upload_folder = client

    response = test_client.post(
        "/predict",
        data={"image": (io.BytesIO(b"not-an-image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["error"]["code"] == "invalid_file_type"


def test_predict_success_cleans_up_temp_upload(client):
    test_client, upload_folder = client

    response = test_client.post(
        "/predict",
        data={"image": (io.BytesIO(b"fake image bytes"), "leaf.jpg")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["crop"] == "tomato"
    assert list(Path(upload_folder).glob("*")) == []
