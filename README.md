# PlantGuard AI - Two-Stage Crop Disease Predictor

A Flask and TensorFlow web app that classifies a crop leaf image in two stages:
1. identify the crop,
2. run the matching disease model,
3. return treatment and prevention guidance.

## Supported crops
- Tomato
- Potato
- Corn

## Quick start
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Tests
```bash
pytest
```

## Runtime configuration
These environment variables are supported:
- `UPLOAD_FOLDER` default `static/uploads`
- `MAX_CONTENT_LENGTH` default `16777216`
- `ALLOWED_EXTENSIONS` default `png,jpg,jpeg,webp,bmp`
- `FLASK_HOST` default `0.0.0.0`
- `FLASK_PORT` default `5000`
- `FLASK_DEBUG` default `false`

The `/health` endpoint now reports startup readiness, supported crops, warnings, and missing files.

## API
`POST /predict`
- Request: `multipart/form-data` with `image`
- Success response includes crop, disease, confidences, and treatment metadata.
- Error responses are structured:

```json
{
  "success": false,
  "error": {
    "code": "invalid_file_type",
    "message": "Invalid file type. Allowed types: png, jpg, jpeg, webp, bmp"
  }
}
```

## Notes
- Uploaded files are stored only temporarily during inference and are deleted after each request.
- TensorFlow is pinned to the tested `2.15.x` line to reduce runtime compatibility issues.
- If required model files are missing, startup health will report a degraded state.
