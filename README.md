# 🌿 PlantGuard AI — Two-Stage Crop Disease Predictor

A production-ready Flask + TensorFlow web application that diagnoses plant diseases from leaf images using a **two-stage deep learning pipeline**.

```
📷 Leaf Image → 🧠 Stage 1: Crop ID → 🔬 Stage 2: Disease ID → 💊 Treatment
```

## ✨ Key Features

- **Two-Stage Classification**: MobileNetV2-based pipeline — first identifies the crop, then runs a crop-specific disease classifier
- **Grad-CAM Explainability**: Visual heatmaps showing which leaf regions the model focuses on
- **Test-Time Augmentation**: Optional TTA for improved inference accuracy
- **Treatment Recommendations**: Actionable guidance for each detected disease
- **Production-Ready**: Dockerized, with health checks, temp file cleanup, and structured error handling

## 🌾 Supported Crops

| Crop | Diseases Detected |
|------|-------------------|
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, Yellow Curl Virus, Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🌽 Corn | Common Rust, Gray Leaf Spot, Northern Leaf Blight |

## 🚀 Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/Kedar-1118/plant-disease-predictor.git
cd plant-disease-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

### 🐳 Docker

```bash
docker build -t plantguard-ai .
docker run -p 5000:5000 plantguard-ai
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Flask Web Server (app.py)                               │
│  ├── POST /predict     → Two-stage inference pipeline    │
│  ├── GET  /health      → System health + model status    │
│  └── GET  /            → Frontend UI                     │
├──────────────────────────────────────────────────────────┤
│  Prediction Pipeline (prediction/predict.py)             │
│  ├── Stage 1: Crop Classifier (MobileNetV2)              │
│  ├── Stage 2: Disease Classifier (crop-specific model)   │
│  ├── Grad-CAM Heatmap Generation                         │
│  └── Treatment Lookup (data/treatments.json)             │
├──────────────────────────────────────────────────────────┤
│  Training Scripts (training/)                            │
│  ├── train_crop_classifier.py                            │
│  ├── train_disease_classifier.py                         │
│  ├── evaluate_model.py                                   │
│  ├── export_tflite.py     → Convert to TFLite/mobile     │
│  ├── benchmark.py         → Inference latency profiling   │
│  └── save_metadata.py     → Model versioning             │
└──────────────────────────────────────────────────────────┘
```

## 🧪 Training from Scratch

```bash
# 1. Download and organize datasets
python data/organize_datasets.py

# 2. Train the crop classifier (Stage 1)
python training/train_crop_classifier.py

# 3. Train disease classifiers (Stage 2)
python training/train_disease_classifier.py --crop all
# Or train individually: --crop tomato | --crop potato | --crop corn

# 4. Evaluate model performance
python training/evaluate_model.py --model models/crop_model.h5 --data data/raw/val

# 5. Export to TFLite for mobile deployment
python training/export_tflite.py                    # All models, float16
python training/export_tflite.py --quantize int8    # INT8 quantization

# 6. Benchmark inference latency
python training/benchmark.py --image test_leaf.jpg --iterations 100
```

## 📡 API Reference

### `POST /predict`

Predict disease from a leaf image. Supports optional Test-Time Augmentation.

```bash
# Standard prediction
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict

# With TTA (slower but more accurate)
curl -X POST -F "image=@leaf.jpg" "http://localhost:5000/predict?tta=true"
```

**Success Response:**
```json
{
  "success": true,
  "crop": "tomato",
  "crop_confidence": 98.45,
  "disease": "early_blight",
  "disease_confidence": 94.12,
  "treatment": { "display_name": "Early Blight", "severity": "Moderate", ... },
  "gradcam_image": "data:image/png;base64,...",
  "latency_ms": 342.5
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "invalid_file_type",
    "message": "Invalid file type. Allowed types: png, jpg, jpeg, webp, bmp"
  }
}
```

### `GET /health`

Returns system health, model status, supported crops, and diagnostics.

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_FOLDER` | `static/uploads` | Upload directory |
| `MAX_CONTENT_LENGTH` | `16777216` (16MB) | Max upload size |
| `ALLOWED_EXTENSIONS` | `png,jpg,jpeg,webp,bmp` | Accepted formats |
| `FLASK_HOST` | `0.0.0.0` | Server host |
| `FLASK_PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `false` | Debug mode |

## 🧪 Testing

```bash
pytest                      # Run all tests
pytest -v                   # Verbose output
pytest --cov=prediction     # With coverage
```

## 📁 Project Structure

```
plant-disease-predictor/
├── app.py                          # Flask web server
├── config.py                       # Configuration management
├── Dockerfile                      # Docker containerization
├── requirements.txt                # Python dependencies
├── prediction/
│   └── predict.py                  # Two-stage inference + Grad-CAM
├── utils/
│   ├── preprocessing.py            # Image loading + augmentation
│   └── gradcam.py                  # Grad-CAM explainability
├── training/
│   ├── train_crop_classifier.py    # Stage 1 training
│   ├── train_disease_classifier.py # Stage 2 training
│   ├── evaluate_model.py           # Model evaluation
│   ├── export_tflite.py            # TFLite conversion
│   ├── benchmark.py                # Performance profiling
│   └── save_metadata.py           # Model metadata
├── data/
│   ├── organize_datasets.py        # Dataset preparation
│   └── treatments.json             # Treatment database
├── models/                         # Trained .h5 models
├── templates/index.html            # Frontend UI
├── static/
│   ├── css/style.css               # Styling
│   └── js/app.js                   # Frontend logic
└── tests/
    └── test_app.py                 # Unit tests
```

## 📝 Notes

- Uploaded images are stored temporarily during inference and deleted immediately after
- TensorFlow is pinned to `2.15.x` for runtime compatibility
- Models are lazy-loaded and cached in memory for fast subsequent predictions
- Grad-CAM uses the last convolutional layer of MobileNetV2 for heatmap generation
- If model files are missing, the `/health` endpoint reports degraded state
