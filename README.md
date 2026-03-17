# 🌿 PlantGuard AI — Two-Stage Crop Disease Predictor

A production-structured, beginner-friendly deep learning web application that:
1. **Stage 1**: Identifies the crop type from a leaf image (Tomato, Potato, Rice, Corn)
2. **Stage 2**: Detects the specific disease using a crop-specific model
3. **Recommends** treatment and prevention methods

**Tech Stack**: Python · TensorFlow/Keras · MobileNetV2 · Flask · HTML/CSS/JS

---

## 📁 Project Structure

```
plant disease predictor/
├── data/
│   ├── raw/                         # ← Put your dataset here
│   │   ├── train/
│   │   │   ├── tomato/
│   │   │   │   ├── early_blight/
│   │   │   │   ├── late_blight/
│   │   │   │   ├── leaf_mold/
│   │   │   │   ├── septoria_leaf_spot/
│   │   │   │   └── healthy/
│   │   │   ├── potato/
│   │   │   ├── rice/
│   │   │   └── corn/
│   │   └── val/                     # Same structure as train/
│   └── treatments.json              # Disease treatment database
├── models/                          # Trained .h5 models saved here
├── utils/
│   └── preprocessing.py             # Image preprocessing & augmentation
├── training/
│   ├── train_crop_classifier.py     # Stage 1 training
│   ├── train_disease_classifier.py  # Stage 2 training
│   └── evaluate_model.py            # Model evaluation
├── prediction/
│   └── predict.py                   # Two-stage prediction pipeline
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── uploads/                     # Uploaded images stored here
├── templates/
│   └── index.html
├── app.py                           # Flask web server
└── requirements.txt
```

---

## ⚙️ Environment Setup

### Step 1 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU Support (Optional)**: For faster training, install the GPU version:
> ```bash
> pip install tensorflow-gpu
> ```

---

## 📦 Dataset Preparation

### Option A — PlantVillage Dataset (Recommended)

1. Download from Kaggle: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Organize into the folder structure shown above:
   ```
   data/raw/train/tomato/early_blight/  ← put tomato early blight images here
   data/raw/train/tomato/healthy/       ← put healthy tomato images here
   data/raw/val/tomato/early_blight/    ← validation images (20% split)
   ```
3. Recommended split: **80% train / 20% val**

### Option B — Use the organize script

After downloading PlantVillage, you can manually sort images into the folder structure. Aim for at least **200 images per class** for good results.

### Supported Disease Classes

| Crop   | Diseases |
|--------|----------|
| Tomato | Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Rice   | Brown Spot, Bacterial Blight, Leaf Smut, Healthy |
| Corn   | Common Rust, Northern Leaf Blight, Healthy |

---

## 🧠 Training the Models

### Stage 1 — Crop Classifier

```bash
python training/train_crop_classifier.py
```

- Trains MobileNetV2 on `data/raw/train/` (all crop folders)
- Saves model → `models/crop_model.h5`
- Saves class mapping → `models/crop_classes.json`
- Saves training plot → `models/crop_training_history.png`

### Stage 2 — Disease Classifiers

```bash
# Train for a specific crop:
python training/train_disease_classifier.py --crop tomato
python training/train_disease_classifier.py --crop potato
python training/train_disease_classifier.py --crop rice
python training/train_disease_classifier.py --crop corn

# OR train all at once:
python training/train_disease_classifier.py --crop all
```

- Saves models → `models/tomato_disease_model.h5`, etc.
- Saves class mappings → `models/tomato_disease_classes.json`, etc.

### Model Evaluation

```bash
# Evaluate crop classifier:
python training/evaluate_model.py --model models/crop_model.h5 --data data/raw/val

# Evaluate tomato disease classifier:
python training/evaluate_model.py --model models/tomato_disease_model.h5 --data data/raw/val/tomato
```

Outputs: accuracy, precision, recall, F1-score, confusion matrix PNG.

---

## 🚀 Running the Web App

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

### Testing the Prediction Pipeline (without browser)

```bash
python prediction/predict.py --image path/to/your/leaf.jpg
```

---

## 🌐 Deployment

### Local Network (share with others on same WiFi)

The Flask app already binds to `0.0.0.0`, so it's accessible on your local network:
```
http://<your-ip-address>:5000
```

### Production Deployment with Gunicorn (Linux/macOS)

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### Deploy to a Cloud VM (e.g., AWS EC2, Google Cloud)

1. SSH into your VM
2. Clone/upload the project
3. Install dependencies: `pip install -r requirements.txt`
4. Copy trained models to `models/` folder
5. Run with Gunicorn:
   ```bash
   gunicorn -w 2 -b 0.0.0.0:80 app:app
   ```

### Deploy to Render (Free Tier)

1. Push project to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`

> ⚠️ **Note**: Trained `.h5` model files must be included in your deployment. They are large (~15 MB each) — consider using Git LFS or uploading them separately.

---

## 📈 Accuracy Improvement Tips

### 1. Data Augmentation (already implemented)
The training scripts use:
- Horizontal flips, rotation, zoom, brightness variation
- These prevent overfitting and improve generalization

### 2. Transfer Learning (already implemented)
- MobileNetV2 pretrained on ImageNet provides strong feature extraction
- Two-phase training: frozen base → fine-tuning top layers

### 3. Class Balancing (already implemented)
- `compute_class_weight("balanced")` automatically handles imbalanced datasets

### 4. Additional Tips
- **More data**: Aim for 500+ images per class for best results
- **Higher resolution**: Try 256×256 or 299×299 input (update `IMG_SIZE` in `preprocessing.py`)
- **Ensemble**: Train multiple models and average their predictions
- **ResNet50**: Swap MobileNetV2 for ResNet50 in the training scripts for potentially higher accuracy (slower training)
- **Learning rate scheduling**: Already implemented via `ReduceLROnPlateau`

---

## 🔧 Configuration

| Setting | File | Default |
|---------|------|---------|
| Image size | `utils/preprocessing.py` | 224×224 |
| Batch size | `utils/preprocessing.py` | 32 |
| Training epochs | `training/train_*.py` | 10 + 20 |
| Confidence threshold | `prediction/predict.py` | 60% |
| Max upload size | `app.py` | 16 MB |
| Flask port | `app.py` | 5000 |

---

## 📋 API Reference

### `POST /predict`

Upload a leaf image and get a prediction.

**Request**: `multipart/form-data` with field `image`

**Response**:
```json
{
  "success": true,
  "crop": "tomato",
  "crop_confidence": 97.3,
  "disease": "early_blight",
  "disease_confidence": 89.1,
  "treatment": {
    "display_name": "Early Blight",
    "description": "...",
    "symptoms": ["..."],
    "treatment": ["..."],
    "prevention": ["..."],
    "severity": "Moderate"
  },
  "image_url": "/static/uploads/abc123_leaf.jpg",
  "low_confidence_warning": false
}
```

---

## 📚 References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Flask Documentation](https://flask.palletsprojects.com/)
#   p l a n t - d i s e a s e - p r e d i c t o r  
 #   p l a n t - d i s e a s e - p r e d i c t o r  
 #   p l a n t - d i s e a s e - p r e d i c t o r  
 #   p l a n t - d i s e a s e - p r e d i c t o r  
 