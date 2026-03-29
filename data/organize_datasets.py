"""
data/organize_datasets.py
---------------------------
Automatically organizes the downloaded plant disease datasets into a
unified directory structure expected by the training scripts.

OUTPUT STRUCTURE:
    data/raw/
    ├── train/
    │   ├── tomato/
    │   │   ├── early_blight/
    │   │   ├── late_blight/
    │   │   └── ...
    │   ├── potato/
    │   ├── corn/
    │   └── rice/ (if available)
    └── val/
        └── (same structure)

HOW TO RUN:
    python data/organize_datasets.py
"""

import os
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

OUTPUT_DIR = BASE_DIR / "raw"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

random.seed(SEED)

# ─────────────────────────────────────────────
# Source dataset paths
# ─────────────────────────────────────────────
PLANTVILLAGE_DIR = BASE_DIR / "plantdisease" / "PlantVillage"
PLANTDOC_TRAIN   = BASE_DIR / "plant-doc-dataset" / "PlantDoc-Dataset" / "train"
PLANTDOC_TEST    = BASE_DIR / "plant-doc-dataset" / "PlantDoc-Dataset" / "test"
TOMATOLEAF_TRAIN = BASE_DIR / "tomatoleaf" / "tomato" / "train"
TOMATOLEAF_VAL   = BASE_DIR / "tomatoleaf" / "tomato" / "val"

# ─────────────────────────────────────────────
# Mapping: source folder name → (crop, disease)
# ─────────────────────────────────────────────
FOLDER_MAPPING = {
    "Tomato_Early_blight": ("tomato", "early_blight"),
    "Tomato_Late_blight": ("tomato", "late_blight"),
    "Tomato_Leaf_Mold": ("tomato", "leaf_mold"),
    "Tomato_Septoria_leaf_spot": ("tomato", "septoria_leaf_spot"),
    "Tomato_healthy": ("tomato", "healthy"),
    "Tomato_Bacterial_spot": ("tomato", "bacterial_spot"),
    "Tomato__Target_Spot": ("tomato", "target_spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("tomato", "yellow_leaf_curl_virus"),
    "Tomato__Tomato_mosaic_virus": ("tomato", "mosaic_virus"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("tomato", "spider_mites"),
    "Potato___Early_blight": ("potato", "early_blight"),
    "Potato___Late_blight": ("potato", "late_blight"),
    "Potato___healthy": ("potato", "healthy"),

    "Tomato Early blight leaf": ("tomato", "early_blight"),
    "Tomato leaf late blight": ("tomato", "late_blight"),
    "Tomato mold leaf": ("tomato", "leaf_mold"),
    "Tomato Septoria leaf spot": ("tomato", "septoria_leaf_spot"),
    "Tomato leaf": ("tomato", "healthy"),
    "Tomato leaf bacterial spot": ("tomato", "bacterial_spot"),
    "Tomato leaf mosaic virus": ("tomato", "mosaic_virus"),
    "Tomato leaf yellow virus": ("tomato", "yellow_leaf_curl_virus"),
    "Potato leaf early blight": ("potato", "early_blight"),
    "Potato leaf late blight": ("potato", "late_blight"),
    "Corn rust leaf": ("corn", "common_rust"),
    "Corn Gray leaf spot": ("corn", "gray_leaf_spot"),
    "Corn leaf blight": ("corn", "northern_leaf_blight"),

    "Tomato___Early_blight": ("tomato", "early_blight"),
    "Tomato___Late_blight": ("tomato", "late_blight"),
    "Tomato___Leaf_Mold": ("tomato", "leaf_mold"),
    "Tomato___Septoria_leaf_spot": ("tomato", "septoria_leaf_spot"),
    "Tomato___healthy": ("tomato", "healthy"),
    "Tomato___Bacterial_spot": ("tomato", "bacterial_spot"),
    "Tomato___Target_Spot": ("tomato", "target_spot"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("tomato", "yellow_leaf_curl_virus"),
    "Tomato___Tomato_mosaic_virus": ("tomato", "mosaic_virus"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("tomato", "spider_mites"),
}

# ─────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def clean_and_save(src: Path, dst: Path) -> bool:
    try:
        img = Image.open(src)
        img.verify()

        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img.save(dst, quality=95)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
# Dataset collection
# ─────────────────────────────────────────────
def collect_images_from_folder(source_dir: Path):
    collection = defaultdict(list)

    if not source_dir.exists():
        print(f"[SKIP] {source_dir} not found")
        return collection

    for folder in source_dir.iterdir():
        if not folder.is_dir():
            continue

        if folder.name not in FOLDER_MAPPING:
            continue

        crop, disease = FOLDER_MAPPING[folder.name]

        for file in folder.iterdir():
            if file.is_file() and is_image_file(file):
                collection[(crop, disease)].append(file)

        print(f"{folder.name} → {crop}/{disease} ({len(collection[(crop,disease)])})")

    return collection


# ─────────────────────────────────────────────
# Main organizer
# ─────────────────────────────────────────────
def organize():

    print("Collecting images...")

    plantvillage = collect_images_from_folder(PLANTVILLAGE_DIR)
    plantdoc_train = collect_images_from_folder(PLANTDOC_TRAIN)
    plantdoc_val = collect_images_from_folder(PLANTDOC_TEST)
    tomleaf_train = collect_images_from_folder(TOMATOLEAF_TRAIN)
    tomleaf_val = collect_images_from_folder(TOMATOLEAF_VAL)

    train_images = defaultdict(list)
    val_images = defaultdict(list)
    test_images = defaultdict(list)

    # PlantVillage split (70/15/15)
    for key, imgs in plantvillage.items():
        random.shuffle(imgs)
        n = len(imgs)
        n_val = int(n * VAL_SPLIT)
        n_test = int(n * TEST_SPLIT)
        val_images[key] += imgs[:n_val]
        test_images[key] += imgs[n_val:n_val + n_test]
        train_images[key] += imgs[n_val + n_test:]

    # PlantDoc
    for k,v in plantdoc_train.items():
        train_images[k]+=v

    for k,v in plantdoc_val.items():
        val_images[k]+=v

    # TomatoLeaf
    for k,v in tomleaf_train.items():
        train_images[k]+=v

    for k,v in tomleaf_val.items():
        val_images[k]+=v


    print("Creating directories...")

    all_keys=set(list(train_images.keys())+list(val_images.keys())+list(test_images.keys()))

    for crop,disease in all_keys:
        (TRAIN_DIR/crop/disease).mkdir(parents=True,exist_ok=True)
        (VAL_DIR/crop/disease).mkdir(parents=True,exist_ok=True)
        (TEST_DIR/crop/disease).mkdir(parents=True,exist_ok=True)


    print("Processing images...")

    for split_name,split_images,split_dir in [
        ("train",train_images,TRAIN_DIR),
        ("val",val_images,VAL_DIR),
        ("test",test_images,TEST_DIR)
    ]:

        print(f"Processing {split_name}")

        for (crop,disease),paths in split_images.items():

            dest=split_dir/crop/disease

            for i,src in enumerate(paths):

                base=f"{crop}_{disease}_{i:05d}"
                ext=src.suffix.lower()

                dst=dest/f"{base}{ext}"

                counter=1
                while dst.exists():
                    dst=dest/f"{base}_{counter}{ext}"
                    counter+=1

                clean_and_save(src,dst)


    print("Dataset organization complete.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    organize()