"""
training/train_crop_classifier.py
-----------------------------------
Stage 1: Train a MobileNetV2-based crop classifier.

This version automatically uses GPU acceleration if available.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────
# GPU Setup
# ─────────────────────────────────────────────
print("\nChecking for GPU...")

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")

        print(f"GPU detected: {gpus}")
        print("Mixed precision enabled (FP16).")

    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
TRAIN_DIR   = "data/raw/train"
VAL_DIR     = "data/raw/val"
MODEL_SAVE  = "models/crop_model.h5"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_FROZEN   = 10
EPOCHS_FINETUNE = 20

LEARNING_RATE   = 1e-4
FINETUNE_LR     = 1e-5


# ─────────────────────────────────────────────
# Dataset Loader
# ─────────────────────────────────────────────
def collect_crop_dataframe(base_dir):

    records = []

    for crop in sorted(os.listdir(base_dir)):

        crop_path = os.path.join(base_dir, crop)

        if not os.path.isdir(crop_path):
            continue

        for disease in sorted(os.listdir(crop_path)):

            disease_path = os.path.join(crop_path, disease)

            if not os.path.isdir(disease_path):
                continue

            for img_name in os.listdir(disease_path):

                img_path = os.path.join(disease_path, img_name)

                if os.path.isfile(img_path):

                    records.append({
                        "filepath": img_path,
                        "crop": crop
                    })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────
def create_crop_generators(train_dir, val_dir):

    train_df = collect_crop_dataframe(train_dir)
    val_df = collect_crop_dataframe(val_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8,1.2],
        shear_range=0.1,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="crop",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="crop",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen


# ─────────────────────────────────────────────
# Class Weight Calculation
# ─────────────────────────────────────────────
def get_class_weights_from_gen(generator):

    from sklearn.utils.class_weight import compute_class_weight

    labels = generator.classes
    class_indices = list(range(len(generator.class_indices)))

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_indices),
        y=labels
    )

    return dict(enumerate(weights))


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────
def build_model(num_classes):

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# ─────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────
def unfreeze_top_layers(model, base_model, layers=30):

    base_model.trainable = True

    for layer in base_model.layers[:-layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"Fine-tuning last {layers} layers")


# ─────────────────────────────────────────────
# Plot Training
# ─────────────────────────────────────────────
def plot_training_history(h1, h2):

    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]

    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    epochs = range(1,len(acc)+1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs,acc,label="Train")
    plt.plot(epochs,val_acc,label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs,loss,label="Train")
    plt.plot(epochs,val_loss,label="Val")
    plt.title("Loss")
    plt.legend()

    os.makedirs("models",exist_ok=True)
    plt.savefig("models/crop_training_history.png")
    plt.close()


# ─────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────
def main():

    print("\nLoading dataset...")

    train_gen, val_gen = create_crop_generators(TRAIN_DIR,VAL_DIR)

    num_classes = len(train_gen.class_indices)

    print(f"Classes: {train_gen.class_indices}")

    os.makedirs("models",exist_ok=True)

    with open("models/crop_classes.json","w") as f:
        json.dump(train_gen.class_indices,f,indent=2)

    class_weights = get_class_weights_from_gen(train_gen)

    model, base_model = build_model(num_classes)

    print("\nTraining classification head...")

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=5,restore_best_weights=True),
            ReduceLROnPlateau(patience=3)
        ]
    )

    print("\nFine tuning...")

    unfreeze_top_layers(model,base_model)

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=7,restore_best_weights=True),
            ModelCheckpoint(MODEL_SAVE,save_best_only=True),
            ReduceLROnPlateau(patience=4)
        ]
    )

    val_loss,val_acc = model.evaluate(val_gen)

    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")

    plot_training_history(history1,history2)

    # ── Save metadata ─────────────────────────────────────────────────────
    from save_metadata import save_model_metadata
    total_epochs = len(history1.history["accuracy"]) + len(history2.history["accuracy"])
    save_model_metadata(
        model_path=MODEL_SAVE,
        val_accuracy=val_acc,
        val_loss=val_loss,
        epochs_run=total_epochs,
        num_classes=num_classes,
        class_names=list(train_gen.class_indices.keys()),
        hyperparameters={
            "learning_rate": LEARNING_RATE,
            "finetune_lr": FINETUNE_LR,
            "batch_size": BATCH_SIZE,
            "epochs_frozen": EPOCHS_FROZEN,
            "epochs_finetune": EPOCHS_FINETUNE,
            "unfrozen_layers": 30,
        },
        dataset_info={
            "train_samples": train_gen.samples,
            "val_samples": val_gen.samples,
            "train_dir": TRAIN_DIR,
            "val_dir": VAL_DIR,
        },
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()