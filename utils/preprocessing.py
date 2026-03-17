"""
utils/preprocessing.py
-----------------------
Image preprocessing utilities for the Plant Disease Predictor.
Handles image loading, resizing, normalization, and data augmentation.
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

# Monkey-patch to fix Keras mixed precision model loading bug (quantization_mode)
if not hasattr(mixed_precision.Policy, 'quantization_mode'):
    mixed_precision.Policy.quantization_mode = property(lambda self: None)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)          # Input size expected by MobileNetV2
BATCH_SIZE = 32


def setup_gpu(enable_mixed_precision: bool = True):
    """
    Configure TensorFlow to use GPU efficiently.
    - Enables memory growth to prevent TF from allocating all VRAM at once.
    - Optionally enables mixed precision (FP16) for faster training on supported GPUs.
    """
    print("[GPU] Configuring GPU settings...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth prevents TF from grabbing all VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Use mixed precision if requested (best for Volta, Turing, Ampere+ architectures)
            if enable_mixed_precision:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                print("      Mixed precision enabled (float16).")

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"      Found {len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)")
            print("      Memory growth enabled.")
        except RuntimeError as e:
            print(f"      [Warning] GPU configuration error: {e}")
    else:
        print("      No GPU detected. Running on CPU.")


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk, resize it to 224x224, and normalize pixel values
    to the range [0, 1] as expected by MobileNetV2.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
                    The batch dimension is added so it can be fed directly
                    into model.predict().
    """
    # Open image using PIL (handles JPEG, PNG, BMP, etc.)
    img = Image.open(image_path).convert("RGB")

    # Resize to the target size
    img = img.resize(IMG_SIZE)

    # Convert PIL image to NumPy array
    img_array = np.array(img, dtype=np.float32)

    # Normalize pixel values from [0, 255] to [0, 1]
    img_array = img_array / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_opencv(image_path: str) -> np.ndarray:
    """
    Alternative preprocessing using OpenCV.
    Useful for real-time camera feed or when PIL is not available.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image array of shape (1, 224, 224, 3).
    """
    # Read image with OpenCV (BGR format by default)
    img = cv2.imread(image_path)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to target size
    img = cv2.resize(img, IMG_SIZE)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


def create_data_generators(train_dir: str, val_dir: str):
    """
    Create Keras ImageDataGenerators for training and validation.

    Training generator applies data augmentation to improve model generalization:
    - Random horizontal/vertical flips
    - Random rotation up to 20 degrees
    - Random zoom up to 20%
    - Random brightness adjustment
    - Width and height shifts

    Validation generator only normalizes (no augmentation).

    Args:
        train_dir (str): Path to the training data directory.
                         Expected structure: train_dir/class_name/image.jpg
        val_dir   (str): Path to the validation data directory.
                         Expected structure: val_dir/class_name/image.jpg

    Returns:
        tuple: (train_generator, val_generator)
    """

    # ── Training augmentation ──────────────────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,           # Normalize pixel values to [0, 1]
        horizontal_flip=True,        # Randomly flip images left-right
        vertical_flip=False,         # Vertical flip rarely helps for leaves
        rotation_range=20,           # Rotate images up to ±20 degrees
        zoom_range=0.2,              # Randomly zoom in/out by up to 20%
        width_shift_range=0.1,       # Shift image horizontally by up to 10%
        height_shift_range=0.1,      # Shift image vertically by up to 10%
        brightness_range=[0.8, 1.2], # Vary brightness (simulates lighting)
        shear_range=0.1,             # Shear transformation
        fill_mode="nearest"          # Fill empty pixels with nearest value
    )

    # ── Validation: only normalize, no augmentation ────────────────────────
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # ── Create generators ──────────────────────────────────────────────────
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",    # One-hot encoded labels
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False                # Don't shuffle validation data
    )

    return train_generator, val_generator


def get_class_weights(generator) -> dict:
    """
    Compute class weights to handle imbalanced datasets.
    Classes with fewer samples get higher weights so the model
    doesn't ignore minority classes.

    Args:
        generator: A Keras ImageDataGenerator flow object.

    Returns:
        dict: Mapping from class index to weight value.
    """
    from sklearn.utils.class_weight import compute_class_weight

    # Get all class labels from the generator
    labels = generator.classes
    class_indices = list(generator.class_indices.values())

    # Compute balanced class weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_indices),
        y=labels
    )

    # Return as dictionary {class_index: weight}
    return dict(enumerate(weights))
