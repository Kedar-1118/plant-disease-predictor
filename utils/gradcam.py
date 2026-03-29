"""
utils/gradcam.py
-----------------
Gradient-weighted Class Activation Mapping (Grad-CAM) for model explainability.

Generates a heatmap overlay showing which regions of the leaf image
the CNN focused on when making its prediction.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
           https://arxiv.org/abs/1610.02391
"""

from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Default target layer — last convolutional block in MobileNetV2
MOBILENETV2_LAST_CONV = "Conv_1"


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Walk the model layers in reverse and return the name of the
    last Conv2D / DepthwiseConv2D layer.  Falls back to the
    hard-coded MobileNetV2 default if nothing is found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    LOGGER.warning(
        "Could not auto-detect last conv layer; "
        "falling back to '%s'.", MOBILENETV2_LAST_CONV,
    )
    return MOBILENETV2_LAST_CONV


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: int | None = None,
    layer_name: str | None = None,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Args:
        model:      A compiled Keras model.
        img_array:  Pre-processed image tensor of shape (1, 224, 224, 3).
        pred_index: Class index to explain.  Defaults to argmax (top prediction).
        layer_name: Name of the convolutional layer to use.
                    Defaults to the last conv layer detected in the model.

    Returns:
        A 2-D numpy array (H, W) with values in [0, 1] representing the heatmap.
    """
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        LOGGER.warning("Layer '%s' not found, trying auto-detect.", layer_name)
        layer_name = _find_last_conv_layer(model)
        target_layer = model.get_layer(layer_name)

    # Build a mini-model that outputs both the conv layer activations and the
    # final predictions in a single forward pass.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    # Forward pass + gradient computation
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the predicted class w.r.t. the conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global-average-pool the gradients to get per-filter importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of activation maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    original_image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_image_path: Path to the original leaf image.
        heatmap:             2-D heatmap from generate_gradcam_heatmap().
        alpha:               Blending factor (0 = image only, 1 = heatmap only).
        colormap:            OpenCV colormap for the heatmap.
        output_size:         Final image size (width, height).

    Returns:
        An (H, W, 3) uint8 numpy array with the overlay.
    """
    # Load and resize the original image
    img = Image.open(original_image_path).convert("RGB")
    img = img.resize(output_size)
    img_array = np.array(img, dtype=np.uint8)

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, output_size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * img_array)
    return overlay


def gradcam_to_base64(overlay: np.ndarray) -> str:
    """
    Convert a Grad-CAM overlay (numpy array) to a base64-encoded PNG string.

    Returns:
        A data-URI string: "data:image/png;base64,..."
    """
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_gradcam_overlay_b64(
    model: tf.keras.Model,
    img_array: np.ndarray,
    original_image_path: str,
    pred_index: int | None = None,
    layer_name: str | None = None,
    alpha: float = 0.4,
) -> str:
    """
    Convenience function: generate Grad-CAM heatmap → overlay on original
    image → return as base64 PNG data URI.

    This is the main function called from the prediction pipeline.
    """
    heatmap = generate_gradcam_heatmap(model, img_array, pred_index, layer_name)
    overlay = overlay_gradcam(original_image_path, heatmap, alpha=alpha)
    return gradcam_to_base64(overlay)
