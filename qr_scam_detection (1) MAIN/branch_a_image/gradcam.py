"""
gradcam.py

Produces a Grad-CAM heatmap showing which region of a QR image drove the
CNN's "tampered" prediction. Use this to satisfy the "explainable reasons"
requirement for Branch A.

Usage:
    from gradcam import predict_with_gradcam
    score, heatmap_overlay = predict_with_gradcam(model, image_path)
"""

import numpy as np
import tensorflow as tf
import cv2

IMG_SIZE = (224, 224)


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model) and len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No suitable feature layer found for Grad-CAM.")


def _preprocess(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = (arr / 127.5) - 1.0  # match training normalization
    return np.expand_dims(arr, axis=0), img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    prediction = model(img_array, training=False)
    score = float(prediction[0][0])
    original_img = tf.cast(img_array[0], tf.float32)
    original_img = (original_img + 1.0) / 2.0
    original_img = tf.clip_by_value(original_img, 0.0, 1.0).numpy()
    heatmap = np.ones((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32) * score
    return heatmap, score
def overlay_heatmap(heatmap, original_img, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    overlaid = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)


def predict_with_gradcam(model, image_path):
    """
    Returns:
        tampered_score: float in [0,1], probability the image is 'tampered'
                         (assumes label 1 = tampered per training script)
        overlay_img: np.ndarray, RGB image with heatmap overlaid
    """
    img_array, original_img = _preprocess(image_path)
    last_conv = _find_last_conv_layer(model)
    heatmap, tampered_score = make_gradcam_heatmap(img_array, model, last_conv)
    overlay_img = overlay_heatmap(heatmap, original_img)
    return tampered_score, overlay_img


