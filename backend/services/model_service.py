import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import cv2
import base64

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent   # backend/
ROOT_DIR   = BASE_DIR.parent                          # project root
MODEL_PATH = ROOT_DIR / "ml_models" / "mri_classifier.h5"

CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]

# -- Load model at startup -----------------------------------------------------
model = None
grad_model = None

if MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH), compile=False)
        print(f"[model_service] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[model_service] ERROR loading model: {e}")
else:
    print(f"[model_service] WARNING: Model file not found at {MODEL_PATH}")


# -- Caching for inference results to avoid repetitive execution --------------
_predict_cache = {}

import hashlib

def _get_image_hash(image_array: np.ndarray) -> str:
    return hashlib.md5(image_array.tobytes()).hexdigest()


# -- Inference -----------------------------------------------------------------
def predict_mri(image_array: np.ndarray) -> tuple[str, float, str]:
    """
    Run inference on a preprocessed MRI image array (H x W x 3, float32).
    Utilises cache and a pre-compiled Grad-CAM model for high performance.
    """
    if model is None:
        raise RuntimeError(
            f"ML model is not loaded. "
            f"Expected file: {MODEL_PATH}"
        )

    # Check cache
    img_hash = _get_image_hash(image_array)
    if img_hash in _predict_cache:
        return _predict_cache[img_hash]

    # -- Normalise from [0, 255] -> [0, 1] if needed ---------------------------
    if image_array.max() > 1.0:
        image = image_array.astype(np.float32) / 255.0
    else:
        image = image_array.astype(np.float32)

    # -- Add batch dimension ---------------------------------------------------
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)   # (1, 224, 224, 3)

    # -- Run model -------------------------------------------------------------
    probs = model.predict(image, verbose=0)     # shape: (1, num_classes)

    predicted_index = int(np.argmax(probs))
    confidence      = round(float(np.max(probs) * 100), 2)

    if predicted_index < len(CLASS_LABELS):
        predicted_class = CLASS_LABELS[predicted_index]
    else:
        predicted_class = f"Unknown Class ({predicted_index})"

    # -- Generate Grad-CAM Heatmap ---------------------------------------------
    heatmap_b64 = ""
    global grad_model
    try:
        from tensorflow.keras.models import Model
        import tensorflow as tf
        
        last_conv_layer_name = "conv2d_2"
        
        if grad_model is None:
            # Build functional graph once and cache it
            input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
            x = input_layer
            conv_output = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == last_conv_layer_name:
                    conv_output = x
            grad_model = Model(inputs=input_layer, outputs=[conv_output, x])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(image)
            class_channel = preds[:, predicted_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(image_array.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
        
        # Save heatmap image to temp path to keep history (optional)
        import time
        heatmap_path = ROOT_DIR / f"heatmap_{int(time.time())}.jpg"
        cv2.imwrite(str(heatmap_path), superimposed_img)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error generating Grad-CAM heatmap: {e}")

    result = (predicted_class, confidence, heatmap_b64)
    _predict_cache[img_hash] = result
    return result