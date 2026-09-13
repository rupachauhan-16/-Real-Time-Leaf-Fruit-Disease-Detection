"""
Preprocessing Module.

Implements model-specific normalization, RGB integrity checks,
and safe image transformation for both training pipelines and single-image inference.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
from PIL import Image

import tensorflow as tf
from src.config import config
from src.utils import setup_logger

logger = setup_logger("preprocessing", config.LOGS_PATH / "preprocessing.log")


def get_model_preprocessing_layer(model_name: str) -> tf.keras.layers.Layer:
    """
    Get architecture-tailored normalization layer to embed directly inside the model.

    - MobileNetV3 / MobileNetV2: Normalizes pixel values from [0, 255] to [-1, 1].
    - EfficientNetB0: The Keras EfficientNet backbone natively incorporates Rescaling(1/255)
      or expects standard [0, 255] inputs.

    Args:
        model_name: Architecture name.

    Returns:
        tf.keras.layers.Layer for preprocessing.
    """
    m_name = model_name.lower()
    if "mobilenet_v3" in m_name:
        # MobileNetV3 natively incorporates internal Rescaling(1/127.5, offset=-1.0)
        return tf.keras.layers.Identity(name="mobilenet_v3_passthrough")
    elif "efficientnet" in m_name:
        # EfficientNet natively incorporates internal normalization
        return tf.keras.layers.Identity(name="efficientnet_passthrough")
    elif "mobilenet_v2" in m_name:
        # MobileNetV2 expects [-1, 1] scaling
        return tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_v2_rescaling")
    else:
        return tf.keras.layers.Identity(name="default_passthrough")


def load_and_preprocess_single_image(
    image_input: Union[str, Path, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    model_name: str = "mobilenet_v3_small"
) -> Tuple[np.ndarray, Image.Image]:
    """
    Safely load, format, and preprocess a single image for inference.

    Args:
        image_input: File path or PIL Image instance.
        target_size: Desired (height, width).
        model_name: Backbone architecture name for preprocessing.

    Returns:
        Tuple of (batch_tensor_np: (1, H, W, 3), pil_preview_image).
    """
    try:
        if isinstance(image_input, (str, Path)):
            str_path = str(image_input).strip()
            if str_path.startswith(("http://", "https://")):
                import io
                import requests
                resp = requests.get(str_path, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content))
            else:
                pil_img = Image.open(str_path)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Ensure RGB format
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize using high-quality resampling
        resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)

        # Convert to numpy array float32
        img_array = np.array(resized_pil, dtype=np.float32)

        # Normalize based on architecture if done outside model graph
        m_name = model_name.lower()
        if "mobilenet_v2" in m_name:
            img_array = (img_array / 127.5) - 1.0
        elif "efficientnet" in m_name or "mobilenet_v3" in m_name:
            # EfficientNet and MobileNetV3 expect raw [0, 255] float32 (internal model rescaling)
            pass
        else:
            img_array = img_array / 255.0

        # Expand to batch dimension: (1, H, W, 3)
        batch_tensor = np.expand_dims(img_array, axis=0)
        return batch_tensor, pil_img

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise
