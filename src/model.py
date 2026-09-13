"""
Model Architecture and Transfer Learning Factory.

Constructs lightweight convolutional neural networks (MobileNetV3Small, MobileNetV2,
and EfficientNetB0) optimized for time-efficient leaf and fruit disease classification.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.config import config
from src.preprocessing import get_model_preprocessing_layer
from src.utils import setup_logger

logger = setup_logger("model", config.LOGS_PATH / "model.log")


def build_disease_model(
    model_name: str = "mobilenet_v3_small",
    num_classes: int = 4,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-4,
    weights: str = "imagenet"
) -> tf.keras.Model:
    """
    Construct a transfer learning model with a lightweight CNN backbone
    and a custom classification head.

    Args:
        model_name: "mobilenet_v3_small", "mobilenet_v2", or "efficientnet_b0".
        num_classes: Dynamic number of disease categories.
        input_shape: (height, width, channels) of input images.
        dropout_rate: Dropout probability for regularization.
        l2_reg: L2 weight decay parameter.
        weights: Pretrained weights ("imagenet" or None).

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    m_name = model_name.lower().strip()

    # Define Input Tensor
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Embedded Preprocessing Layer
    preprocessing_layer = get_model_preprocessing_layer(m_name)
    x = preprocessing_layer(inputs)

    # Select Backbone Architecture
    if m_name == "mobilenet_v3_small":
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None
        )
    elif m_name == "mobilenet_v2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None
        )
    elif m_name == "efficientnet_b0":
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling=None
        )
    else:
        supported = config.SUPPORTED_MODELS
        raise ValueError(f"Unsupported model '{model_name}'. Supported options: {supported}")

    # Initially freeze backbone for Stage 1 (Feature Extraction)
    backbone.trainable = False

    # Pass preprocessed tensor through backbone
    x = backbone(x, training=False)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="head_dense_projection"
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_2")(x)

    # Output probabilities
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="disease_predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{m_name}_disease_classifier")
    logger.info(
        f"Built '{model.name}' for {num_classes} classes. "
        f"Stage 1: Backbone frozen ({len(backbone.layers)} backbone layers)."
    )
    return model


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    unfreeze_percent: float = 0.30
) -> tf.keras.Model:
    """
    Unfreeze the top portion of the backbone for Stage 2 Fine-Tuning,
    while keeping BatchNormalization layers frozen for statistical stability.

    Args:
        model: tf.keras.Model with a backbone sub-layer or backbone layers.
        unfreeze_percent: Fraction of backbone layers to unfreeze (0.0 to 1.0).

    Returns:
        Model with selectively unfrozen deeper layers.
    """
    # Locate the backbone model inside the functional graph
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and any(kw in layer.name.lower() for kw in ["mobilenet", "efficientnet"]):
            backbone = layer
            break

    if backbone is None:
        # If backbone layers are directly in model
        layers = model.layers
    else:
        backbone.trainable = True
        layers = backbone.layers

    num_layers = len(layers)
    num_to_unfreeze = int(num_layers * unfreeze_percent)
    split_index = max(0, num_layers - num_to_unfreeze)

    unfrozen_count = 0
    for idx, layer in enumerate(layers):
        if idx >= split_index:
            # Best practice: Do not train BatchNormalization layers during fine-tuning
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
                unfrozen_count += 1
        else:
            layer.trainable = False

    logger.info(
        f"Stage 2 Fine-Tuning: Unfroze top {unfrozen_count} of {num_layers} layers "
        f"({unfreeze_percent*100:.1f}%). BatchNormalization layers kept frozen."
    )
    return model


def get_model_parameter_counts(model: tf.keras.Model) -> Dict[str, int]:
    """
    Calculate total, trainable, and non-trainable parameter counts.

    Args:
        model: tf.keras.Model.

    Returns:
        Dict with 'total', 'trainable', and 'non_trainable' counts.
    """
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    total_params = trainable_params + non_trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }
