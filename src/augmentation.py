"""
Data Augmentation Module for Plant Leaf and Fruit Images.

Applies biologically plausible image augmentations (small rotations, mild zooms,
horizontal flips, subtle illumination variations) without altering pathological lesions.
"""

from typing import Optional
import tensorflow as tf

from src.config import config
from src.utils import setup_logger

logger = setup_logger("augmentation", config.LOGS_PATH / "augmentation.log")


def build_augmentation_pipeline(
    rotation_factor: float = 0.08,
    zoom_range: float = 0.08,
    translation_range: float = 0.05,
    contrast_factor: float = 0.10,
    name: str = "data_augmentation"
) -> tf.keras.Sequential:
    """
    Build a realistic data augmentation layer sequence for plant pathology images.

    Args:
        rotation_factor: Fraction of 2*pi for random rotation (0.08 is ~28 degrees).
        zoom_range: Max percentage of random zoom in/out.
        translation_range: Max fraction of image shift along height/width.
        contrast_factor: Max contrast factor variation.
        name: Name for Keras Sequential layer.

    Returns:
        tf.keras.Sequential model representing the augmentation stage.
    """
    layers = [
        tf.keras.layers.RandomFlip("horizontal", name="aug_random_flip_h"),
        tf.keras.layers.RandomRotation(rotation_factor, fill_mode="reflect", name="aug_random_rotation"),
        tf.keras.layers.RandomZoom(
            height_factor=(-zoom_range, zoom_range),
            fill_mode="reflect",
            name="aug_random_zoom"
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=translation_range,
            width_factor=translation_range,
            fill_mode="reflect",
            name="aug_random_translation"
        ),
        tf.keras.layers.RandomContrast(contrast_factor, name="aug_random_contrast")
    ]

    augmentation_model = tf.keras.Sequential(layers, name=name)
    logger.info("Built biological data augmentation pipeline.")
    return augmentation_model


def augment_training_dataset(
    dataset: tf.data.Dataset,
    augmentation_layer: Optional[tf.keras.layers.Layer] = None
) -> tf.data.Dataset:
    """
    Map data augmentation over an input tf.data.Dataset.

    Args:
        dataset: Input training tf.data.Dataset yielding (images, labels).
        augmentation_layer: Keras layer or Sequential pipeline to apply.

    Returns:
        Augmented tf.data.Dataset.
    """
    if augmentation_layer is None:
        augmentation_layer = build_augmentation_pipeline()

    # Apply augmentation on images only, preserve categorical labels
    def _augment(images, labels):
        return augmentation_layer(images, training=True), labels

    augmented_ds = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return augmented_ds
