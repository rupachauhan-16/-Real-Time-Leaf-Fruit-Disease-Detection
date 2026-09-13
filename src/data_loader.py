"""
Data Loader and Dataset Pipeline Module.

Handles dataset validation, corrupted image filtering, dynamic class discovery,
class imbalance weighting, and optimized tf.data pipeline creation.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from src.config import config, Config
from src.utils import setup_logger, save_json

logger = setup_logger("data_loader", config.LOGS_PATH / "data_loader.log")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def verify_image(image_path: Path) -> bool:
    """Safely verify that an image is readable and not corrupt."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
            if img.mode not in ("RGB", "RGBA", "L"):
                return False
        return True
    except Exception as e:
        logger.warning(f"Corrupted or invalid image detected: {image_path} ({e})")
        return False


def get_class_names_from_directory(directory: Path) -> List[str]:
    """Dynamically scan directory for valid class subfolders in alphabetical order."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    classes = [
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith((".", "_"))
    ]
    classes.sort()
    if not classes:
        raise ValueError(f"No valid class subdirectories found in '{directory}'.")

    return classes


def inspect_and_clean_split(split_dir: Path) -> Dict[str, int]:
    """
    Validate images in a split directory and count valid images per class.

    Args:
        split_dir: Path to split (train, validation, or test).

    Returns:
        Dictionary mapping class name to count of valid images.
    """
    counts = {}
    classes = get_class_names_from_directory(split_dir)

    for cls in classes:
        cls_path = split_dir / cls
        valid_files = 0
        corrupt_files = 0

        for f in cls_path.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                if verify_image(f):
                    valid_files += 1
                else:
                    corrupt_files += 1
                    try:
                        f.unlink()
                        logger.info(f"Removed corrupt image: {f}")
                    except Exception as err:
                        logger.error(f"Failed to remove corrupt image {f}: {err}")

        if corrupt_files > 0:
            logger.warning(f"Removed {corrupt_files} corrupt images in '{split_dir.name}/{cls}'")

        counts[cls] = valid_files

    return counts


def validate_dataset_structure(cfg: Config = config) -> Dict[str, Dict[str, int]]:
    """
    Validate that train, validation, and test directories exist and contain matching classes.

    Args:
        cfg: Configuration instance.

    Returns:
        Dictionary containing counts per split and per class.
    """
    if not cfg.TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory '{cfg.TRAIN_DIR}' does not exist.")
    if not cfg.VAL_DIR.exists():
        raise FileNotFoundError(f"Validation directory '{cfg.VAL_DIR}' does not exist.")

    train_counts = inspect_and_clean_split(cfg.TRAIN_DIR)
    val_counts = inspect_and_clean_split(cfg.VAL_DIR)

    test_counts = {}
    if cfg.TEST_DIR.exists():
        test_counts = inspect_and_clean_split(cfg.TEST_DIR)

    train_classes = sorted(list(train_counts.keys()))
    val_classes = sorted(list(val_counts.keys()))

    if train_classes != val_classes:
        raise ValueError(
            f"Class mismatch between train and validation splits!\n"
            f"Train classes: {train_classes}\n"
            f"Val classes:   {val_classes}"
        )

    # Persist class names for downstream inference
    save_json(train_classes, cfg.CLASS_NAMES_FILE)
    logger.info(f"Validated dataset with {len(train_classes)} classes. Saved classes to '{cfg.CLASS_NAMES_FILE}'.")

    summary = {
        "train": train_counts,
        "validation": val_counts,
        "test": test_counts
    }
    return summary


def compute_imbalance_weights(train_dir: Path, class_names: List[str]) -> Dict[int, float]:
    """
    Compute class weights using balanced inverse frequency to mitigate class imbalance.

    Args:
        train_dir: Path to training split.
        class_names: Sorted list of class names.

    Returns:
        Dictionary mapping integer class index to its floating point weight.
    """
    y_train = []
    for idx, cls in enumerate(class_names):
        cls_path = train_dir / cls
        files = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        y_train.extend([idx] * len(files))

    y_train = np.array(y_train)
    unique_classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )

    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, weights)}
    logger.info(f"Computed class imbalance weights: {class_weight_dict}")
    return class_weight_dict


def load_dataset_pipelines(
    cfg: Config = config,
    batch_size: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], List[str]]:
    """
    Create high-throughput tf.data pipelines for train, validation, and test sets.

    Args:
        cfg: Configuration instance.
        batch_size: Override batch size.
        image_size: Override image dimensions (height, width).

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names).
    """
    b_size = batch_size or cfg.BATCH_SIZE
    img_sz = image_size or cfg.IMAGE_SIZE

    validate_dataset_structure(cfg)
    class_names = get_class_names_from_directory(cfg.TRAIN_DIR)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=b_size,
        image_size=img_sz,
        shuffle=True,
        seed=cfg.RANDOM_SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=b_size,
        image_size=img_sz,
        shuffle=False
    )

    test_ds = None
    if cfg.TEST_DIR.exists() and any(cfg.TEST_DIR.iterdir()):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            cfg.TEST_DIR,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            color_mode="rgb",
            batch_size=b_size,
            image_size=img_sz,
            shuffle=False
        )

    # Performance optimization: cache and prefetch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names
