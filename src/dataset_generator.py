"""
Sample Dataset Generator for Leaf and Fruit Disease Detection.

Generates realistic synthetic plant leaf and fruit images with distinct
pathological features (e.g., Early Blight concentric rings, Black Rot lesions,
Healthy green leaf tissue) for instant verification and pipeline testing.
"""

import sys
import math
import random
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import config
from src.utils import setup_logger, set_random_seeds

logger = setup_logger("dataset_generator", config.LOGS_PATH / "dataset_generator.log")

DEFAULT_CLASSES = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Apple_Black_Rot",
    "Apple_Healthy"
]


def _create_base_leaf(width: int, height: int, is_apple: bool = False) -> Image.Image:
    """Draw a base leaf shape with natural green texture and vein structure."""
    # Background: natural soil or neutral lab background
    bg_color = (
        random.randint(45, 65),
        random.randint(35, 55),
        random.randint(25, 45)
    )
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Leaf dimensions
    if is_apple:
        rx = int(width * random.uniform(0.28, 0.35))
        ry = int(height * random.uniform(0.38, 0.44))
        leaf_color = (
            random.randint(30, 50),
            random.randint(120, 160),
            random.randint(30, 50)
        )
    else:  # Tomato leaf (more elongated, serrated)
        rx = int(width * random.uniform(0.22, 0.30))
        ry = int(height * random.uniform(0.40, 0.46))
        leaf_color = (
            random.randint(40, 60),
            random.randint(140, 180),
            random.randint(40, 60)
        )

    # Draw leaf body as an ellipse
    bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(bbox, fill=leaf_color)

    # Main vein
    vein_color = (
        min(255, leaf_color[0] + 25),
        min(255, leaf_color[1] + 30),
        min(255, leaf_color[2] + 20)
    )
    draw.line([(cx, cy - ry + 10), (cx, cy + ry - 10)], fill=vein_color, width=3)

    # Lateral veins
    for dy in range(-ry + 25, ry - 25, 20):
        y = cy + dy
        draw.line([(cx, y), (cx - int(rx * 0.75), y - 10)], fill=vein_color, width=2)
        draw.line([(cx, y), (cx + int(rx * 0.75), y - 10)], fill=vein_color, width=2)

    # Apply slight blur for realistic organic softness
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    return img


def _add_early_blight_symptoms(img: Image.Image) -> Image.Image:
    """Add characteristic target-board / concentric brown ring lesions for Early Blight."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w // 2, h // 2

    num_spots = random.randint(3, 7)
    for _ in range(num_spots):
        sx = cx + random.randint(-w // 4, w // 4)
        sy = cy + random.randint(-h // 4, h // 4)
        max_r = random.randint(12, 28)

        # Concentric rings: outer chlorotic yellow halo -> brown ring -> necrotic dark center
        # Yellow halo
        draw.ellipse([sx - max_r - 4, sy - max_r - 4, sx + max_r + 4, sy + max_r + 4], fill=(180, 175, 40))
        # Outer brown ring
        draw.ellipse([sx - max_r, sy - max_r, sx + max_r, sy + max_r], fill=(110, 60, 25))
        # Inner tan ring
        draw.ellipse([sx - int(max_r * 0.65), sy - int(max_r * 0.65), sx + int(max_r * 0.65), sy + int(max_r * 0.65)], fill=(140, 85, 35))
        # Dark necrotic center
        draw.ellipse([sx - int(max_r * 0.35), sy - int(max_r * 0.35), sx + int(max_r * 0.35), sy + int(max_r * 0.35)], fill=(50, 25, 10))

    return img.filter(ImageFilter.GaussianBlur(radius=0.7))


def _add_black_rot_symptoms(img: Image.Image) -> Image.Image:
    """Add characteristic irregular necrotic black-brown rot spots for Apple Black Rot."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx, cy = w // 2, h // 2

    num_lesions = random.randint(2, 5)
    for _ in range(num_lesions):
        lx = cx + random.randint(-w // 5, w // 5)
        ly = cy + random.randint(-h // 5, h // 5)
        r = random.randint(18, 35)

        # Irregular black/dark brown patch
        for offset in range(5):
            ox = lx + random.randint(-4, 4)
            oy = ly + random.randint(-4, 4)
            draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=(25, 15, 12))

        # Reddish-brown outer margin
        draw.ellipse([lx - r - 3, ly - r - 3, lx + r + 3, ly + r + 3], outline=(90, 30, 20), width=2)

    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def generate_synthetic_dataset(
    output_dir: Path = config.DATASET_PATH,
    samples_per_split: Tuple[int, int, int] = (30, 10, 10),
    classes: List[str] = None,
    seed: int = 42
) -> None:
    """
    Generate complete train, validation, and test datasets with synthetic leaf/fruit disease images.

    Args:
        output_dir: Root dataset path.
        samples_per_split: (train_count, val_count, test_count) per class.
        classes: List of disease class names.
        seed: Random seed.
    """
    set_random_seeds(seed)
    if classes is None:
        classes = DEFAULT_CLASSES

    splits = [
        ("train", samples_per_split[0]),
        ("validation", samples_per_split[1]),
        ("test", samples_per_split[2]),
    ]

    logger.info(f"Generating synthetic dataset in '{output_dir}' across {len(classes)} classes...")

    for split_name, count in splits:
        split_path = output_dir / split_name
        for cls_name in classes:
            class_folder = split_path / cls_name
            class_folder.mkdir(parents=True, exist_ok=True)

            is_apple = "apple" in cls_name.lower()
            is_blight = "blight" in cls_name.lower()
            is_rot = "rot" in cls_name.lower()

            for i in range(count):
                img = _create_base_leaf(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, is_apple=is_apple)

                if is_blight:
                    img = _add_early_blight_symptoms(img)
                elif is_rot:
                    img = _add_black_rot_symptoms(img)

                # Save as JPG
                filename = f"{cls_name}_{split_name}_{i+1:04d}.jpg"
                img.save(class_folder / filename, "JPEG", quality=95)

            logger.info(f"Created {count} images in '{split_name}/{cls_name}'")

    logger.info("Sample synthetic dataset generation complete.")


if __name__ == "__main__":
    generate_synthetic_dataset()
