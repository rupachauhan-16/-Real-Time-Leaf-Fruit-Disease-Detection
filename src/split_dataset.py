"""
Stratified Dataset Splitter.

Splits an unorganized or single-directory image dataset into
train, validation, and test partitions while maintaining class distribution.
"""

import sys
import os
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from PIL import Image

from src.config import config
from src.utils import setup_logger, set_random_seeds

logger = setup_logger("split_dataset", config.LOGS_PATH / "dataset_split.log")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(file_path: Path) -> bool:
    """Verify that an image file can be opened and decoded without corruption."""
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Ensure it can actually be loaded into memory
        with Image.open(file_path) as img:
            img.load()
        return True
    except Exception as e:
        logger.warning(f"Corrupt or unreadable image skipped: {file_path} ({e})")
        return False


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True
) -> None:
    """
    Perform stratified split of class-folder formatted images into train/val/test splits.

    Args:
        source_dir: Directory containing subdirectories for each disease class.
        output_dir: Destination root directory where train, validation, and test folders will be created.
        train_ratio: Proportion of data for training (e.g., 0.70).
        val_ratio: Proportion of data for validation (e.g., 0.15).
        test_ratio: Proportion of data for testing (e.g., 0.15).
        seed: Random seed for reproducibility.
        copy_files: If True, copy files; if False, create symlinks/moves.
    """
    set_random_seeds(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory '{source_dir}' does not exist.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio:.3f})")

    # Discover classes
    classes = [d.name for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not classes:
        raise ValueError(f"No class subdirectories found inside '{source_dir}'.")

    classes.sort()
    logger.info(f"Found {len(classes)} classes in '{source_dir}': {classes}")

    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            (d / cls).mkdir(parents=True, exist_ok=True)

    summary = []

    for cls in classes:
        cls_dir = source_dir / cls
        all_files = [
            f for f in cls_dir.iterdir()
            if f.is_file() and is_valid_image(f)
        ]

        if len(all_files) < 3:
            logger.warning(f"Class '{cls}' has fewer than 3 valid images ({len(all_files)}). Splitting may be degenerate.")

        # First split: Train vs (Val + Test)
        val_test_ratio = val_ratio + test_ratio
        train_files, temp_files = train_test_split(
            all_files,
            test_size=val_test_ratio,
            random_state=seed,
            shuffle=True
        )

        # Second split: Val vs Test
        relative_test_ratio = test_ratio / val_test_ratio
        val_files, test_files = train_test_split(
            temp_files,
            test_size=relative_test_ratio,
            random_state=seed,
            shuffle=True
        )

        # Copy files to designated splits
        for split_name, file_list, target_root in [
            ("train", train_files, train_dir),
            ("validation", val_files, val_dir),
            ("test", test_files, test_dir),
        ]:
            for f in file_list:
                dest = target_root / cls / f.name
                if copy_files:
                    shutil.copy2(f, dest)
                else:
                    shutil.move(f, dest)

        summary.append({
            "class": cls,
            "total": len(all_files),
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        })

    logger.info("Stratified dataset split completed successfully.")
    for s in summary:
        logger.info(
            f"Class '{s['class']}': Total={s['total']} | "
            f"Train={s['train']} | Val={s['validation']} | Test={s['test']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an unorganized image dataset into train/val/test partitions.")
    parser.add_argument("--source", type=str, required=True, help="Path to raw source directory containing class folders")
    parser.add_argument("--output", type=str, default=str(config.DATASET_PATH), help="Path to output dataset root")
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")

    args = parser.parse_args()
    split_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed
    )
