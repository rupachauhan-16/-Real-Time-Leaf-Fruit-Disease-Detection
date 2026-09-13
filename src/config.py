"""
Centralized Configuration Module for Leaf and Fruit Disease Detection.

Defines all paths, model hyperparameters, camera configurations,
and evaluation settings to prevent hardcoding across the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Master configuration class."""

    # -----------------------------
    # Directory & File Paths
    # -----------------------------
    ROOT_DIR: Path = PROJECT_ROOT
    DATASET_PATH: Path = PROJECT_ROOT / "dataset"
    TRAIN_DIR: Path = PROJECT_ROOT / "dataset" / "train"
    VAL_DIR: Path = PROJECT_ROOT / "dataset" / "validation"
    TEST_DIR: Path = PROJECT_ROOT / "dataset" / "test"

    MODELS_PATH: Path = PROJECT_ROOT / "models"
    RESULTS_PATH: Path = PROJECT_ROOT / "results"
    CAPTURES_PATH: Path = PROJECT_ROOT / "results" / "captures"
    BENCHMARKS_PATH: Path = PROJECT_ROOT / "results" / "benchmarks"
    PLOTS_PATH: Path = PROJECT_ROOT / "results" / "plots"
    LOGS_PATH: Path = PROJECT_ROOT / "logs"

    # Default Model File & Dynamic Class Names File
    DEFAULT_MODEL_FILE: Path = PROJECT_ROOT / "models" / "best_mobilenet_v3_small.keras"
    CLASS_NAMES_FILE: Path = PROJECT_ROOT / "models" / "class_names.json"

    # Aliases for prompt naming consistency
    @property
    def MODEL_PATH(self) -> Path:
        if self.DEFAULT_MODEL_FILE.exists():
            return self.DEFAULT_MODEL_FILE
        keras_files = list(self.MODELS_PATH.glob("*.keras"))
        return keras_files[0] if keras_files else self.DEFAULT_MODEL_FILE

    @property
    def CLASS_NAMES_PATH(self) -> Path:
        return self.CLASS_NAMES_FILE

    # -----------------------------
    # Model Architectures
    # -----------------------------
    SUPPORTED_MODELS: List[str] = field(
        default_factory=lambda: [
            "mobilenet_v3_small",
            "mobilenet_v2",
            "efficientnet_b0",
        ]
    )
    DEFAULT_MODEL: str = "efficientnet_b0"

    # -----------------------------
    # Image & Preprocessing Settings
    # -----------------------------
    IMAGE_HEIGHT: int = 224
    IMAGE_WIDTH: int = 224
    CHANNELS: int = 3

    @property
    def IMAGE_SIZE(self) -> Tuple[int, int]:
        return (self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

    @property
    def INPUT_SHAPE(self) -> Tuple[int, int, int]:
        return (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS)

    # -----------------------------
    # Real-Time Camera Settings
    # -----------------------------
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_SKIP: int = int(os.getenv("FRAME_SKIP", "2"))
    CAMERA_FPS_TARGET: int = 30
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    FPS_SMOOTHING_WINDOW: int = 15
    NUM_WARMUP_RUNS: int = 10

    # -----------------------------
    # Inference & Safety Threshold
    # -----------------------------
    CONFIDENCE_THRESHOLD: float = 0.50

    # -----------------------------
    # Training Hyperparameters
    # -----------------------------
    BATCH_SIZE: int = 32
    STAGE1_EPOCHS: int = 10
    STAGE2_EPOCHS: int = 20
    TOTAL_EPOCHS: int = 30

    INITIAL_LEARNING_RATE: float = 1e-3
    FINE_TUNE_LEARNING_RATE: float = 1e-4

    DROPOUT_RATE: float = 0.2
    L2_REGULARIZATION: float = 1e-4

    # -----------------------------
    # Regularization & Callbacks
    # -----------------------------
    EARLY_STOPPING_PATIENCE: int = 5
    REDUCE_LR_PATIENCE: int = 3
    REDUCE_LR_FACTOR: float = 0.2
    MIN_LEARNING_RATE: float = 1e-6
    FINE_TUNE_UNFREEZE_PERCENT: float = 0.30

    # -----------------------------
    # Benchmarking & Profiling
    # -----------------------------
    BENCHMARK_WARMUP_RUNS: int = 15
    BENCHMARK_ITERATIONS: int = 100

    # -----------------------------
    # Reproducibility
    # -----------------------------
    RANDOM_SEED: int = 42

    def ensure_directories(self) -> None:
        """Ensure all required output and project directories exist."""
        for path in [
            self.DATASET_PATH,
            self.TRAIN_DIR,
            self.VAL_DIR,
            self.TEST_DIR,
            self.MODELS_PATH,
            self.RESULTS_PATH,
            self.CAPTURES_PATH,
            self.BENCHMARKS_PATH,
            self.PLOTS_PATH,
            self.LOGS_PATH,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Default global instance
config = Config()
config.ensure_directories()
