"""
Utility functions for reproducibility, logging, system profiling,
metric plotting, and computational complexity estimation.
"""

import os
import sys
import json
import random
import logging
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seeds(seed: int = 42) -> None:
    """
    Enforce reproducibility across Python runtime, NumPy, and TensorFlow.

    Args:
        seed: Integer random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_logger(name: str = "leaf_disease", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up a structured logger outputting to both console and log file.

    Args:
        name: Logger name.
        log_file: Optional path to log file.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_hardware_info() -> Dict[str, Any]:
    """
    Detect and document hardware environment for scientific benchmarking.

    Returns:
        Dictionary containing OS, CPU, RAM, and GPU specifications.
    """
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cpu": platform.processor() or "Unknown CPU",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpus": []
    }

    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            info["gpus"].append(gpu.name)
    except ImportError:
        info["tensorflow_version"] = "Not Installed"

    return info


def _json_serializable(obj):
    """Serialize numpy types and Paths for json.dump."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(data: Any, filepath: Path) -> None:
    """Save serializable data to JSON file with indentation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=_json_serializable)


def load_json(filepath: Path) -> Any:
    """Load JSON file content safely."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_size_mb(filepath: Path) -> float:
    """Return file size in megabytes (MB)."""
    if not filepath.exists():
        return 0.0
    return round(filepath.stat().st_size / (1024 * 1024), 2)


def estimate_flops(model) -> int:
    """
    Estimate total floating point operations (FLOPs) for a Keras Functional/Sequential model.

    Args:
        model: tf.keras.Model instance.

    Returns:
        Estimated integer count of FLOPs.
    """
    total_flops = 0
    try:
        for layer in model.layers:
            # Dense layers: 2 * in_dim * out_dim
            if "Dense" in layer.__class__.__name__:
                weights = layer.get_weights()
                if weights:
                    w = weights[0]
                    total_flops += 2 * np.prod(w.shape)
            # Conv2D layers: 2 * (k_h * k_w * c_in) * c_out * out_h * out_w
            elif "Conv2D" in layer.__class__.__name__ or "DepthwiseConv2D" in layer.__class__.__name__:
                weights = layer.get_weights()
                if weights and hasattr(layer, "output_shape") and layer.output_shape:
                    w = weights[0]
                    out_shape = layer.output_shape
                    # out_shape can be (None, H, W, C)
                    if len(out_shape) == 4 and out_shape[1] is not None:
                        h, w_dim = out_shape[1], out_shape[2]
                        total_flops += 2 * np.prod(w.shape) * h * w_dim
    except Exception:
        total_flops = -1
    return total_flops


def plot_training_history(
    history_stage1: Dict[str, List[float]],
    history_stage2: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path] = None,
    model_name: str = "Model"
) -> None:
    """
    Plot and save training/validation accuracy and loss curves,
    clearly distinguishing Stage 1 (Frozen Backbone) from Stage 2 (Fine-Tuning).

    Args:
        history_stage1: History dictionary from initial transfer learning stage.
        history_stage2: Optional history dictionary from fine-tuning stage.
        save_path: Output file path for PNG figure.
        model_name: Name of model for plot title.
    """
    acc = list(history_stage1.get("accuracy", []))
    val_acc = list(history_stage1.get("val_accuracy", []))
    loss = list(history_stage1.get("loss", []))
    val_loss = list(history_stage1.get("val_loss", []))

    stage1_len = len(acc)

    if history_stage2:
        acc.extend(history_stage2.get("accuracy", []))
        val_acc.extend(history_stage2.get("val_accuracy", []))
        loss.extend(history_stage2.get("loss", []))
        val_loss.extend(history_stage2.get("val_loss", []))

    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, val_acc, label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    if history_stage2 and stage1_len < len(acc):
        ax1.axvline(x=stage1_len, color="gray", linestyle="--", label="Start Fine-Tuning")
    ax1.set_title(f"{model_name}: Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs_range, loss, label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2)
    if history_stage2 and stage1_len < len(loss):
        ax2.axvline(x=stage1_len, color="gray", linestyle="--", label="Start Fine-Tuning")
    ax2.set_title(f"{model_name}: Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    model_name: str = "Model",
    normalize: bool = True
) -> None:
    """
    Generate and save a publication-quality Confusion Matrix heatmap.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: List of class name strings.
        save_path: File path to save heatmap PNG.
        model_name: Name of model for plot title.
        normalize: Whether to display percentages or raw counts.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        fmt = ".2%"
        data = cm_norm
    else:
        fmt = "d"
        data = cm

    plt.figure(figsize=(max(8, len(class_names) * 0.9), max(6, len(class_names) * 0.7)))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray"
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_model_parameter_counts(model) -> Dict[str, int]:
    """
    Calculate total, trainable, and non-trainable parameter counts for a Keras model.

    Args:
        model: tf.keras.Model.

    Returns:
        Dict with total_parameters, trainable_parameters, and non_trainable_parameters.
    """
    import tensorflow as tf
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    total_params = trainable_params + non_trainable_params

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }
