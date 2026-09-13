"""
Model Loader and Initialization Module.

Loads trained Keras and TFLite models once, performs warmup inference
to eliminate cold-start overhead, verifies hardware execution context (CPU/GPU),
and returns an initialized production inference handle.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.config import config
from src.utils import setup_logger, load_json, get_hardware_info

logger = setup_logger("model_loader", config.LOGS_PATH / "model.log")


class LoadedModelHandle:
    """Encapsulates a loaded neural network ready for real-time inference."""

    def __init__(
        self,
        model: Any,
        is_tflite: bool,
        class_names: List[str],
        model_path: Path,
        device: str,
        input_details: Optional[List[Dict[str, Any]]] = None,
        output_details: Optional[List[Dict[str, Any]]] = None
    ):
        self.model = model
        self.is_tflite = is_tflite
        self.class_names = class_names
        self.model_path = model_path
        self.model_name = model_path.stem
        self.device = device
        self.input_details = input_details
        self.output_details = output_details

    def run_inference(self, tensor_batch: np.ndarray) -> np.ndarray:
        """Run single forward pass and return raw probability array."""
        if self.is_tflite:
            self.model.set_tensor(self.input_details[0]["index"], tensor_batch)
            self.model.invoke()
            return self.model.get_tensor(self.output_details[0]["index"])[0]
        else:
            return self.model(tensor_batch, training=False).numpy()[0]


# Module-level cache to ensure the model is loaded only once in memory
_CACHED_MODEL_HANDLES: Dict[str, LoadedModelHandle] = {}


def detect_compute_device() -> str:
    """Detect available compute device (NVIDIA GPU or CPU)."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return f"NVIDIA GPU ({len(gpus)} device(s))"
    except Exception:
        pass
    return "CPU (Optimized Edge)"


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    class_names_path: Optional[Union[str, Path]] = None,
    warmup: bool = True,
    num_warmup_runs: int = config.NUM_WARMUP_RUNS
) -> LoadedModelHandle:
    """
    Load and initialize the trained deep learning model once.

    Args:
        model_path: Path to .keras or .tflite model. If None, auto-selects from models/.
        class_names_path: Path to class_names.json. If None, uses config.CLASS_NAMES_FILE.
        warmup: Whether to execute warmup inferences to eliminate latency spikes.
        num_warmup_runs: Number of dummy forward passes.

    Returns:
        LoadedModelHandle ready for zero-overhead inference.
    """
    # 1. Resolve Model Path
    if model_path is None:
        target_path = config.MODEL_PATH
        if not target_path.exists():
            # Check for any .keras file in models/
            keras_files = sorted(list(config.MODELS_PATH.glob("*.keras")))
            if keras_files:
                target_path = keras_files[0]
            else:
                tflite_files = sorted(list(config.MODELS_PATH.glob("*.tflite")))
                if tflite_files:
                    target_path = tflite_files[0]
                else:
                    raise FileNotFoundError(
                        f"Trained model not found at '{config.MODELS_PATH}'. "
                        f"Please train a model first using: python src/train.py"
                    )
    else:
        target_path = Path(model_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Trained model file '{target_path}' does not exist.")

    cache_key = str(target_path.resolve())
    if cache_key in _CACHED_MODEL_HANDLES:
        logger.info(f"Returning cached model handle for '{target_path.name}'.")
        return _CACHED_MODEL_HANDLES[cache_key]

    # 2. Resolve Class Names
    cls_path = Path(class_names_path) if class_names_path else config.CLASS_NAMES_PATH
    if not cls_path.exists():
        raise FileNotFoundError(
            f"Class names file not found at '{cls_path}'. "
            f"Ensure dataset validation has run or class_names.json is present."
        )
    class_names = load_json(cls_path)

    # 3. Detect Device & Load Model Architecture
    device_name = detect_compute_device()
    is_tflite = target_path.suffix.lower() == ".tflite"

    logger.info(f"Loading model '{target_path.name}' onto {device_name}...")
    t_start = time.perf_counter()

    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=str(target_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        handle = LoadedModelHandle(
            model=interpreter,
            is_tflite=True,
            class_names=class_names,
            model_path=target_path,
            device=device_name,
            input_details=input_details,
            output_details=output_details
        )
    else:
        keras_model = tf.keras.models.load_model(target_path)
        handle = LoadedModelHandle(
            model=keras_model,
            is_tflite=False,
            class_names=class_names,
            model_path=target_path,
            device=device_name
        )

    load_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    logger.info(f"Model '{target_path.name}' loaded successfully in {load_time_ms} ms.")

    # 4. Warm-Up Inferences
    if warmup and num_warmup_runs > 0:
        logger.info(f"Executing {num_warmup_runs} warmup inference passes...")
        dummy_input = np.zeros((1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS), dtype=np.float32)
        for _ in range(num_warmup_runs):
            _ = handle.run_inference(dummy_input)
        logger.info("Warmup complete. Inference pipeline is primed for real-time execution.")

    _CACHED_MODEL_HANDLES[cache_key] = handle
    return handle

