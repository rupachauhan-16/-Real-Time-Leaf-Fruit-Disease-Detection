"""
Unified Real-Time Prediction Engine for Leaf and Fruit Disease Detection.

Serves as the shared, single source of truth for both:
1. Real-Time Camera Stream Detection (OpenCV video frames)
2. Image Upload Detection (File, PIL, URL)

Eliminates code duplication and guarantees consistent preprocessing,
confidence thresholding, and Out-of-Distribution (OOD) protection.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from src.config import config
from src.utils import setup_logger
from src.model_loader import load_model, LoadedModelHandle
from src.ood_detector import evaluate_crop_and_ood

logger = setup_logger("predictor", config.LOGS_PATH / "predict.log")


def preprocess_image(
    image_input: Union[np.ndarray, Image.Image, str, Path],
    target_size: Tuple[int, int] = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    model_name: str = "mobilenet_v3_small"
) -> Tuple[np.ndarray, Image.Image]:
    """
    Unified high-performance image preprocessor.

    Handles:
    - OpenCV BGR Camera Frames (np.ndarray) -> Converts BGR to RGB!
    - PIL Image objects -> Converts to RGB
    - File paths / URLs -> Reads and validates

    Args:
        image_input: NumPy frame, PIL Image, or path.
        target_size: (height, width) tuple (default: 224x224).
        model_name: Architecture name for proper normalization.

    Returns:
        Tuple of:
            tensor_batch: Shape (1, H, W, 3) float32 ready for model input.
            pil_image: PIL Image instance for display or metadata.
    """
    if isinstance(image_input, np.ndarray):
        # OpenCV camera frame (BGR)
        if len(image_input.shape) == 2:
            frame_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            frame_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        # Fast OpenCV resize
        resized_arr = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        img_array = resized_arr.astype(np.float32)
        pil_image = Image.fromarray(frame_rgb)

    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
        resized_pil = pil_image.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(resized_pil, dtype=np.float32)

    elif isinstance(image_input, (str, Path)):
        str_path = str(image_input).strip()
        if str_path.startswith(("http://", "https://")):
            import io
            import requests
            resp = requests.get(str_path, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            pil_image = Image.open(str_path).convert("RGB")

        resized_pil = pil_image.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(resized_pil, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Architecture-specific normalization
    m_name = model_name.lower()
    if "mobilenet_v2" in m_name:
        img_array = (img_array / 127.5) - 1.0
    elif "efficientnet" in m_name or "mobilenet_v3" in m_name:
        pass  # EfficientNet and MobileNetV3 contain internal normalization layers
    else:
        img_array = img_array / 255.0

    tensor_batch = np.expand_dims(img_array, axis=0)
    return tensor_batch, pil_image


def get_top_predictions(
    raw_probs: np.ndarray,
    class_names: List[str],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Return ranked list of top candidate predictions."""
    top_indices = np.argsort(raw_probs)[::-1][:min(top_k, len(class_names))]
    return [
        {
            "class_name": class_names[i].replace("_", " "),
            "raw_class_name": class_names[i],
            "confidence": float(raw_probs[i]),
            "confidence_percent": round(float(raw_probs[i]) * 100.0, 2)
        }
        for i in top_indices
    ]


def predict_image(
    image_input: Union[np.ndarray, Image.Image, str, Path],
    model_handle: Optional[LoadedModelHandle] = None,
    top_k: int = 3,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    filename_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core prediction function consumed by both Camera and Upload modes.

    Args:
        image_input: OpenCV frame (np.ndarray), PIL Image, or file path.
        model_handle: Initialized LoadedModelHandle (auto-loads default if None).
        top_k: Number of candidate classes to return.
        confidence_threshold: Minimum probability for reliable prediction.
        filename_hint: Optional filename or source descriptor for crop validation.

    Returns:
        Structured prediction result dictionary.
    """
    if model_handle is None:
        model_handle = load_model()

    # Preprocess image
    tensor_batch, pil_img = preprocess_image(
        image_input=image_input,
        target_size=config.IMAGE_SIZE,
        model_name=model_handle.model_name
    )

    # Isolated Inference Latency Timing
    t_start = time.perf_counter()
    raw_probs = model_handle.run_inference(tensor_batch)
    t_end = time.perf_counter()
    inference_time_ms = round((t_end - t_start) * 1000.0, 2)

    # Top Candidate Breakdown
    pred_idx = int(np.argmax(raw_probs))
    raw_confidence = float(raw_probs[pred_idx])
    raw_class = model_handle.class_names[pred_idx]

    # Evaluate Out-of-Distribution & Crop Domain
    ood_result = evaluate_crop_and_ood(
        raw_probabilities=raw_probs,
        class_names=model_handle.class_names,
        image_input=pil_img,
        filename_hint=filename_hint,
        confidence_threshold=confidence_threshold
    )

    top_candidates = get_top_predictions(raw_probs, model_handle.class_names, top_k=top_k)

    # Determine final user-facing disease text
    is_reliable = ood_result["is_reliable"]
    disease_display = ood_result["display_diagnosis"]
    raw_class = ood_result["raw_predicted_class"]
    final_confidence = ood_result["top_confidence"]

    if ood_result["is_out_of_distribution"]:
        status_message = ood_result["rejection_reason"]
    elif not is_reliable:
        status_message = ood_result["rejection_reason"]
    else:
        status_message = "Reliable Diagnosis"

    return {
        "disease": disease_display,
        "predicted_disease": disease_display,
        "raw_class_name": raw_class,
        "confidence": final_confidence,
        "confidence_percent": round(final_confidence * 100.0, 2),
        "inference_time_ms": inference_time_ms,
        "is_reliable": is_reliable,
        "is_out_of_distribution": ood_result["is_out_of_distribution"],
        "decision_category": ood_result["decision_category"],
        "detected_crop": ood_result["detected_crop"],
        "detected_crop_meta": ood_result["detected_crop_meta"],
        "rejection_reason": ood_result["rejection_reason"],
        "status_message": status_message,
        "threshold": confidence_threshold,
        "model_used": model_handle.model_name,
        "device": model_handle.device,
        "top_predictions": top_candidates
    }


class DiseasePredictor:
    """Class wrapper maintaining backward compatibility with existing scripts."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD
    ):
        self.handle = load_model(
            model_path=model_path,
            class_names_path=class_names_path,
            warmup=True
        )
        self.confidence_threshold = confidence_threshold
        self.class_names = self.handle.class_names
        self.model_path = self.handle.model_path

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        top_k: int = 3,
        filename_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        return predict_image(
            image_input=image_input,
            model_handle=self.handle,
            top_k=top_k,
            confidence_threshold=self.confidence_threshold,
            filename_hint=filename_hint
        )

