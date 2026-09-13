"""
Real-Time OpenCV Camera Handler & Visual Analytics Module.

Features:
- Configurable camera index with graceful error handling.
- Rolling FPS calculation using a sliding time window.
- Isolated inference latency measurement.
- Configurable frame skipping (e.g., run inference every 2nd or 3rd frame).
- Real-time HUD visual overlay with color-coded reliability.
- Center Region of Interest (ROI) targeting box for precise positioning.
- Timestamped snapshot capture saved to results/captures/.
- Rolling prediction history buffer.
"""

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.config import config
from src.utils import setup_logger
from src.model_loader import load_model, LoadedModelHandle
from src.predictor import predict_image

logger = setup_logger("camera", config.LOGS_PATH / "predict.log")


class RealTimeCameraHandler:
    """Production OpenCV camera pipeline with frame-skipping and HUD overlays."""

    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        frame_skip: int = config.FRAME_SKIP,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        model_handle: Optional[LoadedModelHandle] = None,
        history_len: int = 10,
        enable_roi: bool = True
    ):
        self.camera_index = camera_index
        self.frame_skip = max(1, frame_skip)
        self.confidence_threshold = confidence_threshold
        self.enable_roi = enable_roi

        # Load model handle (reused across all frames, loaded once)
        self.model_handle = model_handle if model_handle is not None else load_model()

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_counter = 0

        # Performance analytics
        self.fps_window = deque(maxlen=config.FPS_SMOOTHING_WINDOW)
        self.current_fps = 0.0
        self.last_inference_time_ms = 0.0

        # Prediction cache (displayed during skipped frames)
        self.last_prediction: Optional[Dict[str, Any]] = None

        # Prediction history buffer
        self.history = deque(maxlen=history_len)

    def open_camera(self) -> bool:
        """Initialize OpenCV camera capture."""
        logger.info(f"Opening camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        # Set target resolution if supported
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not self.cap.isOpened():
            logger.error(
                f"Unable to access camera (Index: {self.camera_index}). "
                f"Please check camera permissions or connection."
            )
            return False

        logger.info(f"Camera index {self.camera_index} initialized successfully.")
        return True

    def release_camera(self) -> None:
        """Safely release camera hardware."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")
        self.cap = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single raw frame from camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def capture_snapshot(self, frame: np.ndarray) -> Path:
        """Save clean snapshot without HUD overlay to results/captures/."""
        config.CAPTURES_PATH.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"capture_{timestamp}.jpg"
        save_path = config.CAPTURES_PATH / filename
        cv2.imwrite(str(save_path), frame)
        logger.info(f"Captured snapshot saved to '{save_path}'.")
        return save_path

    def process_frame(
        self,
        frame: np.ndarray,
        force_inference: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], float]:
        """
        Process a single camera frame:
        - Updates rolling FPS
        - Executes model inference according to FRAME_SKIP
        - Draws real-time HUD and ROI
        """
        t_frame_start = time.perf_counter()
        self.frame_counter += 1

        h, w, _ = frame.shape
        roi_box = None
        roi_frame = frame

        # Compute Center Region of Interest (ROI) if enabled
        if self.enable_roi:
            roi_size = min(h, w) * 3 // 4
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            roi_box = (x1, y1, x2, y2)
            roi_frame = frame[y1:y2, x1:x2]

        # Execute AI Inference on scheduled frames
        should_infer = force_inference or (self.frame_counter % self.frame_skip == 0)
        if should_infer or self.last_prediction is None:
            pred_result = predict_image(
                image_input=roi_frame,
                model_handle=self.model_handle,
                top_k=3,
                confidence_threshold=self.confidence_threshold,
                filename_hint="camera_feed"
            )
            self.last_prediction = pred_result
            self.last_inference_time_ms = pred_result["inference_time_ms"]

            # Record to history buffer
            self.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "disease": pred_result["disease"],
                "confidence_percent": pred_result["confidence_percent"],
                "is_reliable": pred_result["is_reliable"]
            })

        # Calculate Rolling FPS
        t_frame_end = time.perf_counter()
        dt = t_frame_end - t_frame_start
        self.fps_window.append(dt)
        avg_dt = sum(self.fps_window) / len(self.fps_window)
        self.current_fps = round(1.0 / max(avg_dt, 1e-5), 1)

        # Draw HUD Overlays onto frame
        annotated_frame = self.draw_hud_overlay(frame.copy(), self.last_prediction, roi_box)

        return annotated_frame, self.last_prediction, self.current_fps

    def draw_hud_overlay(
        self,
        frame: np.ndarray,
        pred: Optional[Dict[str, Any]],
        roi_box: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """Render high-contrast, professional real-time HUD on OpenCV frame."""
        h, w, _ = frame.shape

        # 1. Draw Target ROI Center Box
        if roi_box is not None:
            x1, y1, x2, y2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
            # Subtle corner brackets
            corner_len = 24
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 2)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 2)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 2)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 2)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 2)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 255), 2)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 2)

            cv2.putText(
                frame, "Position Leaf / Fruit within Target",
                (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
            )

        # 2. Draw Semi-transparent HUD Header & Performance Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 23, 42), -1)  # Top banner
        cv2.rectangle(overlay, (10, h - 130), (380, h - 10), (15, 23, 42), -1)  # Bottom-left card
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top Header
        cv2.putText(
            frame, "REAL-TIME LEAF & FRUIT DISEASE DETECTION",
            (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
        )

        # Controls Hint (Top-right)
        cv2.putText(
            frame, "[Q] Quit  |  [S] Snapshot",
            (w - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA
        )

        # 3. Diagnostic & Latency Metrics
        if pred is not None:
            is_reliable = pred.get("is_reliable", False)
            disease = pred.get("disease", "Scanning...")
            conf = pred.get("confidence_percent", 0.0)
            latency = pred.get("inference_time_ms", 0.0)
            crop_name = pred.get("detected_crop", "Foliage")

            # Color coding: Green = Reliable, Yellow = Uncertain/OOD, Red = Error
            if is_reliable:
                status_color = (74, 222, 128)  # Vibrant Emerald Green (BGR)
                status_text = f"Disease: {disease}"
            else:
                status_color = (60, 215, 255)  # Amber / Yellow (BGR)
                status_text = f"{disease}"

            # Disease Diagnosis line
            cv2.putText(
                frame, status_text[:38],
                (20, h - 98), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA
            )

            # Confidence line
            conf_str = f"Confidence: {conf:.1f}%"
            if not is_reliable:
                conf_str += " (Uncertain / Low Confidence)"
            cv2.putText(
                frame, conf_str,
                (20, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (241, 245, 249), 1, cv2.LINE_AA
            )

            # Latency and FPS line
            perf_str = f"Inference: {latency:.1f} ms  |  FPS: {self.current_fps}"
            cv2.putText(
                frame, perf_str,
                (20, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1, cv2.LINE_AA
            )

            # Model & Crop tag
            model_tag = f"Model: {pred.get('model_used', 'MobileNet')} ({pred.get('device', 'CPU')[:3]})"
            cv2.putText(
                frame, model_tag,
                (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame, "Initializing Neural Engine...",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 1, cv2.LINE_AA
            )

        return frame

