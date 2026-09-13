"""
Desktop Real-Time Camera Application for Leaf & Fruit Disease Detection.

Direct OpenCV implementation delivering maximum throughput and minimal latency.

Controls:
    [Q] or [ESC] : Quit application
    [S]          : Capture snapshot to results/captures/
    [C]          : Toggle Region of Interest (ROI) positioning guide

Run:
    python app/realtime_camera.py [--camera 0] [--skip 2] [--threshold 0.70]
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.config import config
from src.camera import RealTimeCameraHandler
from src.model_loader import load_model


def run_realtime_desktop_app(
    camera_index: int = config.CAMERA_INDEX,
    frame_skip: int = config.FRAME_SKIP,
    confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
    model_path: str = None
) -> None:
    """Launch the standalone desktop real-time OpenCV camera application."""
    print("\n" + "=" * 62)
    print("   LEAF & FRUIT DISEASE DETECTION - REAL-TIME DESKTOP APP")
    print("=" * 62)
    print(f"Target Camera Index    : {camera_index}")
    print(f"Frame Skipping Factor  : {frame_skip} (Inference every {frame_skip} frames)")
    print(f"Confidence Threshold   : {confidence_threshold * 100:.0f}%")
    print("Controls:")
    print("   [Q] / [ESC] -> Quit")
    print("   [S]         -> Capture Snapshot")
    print("   [C]         -> Toggle Target ROI Box")
    print("=" * 62 + "\n")

    # Load model once before opening camera
    try:
        model_handle = load_model(model_path=model_path, warmup=True)
    except Exception as e:
        print(f"\n[ERROR] Trained model could not be loaded: {e}")
        print("Please train a model first using: python src/train.py\n")
        return

    # Initialize Camera Handler
    camera = RealTimeCameraHandler(
        camera_index=camera_index,
        frame_skip=frame_skip,
        confidence_threshold=confidence_threshold,
        model_handle=model_handle,
        enable_roi=True
    )

    if not camera.open_camera():
        print(f"\n[ERROR] Camera could not be opened (Index {camera_index}).")
        print("Please check camera permissions or verify that your webcam is connected.\n")
        return

    window_name = "Leaf & Fruit Disease Detection - Real-Time Engine"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    snapshot_msg_timer = 0
    snapshot_path_str = ""

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("\n[WARN] Failed to grab frame from camera stream.")
                break

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)

            # Process frame with AI inference and HUD overlay
            annotated_frame, pred_result, fps = camera.process_frame(frame)

            # Display snapshot banner if recently captured
            if snapshot_msg_timer > 0:
                snapshot_msg_timer -= 1
                cv2.rectangle(annotated_frame, (10, 48), (520, 84), (16, 185, 129), -1)
                cv2.putText(
                    annotated_frame, f"Snapshot Saved: {snapshot_path_str}",
                    (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
                )

            cv2.imshow(window_name, annotated_frame)

            # Keyboard listener
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # Q or ESC
                print("\nExiting real-time camera...")
                break
            elif key in [ord("s"), ord("S")]:  # S to snapshot
                saved_path = camera.capture_snapshot(frame)
                snapshot_path_str = saved_path.name
                snapshot_msg_timer = 45  # Show banner for ~1.5s
                print(f"[SNAPSHOT] Saved raw frame to: {saved_path}")
            elif key in [ord("c"), ord("C")]:  # C to toggle ROI
                camera.enable_roi = not camera.enable_roi
                print(f"[ROI] Target guide set to: {camera.enable_roi}")

    finally:
        camera.release_camera()
        cv2.destroyAllWindows()
        print("\n" + "-" * 62)
        print("   CAMERA SESSION CONCLUDED")
        print(f"Total Frames Processed : {camera.frame_counter}")
        print(f"Final Running FPS      : {camera.current_fps}")
        print(f"Inference Latency      : {camera.last_inference_time_ms} ms")
        print("-" * 62 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Real-Time Desktop Camera Disease Classifier.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="OpenCV camera device index")
    parser.add_argument("--skip", type=int, default=config.FRAME_SKIP, help="Frame skipping factor (default: 2)")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model file")

    args = parser.parse_args()
    run_realtime_desktop_app(
        camera_index=args.camera,
        frame_skip=args.skip,
        confidence_threshold=args.threshold,
        model_path=args.model
    )

