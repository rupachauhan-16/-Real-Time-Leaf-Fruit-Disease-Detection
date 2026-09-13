"""
Real-Time Camera Detection Component for Streamlit.

Provides interactive webcam streaming directly inside the Streamlit web dashboard:
- Continuous frame capture and processing loop
- Configurable frame skipping and target ROI guide
- Real-time HUD overlays with latency and rolling FPS
- Snapshot capture to results/captures/
- Rolling prediction history table
- Direct instruction for launching the ultra-low-latency desktop OpenCV mode
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import streamlit as st
import pandas as pd
from PIL import Image

from src.config import config
from src.model_loader import LoadedModelHandle
from src.camera import RealTimeCameraHandler
from src.predictor import predict_image


def render_camera_page(
    model_handle: LoadedModelHandle,
    camera_index: int,
    frame_skip: int,
    confidence_threshold: float,
    show_top3: bool
) -> None:
    """Render the Real-Time Camera Detection interface in Streamlit."""
    st.markdown("### 📷 Real-Time Camera Disease Detection")
    st.markdown(
        "Capture live video from your webcam to classify leaf and fruit pathologies in real time. "
        "The model is loaded once and reused across frames with frame-skipping optimization."
    )

    # Performance & Status Alert
    st.info(
        f"💡 **Active Engine**: `{model_handle.model_name}` on `{model_handle.device}` | "
        f"Frame Skip: `{frame_skip}` | Confidence Threshold: `{confidence_threshold * 100:.0f}%`\n\n"
        f"For maximum frame rate (up to 60+ FPS without browser overhead), run the desktop app: `python app/realtime_camera.py`"
    )

    # Mode Selector for Camera
    cam_mode = st.radio(
        "Select Camera Input Mode:",
        ["🔴 Continuous Live Video Stream", "📸 Single Snapshot Capture (Browser Webcam)"],
        horizontal=True
    )

    if cam_mode == "🔴 Continuous Live Video Stream":
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 2])
        with c_ctrl1:
            start_stream = st.button("▶ Start Camera", type="primary", use_container_width=True)
        with c_ctrl2:
            stop_stream = st.button("⏹ Stop Camera", type="secondary", use_container_width=True)

        if "stream_active" not in st.session_state:
            st.session_state["stream_active"] = False

        if start_stream:
            st.session_state["stream_active"] = True
        if stop_stream:
            st.session_state["stream_active"] = False

        col_cam, col_metrics = st.columns([1.3, 1.0], gap="large")

        with col_cam:
            frame_placeholder = st.empty()
            capture_btn_placeholder = st.empty()

        with col_metrics:
            metrics_placeholder = st.empty()
            alert_placeholder = st.empty()
            history_placeholder = st.empty()

        if st.session_state["stream_active"]:
            camera = RealTimeCameraHandler(
                camera_index=camera_index,
                frame_skip=frame_skip,
                confidence_threshold=confidence_threshold,
                model_handle=model_handle,
                enable_roi=True
            )

            if not camera.open_camera():
                st.error(
                    f"⛔ **Unable to access camera (Index: {camera_index}).**\n\n"
                    f"Please check camera permissions, verify that your webcam is plugged in, "
                    f"or adjust `CAMERA_INDEX` in the sidebar."
                )
                st.session_state["stream_active"] = False
            else:
                try:
                    # Continuous streaming loop inside Streamlit
                    while st.session_state["stream_active"]:
                        ret, frame = camera.read_frame()
                        if not ret or frame is None:
                            st.warning("Video stream ended or dropped.")
                            break

                        frame = cv2.flip(frame, 1)
                        annotated_frame, pred, fps = camera.process_frame(frame)

                        # Convert annotated frame BGR to RGB for Streamlit display
                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        # Update Live Performance Panel
                        if pred is not None:
                            with metrics_placeholder.container():
                                st.markdown("#### ⚡ Real-Time Performance")
                                m_col1, m_col2 = st.columns(2)
                                with m_col1:
                                    st.metric(label="Disease Diagnosis", value=pred["disease"])
                                    st.metric(label="Confidence", value=f"{pred['confidence_percent']:.1f}%")
                                with m_col2:
                                    st.metric(label="Live FPS", value=f"{fps}")
                                    st.metric(label="Inference Latency", value=f"{pred['inference_time_ms']} ms")

                                st.caption(f"Model: `{pred['model_used']}` | Device: `{pred.get('device', 'CPU')}`")

                            # OOD or Low-Confidence Alert
                            with alert_placeholder.container():
                                if pred.get("is_out_of_distribution", False):
                                    st.error(
                                        f"⛔ **Out-of-Distribution Warning**: {pred['status_message']}\n\n"
                                        f"Please position an Apple or Tomato leaf/fruit within the target box."
                                    )
                                elif not pred.get("is_reliable", False):
                                    st.warning(
                                        f"⚠️ **Uncertain Prediction**: {pred['status_message']}\n\n"
                                        f"Hold the leaf steady within the target box for clearer focus."
                                    )

                            # Prediction History Buffer
                            if camera.history:
                                with history_placeholder.container():
                                    st.markdown("##### ⏱️ Recent Live Predictions")
                                    df_hist = pd.DataFrame(list(camera.history)[-5:])
                                    df_hist.rename(
                                        columns={
                                            "time": "Time",
                                            "disease": "Predicted Condition",
                                            "confidence_percent": "Confidence (%)",
                                            "is_reliable": "Reliable"
                                        },
                                        inplace=True
                                    )
                                    st.dataframe(df_hist, use_container_width=True)

                        # Small sleep to yield CPU cycle
                        time.sleep(0.01)

                finally:
                    camera.release_camera()
        else:
            with frame_placeholder:
                st.markdown(
                    """
                    <div style="border: 2px dashed #334155; border-radius: 12px; padding: 60px; text-align: center; color: #64748b;">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">📷</div>
                        <b>Live Camera Inactive</b><br>
                        Click <b>[▶ Start Camera]</b> above to begin real-time diagnosis streaming.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    elif cam_mode == "📸 Single Snapshot Capture (Browser Webcam)":
        cam_snap = st.camera_input("Position leaf/fruit and take a picture")
        if cam_snap:
            img_snap = Image.open(cam_snap)
            col_s1, col_s2 = st.columns([1, 1], gap="large")
            with col_s1:
                st.image(img_snap, caption=f"Captured Frame ({img_snap.width}x{img_snap.height} px)", use_container_width=True)

                if st.button("💾 Save Snapshot to Disk", use_container_width=True):
                    config.CAPTURES_PATH.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    save_path = config.CAPTURES_PATH / f"capture_{timestamp}.jpg"
                    img_snap.save(save_path)
                    st.success(f"Snapshot saved to `{save_path}`")

            with col_s2:
                with st.spinner("Analyzing captured frame..."):
                    res = predict_image(
                        image_input=img_snap,
                        model_handle=model_handle,
                        top_k=3,
                        confidence_threshold=confidence_threshold,
                        filename_hint="webcam_capture.jpg"
                    )

                st.markdown("#### Diagnostic Assessment")
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Predicted Disease", res["predicted_disease"])
                with sc2:
                    st.metric("Confidence", f"{res['confidence_percent']:.1f}%")
                with sc3:
                    st.metric("Inference", f"{res['inference_time_ms']} ms")

                if res["is_out_of_distribution"]:
                    st.error(f"⛔ **Out-of-Distribution Warning**: {res['status_message']}")
                elif not res["is_reliable"]:
                    st.warning(f"⚠️ **Uncertain**: {res['status_message']}")
                else:
                    st.success(f"✔ **Reliable Diagnosis ({res['confidence_percent']}%)**")

                if show_top3:
                    st.markdown("##### Candidate Predictions")
                    df_c = pd.DataFrame(res["top_predictions"])
                    st.bar_chart(data=df_c.set_index("class_name")["confidence_percent"], color="#10b981")

