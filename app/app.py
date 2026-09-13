"""
Master Streamlit Application: Time-Efficient Leaf and Fruit Disease Detection.

Supports:
- Dual Detection Pipeline: 🖼️ Image Upload Mode & 📷 Real-Time Camera Mode
- Unified Preprocessing and Model Inference Engine
- Out-of-Distribution (OOD) & Crop Domain Safety Guard
- Academic Performance Benchmarks & Empirical Latency Profiles
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd

from src.config import config
from src.utils import get_hardware_info
from src.model_loader import load_model, LoadedModelHandle
from app.upload_detection import render_upload_page
from app.realtime_detection import render_camera_page

try:
    from upload_detection import render_upload_page
    from realtime_detection import render_camera_page
except ImportError:
    from app.upload_detection import render_upload_page
    from app.realtime_detection import render_camera_page

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Real-Time Leaf & Fruit Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-sub {
        color: #10b981;
        font-size: 0.80rem;
        margin-top: 4px;
    }
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .mode-card {
        background: #1e293b;
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .mode-card:hover {
        border-color: #10b981;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model_handle(model_path_str: str) -> Optional[LoadedModelHandle]:
    """Cache loaded neural network across Streamlit reruns to eliminate reloading."""
    try:
        return load_model(model_path=model_path_str, warmup=True)
    except Exception as e:
        st.sidebar.error(f"Error loading model '{model_path_str}': {e}")
        return None


def get_available_models() -> List[Path]:
    """Discover all trained .keras and .tflite models in models/."""
    keras_models = sorted(list(config.MODELS_PATH.glob("*.keras")))
    tflite_models = sorted(list(config.MODELS_PATH.glob("*.tflite")))
    return keras_models + tflite_models


# ==========================================
# Sidebar Configuration
# ==========================================
st.sidebar.markdown("## ⚙️ Navigation & Settings")

# Application Page Selector
if "selected_nav" not in st.session_state:
    st.session_state["selected_nav"] = "🏠 Home"

nav_options = [
    "🏠 Home",
    "🖼️ Image Upload Detection",
    "📷 Real-Time Camera Detection",
    "📊 Model Performance Dashboard",
    "📖 Pathology & Methodology"
]

current_nav = st.sidebar.radio(
    "Choose Section:",
    nav_options,
    index=nav_options.index(st.session_state["selected_nav"]) if st.session_state["selected_nav"] in nav_options else 0
)
st.session_state["selected_nav"] = current_nav

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Selection")

models = get_available_models()
if not models:
    st.sidebar.error("No models found in `models/`. Please train a model first.")
    active_handle = None
else:
    model_labels = [m.name for m in models]
    # Default to best_efficientnet_b0 or first available model
    default_idx = 0
    for idx, name in enumerate(model_labels):
        if "efficientnet" in name.lower():
            default_idx = idx
            break

    selected_model_name = st.sidebar.selectbox("Active Model", model_labels, index=default_idx)
    selected_model_path = config.MODELS_PATH / selected_model_name
    active_handle = get_cached_model_handle(str(selected_model_path))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Inference Controls")

confidence_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.30,
    max_value=0.95,
    value=config.CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Flags predictions below this threshold as Uncertain / Low Confidence."
)

frame_skip_slider = st.sidebar.slider(
    "Frame Skipping Factor",
    min_value=1,
    max_value=5,
    value=config.FRAME_SKIP,
    step=1,
    help="Run AI inference every N frames to maximize real-time video smoothness."
)

show_top3_toggle = st.sidebar.checkbox(
    "Show Top-3 Predictions",
    value=True,
    help="Display candidate probability breakdown."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Hardware Diagnostics")
hw = get_hardware_info()
st.sidebar.text(f"OS: {hw['os']}")
st.sidebar.text(f"CPU: {hw['cpu'][:20]}...")
st.sidebar.text(f"RAM: {hw['total_ram_gb']} GB")
if hw["gpus"]:
    st.sidebar.success(f"GPU: {hw['gpus'][0]}")
else:
    st.sidebar.info("Device: CPU (Edge Inference)")


# ==========================================
# Main Content Router
# ==========================================

# ------------------------------------------
# PAGE 1: HOME
# ------------------------------------------
if current_nav == "🏠 Home":
    st.markdown("""
    <div class="hero-container">
        <h1 style="color: #f8fafc; margin-bottom: 8px;">🌿 Real-Time Leaf & Fruit Disease Detection</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0 auto 24px auto;">
            Time-Efficient Deep Learning Platform for Agricultural Pathology Classification.<br>
            Optimized for ultra-low latency, edge deployment, and live video stream analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Choose Detection Mode:")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="mode-card">
            <div style="font-size: 3.5rem; margin-bottom: 12px;">🖼️</div>
            <h3 style="margin-bottom: 8px;">Image Upload Detection</h3>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 18px;">
                Upload photos of leaves or fruit lesions from your computer, paste web URLs, 
                or test holdout dataset samples with instant diagnostic reporting.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Upload Detection →", use_container_width=True, type="primary"):
            st.session_state["selected_nav"] = "🖼️ Image Upload Detection"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="mode-card">
            <div style="font-size: 3.5rem; margin-bottom: 12px;">📷</div>
            <h3 style="margin-bottom: 8px;">Real-Time Camera Detection</h3>
            <p style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 18px;">
                Stream live video from your webcam for continuous real-time disease classification,
                rolling FPS throughput monitoring, and instant snapshot captures.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Camera Detection →", use_container_width=True, type="primary"):
            st.session_state["selected_nav"] = "📷 Real-Time Camera Detection"
            st.rerun()

    st.markdown("---")
    st.markdown("### 🔬 System Architecture & Real-Time Pipeline")
    st.markdown("""
    ```
    Live Camera Stream / File Upload
                  │
                  ▼
         Unified Preprocessing (224 × 224, BGR → RGB Normalization)
                  │
                  ▼
         Lightweight Deep Neural Network (MobileNetV3 / EfficientNetB0)
                  │
                  ▼
         Real-Time Crop Validation & Out-of-Distribution (OOD) Guard
                  │
         ┌────────┴────────┐
         ▼                 ▼
    [Valid Target]    [Unsupported Crop (e.g., Watermelon)]
    Disease Class      Safely Suppressed False Output
    + Confidence       "Unsupported Crop: Watermelon"
    + Latency & FPS    + Watermelon Agronomic Advisory
    ```
    """)

# ------------------------------------------
# PAGE 2: IMAGE UPLOAD DETECTION
# ------------------------------------------
elif current_nav == "🖼️ Image Upload Detection":
    if active_handle:
        render_upload_page(
            model_handle=active_handle,
            confidence_threshold=confidence_slider,
            show_top3=show_top3_toggle
        )
    else:
        st.error("Model handle is unavailable.")

# ------------------------------------------
# PAGE 3: REAL-TIME CAMERA DETECTION
# ------------------------------------------
elif current_nav == "📷 Real-Time Camera Detection":
    if active_handle:
        render_camera_page(
            model_handle=active_handle,
            camera_index=config.CAMERA_INDEX,
            frame_skip=frame_skip_slider,
            confidence_threshold=confidence_slider,
            show_top3=show_top3_toggle
        )
    else:
        st.error("Model handle is unavailable.")

# ------------------------------------------
# PAGE 4: MODEL PERFORMANCE DASHBOARD
# ------------------------------------------
elif current_nav == "📊 Model Performance Dashboard":
    st.markdown("### 📊 Model Performance & Empirical Benchmark Study")
    st.markdown(
        "Scientific evaluation of classification accuracy, parameters, inference latency, "
        "and throughput across lightweight edge architectures."
    )

    comp_csv = config.RESULTS_PATH / "model_comparison.csv"
    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        st.dataframe(df_comp, use_container_width=True)
    else:
        st.info("Benchmark data not generated yet. Run `python src/benchmark.py`.")

    cm_files = list(config.RESULTS_PATH.glob("confusion_matrix_*.png"))
    curve_files = list(config.RESULTS_PATH.glob("training_curves_*.png"))

    if cm_files or curve_files:
        col_cm, col_crv = st.columns(2)
        with col_cm:
            st.markdown("#### Test Set Confusion Matrices")
            for cm in cm_files:
                st.image(str(cm), caption=f"Confusion Matrix: {cm.stem}", use_container_width=True)
        with col_crv:
            st.markdown("#### Two-Stage Training Curves")
            for crv in curve_files:
                st.image(str(crv), caption=f"Training Progression: {crv.stem}", use_container_width=True)

# ------------------------------------------
# PAGE 5: PATHOLOGY & METHODOLOGY
# ------------------------------------------
elif current_nav == "📖 Pathology & Methodology":
    st.markdown("""
    ### 📖 Pathology Knowledge Base & Technical Methodology

    #### Supported Agricultural Domain
    The primary classification model is trained on distinct pathology classes:
    - **Apple (*Malus domestica*)**: Apple Black Rot (*Botryosphaeria obtusa*), Healthy Foliage.
    - **Tomato (*Solanum lycopersicum*)**: Tomato Early Blight (*Alternaria solani*), Healthy Foliage.

    #### Out-of-Distribution (OOD) Safeguard
    When an unsupported crop (such as a **Watermelon**, banana, or non-plant object) is uploaded or presented to the camera:
    1. **Visual Morphological Inspection**: Analyzes distinctive melon pulp ($R > 1.7G$, $R > 1.7B$) and rind geometry.
    2. **Information-Theoretic Uncertainty**: Calculates Shannon entropy $H(p) = -\\sum p_i \\log_2(p_i)$ and probability margins.
    3. **Domain Guarding**: Safely suppresses false Solanaceae/Rosaceae classifications and displays the specialized **Watermelon Agronomic Advisory** instead of improper chemical recommendations.

    #### Future Roadmap: YOLO Object Detection Integration
    The system architecture is designed for seamless extensibility:
    ```
    Live Camera → YOLO Object Detector → Bounding Box Crop (ROI) → Disease Classifier
    ```
    This modular separation ensures real-time classification works today while allowing future localized bounding box detection.
    """)
