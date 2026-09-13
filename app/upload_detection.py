"""
Image Upload Detection Component for Streamlit.

Handles:
- Static image file upload (JPG, PNG, WEBP, BMP)
- Web Image URL ingestion
- Sample image gallery testing
- Real-time disease classification with OOD & Watermelon protection
- Top-3 candidate probability breakdown
- Pathology Knowledge Base advisory
"""

import io
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
import pandas as pd
from PIL import Image

from src.config import config
from src.model_loader import load_model, LoadedModelHandle
from src.predictor import predict_image

# Comprehensive Pathology Knowledge Base
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Tomato_Early_Blight": {
        "crop": "Tomato (Solanum lycopersicum)",
        "family": "Solanaceae",
        "pathogen": "Alternaria solani (Fungus)",
        "risk_level": "Moderate to High",
        "key_symptoms": [
            "Concentric brown rings resembling a 'target board' or bullseye pattern.",
            "Chlorotic yellow halo expanding around older leaf lesions.",
            "Accelerated lower leaf drop moving progressively up the canopy."
        ],
        "treatment": [
            "Apply bio-fungicides (Bacillus subtilis) or copper hydroxide at first detection.",
            "Remove lower foliage (pruning) to disrupt splash-dispersal of fungal spores.",
            "Enforce strict 3-year crop rotation with non-solanaceous plants."
        ],
        "prevention": "Switch to drip irrigation to prevent wet foliage; apply organic straw mulch around plant base."
    },
    "Tomato_Healthy": {
        "crop": "Tomato (Solanum lycopersicum)",
        "family": "Solanaceae",
        "pathogen": "None (Healthy Tissue)",
        "risk_level": "Minimal / Optimal",
        "key_symptoms": [
            "Lush green coloration with intact epidermal layer.",
            "Absence of necrotic spots, chlorotic yellowing, or fungal pustules.",
            "Turgid, vigorous vegetative and reproductive foliage."
        ],
        "treatment": [
            "No curative chemical or fungicide treatment required.",
            "Maintain balanced nitrogen-phosphorus-potassium (NPK) feeding."
        ],
        "prevention": "Maintain regular scouting schedule and sanitize harvesting shears."
    },
    "Apple_Black_Rot": {
        "crop": "Apple (Malus domestica)",
        "family": "Rosaceae",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "risk_level": "High (can cause fruit rot & branch cankers)",
        "key_symptoms": [
            "Frogeye leaf spots: small purple margins with light tan/brown necrotic centers.",
            "Firm brown decay on fruit expanding in dark concentric circles, later shriveling into black mummies.",
            "Depressed bark cankers on twigs and main scaffold limbs."
        ],
        "treatment": [
            "Prune out dead twigs, cankered wood, and mummified fruit during dormant season.",
            "Apply protective fungicides (Captan, Mancozeb, or Sulfur) from tight cluster through cover sprays.",
            "Immediately destroy or bury all infected prunings."
        ],
        "prevention": "Prune tree canopies to maximize sunlight penetration and air circulation; remove wild cedar/apple hosts nearby."
    },
    "Apple_Healthy": {
        "crop": "Apple (Malus domestica)",
        "family": "Rosaceae",
        "pathogen": "None (Healthy Foliage)",
        "risk_level": "Minimal / Optimal",
        "key_symptoms": [
            "Smooth, uniformly green leaf margins with clean venation.",
            "Fruit surface free from rot, fungal scabs, or insect puncture lesions.",
            "Vigorous spur growth and balanced shoot development."
        ],
        "treatment": [
            "No fungicide intervention needed.",
            "Continue standard seasonal integrated pest management (IPM)."
        ],
        "prevention": "Ensure adequate soil aeration and periodic micronutrient leaf sprays (calcium, boron)."
    },
    "Watermelon_Advisory": {
        "crop": "Watermelon (Citrullus lanatus)",
        "family": "Cucurbitaceae",
        "pathogen": "Multi-Pathogen Profile (Crop Outside Model Scope)",
        "risk_level": "Agronomic Diagnostic Scope Mismatch",
        "key_symptoms": [
            "Gummy Stem Blight (Stagonosporopsis cucurbitacearum): Dark water-soaked lesions exuding amber gum.",
            "Anthracnose (Colletotrichum orbiculare): Circular sunken necrotic lesions on fruit and foliage.",
            "Fusarium Wilt (Fusarium oxysporum f. sp. niveum): Progressive vascular wilting and runner dieback.",
            "Downy Mildew (Pseudoperonospora cubensis): Vein-bounded angular chlorotic leaf spots."
        ],
        "treatment": [
            "Do NOT apply solanaceous/apple fungicides without verifying cucurbit label registration.",
            "Apply broad-spectrum protectants registered for watermelon (e.g., Chlorothalonil or Mancozeb).",
            "Use certified disease-free seeds and disease-resistant rootstocks for grafting."
        ],
        "prevention": "Implement drip irrigation under plastic mulch; enforce 3+ year crop rotation away from all cucurbits (melons, cucumbers, squash)."
    }
}


def render_upload_page(
    model_handle: LoadedModelHandle,
    confidence_threshold: float,
    show_top3: bool
) -> None:
    """Render the Image Upload Detection interface."""
    st.markdown("### 🖼️ Image Upload Disease Detection")
    st.markdown(
        "Upload a leaf or fruit photo from your device, enter a web image URL, "
        "or select a sample image to perform instant neural diagnosis with OOD validation."
    )

    col_input, col_output = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown("#### 1. Input Image")
        source_mode = st.radio(
            "Select Ingestion Method:",
            ["📁 File Upload", "🌐 Web Image URL", "📂 Sample Gallery"],
            horizontal=True
        )

        img_to_diagnose: Optional[Image.Image] = None
        img_source_desc = ""

        # Option A: File Upload
        if source_mode == "📁 File Upload":
            uploaded_file = st.file_uploader(
                "Choose a leaf or fruit photo",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Supports JPG, PNG, WEBP, and BMP formats."
            )
            if uploaded_file:
                try:
                    img_to_diagnose = Image.open(uploaded_file)
                    img_source_desc = f"File: {uploaded_file.name}"
                except Exception as e:
                    st.error(f"Invalid image file: {e}")

        # Option B: Web URL
        elif source_mode == "🌐 Web Image URL":
            img_url = st.text_input(
                "Paste public image URL:",
                placeholder="https://example.com/leaf_photo.jpg"
            )
            if img_url:
                try:
                    with st.spinner("Fetching image from URL..."):
                        resp = requests.get(img_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                        resp.raise_for_status()
                        img_to_diagnose = Image.open(io.BytesIO(resp.content))
                        img_source_desc = "Web URL Ingested"
                except Exception as e:
                    st.error(f"Failed to fetch image: {e}")

        # Option C: Sample Gallery
        elif source_mode == "📂 Sample Gallery":
            sample_options = {}
            if config.TEST_DIR.exists():
                for cls_folder in config.TEST_DIR.iterdir():
                    if cls_folder.is_dir():
                        files = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
                        if files:
                            sample_options[f"{cls_folder.name} (Test Set)"] = files[0]

            # Also offer the Watermelon sample from Desktop if available
            desktop_wm = Path(r"C:\Users\himan\OneDrive\Desktop\watermelon_PNG2656.png")
            if desktop_wm.exists():
                sample_options["Watermelon (Out-of-Distribution Sample)"] = desktop_wm

            if sample_options:
                selected_sample_label = st.selectbox("Choose sample image:", list(sample_options.keys()))
                chosen_path = sample_options[selected_sample_label]
                img_to_diagnose = Image.open(chosen_path)
                img_source_desc = f"Sample: {chosen_path.name}"

        # Display Image Preview & File Properties
        if img_to_diagnose is not None:
            st.image(
                img_to_diagnose,
                caption=f"Preview ({img_to_diagnose.width}x{img_to_diagnose.height} px, {img_to_diagnose.mode})",
                use_container_width=True
            )
            st.caption(f"Source: {img_source_desc}")

        predict_btn = st.button("🚀 Run Disease Diagnosis", type="primary", use_container_width=True)

    with col_output:
        st.markdown("#### 2. Diagnostic Assessment")

        if img_to_diagnose is not None and (predict_btn or "last_upload_res" in st.session_state):
            with st.spinner("Executing neural inference and latency profiling..."):
                res = predict_image(
                    image_input=img_to_diagnose,
                    model_handle=model_handle,
                    top_k=3,
                    confidence_threshold=confidence_threshold,
                    filename_hint=img_source_desc
                )
                st.session_state["last_upload_res"] = res

            # Metric Cards Display
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Condition</div>
                    <div class="metric-value" style="font-size: 1.15rem;">{res['predicted_disease']}</div>
                    <div class="metric-sub">{'Domain Protected' if res['is_out_of_distribution'] else 'Pathology Class'}</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                conf_display = f"{res['confidence_percent']:.1f}%"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Confidence Score</div>
                    <div class="metric-value">{conf_display}</div>
                    <div class="metric-sub">{'Out-of-Scope' if res['is_out_of_distribution'] else 'Softmax Prob'}</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                fps_val = round(1000.0 / max(res['inference_time_ms'], 1e-4), 1)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Latency</div>
                    <div class="metric-value">{res['inference_time_ms']} <span style="font-size: 0.85rem;">ms</span></div>
                    <div class="metric-sub">{fps_val} FPS Throughput</div>
                </div>
                """, unsafe_allow_html=True)

            # Reliability & OOD Status Alert
            if res["is_out_of_distribution"]:
                is_wm = "watermelon" in res["predicted_disease"].lower()
                crop_title = "Watermelon (Citrullus lanatus)" if is_wm else res.get("detected_crop", "Unsupported Crop")
                st.error(
                    f"⛔ **Out-of-Distribution Crop Intercepted: {crop_title}**\n\n"
                    f"{res['status_message']}\n\n"
                    f"*The active neural network is trained exclusively on **Apple** and **Tomato** pathologies. "
                    f"Real-time diagnostic guards safely suppressed false Solanaceae/Rosaceae classification to prevent improper agronomic treatment.*"
                )
            elif not res["is_reliable"]:
                st.warning(
                    f"⚠️ **Caution: Uncertain Diagnosis ({res['confidence_percent']}%)**\n\n"
                    f"{res['status_message']}\n\n"
                    f"*Please upload a clearer, well-lit photo of the leaf or fruit lesion.*"
                )
            else:
                st.success(f"✔ **High-Confidence Pathology Diagnosis ({res['confidence_percent']}%)**")

            # Top Candidates Breakdown
            if show_top3:
                st.markdown("##### Candidate Probability Breakdown")
                df_cands = pd.DataFrame(res["top_predictions"])
                df_cands["Candidate"] = df_cands["class_name"]
                df_cands["Confidence (%)"] = df_cands["confidence_percent"]

                st.bar_chart(
                    data=df_cands.set_index("Candidate")["Confidence (%)"],
                    color="#10b981",
                    use_container_width=True
                )
                if res["is_out_of_distribution"]:
                    st.caption(
                        "ℹ️ *Note: Closed-set neural networks distribute remaining probability across trained classes. "
                        "These scores are shown for empirical transparency but were rejected from diagnosis.*"
                    )

            # Agronomic Advisory Section
            st.markdown("---")
            if res["is_out_of_distribution"] and "watermelon" in res["predicted_disease"].lower():
                advisory = DISEASE_KNOWLEDGE_BASE["Watermelon_Advisory"]
                st.markdown("### 📋 Watermelon (Citrullus lanatus) Pathology Profile & Agronomic Advisory")
                a_col1, a_col2 = st.columns(2)
                with a_col1:
                    st.markdown(f"**Host Crop**: `{advisory['crop']}`")
                    st.markdown(f"**Botanical Family**: `{advisory['family']}`")
                    st.markdown(f"**Classification Scope**: `{advisory['risk_level']}`")
                    st.markdown("**Common Watermelon Diseases & Symptoms Checklist:**")
                    for sym in advisory["key_symptoms"]:
                        st.markdown(f"- {sym}")
                with a_col2:
                    st.markdown("**Agronomic Management & Prevention Advice:**")
                    for trt in advisory["treatment"]:
                        st.markdown(f"- {trt}")
                    st.markdown(f"**Field Strategy:** {advisory['prevention']}")

            elif res["is_reliable"]:
                raw_cls = res["raw_class_name"]
                advisory = DISEASE_KNOWLEDGE_BASE.get(raw_cls, None)
                if advisory:
                    st.markdown(f"### 📋 Agronomic Treatment & Advisory: *{res['predicted_disease']}*")
                    a_col1, a_col2 = st.columns(2)
                    with a_col1:
                        st.markdown(f"**Host Crop**: {advisory['crop']}")
                        st.markdown(f"**Pathogen**: `{advisory['pathogen']}`")
                        st.markdown(f"**Severity Risk**: `{advisory['risk_level']}`")
                        st.markdown("**Visual Symptoms Checklist:**")
                        for sym in advisory["key_symptoms"]:
                            st.markdown(f"- {sym}")
                    with a_col2:
                        st.markdown("**Recommended Management & Treatment:**")
                        for trt in advisory["treatment"]:
                            st.markdown(f"- {trt}")
                        st.markdown(f"**Prevention Advice:** {advisory['prevention']}")

        elif img_to_diagnose is None:
            st.info("👈 Upload an image or select a sample on the left panel to receive instant disease diagnosis.")

