"""
Real-Time Crop Identification & Out-of-Distribution (OOD) Detection Module.

Guards the diagnostic pipeline against out-of-domain crops (e.g., Watermelon,
banana, citrus) and non-plant objects. Prevents closed-set softmax classifiers
from falsely assigning Apple or Tomato pathologies to unsupported samples.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

# Supported crops and their associated pathology classes in the current model
SUPPORTED_CROPS = {
    "apple": {
        "scientific_name": "Malus domestica",
        "family": "Rosaceae",
        "classes": ["Apple_Black_Rot", "Apple_Healthy"]
    },
    "tomato": {
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "classes": ["Tomato_Early_Blight", "Tomato_Healthy"]
    }
}

# Known common agricultural crops outside active model training domain
KNOWN_UNSUPPORTED_CROPS = {
    "watermelon": {
        "common_name": "Watermelon",
        "scientific_name": "Citrullus lanatus",
        "family": "Cucurbitaceae",
        "keywords": ["watermelon", "citrullus", "tarbooz", "water-melon"]
    },
    "melon": {
        "common_name": "Melon / Cucurbit",
        "scientific_name": "Cucumis melo",
        "family": "Cucurbitaceae",
        "keywords": ["cantaloupe", "honeydew", "muskmelon", "melon"]
    },
    "banana": {
        "common_name": "Banana",
        "scientific_name": "Musa acuminata",
        "family": "Musaceae",
        "keywords": ["banana", "plantain", "musa"]
    },
    "orange": {
        "common_name": "Orange / Citrus",
        "scientific_name": "Citrus sinensis",
        "family": "Rutaceae",
        "keywords": ["orange", "citrus", "mandarin", "tangerine"]
    },
    "lemon": {
        "common_name": "Lemon / Lime",
        "scientific_name": "Citrus limon",
        "family": "Rutaceae",
        "keywords": ["lemon", "lime"]
    },
    "potato": {
        "common_name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "family": "Solanaceae (Tuber)",
        "keywords": ["potato", "tuber", "spud"]
    },
    "grape": {
        "common_name": "Grape",
        "scientific_name": "Vitis vinifera",
        "family": "Vitaceae",
        "keywords": ["grape", "vine", "grapevine"]
    },
    "cucumber": {
        "common_name": "Cucumber",
        "scientific_name": "Cucumis sativus",
        "family": "Cucurbitaceae",
        "keywords": ["cucumber", "gherkin"]
    },
    "mango": {
        "common_name": "Mango",
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "keywords": ["mango"]
    },
    "strawberry": {
        "common_name": "Strawberry",
        "scientific_name": "Fragaria ananassa",
        "family": "Rosaceae (Berry)",
        "keywords": ["strawberry", "berry"]
    }
}


def _calculate_shannon_entropy(probabilities: np.ndarray) -> float:
    """Calculate Shannon entropy in bits for probability distribution."""
    eps = 1e-12
    p = np.clip(probabilities, eps, 1.0)
    return float(-np.sum(p * np.log2(p)))


def detect_watermelon_visual_features(image: Image.Image) -> Dict[str, Any]:
    """
    Perform fast visual color and morphology screening for watermelon features:
    - Vibrant red/pink pulp with high red-to-green and red-to-blue contrast.
    - Dark seed clusters and distinctive green striped rind.
    """
    img_rgb = image.convert("RGB")
    # Resize thumbnail for high-speed real-time processing (< 2ms)
    thumb = img_rgb.resize((120, 120))
    arr = np.array(thumb, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Non-background pixels (filter out pure black/white canvas)
    non_bg = (r + g + b > 35) & (r + g + b < 720)
    total_valid = max(int(np.sum(non_bg)), 1)

    # Watermelon red/pink interior pulp signature
    red_pulp_mask = (r > 155) & (r > 1.7 * g) & (r > 1.7 * b) & non_bg
    red_pulp_ratio = float(np.sum(red_pulp_mask) / total_valid)

    # Green rind signature
    green_rind_mask = (g > 85) & (g > 1.15 * r) & (g > 1.15 * b) & non_bg
    green_rind_ratio = float(np.sum(green_rind_mask) / total_valid)

    # Dark seeds: very low intensity pixels
    dark_mask = (r < 40) & (g < 40) & (b < 40) & non_bg
    dark_ratio = float(np.sum(dark_mask) / total_valid)

    is_watermelon = False
    confidence = 0.0

    if red_pulp_ratio > 0.008:
        is_watermelon = True
        confidence = min(0.99, 0.70 + red_pulp_ratio * 10.0)
    elif green_rind_ratio > 0.35 and dark_ratio > 0.02:
        is_watermelon = True
        confidence = 0.75

    return {
        "is_watermelon": is_watermelon,
        "confidence": confidence,
        "red_pulp_ratio": red_pulp_ratio,
        "green_rind_ratio": green_rind_ratio,
        "dark_ratio": dark_ratio
    }


def evaluate_crop_and_ood(
    raw_probabilities: np.ndarray,
    class_names: List[str],
    image_input: Optional[Union[str, Path, Image.Image]] = None,
    filename_hint: Optional[str] = None,
    confidence_threshold: float = 0.60
) -> Dict[str, Any]:
    """
    Multi-signal Real-time Crop Identification and Out-of-Distribution (OOD) Validator.

    Args:
        raw_probabilities: 1D array of softmax outputs from the pathology model.
        class_names: List of class names the model was trained on.
        image_input: The input image (path or PIL image).
        filename_hint: Optional filename or source string for context.
        confidence_threshold: Probability threshold for reliable classification.

    Returns:
        Structured evaluation with OOD flags, detected crop, and agronomic context.
    """
    probs = np.array(raw_probabilities, dtype=np.float32)
    top_idx = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    predicted_raw_class = class_names[top_idx]

    # Calculate Information-Theoretic Uncertainty Metrics
    entropy = _calculate_shannon_entropy(probs)
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    # 1. Filename & Metadata Ingestion
    detected_crop_info: Optional[Dict[str, Any]] = None
    supported_crop_hint: Optional[str] = None
    fn_lower = ""
    if filename_hint:
        fn_lower = str(filename_hint).lower()
    elif isinstance(image_input, (str, Path)):
        fn_lower = Path(image_input).name.lower()

    if fn_lower:
        # Check for known unsupported crops first (e.g. Watermelon)
        for crop_key, crop_meta in KNOWN_UNSUPPORTED_CROPS.items():
            if any(kw in fn_lower for kw in crop_meta["keywords"]):
                detected_crop_info = crop_meta
                break

        # Check for supported target crops
        if not detected_crop_info:
            if any(kw in fn_lower for kw in ["apple", "malus"]):
                supported_crop_hint = "apple"
            elif any(kw in fn_lower for kw in ["tomato", "solanum"]):
                supported_crop_hint = "tomato"

    # 2. Visual Morphological & Color Inspection
    pil_img: Optional[Image.Image] = None
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, (str, Path)) and Path(image_input).exists():
        try:
            pil_img = Image.open(image_input)
        except Exception:
            pil_img = None

    visual_wm_check = {"is_watermelon": False, "confidence": 0.0}
    if pil_img is not None:
        visual_wm_check = detect_watermelon_visual_features(pil_img)
        if visual_wm_check["is_watermelon"] and not detected_crop_info:
            detected_crop_info = KNOWN_UNSUPPORTED_CROPS["watermelon"]

    # Align crop-specific pathology when target crop is confirmed by metadata/hint
    if supported_crop_hint == "apple":
        apple_indices = [i for i, c in enumerate(class_names) if c.startswith("Apple")]
        if apple_indices:
            apple_sub_probs = probs[apple_indices]
            sum_apple = float(np.sum(apple_sub_probs))
            if sum_apple > 0.01:
                best_sub_idx = apple_indices[int(np.argmax(apple_sub_probs))]
                predicted_raw_class = class_names[best_sub_idx]
                top_conf = float(probs[best_sub_idx] / sum_apple)
    elif supported_crop_hint == "tomato":
        tomato_indices = [i for i, c in enumerate(class_names) if c.startswith("Tomato")]
        if tomato_indices:
            tomato_sub_probs = probs[tomato_indices]
            sum_tomato = float(np.sum(tomato_sub_probs))
            if sum_tomato > 0.01:
                best_sub_idx = tomato_indices[int(np.argmax(tomato_sub_probs))]
                predicted_raw_class = class_names[best_sub_idx]
                top_conf = float(probs[best_sub_idx] / sum_tomato)

    # 3. Decision Logic for Out-of-Distribution Rejection vs Confidence Gating
    is_out_of_distribution = False
    is_reliable = True
    rejection_reason = None
    decision_category = "IN_DISTRIBUTION"

    if detected_crop_info:
        is_out_of_distribution = True
        is_reliable = False
        decision_category = "UNSUPPORTED_CROP"
        crop_name = detected_crop_info["common_name"]
        sci_name = detected_crop_info["scientific_name"]
        family = detected_crop_info["family"]
        rejection_reason = (
            f"Detected {crop_name} ({sci_name}, family {family}), which is outside "
            f"the active Apple & Tomato pathology domain."
        )
    elif entropy > 1.95 and margin < 0.05 and not supported_crop_hint:
        # True uniform uncertainty across all 4 classes (~25% each) indicating non-crop/random noise
        is_out_of_distribution = True
        is_reliable = False
        decision_category = "HIGH_ENTROPY"
        rejection_reason = (
            f"Extreme predictive entropy ({entropy:.2f} bits) indicates an unrecognized non-target object."
        )
    elif top_conf < confidence_threshold:
        # In-distribution plant sample, but below strict confidence threshold
        is_out_of_distribution = False
        is_reliable = False
        decision_category = "LOW_CONFIDENCE"
        rejection_reason = (
            f"Prediction confidence ({top_conf * 100:.1f}%) is below optimal threshold "
            f"({confidence_threshold * 100:.0f}%). Consider testing a clearer photo."
        )

    # Resolve Display Name
    if is_out_of_distribution:
        if detected_crop_info:
            display_diagnosis = f"Unsupported Crop: {detected_crop_info['common_name']}"
            detected_crop_str = f"{detected_crop_info['common_name']} ({detected_crop_info['scientific_name']})"
        else:
            display_diagnosis = "Unrecognized / Unsupported Sample"
            detected_crop_str = "Unknown / Non-Target Sample"
    else:
        display_diagnosis = predicted_raw_class.replace("_", " ")
        detected_crop_str = "Apple" if "Apple" in predicted_raw_class else "Tomato"

    return {
        "is_out_of_distribution": is_out_of_distribution,
        "is_reliable": is_reliable,
        "decision_category": decision_category,
        "display_diagnosis": display_diagnosis,
        "detected_crop": detected_crop_str,
        "detected_crop_meta": detected_crop_info,
        "rejection_reason": rejection_reason,
        "entropy": round(entropy, 3),
        "margin": round(margin, 3),
        "top_confidence": round(top_conf, 4),
        "raw_predicted_class": predicted_raw_class
    }

