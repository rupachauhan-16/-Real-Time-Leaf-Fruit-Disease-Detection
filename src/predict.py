"""
CLI Prediction Script for Leaf & Fruit Disease Detection.

Usage:
    python src/predict.py --image <path_to_image> [--model <model_path>] [--threshold 0.70]
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.predictor import DiseasePredictor, predict_image, load_model


def format_cli_output(res: dict) -> str:
    """Format prediction dictionary for terminal display."""
    status_icon = "[OK]" if res.get("is_reliable", False) else "[!]"
    if res.get("is_out_of_distribution", False):
        rel_text = "REJECTED / OUT-OF-DISTRIBUTION"
    elif not res.get("is_reliable", False):
        rel_text = "UNCERTAIN / LOW CONFIDENCE"
    else:
        rel_text = "Reliable Diagnosis"

    lines = [
        "\n" + "=" * 62,
        "   REAL-TIME LEAF & FRUIT DISEASE DETECTION RESULT",
        "=" * 62,
        f"Detected Crop     : {res.get('detected_crop', 'N/A')}",
        f"Diagnosis Result  : {res['predicted_disease']}",
        f"Confidence Score  : {res['confidence_percent']:.2f}%",
        f"Inference Latency : {res['inference_time_ms']} ms",
        f"Model Executed    : {res['model_used']}",
        f"Compute Device    : {res.get('device', 'CPU')}",
        f"Reliability Status: {status_icon} {rel_text}",
        "-" * 62,
        f"Diagnostic Notice : {res['status_message']}",
        "-" * 62,
        "Top Predictions:"
    ]
    for idx, top in enumerate(res["top_predictions"], 1):
        lines.append(f"  {idx}. {top['class_name']:<28}: {top['confidence_percent']:>6.2f}%")
    lines.append("=" * 62 + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict plant leaf or fruit disease from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras or .tflite model file")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of candidate classes to show")

    args = parser.parse_args()

    predictor = DiseasePredictor(
        model_path=args.model,
        confidence_threshold=args.threshold
    )

    prediction = predictor.predict(
        image_input=Path(args.image),
        top_k=args.top_k,
        filename_hint=Path(args.image).name
    )

    print(format_cli_output(prediction))
