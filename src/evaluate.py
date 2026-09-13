"""
Model Evaluation and Error Analysis Pipeline.

Computes comprehensive classification metrics (accuracy, macro/weighted precision,
recall, F1-score), exports per-class classification reports to CSV/text, and generates
annotated confusion matrix heatmaps saved to results/.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import tensorflow as tf
from src.config import config
from src.utils import (
    setup_logger,
    load_json,
    save_json,
    plot_confusion_matrix
)
from src.data_loader import get_class_names_from_directory

logger = setup_logger("evaluate", config.LOGS_PATH / "evaluate.log")


def evaluate_model(
    model_path: Path,
    test_dir: Optional[Path] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform full test set evaluation on a saved Keras model.

    Args:
        model_path: Path to .keras model checkpoint.
        test_dir: Directory containing test split. Defaults to config.TEST_DIR (or VAL_DIR fallback).
        class_names: List of class names. If None, loaded from models/class_names.json.

    Returns:
        Dictionary containing overall and per-class evaluation metrics.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    model_name = model_path.stem.replace("best_", "")
    logger.info(f"=== Evaluating Model '{model_name}' ({model_path.name}) ===")

    # Determine evaluation split
    eval_dir = test_dir or config.TEST_DIR
    if not eval_dir.exists() or not any(eval_dir.iterdir()):
        logger.warning(f"Test directory '{eval_dir}' is empty or missing. Falling back to validation set.")
        eval_dir = config.VAL_DIR

    # Load class names
    if class_names is None:
        if config.CLASS_NAMES_FILE.exists():
            class_names = load_json(config.CLASS_NAMES_FILE)
        else:
            class_names = get_class_names_from_directory(eval_dir)

    logger.info(f"Evaluating across {len(class_names)} classes: {class_names}")

    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Prepare dataset with shuffle=False to align labels with predictions
    test_ds = tf.keras.utils.image_dataset_from_directory(
        eval_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        shuffle=False
    )

    y_true_list = []
    y_pred_probs_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels.numpy())

    y_true_cat = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

    y_true = np.argmax(y_true_cat, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute macro and weighted metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # Save classification report as text
    text_report_path = config.RESULTS_PATH / f"classification_report_{model_name}.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    # Save classification report as CSV
    df_report = pd.DataFrame(report_dict).transpose()
    csv_report_path = config.RESULTS_PATH / f"classification_report_{model_name}.csv"
    df_report.to_csv(csv_report_path, index=True)

    # Generate and save Confusion Matrix plot
    cm_plot_path = config.RESULTS_PATH / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=cm_plot_path,
        model_name=model_name.replace("_", " ").title(),
        normalize=True
    )

    metrics_summary = {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "total_test_samples": int(len(y_true)),
        "report_csv": str(csv_report_path),
        "confusion_matrix_png": str(cm_plot_path)
    }

    metrics_json_path = config.RESULTS_PATH / f"metrics_{model_name}.json"
    save_json(metrics_summary, metrics_json_path)

    logger.info(f"Accuracy:           {acc * 100:.2f}%")
    logger.info(f"Precision (Macro):  {prec_macro * 100:.2f}% | (Weighted): {prec_weighted * 100:.2f}%")
    logger.info(f"Recall (Macro):     {rec_macro * 100:.2f}% | (Weighted): {rec_weighted * 100:.2f}%")
    logger.info(f"F1-Score (Macro):   {f1_macro * 100:.2f}% | (Weighted): {f1_weighted * 100:.2f}%")
    logger.info(f"Saved artifacts to '{config.RESULTS_PATH}'.")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained leaf/fruit disease detection model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.MODELS_PATH / f"best_{config.DEFAULT_MODEL}.keras"),
        help="Path to saved .keras model file"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=str(config.TEST_DIR),
        help="Path to test directory"
    )

    args = parser.parse_args()
    evaluate_model(
        model_path=Path(args.model_path),
        test_dir=Path(args.test_dir)
    )
