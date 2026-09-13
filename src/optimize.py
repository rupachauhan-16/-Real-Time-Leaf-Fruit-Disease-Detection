"""
Model Optimization and TensorFlow Lite Quantization Module.

Implements:
1. Standard Float32 TFLite conversion.
2. Post-Training Dynamic Range Quantization (INT8 weight quantization).
3. Comparative analysis between Original Keras and Quantized TFLite models.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import tensorflow as tf
from src.config import config
from src.utils import setup_logger, save_json, get_file_size_mb
from src.benchmark import benchmark_keras_model, benchmark_tflite_model

logger = setup_logger("optimize", config.LOGS_PATH / "optimize.log")


def convert_to_tflite(
    keras_model_path: Path,
    quantize_int8: bool = True
) -> Tuple[Path, Path]:
    """
    Convert a saved Keras model into standard Float32 and INT8 Dynamic Range Quantized TFLite formats.

    Args:
        keras_model_path: Path to .keras file.
        quantize_int8: Whether to generate dynamic range quantized model.

    Returns:
        Tuple of (float32_tflite_path, quantized_tflite_path).
    """
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model '{keras_model_path}' not found.")

    model_name = keras_model_path.stem.replace("best_", "")
    logger.info(f"Loading Keras model '{keras_model_path}' for TFLite conversion...")
    model = tf.keras.models.load_model(keras_model_path)

    # 1. Standard Float32 TFLite
    converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter_float.convert()

    float_path = config.MODELS_PATH / f"{model_name}_float32.tflite"
    with open(float_path, "wb") as f:
        f.write(tflite_float_model)
    logger.info(f"Standard Float32 TFLite model saved to '{float_path}' ({get_file_size_mb(float_path)} MB).")

    # 2. Dynamic Range INT8 Quantization
    quantized_path = None
    if quantize_int8:
        converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter_quant.convert()

        quantized_path = config.MODELS_PATH / f"{model_name}_quantized_int8.tflite"
        with open(quantized_path, "wb") as f:
            f.write(tflite_quant_model)
        logger.info(
            f"INT8 Quantized TFLite model saved to '{quantized_path}' "
            f"({get_file_size_mb(quantized_path)} MB)."
        )

    return float_path, quantized_path


def compare_original_vs_optimized(
    keras_model_path: Path,
    quantize_int8: bool = True
) -> pd.DataFrame:
    """
    Perform rigorous side-by-side benchmark of original Keras model vs TFLite optimized models.

    Args:
        keras_model_path: Path to original .keras model.
        quantize_int8: Whether to include INT8 quantized model in comparison.

    Returns:
        Comparison DataFrame with size, latency, throughput, and reduction ratio.
    """
    model_name = keras_model_path.stem.replace("best_", "")
    logger.info(f"=== Profiling Optimization for '{model_name}' ===")

    float_tflite_path, quant_tflite_path = convert_to_tflite(keras_model_path, quantize_int8)

    # Benchmark original
    res_orig = benchmark_keras_model(keras_model_path)

    # Benchmark Float32 TFLite
    res_float = benchmark_tflite_model(float_tflite_path)

    comparison_records = [
        {
            "Variant": "Original Model",
            "Format": "Keras (.keras)",
            "Size (MB)": res_orig["model_size_mb"],
            "Mean Latency (ms)": res_orig["mean_latency_ms"],
            "Median Latency (ms)": res_orig["median_latency_ms"],
            "Throughput (FPS)": res_orig["throughput_fps"],
            "Size Reduction": "0.0%",
            "Latency Speedup": "1.00x"
        },
        {
            "Variant": "TFLite Standard",
            "Format": "TensorFlow Lite (.tflite)",
            "Size (MB)": res_float["model_size_mb"],
            "Mean Latency (ms)": res_float["mean_latency_ms"],
            "Median Latency (ms)": res_float["median_latency_ms"],
            "Throughput (FPS)": res_float["throughput_fps"],
            "Size Reduction": f"{((res_orig['model_size_mb'] - res_float['model_size_mb']) / max(res_orig['model_size_mb'], 1e-5)) * 100:.1f}%",
            "Latency Speedup": f"{res_orig['mean_latency_ms'] / max(res_float['mean_latency_ms'], 1e-5):.2f}x"
        }
    ]

    # Benchmark INT8 Quantized TFLite
    if quant_tflite_path and quant_tflite_path.exists():
        res_quant = benchmark_tflite_model(quant_tflite_path)
        comparison_records.append({
            "Variant": "TFLite Dynamic Range (INT8)",
            "Format": "Quantized TFLite (.tflite)",
            "Size (MB)": res_quant["model_size_mb"],
            "Mean Latency (ms)": res_quant["mean_latency_ms"],
            "Median Latency (ms)": res_quant["median_latency_ms"],
            "Throughput (FPS)": res_quant["throughput_fps"],
            "Size Reduction": f"{((res_orig['model_size_mb'] - res_quant['model_size_mb']) / max(res_orig['model_size_mb'], 1e-5)) * 100:.1f}%",
            "Latency Speedup": f"{res_orig['mean_latency_ms'] / max(res_quant['mean_latency_ms'], 1e-5):.2f}x"
        })

    df = pd.DataFrame(comparison_records)

    csv_path = config.RESULTS_PATH / f"optimization_comparison_{model_name}.csv"
    json_path = config.RESULTS_PATH / f"optimization_comparison_{model_name}.json"
    df.to_csv(csv_path, index=False)
    save_json(comparison_records, json_path)

    logger.info(f"Optimization comparison saved to '{csv_path}'.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize and quantize leaf/fruit disease detection models.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.MODELS_PATH / f"best_{config.DEFAULT_MODEL}.keras"),
        help="Path to saved .keras model file"
    )

    args = parser.parse_args()
    df_opt = compare_original_vs_optimized(Path(args.model_path))
    print("\n=== Model Optimization Benchmark ===")
    print(df_opt.to_string(index=False))
