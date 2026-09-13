"""
Scientific Time-Efficiency and Computational Complexity Benchmark.

Measures per-image inference latency (warm-up, mean, median, min, max, std dev, FPS),
model storage footprint (MB), parameter counts, and FLOPs.
Generates academic comparison tables across architectures.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import tensorflow as tf
from src.config import config
from src.utils import (
    setup_logger,
    load_json,
    save_json,
    get_file_size_mb,
    get_hardware_info,
    estimate_flops,
    get_model_parameter_counts
)

logger = setup_logger("benchmark", config.LOGS_PATH / "benchmark.log")


def benchmark_keras_model(
    model_path: Path,
    warmup_runs: int = config.BENCHMARK_WARMUP_RUNS,
    benchmark_iterations: int = config.BENCHMARK_ITERATIONS,
    input_shape: tuple = (1, 224, 224, 3)
) -> Dict[str, Any]:
    """
    Profile a Keras model for inference latency, throughput, model size, and parameters.

    Args:
        model_path: Path to .keras file.
        warmup_runs: Number of unmeasured runs to prime cache and kernel graph.
        benchmark_iterations: Number of timed inference iterations.
        input_shape: Single-sample batch shape (1, H, W, C).

    Returns:
        Dictionary of latency and complexity statistics.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model_name = model_path.stem.replace("best_", "")
    logger.info(f"Benchmarking Keras Model: {model_name} ({model_path.name})")

    # Hardware profiling
    hw_info = get_hardware_info()

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Size on disk
    size_mb = get_file_size_mb(model_path)

    # Parameter counts
    param_counts = get_model_parameter_counts(model)

    # FLOPs estimation
    flops = estimate_flops(model)

    # Generate dummy input tensor
    dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

    # 1. Warm-up Phase (eliminates trace & memory allocation latency)
    logger.info(f"Executing {warmup_runs} warm-up iterations...")
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)

    # 2. Benchmark Phase
    logger.info(f"Executing {benchmark_iterations} timed benchmark iterations...")
    latencies_ms = []

    for _ in range(benchmark_iterations):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    std_lat = float(np.std(latencies))
    throughput_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    result = {
        "model_name": model_name,
        "format": "Keras (.keras)",
        "model_size_mb": size_mb,
        "total_parameters": param_counts["total_parameters"],
        "trainable_parameters": param_counts["trainable_parameters"],
        "non_trainable_parameters": param_counts["non_trainable_parameters"],
        "estimated_flops": flops,
        "warmup_runs": warmup_runs,
        "benchmark_iterations": benchmark_iterations,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "throughput_fps": round(throughput_fps, 2),
        "hardware": hw_info
    }

    # Merge with test metrics if already evaluated
    metrics_file = config.RESULTS_PATH / f"metrics_{model_name}.json"
    if metrics_file.exists():
        eval_metrics = load_json(metrics_file)
        result["accuracy"] = eval_metrics.get("accuracy", None)
        result["precision_weighted"] = eval_metrics.get("precision_weighted", None)
        result["recall_weighted"] = eval_metrics.get("recall_weighted", None)
        result["f1_weighted"] = eval_metrics.get("f1_weighted", None)

    save_path = config.RESULTS_PATH / f"benchmark_{model_name}.json"
    save_json(result, save_path)
    logger.info(
        f"Benchmark Complete for '{model_name}': "
        f"Latency: {mean_lat:.2f} ms (Median: {median_lat:.2f} ms, Std: {std_lat:.2f} ms) | "
        f"Throughput: {throughput_fps:.1f} FPS | Size: {size_mb:.2f} MB"
    )
    return result


def benchmark_tflite_model(
    tflite_path: Path,
    warmup_runs: int = config.BENCHMARK_WARMUP_RUNS,
    benchmark_iterations: int = config.BENCHMARK_ITERATIONS
) -> Dict[str, Any]:
    """
    Profile a TensorFlow Lite quantized model.

    Args:
        tflite_path: Path to .tflite file.
        warmup_runs: Number of warm-up iterations.
        benchmark_iterations: Number of timed iterations.

    Returns:
        Dictionary of TFLite latency and size metrics.
    """
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite file '{tflite_path}' not found.")

    model_name = tflite_path.stem
    logger.info(f"Benchmarking TFLite Model: {model_name} ({tflite_path.name})")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(input_details[0]["dtype"])

    size_mb = get_file_size_mb(tflite_path)

    # Warm-up
    for _ in range(warmup_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])

    # Timed iterations
    latencies_ms = []
    for _ in range(benchmark_iterations):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    std_lat = float(np.std(latencies))
    throughput_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    result = {
        "model_name": model_name,
        "format": "TensorFlow Lite (.tflite)",
        "model_size_mb": size_mb,
        "total_parameters": "N/A (Compressed FlatBuffer)",
        "trainable_parameters": 0,
        "non_trainable_parameters": "N/A",
        "estimated_flops": "N/A",
        "warmup_runs": warmup_runs,
        "benchmark_iterations": benchmark_iterations,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "throughput_fps": round(throughput_fps, 2),
        "hardware": get_hardware_info()
    }

    save_path = config.RESULTS_PATH / f"benchmark_{model_name}.json"
    save_json(result, save_path)
    return result


def generate_comparison_report(benchmark_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Consolidate all model benchmarks into a comparative analysis table,
    compute efficiency scores, and persist to CSV and Markdown.

    Args:
        benchmark_results: List of benchmark result dictionaries.

    Returns:
        Formatted pandas DataFrame.
    """
    records = []
    for r in benchmark_results:
        rec = {
            "Model": r.get("model_name"),
            "Format": r.get("format"),
            "Accuracy": f"{r.get('accuracy', 0) * 100:.2f}%" if r.get("accuracy") is not None else "[Run Eval]",
            "F1-Score": f"{r.get('f1_weighted', 0) * 100:.2f}%" if r.get("f1_weighted") is not None else "[Run Eval]",
            "Parameters": f"{r.get('total_parameters'):,}" if isinstance(r.get("total_parameters"), (int, float)) else r.get("total_parameters"),
            "Size (MB)": r.get("model_size_mb"),
            "Mean Latency (ms)": r.get("mean_latency_ms"),
            "Median Latency (ms)": r.get("median_latency_ms"),
            "Throughput (FPS)": r.get("throughput_fps"),
        }
        records.append(rec)

    df = pd.DataFrame(records)

    # Save to both RESULTS_PATH and BENCHMARKS_PATH as requested by project specification
    csv_path = config.RESULTS_PATH / "model_comparison.csv"
    md_path = config.RESULTS_PATH / "model_comparison.md"
    bench_csv = config.BENCHMARKS_PATH / "benchmark_results.csv"

    config.BENCHMARKS_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_csv(bench_csv, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Comparison: Accuracy vs. Time-Efficiency\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n### Architectural Trade-off Analysis\n")
        f.write(
            "- **MobileNetV3-Small**: Tailored for resource-constrained edge systems. "
            "Delivers minimum latency and parameter count.\n"
            "- **EfficientNetB0**: Higher computational depth via compound scaling; "
            "often attains higher boundary discrimination but requires greater FLOPs and inference latency.\n"
            "- **TFLite Quantization**: Drastically lowers disk footprint and latency with minimal accuracy loss.\n"
        )

    logger.info(f"Comparative report generated at '{csv_path}' and '{bench_csv}'.")
    return df


def run_offline_vs_realtime_comparison(
    model_path: Optional[Path] = None,
    iterations: int = 50
) -> pd.DataFrame:
    """
    Empirical study comparing Offline Single-Image Prediction vs. Real-Time Camera Stream Prediction.
    
    Demonstrates the impact of frame acquisition, BGR->RGB conversion, ROI cropping,
    and frame-skipping on overall system throughput.
    """
    from src.model_loader import load_model
    from src.predictor import predict_image

    m_path = model_path if model_path else config.MODEL_PATH
    handle = load_model(model_path=m_path, warmup=True)
    m_name = handle.model_name

    # 1. Offline Single Image Prediction
    dummy_offline_img = np.random.randint(0, 256, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3), dtype=np.uint8)
    offline_latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = predict_image(dummy_offline_img, model_handle=handle, filename_hint="offline_test.jpg")
        offline_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. Real-Time Simulated Stream (including 640x480 frame capture, ROI crop, BGR->RGB, frame skip factor = 2)
    stream_frame = np.random.randint(0, 256, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    realtime_latencies = []
    frame_skip = config.FRAME_SKIP

    for frame_idx in range(iterations):
        t0 = time.perf_counter()
        # Simulated camera frame handling
        h, w, _ = stream_frame.shape
        roi_size = min(h, w) * 3 // 4
        x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
        roi_crop = stream_frame[y1:y1 + roi_size, x1:x1 + roi_size]

        if frame_idx % frame_skip == 0:
            _ = predict_image(roi_crop, model_handle=handle, filename_hint="camera_feed")
        realtime_latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_offline_ms = round(float(np.mean(offline_latencies)), 2)
    avg_realtime_ms = round(float(np.mean(realtime_latencies)), 2)
    realtime_fps = round(1000.0 / max(avg_realtime_ms, 1e-4), 1)

    exp_records = [
        {
            "Experiment": "Offline Image Prediction",
            "Model": m_name,
            "Device": handle.device,
            "Latency (ms)": avg_offline_ms,
            "Throughput (FPS)": round(1000.0 / max(avg_offline_ms, 1e-4), 1),
            "Characteristics": "Direct 224x224 input, no camera overhead"
        },
        {
            "Experiment": "Real-Time Camera Stream (Frame Skip = 2)",
            "Model": m_name,
            "Device": handle.device,
            "Latency (ms)": avg_realtime_ms,
            "Throughput (FPS)": realtime_fps,
            "Characteristics": f"640x480 frame capture, ROI targeting, inference every {frame_skip} frames"
        }
    ]

    df_exp = pd.DataFrame(exp_records)
    exp_csv = config.BENCHMARKS_PATH / "offline_vs_realtime_comparison.csv"
    df_exp.to_csv(exp_csv, index=False)
    logger.info(f"Offline vs. Real-Time comparison saved to '{exp_csv}'.")
    return df_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark deep learning models for leaf/fruit disease detection.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific model file to benchmark")
    parser.add_argument("--all_models", action="store_true", help="Benchmark all discovered models in models/")

    args = parser.parse_args()

    results = []
    if args.all_models or args.model_path is None:
        # Search all .keras and .tflite files in models directory
        keras_models = list(config.MODELS_PATH.glob("*.keras"))
        tflite_models = list(config.MODELS_PATH.glob("*.tflite"))

        for km in keras_models:
            res = benchmark_keras_model(km)
            results.append(res)
        for tm in tflite_models:
            res = benchmark_tflite_model(tm)
            results.append(res)
    else:
        target_path = Path(args.model_path)
        if target_path.suffix == ".keras":
            results.append(benchmark_keras_model(target_path))
        elif target_path.suffix == ".tflite":
            results.append(benchmark_tflite_model(target_path))
        else:
            raise ValueError(f"Unsupported model format: {target_path.suffix}")

    if results:
        df_comp = generate_comparison_report(results)
        print("\n=== Model Comparison Table ===")
        print(df_comp.to_string(index=False))

    print("\n=== Running Offline vs. Real-Time Benchmark Experiment ===")
    df_exp = run_offline_vs_realtime_comparison()
    print(df_exp.to_string(index=False))
    print("\nBenchmark completed. Results saved to 'results/benchmarks/'.\n")
