# Time-Efficient Deep Learning Model for Real-Time Leaf and Fruit Disease Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.17+](https://img.shields.io/badge/TensorFlow-2.17%2B-orange.svg)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0%2B-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Academic Project](https://img.shields.io/badge/Academic-Final%20Year%20Project-success.svg)](#)

---

## 1. Project Overview & Abstract

Early and accurate diagnosis of plant leaf and fruit diseases is vital to safeguarding agricultural productivity and preventing severe crop loss. While heavy convolutional neural networks (e.g., ResNet-50, VGG-16) achieve high classification accuracy on static benchmarks, their computational footprint and inference latency severely limit real-time continuous deployment on edge devices such as agricultural scouting robots, drones, and handheld field devices.

This project delivers an end-to-end, production-grade **real-time disease detection and diagnosis platform** supporting two unified input modes:
1. **📷 Real-Time Camera Detection**: Live continuous webcam/camera streaming with low-latency inference, rolling FPS throughput, frame-skipping optimization, color-coded HUD overlays, target ROI positioning, and snapshot captures.
2. **🖼️ Image Upload Detection**: High-resolution file uploads, web image URL ingestion, and sample testing with instant latency profiling, confidence scoring, and candidate breakdowns.

Both modes are powered by a shared, single-instance prediction engine loaded once in memory with warm-up optimization, coupled with an **Out-of-Distribution (OOD) & Crop Domain Guard** that prevents false classifications when unsupported crops (such as watermelons or non-target foliage) are evaluated.

---

## 2. Real-Time Detection Architecture

```
                 ┌───────────────────────────┐
                 │ Live Camera / Image File  │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  OpenCV Video Acquisition │
                 │  & Configurable Frame Skip│
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │    Target ROI Cropping    │
                 │   & 224x224 Normalization │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │  Lightweight Neural Net   │
                 │    (MobileNetV3-Small)    │
                 └─────────────┬─────────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Multi-Signal OOD Guard    │
                 │ (Entropy + Crop Validation)│
                 └─────────────┬─────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        [Supported Crop]            [Unsupported Crop (Watermelon)]
      Disease: Early Blight         Unsupported Crop: Watermelon
      Confidence: 96.4%             Out-of-Distribution Intercepted
      Inference: 32.7 ms            Fungicide Advisory Suppressed
      Throughput: 28.6 FPS          Watermelon Pathology Profile
```

---

## 3. Technology Stack

- **Core Language**: Python 3.10+ (tested on Python 3.13)
- **Deep Learning Framework**: TensorFlow 2.17+ / Keras 3
- **Computer Vision**: OpenCV (`cv2`)
- **Data & Numerical Processing**: NumPy, Pandas, Scikit-Learn, Pillow
- **Visualization**: Matplotlib, Seaborn
- **Deployment & Applications**: Streamlit (Web Dashboard) & Native OpenCV (Desktop Real-Time Engine)
- **Edge Quantization**: TensorFlow Lite (TFLite Dynamic Range INT8)

---

## 4. Project Directory Structure

```
leaf_fruit_disease_detection/
│
├── dataset/                         # Dataset root
│   ├── train/                       # Training split (class folders)
│   ├── validation/                  # Validation split (class folders)
│   └── test/                        # Holdout test split (class folders)
│
├── models/                          # Saved models & metadata
│   ├── best_mobilenet_v3_small.keras
│   ├── best_efficientnet_b0.keras
│   ├── mobilenet_v3_small_float32.tflite
│   ├── mobilenet_v3_small_quantized_int8.tflite
│   └── class_names.json             # Dynamic pathology classes
│
├── src/                             # Core modular pipeline
│   ├── __init__.py
│   ├── config.py                    # Centralized settings & hyperparameters
│   ├── model_loader.py              # Zero-overhead single model loading & warmup
│   ├── preprocessing.py             # Model-specific tensor normalization
│   ├── predictor.py                 # Unified shared prediction engine
│   ├── ood_detector.py              # Real-time crop validation & OOD guard
│   ├── camera.py                    # OpenCV CameraHandler with HUD & ROI
│   ├── benchmark.py                 # Latency, FPS, and offline vs. realtime study
│   ├── train.py                     # Two-stage transfer learning pipeline
│   ├── evaluate.py                  # Test set metrics & confusion matrix
│   ├── optimize.py                  # TFLite conversion & INT8 quantization
│   └── utils.py                     # Logging, seeds, system diagnostics
│
├── app/                             # User applications
│   ├── app.py                       # Master Streamlit web portal
│   ├── upload_detection.py          # Streamlit image upload component
│   ├── realtime_detection.py        # Streamlit live webcam component
│   └── realtime_camera.py           # Native low-latency desktop OpenCV camera app
│
├── results/                         # Evaluation artifacts & outputs
│   ├── captures/                    # Saved camera snapshots (timestamped)
│   ├── benchmarks/                  # Latency & comparison CSVs
│   └── plots/                       # Confusion matrices & loss curves
│
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Project documentation
```

---

## 5. Installation & Setup

### 5.1 Activate Virtual Environment
```bash
# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 5.2 Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. How to Run

### 6.1 Launch Master Web Platform (Streamlit)
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`. Features include:
- **🏠 Home**: Interactive portal with direct mode selectors.
- **🖼️ Image Upload Detection**: Ingest images via drag-and-drop, URL, or sample gallery.
- **📷 Real-Time Camera Detection**: Browser webcam stream with real-time HUD and snapshot capture.
- **📊 Model Performance Dashboard**: Empirical comparison table, confusion matrices, and loss curves.
- **📖 Pathology & Methodology**: Agronomic treatment and prevention guide.

### 6.2 Launch Desktop Real-Time Camera Application (OpenCV)
For maximum FPS and zero-overhead edge performance, launch the direct OpenCV desktop application:
```bash
python app/realtime_camera.py
```
**Desktop Controls**:
- `[Q]` or `[ESC]`: Exit application.
- `[S]`: Capture snapshot to `results/captures/capture_YYYY_MM_DD_HH_MM_SS.jpg`.
- `[C]`: Toggle center Region of Interest (ROI) targeting guide.

**CLI Options**:
```bash
# Custom camera index and frame skipping
python app/realtime_camera.py --camera 0 --skip 2 --threshold 0.70
```

### 6.3 Run Offline Single-Image Prediction (CLI)
```bash
python src/predict.py --image "dataset/test/Tomato_Early_Blight/Tomato_Early_Blight_test_0001.jpg"
```

### 6.4 Execute Empirical Benchmarking Suite
```bash
python src/benchmark.py --all_models
```
Generates latency distributions, throughput (FPS), and offline vs. real-time comparative studies saved to `results/benchmarks/`.

---

## 7. Out-of-Distribution (OOD) & Crop Domain Protection

A frequent critical flaw in agricultural computer vision classifiers is the **closed-set softmax assumption**: when an unsupported crop (such as a **Watermelon**, banana, or shoe) is uploaded, standard softmax forces probabilities to sum to 1.0 across trained classes, falsely misdiagnosing the watermelon as Tomato Early Blight or Apple Black Rot.

This platform implements a multi-signal **Crop & OOD Validator** (`src/ood_detector.py`):
1. **Metadata & Filename Inspection**: Flags known non-target agricultural keywords.
2. **Visual Morphology & Pulp Screening**: Analyzes watermelon-specific red interior flesh contrast ($R > 155, R > 1.7G, R > 1.7B$), seeds, and rind geometry.
3. **Information-Theoretic Uncertainty Gating**: Calculates predictive Shannon entropy $H(p)$ and probability margins. Samples with high entropy or confidence below threshold are rejected.
4. **Agronomic Protection**: When an unsupported crop is detected, false solanaceous/apple diagnoses and chemical spray recommendations are safely intercepted and replaced with specialized crop advisories.

---

## 8. Extensibility: Path to YOLO Object Detection

The system is structured so that object detection can be introduced as a modular pre-stage:
```
Camera Feed → YOLO Object Detector → Bounding Box Crop (ROI) → Disease Classifier
```
Currently, the application implements the center-targeting ROI mode, allowing users to position the leaf/fruit within the target box for optimal real-time classification.

---

## 9. Academic Project Deliverables & Research Questions

1. *"Can a lightweight CNN (MobileNetV3-Small) deliver clinically sufficient plant disease classification accuracy while maintaining real-time inference latency?"*
   - **Yes**: MobileNetV3-Small slashes parameter count to ~1.01M with real-time frame rates, while TFLite quantization compresses model footprint down to ~1.17 MB.
2. *"How does real-time streaming latency compare to offline inference?"*
   - Real-time frame skipping ensures 30+ FPS video smoothness while maintaining dedicated per-frame inference profiling.
