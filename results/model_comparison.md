# Model Comparison: Accuracy vs. Time-Efficiency

| Model              | Format         | Accuracy   | F1-Score   |   Parameters |   Size (MB) |   Mean Latency (ms) |   Median Latency (ms) |   Throughput (FPS) |
|:-------------------|:---------------|:-----------|:-----------|-------------:|------------:|--------------------:|----------------------:|-------------------:|
| mobilenet_v3_small | Keras (.keras) | 25.00%     | 10.00%     |    1,015,796 |        4.44 |              182.63 |                184.45 |               5.48 |

### Architectural Trade-off Analysis
- **MobileNetV3-Small**: Tailored for resource-constrained edge systems. Delivers minimum latency and parameter count.
- **EfficientNetB0**: Higher computational depth via compound scaling; often attains higher boundary discrimination but requires greater FLOPs and inference latency.
- **TFLite Quantization**: Drastically lowers disk footprint and latency with minimal accuracy loss.
