# Model Comparison and Selection Report

## 1. Architecture Overview

### YOLOv8 (Recommended)
- **Type:** Single-stage object detector
- **Architecture:** CSPDarknet backbone + C2f modules + PANet neck + Decoupled head
- **Variants:** Nano (3.2M params), Small (11.2M), Medium (25.9M), Large (43.7M), XLarge (68.2M)
- **Output:** Bounding boxes + class labels + confidence scores
- **Key advantage:** End-to-end detection and localization in a single pass

### DenseNet-121
- **Type:** Image classifier (no localization)
- **Architecture:** 121 layers, dense connections between blocks, ~8M parameters
- **Output:** Class probabilities only (no bounding boxes)
- **Key limitation:** Cannot localize wounds; only classifies entire image

### EfficientNet-B0
- **Type:** Image classifier (no localization)
- **Architecture:** Compound scaling (depth, width, resolution), ~5.3M parameters
- **Output:** Class probabilities only (no bounding boxes)
- **Key limitation:** Same as DenseNet; no localization capability

### Faster R-CNN
- **Type:** Two-stage object detector
- **Architecture:** ResNet backbone + Region Proposal Network + classification head
- **Output:** Bounding boxes + class labels
- **Key limitation:** Slower inference, not real-time capable

## 2. Literature Performance Comparison

| Study | Model | Dataset Size | mAP@50 | Accuracy | Year |
|-------|-------|-------------|--------|----------|------|
| Lau et al. | YOLOv8 | 2,811 images | 90.8% | - | 2024 |
| Ay et al. | DenseNet-121 | 1,091 images (PIID) | - | 83.6% | 2022 |
| Ay et al. | EfficientNet-B0 | 1,091 images (PIID) | - | 80.2% | 2022 |
| Ahmad et al. | Faster R-CNN | ~1,500 images | 78.3% | - | 2023 |
| Zahia et al. | CNN (custom) | 600 images | - | 91.4% | 2020 |

## 3. Comparison Criteria

| Criterion | YOLOv8 | DenseNet-121 | EfficientNet-B0 | Faster R-CNN |
|-----------|--------|-------------|----------------|-------------|
| Detection + Localization | Yes | No | No | Yes |
| Real-time inference (>15 FPS) | Yes (~100+ FPS) | Yes (classification only) | Yes (classification only) | No (~5-10 FPS) |
| Parameters (smallest variant) | 3.2M (nano) | 8.0M | 5.3M | 41.1M |
| Inference speed | ~2-5ms | ~10-15ms | ~8-12ms | ~50-100ms |
| COCO pretrained weights | Yes | ImageNet | ImageNet | Yes |
| Published PU results | mAP@50=90.8% | Acc=83.6% | Acc=80.2% | mAP@50=78.3% |
| Ease of fine-tuning | Very easy (ultralytics) | Medium (torchvision) | Medium (torchvision) | Complex |
| ONNX/TensorRT export | Built-in | Manual | Manual | Manual |
| NaMeKI real-time capability | Excellent | N/A (no localization) | N/A (no localization) | Marginal |

## 4. Scoring Matrix

| Criterion | Weight | YOLOv8 | DenseNet | EfficientNet | Faster R-CNN |
|-----------|--------|--------|----------|-------------|-------------|
| Detection + Localization | 30% | 10 | 0 | 0 | 10 |
| Published PU performance | 25% | 10 | 7 | 6 | 6 |
| Real-time capability | 20% | 10 | 5 | 5 | 3 |
| Ease of implementation | 15% | 10 | 7 | 7 | 4 |
| Deployment readiness | 10% | 10 | 5 | 5 | 4 |
| **Weighted Score** | **100%** | **10.0** | **4.3** | **3.7** | **6.2** |

## 5. Recommendation

**YOLOv8s (Small variant, 11.2M parameters)** is the recommended model for the following reasons:

1. **Best published results for PU detection:** Lau et al. (2024) achieved 90.8% mAP@50 with only 2,811 images, demonstrating that YOLOv8 handles small PU datasets effectively.

2. **Detection AND localization:** Unlike DenseNet/EfficientNet which can only classify images, YOLOv8 simultaneously detects, localizes, and classifies pressure ulcers, which is essential for the NaMeKI system.

3. **Real-time performance:** YOLOv8s achieves 100+ FPS on GPU, making it suitable for real-time camera-based monitoring in the NaMeKI integrated care platform.

4. **Strong augmentation synergy:** Our 53K+ augmented dataset (expandable to 300K+ for BA) aligns well with YOLO's data-hungry architecture, and the Ultralytics framework provides built-in online augmentation (Mosaic, MixUp).

5. **Easy deployment:** Built-in ONNX and TensorRT export makes integration with the NaMeKI system straightforward.

6. **Active community and documentation:** The Ultralytics ecosystem provides extensive documentation, pretrained weights, and community support.

### Recommended variant: YOLOv8s
- **Why not YOLOv8n (nano)?** Too few parameters for nuanced PU staging (4 classes with subtle visual differences)
- **Why not YOLOv8m (medium)?** Diminishing returns for our dataset size; YOLOv8s provides the best accuracy/speed trade-off
- **Fallback:** If YOLOv8s doesn't achieve sufficient accuracy, upgrade to YOLOv8m

### Training Plan for BA Phase
```
yolo detect train data=dataset.yaml model=yolov8s.pt epochs=200 imgsz=640 batch=16 device=mps
```
- Pre-trained weights: yolov8s.pt (COCO)
- Image size: 640x640
- Batch size: 16
- Epochs: 200 (with early stopping, patience=50)
- Optimizer: AdamW (lr=0.001)
- Built-in augmentation: Mosaic=1.0, MixUp=0.15
