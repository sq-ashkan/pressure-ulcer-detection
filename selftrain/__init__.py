"""selftrain -- self-retraining pressure-ulcer segmentation system (HAWK Bachelor thesis).

Modules
  config      paths and default hyper-parameters (one place, versioned with the code)
  letterbox   aspect-preserving resize/pad and its exact inverse
  data        PyTorch dataset over data/seg (images + uint8 stage masks)
  unet        U-Net (segmentation_models_pytorch) training and inference
  yoloseg     YOLOv8-seg (Ultralytics) training and inference
  metrics     Dice/IoU, per-stage pooled scores, image-level rates, bootstrap CIs
  evaluate    one evaluator for both architectures (threshold chosen on val only)
"""
__version__ = "0.1.0"
