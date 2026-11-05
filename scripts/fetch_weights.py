"""
Pre-download every pretrained checkpoint the thesis experiments start from, so
training never stalls on a download (and runs offline afterwards).

  models/yolov8/   YOLOv8 detection + segmentation (n/s/m), COCO-pretrained
  models/sam/      SAM 2.1 base + MobileSAM, for model-assisted mask annotation
  ~/.cache/        U-Net encoders (ImageNet) via segmentation_models_pytorch/timm

Usage: python scripts/fetch_weights.py
"""
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

ROOT = Path(__file__).resolve().parents[1]
YOLO = ["yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
SAM = ["sam2.1_b.pt", "mobile_sam.pt"]
UNET_ENCODERS = ["resnet34", "resnet50", "efficientnet-b4"]


def main():
    for name, sub in [(n, "yolov8") for n in YOLO] + [(n, "sam") for n in SAM]:
        dest = ROOT / "models" / sub / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  ok      {dest.relative_to(ROOT)}")
            continue
        path = Path(attempt_download_asset(str(dest)))
        print(f"  fetched {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")

    import segmentation_models_pytorch as smp
    for enc in UNET_ENCODERS:
        smp.Unet(encoder_name=enc, encoder_weights="imagenet", classes=1)
        print(f"  ok      U-Net encoder {enc} (imagenet)")


if __name__ == "__main__":
    main()
