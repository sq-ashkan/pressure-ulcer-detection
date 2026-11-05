"""
Fine-tune YOLOv8s for four-class pressure ulcer detection on Apple Silicon (MPS).

Usage:
    python scripts/train_yolo.py --data data/augmented/dataset.yaml --epochs 60
    python scripts/train_yolo.py --benchmark        # short speed probe, no real training

The script pins the device to MPS when available, records the full run
configuration, and writes all artefacts under runs/detect/<name>/.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def describe_dataset(data_yaml: Path) -> dict:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    counts = {}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if not rel:
            continue
        d = root / rel
        counts[split] = sum(
            1 for p in d.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) if d.exists() else 0
    return {"classes": cfg.get("names"), "nc": cfg.get("nc"), "counts": counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/augmented/dataset.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="pu_yolov8s")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run 1 epoch on a tiny subset to measure throughput")
    args = ap.parse_args()

    data_yaml = (PROJECT_ROOT / args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"dataset config not found: {data_yaml}")

    device = pick_device(args.device)
    info = describe_dataset(data_yaml)

    print("=" * 64)
    print("YOLOv8 TRAINING")
    print("=" * 64)
    print(f"  platform   : {platform.platform()}")
    print(f"  torch      : {torch.__version__}")
    print(f"  device     : {device}  (mps_available={torch.backends.mps.is_available()})")
    print(f"  data       : {data_yaml}")
    print(f"  classes    : {info['nc']} -> {info['classes']}")
    print(f"  images     : {info['counts']}")
    print(f"  model      : {args.model}")
    print(f"  epochs     : {args.epochs}   batch: {args.batch}   imgsz: {args.imgsz}")
    print("=" * 64)

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=1 if args.benchmark else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        patience=args.patience,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=f"{args.name}_benchmark" if args.benchmark else args.name,
        exist_ok=True,
        seed=42,
        deterministic=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        val=True,
        plots=True,
        save=True,
        # The offline pipeline already applied geometric/colour augmentation,
        # so the online policy is kept mild; mosaic is retained because it
        # changes object context rather than duplicating those transforms.
        mosaic=1.0,
        mixup=0.10,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=10,
        fraction=0.02 if args.benchmark else 1.0,
    )

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    (save_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "device": device, "dataset": info,
                    "train_kwargs": {k: str(v) for k, v in train_kwargs.items()}}, indent=2)
    )
    print(f"\nArtefacts written to: {save_dir}")
    best = save_dir / "weights" / "best.pt"
    print(f"Best weights: {best}  (exists={best.exists()})")


if __name__ == "__main__":
    main()
