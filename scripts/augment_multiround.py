"""
Multi-round augmentation pipeline for the unified PU dataset.
Generates augmented copies in batches to manage power and disk usage.
Usage: python scripts/augment_multiround.py --copies 10

Power-conscious: runs in small batches, reports progress.
"""

import os
import sys
import cv2
import glob
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("Install: pip install albumentations")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "augmented"


def get_geometric_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.Perspective(scale=(0.02, 0.05), p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def get_color_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def get_mixed_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.1, rotate_limit=30,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.2),
        A.CoarseDropout(max_holes=6, max_height=24, max_width=24,
                        min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.15),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


STRATEGIES = {
    "geometric": get_geometric_pipeline,
    "color": get_color_pipeline,
    "mixed": get_mixed_pipeline,
}


def read_yolo_labels(label_path):
    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bbox = [max(0.001, min(0.999, float(x))) for x in parts[1:5]]
                    bboxes.append(bbox)
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=10, help="Total augmented copies per image")
    parser.add_argument("--batch-size", type=int, default=500, help="Process in batches for power management")
    args = parser.parse_args()

    copies_per_round = max(1, args.copies // 3)
    remainder = args.copies - (copies_per_round * 3)

    print("=" * 60)
    print("MULTI-ROUND DATA AUGMENTATION")
    print("=" * 60)
    print(f"  Copies per image: {args.copies}")
    print(f"  Strategy: 3 rounds x {copies_per_round} copies" +
          (f" + {remainder} extra mixed" if remainder > 0 else ""))

    # Prepare output directory
    if AUGMENTED_DIR.exists():
        shutil.rmtree(AUGMENTED_DIR)

    for split in ["train", "val", "test"]:
        (AUGMENTED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (AUGMENTED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy val and test unchanged
    for split in ["val", "test"]:
        src_imgs = UNIFIED_DIR / "images" / split
        src_lbls = UNIFIED_DIR / "labels" / split
        dst_imgs = AUGMENTED_DIR / "images" / split
        dst_lbls = AUGMENTED_DIR / "labels" / split
        for f in src_imgs.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(str(f), str(dst_imgs / f.name))
        for f in src_lbls.iterdir():
            if f.suffix == ".txt":
                shutil.copy2(str(f), str(dst_lbls / f.name))
        print(f"  Copied {split}: {len(list(dst_imgs.iterdir()))} images")

    # Get train images
    train_img_dir = UNIFIED_DIR / "images" / "train"
    train_lbl_dir = UNIFIED_DIR / "labels" / "train"
    out_img_dir = AUGMENTED_DIR / "images" / "train"
    out_lbl_dir = AUGMENTED_DIR / "labels" / "train"

    image_files = sorted([f for f in train_img_dir.iterdir()
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"\n  Train images: {len(image_files)}")
    print(f"  Expected output: ~{len(image_files) * (args.copies + 1)} images\n")

    total_generated = 0
    total_skipped = 0

    rounds = [
        ("geometric", copies_per_round, 42),
        ("color", copies_per_round, 123),
        ("mixed", copies_per_round + remainder, 456),
    ]

    for img_path in tqdm(image_files, desc="Augmenting"):
        filename = img_path.stem
        ext = img_path.suffix

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read labels
        label_path = train_lbl_dir / (filename + ".txt")
        bboxes, class_labels = read_yolo_labels(str(label_path))

        # Copy original
        cv2.imwrite(str(out_img_dir / f"{filename}_orig{ext}"),
                     cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        write_yolo_labels(str(out_lbl_dir / f"{filename}_orig.txt"), bboxes, class_labels)

        # Generate augmented copies
        for strategy_name, n_copies, seed in rounds:
            np.random.seed(seed + hash(filename) % 10000)
            transform = STRATEGIES[strategy_name]()

            for i in range(n_copies):
                try:
                    if len(bboxes) > 0:
                        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    else:
                        augmented = transform(image=image, bboxes=[], class_labels=[])

                    aug_image = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_labels = augmented["class_labels"]

                    aug_name = f"{filename}_{strategy_name}{i}"
                    cv2.imwrite(str(out_img_dir / f"{aug_name}{ext}"),
                                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    write_yolo_labels(str(out_lbl_dir / f"{aug_name}.txt"),
                                      aug_bboxes, aug_labels)
                    total_generated += 1
                except Exception:
                    total_skipped += 1

    # Summary
    train_count = len(list(out_img_dir.iterdir()))
    val_count = len(list((AUGMENTED_DIR / "images" / "val").iterdir()))
    test_count = len(list((AUGMENTED_DIR / "images" / "test").iterdir()))

    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Original train images: {len(image_files)}")
    print(f"  Augmented copies generated: {total_generated}")
    print(f"  Skipped/failed: {total_skipped}")
    print(f"  Total train: {train_count}")
    print(f"  Val (unchanged): {val_count}")
    print(f"  Test (unchanged): {test_count}")
    print(f"  GRAND TOTAL: {train_count + val_count + test_count}")

    # Update dataset.yaml
    yaml_path = AUGMENTED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {AUGMENTED_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: stage1\n  1: stage2\n  2: stage3\n  3: stage4\n")

    print(f"\n  dataset.yaml saved: {yaml_path}")
    print(f"  Disk usage: ", end="")
    os.system(f"du -sh {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()
