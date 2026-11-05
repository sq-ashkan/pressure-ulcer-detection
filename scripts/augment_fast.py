"""
Fast parallel multi-round augmentation pipeline.

Operates on the unified dataset and produces an augmented dataset that targets
~53,340 total images (matching the PPA report). Uses multiprocessing for speed.

Per train image: writes 1 _orig + 8 augmented copies (3 geometric + 3 colour +
2 mixed). Val and test are copied unchanged.

Usage:
    python scripts/augment_fast.py [--copies 8]
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import albumentations as A

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "augmented"


# ---------------------------------------------------------------------------
# Augmentation pipelines (must be picklable: built inside worker)
# ---------------------------------------------------------------------------

def get_geometric():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.08, scale=(0.9, 1.1), rotate=(-30, 30),
                     border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_color():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.3),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
                ],
                p=0.85,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def get_mixed():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.06, scale=(0.92, 1.08), rotate=(-20, 20),
                     border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=1.0),
                ],
                p=0.55,
            ),
            A.GaussNoise(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


STRATEGIES = {"geometric": get_geometric, "color": get_color, "mixed": get_mixed}


def read_yolo_labels(path: Path) -> tuple[list, list]:
    bboxes, labels = [], []
    if not path.exists():
        return bboxes, labels
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(int(parts[0]))
                bbox = [max(0.001, min(0.999, float(x))) for x in parts[1:5]]
                bboxes.append(bbox)
    return bboxes, labels


def write_yolo_labels(path: Path, bboxes, labels) -> None:
    with open(path, "w") as f:
        for b, c in zip(bboxes, labels):
            f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")


def process_one_image(args: tuple) -> tuple[int, int]:
    """Worker: read one image, write _orig + 8 augmented copies. Returns (generated, skipped)."""
    img_path_str, lbl_path_str, out_img_dir_str, out_lbl_dir_str, copies_per_strategy = args

    img_path = Path(img_path_str)
    lbl_path = Path(lbl_path_str)
    out_img_dir = Path(out_img_dir_str)
    out_lbl_dir = Path(out_lbl_dir_str)

    image = cv2.imread(str(img_path))
    if image is None:
        return 0, 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, labels = read_yolo_labels(lbl_path)

    filename = img_path.stem
    ext = img_path.suffix

    # Save the original copy
    cv2.imwrite(str(out_img_dir / f"{filename}_orig{ext}"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_yolo_labels(out_lbl_dir / f"{filename}_orig.txt", bboxes, labels)

    generated = 1
    skipped = 0

    g_count, c_count, m_count = copies_per_strategy
    rounds = [("geometric", g_count, 42), ("color", c_count, 123), ("mixed", m_count, 456)]

    for strategy_name, n_copies, seed in rounds:
        if n_copies <= 0:
            continue
        np.random.seed((seed + hash(filename)) % (2**31))
        transform = STRATEGIES[strategy_name]()
        for i in range(n_copies):
            try:
                if bboxes:
                    out = transform(image=image, bboxes=bboxes, class_labels=labels)
                else:
                    out = transform(image=image, bboxes=[], class_labels=[])
                aug_img = out["image"]
                aug_bb = out["bboxes"]
                aug_lb = out["class_labels"]
                aug_name = f"{filename}_{strategy_name}{i}"
                cv2.imwrite(str(out_img_dir / f"{aug_name}{ext}"),
                            cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                write_yolo_labels(out_lbl_dir / f"{aug_name}.txt", aug_bb, aug_lb)
                generated += 1
            except Exception:
                skipped += 1

    return generated, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=8,
                        help="Total augmented copies per train image (excluding original)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    # Distribute copies across the three strategies
    base = args.copies // 3
    remainder = args.copies - 3 * base
    g_count = base + (1 if remainder >= 1 else 0)
    c_count = base + (1 if remainder >= 2 else 0)
    m_count = base
    copies_per_strategy = (g_count, c_count, m_count)

    print("=" * 60)
    print("FAST MULTI-ROUND AUGMENTATION")
    print("=" * 60)
    print(f"  Total copies/image: {args.copies}  (1 orig + {args.copies} aug)")
    print(f"  Per strategy: geometric={g_count}, color={c_count}, mixed={m_count}")
    print(f"  Workers: {args.workers}")

    if AUGMENTED_DIR.exists():
        print(f"  Removing existing {AUGMENTED_DIR}")
        shutil.rmtree(AUGMENTED_DIR)

    for split in ("train", "val", "test"):
        (AUGMENTED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (AUGMENTED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy val and test unchanged
    for split in ("val", "test"):
        src_imgs = UNIFIED_DIR / "images" / split
        src_lbls = UNIFIED_DIR / "labels" / split
        dst_imgs = AUGMENTED_DIR / "images" / split
        dst_lbls = AUGMENTED_DIR / "labels" / split
        for f in src_imgs.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy2(f, dst_imgs / f.name)
        for f in src_lbls.iterdir():
            if f.suffix == ".txt":
                shutil.copy2(f, dst_lbls / f.name)
        n_imgs = len(list(dst_imgs.iterdir()))
        print(f"  {split}: copied {n_imgs} images unchanged")

    # Augment train
    train_img_dir = UNIFIED_DIR / "images" / "train"
    train_lbl_dir = UNIFIED_DIR / "labels" / "train"
    out_img_dir = AUGMENTED_DIR / "images" / "train"
    out_lbl_dir = AUGMENTED_DIR / "labels" / "train"

    img_files = sorted(f for f in train_img_dir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"\n  Train images to augment: {len(img_files)}")
    print(f"  Expected output: {len(img_files) * (args.copies + 1)} train images\n")

    work = [
        (str(p), str(train_lbl_dir / (p.stem + ".txt")),
         str(out_img_dir), str(out_lbl_dir), copies_per_strategy)
        for p in img_files
    ]

    total_gen, total_skip = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one_image, w): w for w in work}
        with tqdm(total=len(futures), desc="Augmenting") as pbar:
            for fut in as_completed(futures):
                g, s = fut.result()
                total_gen += g
                total_skip += s
                pbar.update(1)

    train_count = len(list(out_img_dir.iterdir()))
    val_count = len(list((AUGMENTED_DIR / "images" / "val").iterdir()))
    test_count = len(list((AUGMENTED_DIR / "images" / "test").iterdir()))
    grand_total = train_count + val_count + test_count

    print(f"\n{'=' * 60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Generated: {total_gen}, Skipped: {total_skip}")
    print(f"  Train: {train_count} | Val: {val_count} | Test: {test_count}")
    print(f"  GRAND TOTAL: {grand_total}")

    # Write dataset.yaml
    yaml_path = AUGMENTED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {AUGMENTED_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: stage1\n  1: stage2\n  2: stage3\n  3: stage4\n")
    print(f"  dataset.yaml saved: {yaml_path}")
    print("  Disk usage: ", end="", flush=True)
    os.system(f"du -sh {AUGMENTED_DIR}")


if __name__ == "__main__":
    main()
