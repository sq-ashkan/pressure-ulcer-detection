"""
Demonstration dataset generator for Decubitus Detection project.

Builds a fully-structured demonstration dataset that mirrors the unified pipeline
output described in the PPA report. Generates synthetic skin/wound patches per
NPUAP/EPUAP stage with proper YOLO labels and 70/20/10 train/val/test split.

Usage:
    python scripts/generate_dataset.py

Output:
    data/unified/
        images/{train,val,test}/  --  7,985 PNG files
        labels/{train,val,test}/  --  7,985 TXT files (YOLO format)
        dataset.yaml
"""

from __future__ import annotations

import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"

IMG_SIZE = 224

# Match the EDA report distribution
CLASS_COUNTS = {
    0: 2028,  # Stage 1
    1: 1921,  # Stage 2
    2: 2386,  # Stage 3
    3: 2142,  # Stage 4
}
TOTAL = sum(CLASS_COUNTS.values())  # 8,477 -> we cap at 7985 by trimming
TARGET_TOTAL = 7985

# 70/20/10 split
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

# Skin tones (RGB)
SKIN_TONES = [
    (245, 213, 191),  # light
    (228, 178, 152),
    (210, 165, 135),
    (193, 142, 109),
    (161, 113, 84),
    (132, 92, 70),    # dark
]


def random_skin_base(rng: np.random.Generator, size: int) -> np.ndarray:
    """Generate a skin-tone background with subtle texture."""
    base = SKIN_TONES[rng.integers(0, len(SKIN_TONES))]
    # Build base image
    img = np.full((size, size, 3), base, dtype=np.float32)
    # Add gentle gradient
    grad = rng.uniform(-15, 15)
    g = np.linspace(0, grad, size).reshape(-1, 1)
    img += g
    # Add multi-scale noise (subtle skin texture)
    noise = rng.normal(0, 8, (size, size, 3))
    img += noise
    # Add a few low-frequency blotches
    for _ in range(rng.integers(2, 6)):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(20, 60)
        delta = rng.uniform(-12, 12, 3)
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        img[mask] += delta
    return np.clip(img, 0, 255).astype(np.uint8)


def add_wound_patch(img: np.ndarray, stage: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Stamp a stage-specific wound shape onto the skin patch.
    Returns YOLO bbox: (cx, cy, w, h) all normalised to [0, 1].
    """
    size = img.shape[0]
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil, mode="RGBA")

    # Wound center & size (in pixels)
    cx_px = rng.integers(int(size * 0.35), int(size * 0.65))
    cy_px = rng.integers(int(size * 0.35), int(size * 0.65))
    radius = rng.integers(int(size * 0.18), int(size * 0.32))

    # Bounding box
    bx0 = max(0, cx_px - radius)
    by0 = max(0, cy_px - radius)
    bx1 = min(size, cx_px + radius)
    by1 = min(size, cy_px + radius)

    # Stage-specific drawing
    if stage == 0:
        # Stage 1: erythema = red/pink translucent patch on intact skin
        for r_off, alpha in [(0, 90), (-6, 60), (-12, 35)]:
            r = max(2, radius + r_off)
            colour = (200 + rng.integers(-20, 20), 90 + rng.integers(-20, 20),
                      85 + rng.integers(-15, 15), alpha)
            draw.ellipse([cx_px - r, cy_px - r, cx_px + r, cy_px + r], fill=colour)
    elif stage == 1:
        # Stage 2: shallow open wound, red/pink centre with darker edge
        # Outer dark rim
        draw.ellipse([cx_px - radius, cy_px - radius, cx_px + radius, cy_px + radius],
                     fill=(140, 50, 45, 200))
        # Inner wound bed
        ir = int(radius * 0.75)
        draw.ellipse([cx_px - ir, cy_px - ir, cx_px + ir, cy_px + ir],
                     fill=(190, 95, 90, 230))
        # A few highlight spots (serum)
        for _ in range(rng.integers(1, 3)):
            sx = cx_px + rng.integers(-ir // 2, ir // 2)
            sy = cy_px + rng.integers(-ir // 2, ir // 2)
            sr = rng.integers(2, 6)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(220, 180, 170, 200))
    elif stage == 2:
        # Stage 3: deeper wound, exposed subcutaneous fat (yellow), some slough
        draw.ellipse([cx_px - radius, cy_px - radius, cx_px + radius, cy_px + radius],
                     fill=(110, 35, 30, 220))
        ir = int(radius * 0.7)
        draw.ellipse([cx_px - ir, cy_px - ir, cx_px + ir, cy_px + ir],
                     fill=(160, 70, 60, 230))
        # Fat patches (yellow-cream)
        for _ in range(rng.integers(2, 5)):
            fx = cx_px + rng.integers(-ir, ir)
            fy = cy_px + rng.integers(-ir, ir)
            fr = rng.integers(4, 12)
            draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=(220, 200, 140, 220))
        # Slough (yellowish)
        for _ in range(rng.integers(1, 3)):
            sx = cx_px + rng.integers(-ir, ir)
            sy = cy_px + rng.integers(-ir, ir)
            sr = rng.integers(3, 8)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(200, 180, 110, 200))
    else:
        # Stage 4: full-thickness, exposed muscle (deep red) and bone (whitish)
        draw.ellipse([cx_px - radius, cy_px - radius, cx_px + radius, cy_px + radius],
                     fill=(80, 25, 20, 230))
        ir = int(radius * 0.7)
        draw.ellipse([cx_px - ir, cy_px - ir, cx_px + ir, cy_px + ir],
                     fill=(130, 45, 40, 230))
        # Exposed muscle patches (deep red-brown)
        for _ in range(rng.integers(2, 4)):
            mx = cx_px + rng.integers(-ir, ir)
            my = cy_px + rng.integers(-ir, ir)
            mr = rng.integers(5, 12)
            draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(150, 60, 50, 230))
        # Bone/tendon exposure (white/cream)
        for _ in range(rng.integers(1, 3)):
            bx = cx_px + rng.integers(-ir // 2, ir // 2)
            by = cy_px + rng.integers(-ir // 2, ir // 2)
            br = rng.integers(4, 9)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(230, 220, 195, 230))
        # Eschar (dark patches)
        for _ in range(rng.integers(0, 2)):
            ex = cx_px + rng.integers(-radius, radius)
            ey = cy_px + rng.integers(-radius, radius)
            er = rng.integers(3, 8)
            draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(40, 25, 20, 200))

    pil = pil.filter(ImageFilter.SMOOTH)
    img[:] = np.array(pil)

    # YOLO bbox (normalised)
    cx = (bx0 + bx1) / 2 / size
    cy = (by0 + by1) / 2 / size
    w = (bx1 - bx0) / size
    h = (by1 - by0) / size
    return cx, cy, w, h


def generate_one(args: tuple[int, int, str, str]) -> tuple[str, str]:
    """Generate one image + label. args = (idx, stage, split, out_root_str)."""
    idx, stage, split, out_root_str = args
    rng = np.random.default_rng(seed=idx * 7 + stage * 13 + 17)
    out_root = Path(out_root_str)

    img = random_skin_base(rng, IMG_SIZE)
    cx, cy, w, h = add_wound_patch(img, stage, rng)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_path = img_dir / f"unified_{idx:05d}_stage{stage + 1}.png"
    lbl_path = lbl_dir / f"unified_{idx:05d}_stage{stage + 1}.txt"

    Image.fromarray(img).save(img_path, format="PNG", optimize=False, compress_level=1)
    with open(lbl_path, "w") as f:
        f.write(f"{stage} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return str(img_path), str(lbl_path)


def main():
    print("=" * 60)
    print("DEMONSTRATION DATASET GENERATOR")
    print("=" * 60)

    if UNIFIED_DIR.exists():
        print(f"  Removing existing {UNIFIED_DIR}")
        shutil.rmtree(UNIFIED_DIR)

    for split in SPLIT_RATIOS:
        (UNIFIED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Build the per-split, per-class assignment list
    rng = random.Random(2026)

    # Build full list of (stage,) indices summing to TARGET_TOTAL
    # Use proportional class counts
    actual_counts = {}
    raw_total = sum(CLASS_COUNTS.values())
    cumulative = 0
    items = list(CLASS_COUNTS.items())
    for i, (cls, cnt) in enumerate(items):
        if i == len(items) - 1:
            actual_counts[cls] = TARGET_TOTAL - cumulative
        else:
            actual_counts[cls] = round(cnt / raw_total * TARGET_TOTAL)
            cumulative += actual_counts[cls]

    print(f"  Target counts per class: {actual_counts}")

    all_items = []
    for cls, count in actual_counts.items():
        all_items.extend([cls] * count)
    rng.shuffle(all_items)

    # Split 70/20/10
    n_total = len(all_items)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_val = int(n_total * SPLIT_RATIOS["val"])
    n_test = n_total - n_train - n_val

    splits_assignment = (
        [(i, stage, "train") for i, stage in enumerate(all_items[:n_train])]
        + [(i, stage, "val") for i, stage in enumerate(all_items[n_train:n_train + n_val], start=n_train)]
        + [(i, stage, "test") for i, stage in enumerate(all_items[n_train + n_val:], start=n_train + n_val)]
    )

    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")
    print(f"  Total: {n_total}")

    # Generate using process pool
    work = [(idx, stage, split, str(UNIFIED_DIR)) for idx, stage, split in splits_assignment]

    print(f"\n  Generating {len(work)} images using process pool...")
    completed = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futures = [ex.submit(generate_one, w) for w in work]
        for fut in as_completed(futures):
            try:
                fut.result()
                completed += 1
                if completed % 500 == 0:
                    print(f"    {completed}/{len(work)}")
            except Exception as e:
                print(f"  ! error: {e}")

    print(f"\n  Done. Generated {completed} image+label pairs.")

    # Write dataset.yaml
    yaml_path = UNIFIED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {UNIFIED_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: stage1\n  1: stage2\n  2: stage3\n  3: stage4\n")
    print(f"  dataset.yaml saved: {yaml_path}")


if __name__ == "__main__":
    main()
