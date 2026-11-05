"""
Generate per-class sample grid images and an overall preview grid for the
demonstration dataset. Useful for slides and quick visual verification.

Outputs (under results/sample_grids/):
    grid_stage1.png ... grid_stage4.png  --  4x4 sample grid per stage
    grid_overview.png                    --  1 row per stage (5 columns)
    grid_augmentation_demo.png           --  one image + 7 augmented variants

Usage:
    python scripts/make_sample_grids.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
AUGMENTED_DIR = PROJECT_ROOT / "data" / "augmented"
OUT_DIR = PROJECT_ROOT / "results" / "sample_grids"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def collect_per_stage(root: Path) -> dict[int, list[Path]]:
    """Return a dict {stage_idx -> [image paths]} from a unified-style dataset."""
    out: dict[int, list[Path]] = {0: [], 1: [], 2: [], 3: []}
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        if not img_dir.exists():
            continue
        for p in img_dir.iterdir():
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            for stage in range(4):
                if f"_stage{stage + 1}" in p.stem or f"_stage{stage + 1}_" in p.stem:
                    out[stage].append(p)
                    break
    return out


def make_grid(images: list[Path], rows: int, cols: int, cell: int, gap: int,
              title: str, subtitle: str = "", bg=(255, 255, 255)) -> Image.Image:
    W = cols * cell + (cols + 1) * gap
    H_top = 90 if not subtitle else 130
    H = rows * cell + (rows + 1) * gap + H_top
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)

    f_title = load_font(36, bold=True)
    f_sub = load_font(18)

    d.text((gap, 14), title, fill=(30, 30, 30), font=f_title)
    if subtitle:
        d.text((gap, 60), subtitle, fill=(110, 110, 110), font=f_sub)

    rng = random.Random(2026)
    sample = rng.sample(images, min(rows * cols, len(images)))
    while len(sample) < rows * cols:
        sample.append(sample[0])

    for i, p in enumerate(sample):
        r, c = divmod(i, cols)
        x = gap + c * (cell + gap)
        y = H_top + gap + r * (cell + gap)
        try:
            im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
            canvas.paste(im, (x, y))
        except Exception:
            d.rectangle([x, y, x + cell, y + cell], fill=(220, 220, 220))

    return canvas


def stage_label(stage_idx: int) -> str:
    return ["Stage I (Erythema)", "Stage II (Partial Thickness)",
            "Stage III (Full Thickness)", "Stage IV (Tissue Loss)"][stage_idx]


def main() -> None:
    print("Collecting samples per stage from unified dataset...")
    samples = collect_per_stage(UNIFIED_DIR)
    for s, paths in samples.items():
        print(f"  Stage {s + 1}: {len(paths)} images")

    # Per-stage 4x4 grids
    for stage_idx in range(4):
        if not samples[stage_idx]:
            continue
        title = f"Sample Patches  -  {stage_label(stage_idx)}"
        subtitle = "16 randomly selected examples from the demonstration dataset"
        grid = make_grid(samples[stage_idx], rows=4, cols=4, cell=200, gap=12,
                         title=title, subtitle=subtitle)
        out_path = OUT_DIR / f"grid_stage{stage_idx + 1}.png"
        grid.save(out_path, optimize=True)
        print(f"  Saved {out_path}")

    # Overview: one row per stage, 5 columns
    rows = 4
    cols = 5
    cell = 200
    gap = 12
    H_top = 130
    W = cols * cell + (cols + 1) * gap + 220  # space for stage label on left
    H = rows * cell + (rows + 1) * gap + H_top
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f_title = load_font(34, bold=True)
    f_sub = load_font(18)
    f_label = load_font(20, bold=True)
    d.text((20, 18), "Overview - all four NPUAP/EPUAP stages", fill=(30, 30, 30), font=f_title)
    d.text((20, 64), "Five random examples per stage; identical scale and resolution.",
           fill=(110, 110, 110), font=f_sub)

    rng = random.Random(2026)
    for r, stage_idx in enumerate(range(4)):
        # Stage label on the left
        d.text((20, H_top + gap + r * (cell + gap) + cell // 2 - 30),
               f"Stage {stage_idx + 1}", fill=(30, 30, 30), font=f_label)
        if not samples[stage_idx]:
            continue
        chosen = rng.sample(samples[stage_idx], min(cols, len(samples[stage_idx])))
        for c, p in enumerate(chosen):
            x = 220 + gap + c * (cell + gap)
            y = H_top + gap + r * (cell + gap)
            try:
                im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
                canvas.paste(im, (x, y))
            except Exception:
                d.rectangle([x, y, x + cell, y + cell], fill=(220, 220, 220))
    out_path = OUT_DIR / "grid_overview.png"
    canvas.save(out_path, optimize=True)
    print(f"  Saved {out_path}")

    # Augmentation demo — pick 1 train image and 7 augmented variants
    if AUGMENTED_DIR.exists():
        train_aug = AUGMENTED_DIR / "images" / "train"
        if train_aug.exists():
            # Pick an _orig file and find its variants
            orig_files = [p for p in train_aug.iterdir() if "_orig.png" in p.name][:50]
            if orig_files:
                rng2 = random.Random(7)
                pick = rng2.choice(orig_files)
                stem_base = pick.stem.replace("_orig", "")
                variants = sorted(p for p in train_aug.iterdir()
                                  if p.stem.startswith(stem_base + "_") and "_orig" not in p.stem)
                chosen = [pick] + variants[:7]
                if len(chosen) >= 8:
                    cell = 220
                    gap = 12
                    cols = 4
                    rows = 2
                    H_top = 130
                    W = cols * cell + (cols + 1) * gap
                    H = rows * cell + (rows + 1) * gap + H_top
                    canvas = Image.new("RGB", (W, H), (255, 255, 255))
                    d = ImageDraw.Draw(canvas)
                    d.text((gap, 18), "Augmentation Demo  -  1 original + 7 variants",
                           fill=(30, 30, 30), font=f_title)
                    d.text((gap, 64), "Geometric, colour and mixed pipelines applied to a single training image.",
                           fill=(110, 110, 110), font=f_sub)
                    for i, p in enumerate(chosen[:8]):
                        r, c = divmod(i, cols)
                        x = gap + c * (cell + gap)
                        y = H_top + gap + r * (cell + gap)
                        im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
                        canvas.paste(im, (x, y))
                        # Caption
                        cap = "Original" if i == 0 else p.stem.split("_")[-1]
                        d.rectangle([x, y + cell - 30, x + cell, y + cell],
                                    fill=(0, 0, 0, 160))
                        d.text((x + 8, y + cell - 26), cap, fill=(255, 255, 255),
                               font=load_font(14, bold=True))
                    out_path = OUT_DIR / "grid_augmentation_demo.png"
                    canvas.save(out_path, optimize=True)
                    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
