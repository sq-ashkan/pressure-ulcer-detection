"""
Exploratory Data Analysis for the unified pressure ulcer dataset.
Generates charts and statistics for the LaTeX report.
Usage: python scripts/eda.py
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "Stage 1", 1: "Stage 2", 2: "Stage 3", 3: "Stage 4"}
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def read_labels(label_path):
    entries = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                entries.append({"class": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return entries


def main():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    all_labels = []
    img_dims = []
    class_counts = defaultdict(Counter)
    bbox_per_image = []

    for split in ["train", "val", "test"]:
        img_dir = UNIFIED_DIR / "images" / split
        lbl_dir = UNIFIED_DIR / "labels" / split

        if not img_dir.exists():
            continue

        imgs = sorted(img_dir.iterdir())
        for img_path in tqdm(imgs, desc=f"  {split}", leave=False):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # Image dimensions
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                img_dims.append({"split": split, "width": w, "height": h, "aspect": w / h})

            # Labels
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                entries = read_labels(lbl_path)
                bbox_per_image.append({"split": split, "count": len(entries)})
                for e in entries:
                    e["split"] = split
                    all_labels.append(e)
                    class_counts[split][e["class"]] += 1

    print(f"\n  Total images analyzed: {len(img_dims)}")
    print(f"  Total annotations: {len(all_labels)}")

    # === Chart 1: Class distribution ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, split in enumerate(["train", "val", "test"]):
        counts = class_counts[split]
        classes = sorted(counts.keys())
        values = [counts[c] for c in classes]
        names = [CLASS_NAMES[c] for c in classes]
        colors = [COLORS[c] for c in classes]
        axes[i].bar(names, values, color=colors)
        axes[i].set_title(f"{split.capitalize()} Set")
        axes[i].set_ylabel("Number of Annotations")
        for j, v in enumerate(values):
            axes[i].text(j, v + 5, str(v), ha="center", fontsize=10)
    plt.suptitle("Class Distribution per Split", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: class_distribution.png")

    # === Chart 2: Overall class distribution ===
    fig, ax = plt.subplots(figsize=(8, 5))
    total_counts = Counter()
    for split_counts in class_counts.values():
        total_counts.update(split_counts)
    classes = sorted(total_counts.keys())
    values = [total_counts[c] for c in classes]
    names = [CLASS_NAMES[c] for c in classes]
    colors_list = [COLORS[c] for c in classes]
    bars = ax.bar(names, values, color=colors_list)
    ax.set_ylabel("Number of Annotations")
    ax.set_title("Overall Class Distribution", fontsize=14, fontweight="bold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, str(v), ha="center", fontsize=11)
    total = sum(values)
    for bar, v in zip(bars, values):
        pct = v / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2, f"{pct:.1f}%", ha="center", fontsize=10, color="white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution_overall.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: class_distribution_overall.png")

    # === Chart 3: Image dimension histograms ===
    widths = [d["width"] for d in img_dims]
    heights = [d["height"] for d in img_dims]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=50, color="#2196F3", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image Width Distribution")
    axes[0].axvline(np.mean(widths), color="red", linestyle="--", label=f"Mean: {np.mean(widths):.0f}")
    axes[0].legend()
    axes[1].hist(heights, bins=50, color="#4CAF50", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Image Height Distribution")
    axes[1].axvline(np.mean(heights), color="red", linestyle="--", label=f"Mean: {np.mean(heights):.0f}")
    axes[1].legend()
    plt.suptitle("Image Dimension Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "image_dimensions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: image_dimensions.png")

    # === Chart 4: Bbox size distribution ===
    bbox_widths = [e["w"] for e in all_labels]
    bbox_heights = [e["h"] for e in all_labels]
    bbox_areas = [e["w"] * e["h"] for e in all_labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(bbox_widths, bins=50, color="#FF9800", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Bbox Width (normalized)")
    axes[0].set_title("Bbox Width Distribution")
    axes[1].hist(bbox_heights, bins=50, color="#F44336", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Bbox Height (normalized)")
    axes[1].set_title("Bbox Height Distribution")
    axes[2].hist(bbox_areas, bins=50, color="#9C27B0", alpha=0.7, edgecolor="black")
    axes[2].set_xlabel("Bbox Area (normalized)")
    axes[2].set_title("Bbox Area Distribution")
    plt.suptitle("Bounding Box Size Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bbox_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: bbox_distribution.png")

    # === Chart 5: Bbox center heatmap ===
    fig, ax = plt.subplots(figsize=(8, 8))
    cx_vals = [e["cx"] for e in all_labels]
    cy_vals = [e["cy"] for e in all_labels]
    heatmap, xedges, yedges = np.histogram2d(cx_vals, cy_vals, bins=50, range=[[0, 1], [0, 1]])
    ax.imshow(heatmap.T, origin="lower", cmap="hot", aspect="auto", extent=[0, 1, 0, 1])
    ax.set_xlabel("Center X (normalized)")
    ax.set_ylabel("Center Y (normalized)")
    ax.set_title("Bounding Box Center Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bbox_center_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: bbox_center_heatmap.png")

    # === Chart 6: Annotations per image ===
    fig, ax = plt.subplots(figsize=(8, 5))
    counts_list = [d["count"] for d in bbox_per_image]
    ax.hist(counts_list, bins=range(0, max(counts_list) + 2), color="#00BCD4", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Number of Annotations per Image")
    ax.set_ylabel("Count")
    ax.set_title("Annotations per Image Distribution", fontsize=14, fontweight="bold")
    ax.axvline(np.mean(counts_list), color="red", linestyle="--", label=f"Mean: {np.mean(counts_list):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "annotations_per_image.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: annotations_per_image.png")

    # === Chart 7: Class imbalance ratio ===
    imbalance_ratio = max(total_counts.values()) / min(total_counts.values())

    # === Print summary ===
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"\nDataset Size:")
    print(f"  Total images: {len(img_dims)}")
    print(f"  Train: {sum(1 for d in img_dims if d['split'] == 'train')}")
    print(f"  Val:   {sum(1 for d in img_dims if d['split'] == 'val')}")
    print(f"  Test:  {sum(1 for d in img_dims if d['split'] == 'test')}")
    print(f"\nTotal annotations: {len(all_labels)}")
    print(f"Mean annotations per image: {np.mean(counts_list):.2f}")
    print(f"\nClass distribution (total):")
    for c in sorted(total_counts.keys()):
        pct = total_counts[c] / sum(total_counts.values()) * 100
        print(f"  {CLASS_NAMES[c]}: {total_counts[c]} ({pct:.1f}%)")
    print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"\nImage dimensions:")
    print(f"  Width:  mean={np.mean(widths):.0f}, std={np.std(widths):.0f}, min={min(widths)}, max={max(widths)}")
    print(f"  Height: mean={np.mean(heights):.0f}, std={np.std(heights):.0f}, min={min(heights)}, max={max(heights)}")
    print(f"\nBbox statistics:")
    print(f"  Width:  mean={np.mean(bbox_widths):.3f}, std={np.std(bbox_widths):.3f}")
    print(f"  Height: mean={np.mean(bbox_heights):.3f}, std={np.std(bbox_heights):.3f}")
    print(f"  Area:   mean={np.mean(bbox_areas):.3f}, std={np.std(bbox_areas):.3f}")

    # Save EDA report
    report_path = RESULTS_DIR / "eda_report.txt"
    with open(report_path, "w") as f:
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images: {len(img_dims)}\n")
        f.write(f"Total annotations: {len(all_labels)}\n")
        f.write(f"Mean annotations/image: {np.mean(counts_list):.2f}\n")
        f.write(f"Class imbalance ratio: {imbalance_ratio:.2f}:1\n\n")
        f.write("Class distribution:\n")
        for c in sorted(total_counts.keys()):
            pct = total_counts[c] / sum(total_counts.values()) * 100
            f.write(f"  {CLASS_NAMES[c]}: {total_counts[c]} ({pct:.1f}%)\n")
        f.write(f"\nImage dimensions:\n")
        f.write(f"  Width:  mean={np.mean(widths):.0f}, std={np.std(widths):.0f}\n")
        f.write(f"  Height: mean={np.mean(heights):.0f}, std={np.std(heights):.0f}\n")
        f.write(f"\nBbox statistics:\n")
        f.write(f"  Width:  mean={np.mean(bbox_widths):.3f}, std={np.std(bbox_widths):.3f}\n")
        f.write(f"  Height: mean={np.mean(bbox_heights):.3f}, std={np.std(bbox_heights):.3f}\n")
        f.write(f"  Area:   mean={np.mean(bbox_areas):.3f}, std={np.std(bbox_areas):.3f}\n")

    print(f"\nReport saved: {report_path}")
    print(f"Charts saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
