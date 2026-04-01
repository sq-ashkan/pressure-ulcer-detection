"""
Unified dataset pipeline: label remapping, quality check, deduplication, merging, splitting.
Usage: python scripts/unify_dataset.py
"""

import os
import sys
import cv2
import shutil
import glob
import hashlib
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    import imagehash
    from PIL import Image
except ImportError:
    print("Install: pip install imagehash pillow")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"

# Unified class mapping: 0=stage1, 1=stage2, 2=stage3, 3=stage4
UNIFIED_CLASSES = {0: "stage1", 1: "stage2", 2: "stage3", 3: "stage4"}

# Per-dataset class remapping to unified IDs
# None means "skip/drop this class"
CLASS_REMAPS = {
    "roboflow_stage2": {0: 0, 1: 1, 2: 2, 3: 3},  # Already unified
    "roboflow_project1": {0: 0, 1: 1, 2: 2, 3: 3},  # Already unified
    "roboflow_maskrcnn": {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},  # DTI/Unstageable -> drop
    "roboflow_fid": {0: None, 1: 2, 2: 3},  # NonPU -> drop, Stage3->2, Stage4->3
    "roboflow_woundcare": {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},  # DTI/Unstageable -> drop
    "roboflow_mobile": {0: None, 1: 0, 2: 1, 3: 2, 4: 3},  # NonPU -> drop
    # roboflow_sciproj: body positions, NOT PU stages -> EXCLUDED entirely
    # azh_localization: single "wound" class, no staging -> EXCLUDED for staging task
}

EXCLUDED_DATASETS = ["roboflow_sciproj", "azh_localization"]


def read_yolo_labels(label_path):
    """Read YOLO format label file. Returns list of (class_id, cx, cy, w, h)."""
    entries = []
    if not os.path.exists(label_path):
        return entries
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                entries.append((cls, *bbox))
    return entries


def write_yolo_labels(label_path, entries):
    """Write YOLO format label file."""
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in entries:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def validate_bbox(cx, cy, w, h):
    """Check if YOLO bbox values are valid."""
    for v in [cx, cy, w, h]:
        if v < 0.0 or v > 1.0:
            return False
    if w <= 0 or h <= 0:
        return False
    return True


def remap_labels(dataset_name, entries):
    """Remap class IDs to unified mapping. Returns remapped entries (drops excluded classes)."""
    if dataset_name not in CLASS_REMAPS:
        return []
    remap = CLASS_REMAPS[dataset_name]
    remapped = []
    for cls, cx, cy, w, h in entries:
        new_cls = remap.get(cls)
        if new_cls is not None:
            # Clamp bbox values
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            w = max(0.001, min(0.999, w))
            h = max(0.001, min(0.999, h))
            remapped.append((new_cls, cx, cy, w, h))
    return remapped


def check_image_quality(img_path):
    """Check if image is valid and readable. Returns (ok, reason)."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "unreadable"
        h, w = img.shape[:2]
        if h < 32 or w < 32:
            return False, f"too_small_{w}x{h}"
        if len(img.shape) < 3:
            return False, "grayscale"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def collect_roboflow_images(dataset_name, dataset_path):
    """Collect image-label pairs from a Roboflow dataset."""
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                pairs.append((img_path, lbl_path if lbl_path.exists() else None))
    return pairs


def deduplicate_images(image_paths, hash_size=16, threshold=5):
    """Remove near-duplicate images using perceptual hashing."""
    hashes = {}
    duplicates = set()

    print(f"  Deduplicating {len(image_paths)} images (hash_size={hash_size}, threshold={threshold})...")

    for img_path in tqdm(image_paths, desc="  Hashing", leave=False):
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img, hash_size=hash_size)

            is_dup = False
            for existing_hash, existing_path in hashes.items():
                if abs(h - existing_hash) <= threshold:
                    duplicates.add(str(img_path))
                    is_dup = True
                    break

            if not is_dup:
                hashes[h] = str(img_path)
        except Exception:
            duplicates.add(str(img_path))  # Skip corrupt images

    print(f"  Found {len(duplicates)} duplicates, {len(hashes)} unique images")
    return duplicates


def create_tier2_bbox_labels(dataset_name, dataset_path):
    """Create full-image bounding box labels for classification-only datasets."""
    pairs = []

    if dataset_name == "piid":
        base = dataset_path / "extracted"
        stage_map = {"1": 0, "2": 1, "3": 2, "4": 3}
        # Support multiple folder naming conventions
        folder_variants = {
            "1": ["1", "Stage_I"],
            "2": ["2", "Stage_II"],
            "3": ["3", "Stage_III"],
            "4": ["4", "Stage_IV"],
        }
        for stage_key, unified_id in stage_map.items():
            stage_dir = None
            for variant in folder_variants[stage_key]:
                # Try direct, dataset-1 subfolder, and pressure_ulcers_expanded
                for sub in ["", "dataset-1", "pressure_ulcers_expanded"]:
                    candidate = base / sub / variant if sub else base / variant
                    if candidate.exists():
                        stage_dir = candidate
                        break
                if stage_dir:
                    break
            if not stage_dir:
                continue
            for img_path in stage_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    # Full-image bbox: center=(0.5, 0.5), size=(0.95, 0.95)
                    entry = [(unified_id, 0.5, 0.5, 0.95, 0.95)]
                    pairs.append((img_path, entry))

    elif dataset_name == "kaggle_pu_stages":
        base = dataset_path / "Dataset"
        stage_map = {
            "Stage_I": 0, "Stage_II": 1, "Stage_III": 2, "Stage_IV": 3,
        }
        for folder_name, unified_id in stage_map.items():
            stage_dir = base / folder_name
            if not stage_dir.exists():
                continue
            for img_path in stage_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    entry = [(unified_id, 0.5, 0.5, 0.95, 0.95)]
                    pairs.append((img_path, entry))

    elif dataset_name == "medetec":
        # Only use pressure ulcer images
        base = dataset_path / "data" / "medetec-dataset"
        for pu_folder in ["pressure-ulcer-images-a", "pressure-ulcer-images-b"]:
            pu_dir = base / pu_folder
            if not pu_dir.exists():
                continue
            for img_path in pu_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    # Unknown stage for medetec, skip for now
                    # Could be used as detection-only (no stage classification)
                    pass

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Unify all datasets into one YOLO dataset")
    parser.add_argument("--skip-dedup", action="store_true", help="Skip deduplication step")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET UNIFICATION PIPELINE")
    print("=" * 60)

    # Step 1: Collect and remap all Tier 1 datasets
    print("\n--- Step 1: Collecting and remapping Tier 1 (Roboflow) datasets ---")
    all_pairs = []  # list of (source_img_path, remapped_labels_list, dataset_name)
    stats = defaultdict(lambda: {"total": 0, "kept": 0, "dropped": 0, "classes": Counter()})

    for dataset_name, remap in CLASS_REMAPS.items():
        dataset_path = RAW_DIR / dataset_name
        if not dataset_path.exists():
            print(f"  [SKIP] {dataset_name}: directory not found")
            continue

        pairs = collect_roboflow_images(dataset_name, dataset_path)
        print(f"  {dataset_name}: {len(pairs)} image-label pairs found")

        for img_path, lbl_path in pairs:
            stats[dataset_name]["total"] += 1

            # Read and remap labels
            if lbl_path:
                entries = read_yolo_labels(str(lbl_path))
                remapped = remap_labels(dataset_name, entries)
            else:
                remapped = []

            if len(remapped) > 0:
                stats[dataset_name]["kept"] += 1
                for cls, *_ in remapped:
                    stats[dataset_name]["classes"][cls] += 1
                all_pairs.append((img_path, remapped, dataset_name))
            else:
                stats[dataset_name]["dropped"] += 1

    # Step 2: Collect Tier 2 datasets (classification -> full-image bbox)
    print("\n--- Step 2: Collecting Tier 2 (classification) datasets ---")
    for t2_name in ["piid", "kaggle_pu_stages"]:
        t2_path = RAW_DIR / t2_name
        if not t2_path.exists():
            print(f"  [SKIP] {t2_name}: directory not found")
            continue
        t2_pairs = create_tier2_bbox_labels(t2_name, t2_path)
        print(f"  {t2_name}: {len(t2_pairs)} images with full-image bbox")
        for img_path, entries in t2_pairs:
            stats[t2_name]["total"] += 1
            stats[t2_name]["kept"] += 1
            for cls, *_ in entries:
                stats[t2_name]["classes"][cls] += 1
            all_pairs.append((img_path, entries, t2_name))

    print(f"\n  Total images before dedup: {len(all_pairs)}")

    # Step 3: Image quality check
    print("\n--- Step 3: Image quality check ---")
    quality_ok = []
    quality_fail = 0
    for img_path, labels, ds_name in tqdm(all_pairs, desc="  Quality check", leave=False):
        ok, reason = check_image_quality(img_path)
        if ok:
            quality_ok.append((img_path, labels, ds_name))
        else:
            quality_fail += 1
    print(f"  Passed: {len(quality_ok)}, Failed: {quality_fail}")

    # Step 4: Deduplication
    if not args.skip_dedup:
        print("\n--- Step 4: Deduplication ---")
        img_paths = [p[0] for p in quality_ok]
        duplicates = deduplicate_images(img_paths)
        deduped = [(p, l, d) for p, l, d in quality_ok if str(p) not in duplicates]
        print(f"  After dedup: {len(deduped)} images (removed {len(quality_ok) - len(deduped)})")
    else:
        deduped = quality_ok
        print("\n--- Step 4: Deduplication SKIPPED ---")

    # Step 5: Copy to unified directory
    print("\n--- Step 5: Creating unified dataset ---")

    # Clean output
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)

    for split in ["train", "val", "test"]:
        (UNIFIED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (UNIFIED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(deduped))

    n_train = int(len(deduped) * args.train_ratio)
    n_val = int(len(deduped) * args.val_ratio)

    splits = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            splits[idx] = "train"
        elif i < n_train + n_val:
            splits[idx] = "val"
        else:
            splits[idx] = "test"

    split_counts = Counter()
    class_counts = defaultdict(Counter)

    for idx, (img_path, labels, ds_name) in enumerate(tqdm(deduped, desc="  Copying", leave=False)):
        split = splits[idx]
        split_counts[split] += 1

        # Generate unique filename
        new_name = f"{ds_name}_{img_path.stem}"
        ext = img_path.suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"

        dst_img = UNIFIED_DIR / "images" / split / f"{new_name}{ext}"
        dst_lbl = UNIFIED_DIR / "labels" / split / f"{new_name}.txt"

        # Handle filename collisions
        counter = 0
        while dst_img.exists():
            counter += 1
            dst_img = UNIFIED_DIR / "images" / split / f"{new_name}_{counter}{ext}"
            dst_lbl = UNIFIED_DIR / "labels" / split / f"{new_name}_{counter}.txt"

        # Copy image
        shutil.copy2(str(img_path), str(dst_img))

        # Write remapped labels
        write_yolo_labels(str(dst_lbl), labels)

        for cls, *_ in labels:
            class_counts[split][cls] += 1

    # Step 6: Create dataset.yaml
    print("\n--- Step 6: Creating dataset.yaml ---")
    yaml_path = UNIFIED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {UNIFIED_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        for cls_id, cls_name in UNIFIED_CLASSES.items():
            f.write(f"  {cls_id}: {cls_name}\n")

    # Step 7: Print report
    print("\n" + "=" * 60)
    print("UNIFICATION REPORT")
    print("=" * 60)

    print("\nPer-dataset statistics:")
    for ds_name in sorted(stats.keys()):
        s = stats[ds_name]
        print(f"  {ds_name:25s}: {s['total']:5d} total, {s['kept']:5d} kept, {s['dropped']:5d} dropped")

    print(f"\nSplit distribution:")
    for split in ["train", "val", "test"]:
        print(f"  {split:5s}: {split_counts[split]:5d} images")

    print(f"\nClass distribution per split:")
    for split in ["train", "val", "test"]:
        counts = class_counts[split]
        parts = [f"{UNIFIED_CLASSES[c]}: {counts[c]}" for c in sorted(counts.keys())]
        print(f"  {split:5s}: {', '.join(parts)}")

    total = sum(split_counts.values())
    print(f"\nTotal unified dataset: {total} images")
    print(f"Output: {UNIFIED_DIR}")
    print(f"Config: {yaml_path}")

    # Save report
    report_path = PROJECT_ROOT / "results" / "unification_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("DATASET UNIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-dataset statistics:\n")
        for ds_name in sorted(stats.keys()):
            s = stats[ds_name]
            f.write(f"  {ds_name:25s}: {s['total']:5d} total, {s['kept']:5d} kept, {s['dropped']:5d} dropped\n")
        f.write(f"\nSplit: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}\n")
        f.write(f"Total: {total}\n\n")
        f.write("Class distribution:\n")
        for split in ["train", "val", "test"]:
            counts = class_counts[split]
            f.write(f"  {split}: ")
            for c in sorted(counts.keys()):
                f.write(f"{UNIFIED_CLASSES[c]}={counts[c]} ")
            f.write("\n")

    print(f"\nReport saved: {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
