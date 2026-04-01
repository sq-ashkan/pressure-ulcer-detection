"""
Remove duplicate images across merged datasets using perceptual hashing.
Usage: python scripts/deduplicate.py --input data/raw --output data/unified
"""

import imagehash
from PIL import Image
import os
import glob
import shutil
import argparse
from tqdm import tqdm


def find_duplicates(image_dir, hash_size=16, threshold=5):
    """Find duplicate images using perceptual hashing."""
    hashes = {}
    duplicates = []

    image_files = (
        glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(image_dir, "**", "*.jpeg"), recursive=True)
        + glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True)
    )

    print(f"Scanning {len(image_files)} images for duplicates...")

    for img_path in tqdm(image_files):
        try:
            img = Image.open(img_path)
            h = imagehash.phash(img, hash_size=hash_size)

            is_dup = False
            for existing_hash, existing_path in hashes.items():
                if abs(h - existing_hash) <= threshold:
                    duplicates.append((img_path, existing_path))
                    is_dup = True
                    break

            if not is_dup:
                hashes[h] = img_path

        except Exception:
            continue

    print(f"Found {len(duplicates)} duplicate pairs")
    print(f"Unique images: {len(hashes)}")
    return hashes, duplicates


def deduplicate(input_dir, output_dir):
    """Remove duplicates and copy unique images to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    hashes, duplicates = find_duplicates(input_dir)

    # Copy unique images
    img_out = os.path.join(output_dir, "images", "train")
    lbl_out = os.path.join(output_dir, "labels", "train")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    copied = 0
    for h, img_path in tqdm(hashes.items(), desc="Copying unique images"):
        filename = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(img_out, filename))

        # Try to copy corresponding label file
        label_path = img_path.replace("/images/", "/labels/")
        for ext in [".txt"]:
            lbl = os.path.splitext(label_path)[0] + ext
            if os.path.exists(lbl):
                shutil.copy2(lbl, os.path.join(lbl_out, os.path.splitext(filename)[0] + ext))
        copied += 1

    print(f"\nDeduplication complete!")
    print(f"Unique images copied: {copied}")
    print(f"Duplicates removed: {len(duplicates)}")

    # Save duplicate report
    report_path = os.path.join(output_dir, "duplicate_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Total duplicates found: {len(duplicates)}\n\n")
        for dup, orig in duplicates:
            f.write(f"DUPLICATE: {dup}\n  ORIGINAL: {orig}\n\n")
    print(f"Duplicate report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output unified dataset")
    args = parser.parse_args()

    deduplicate(args.input, args.output)
