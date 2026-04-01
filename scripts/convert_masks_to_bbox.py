"""
Convert segmentation masks to YOLO bounding box format.
Usage: python scripts/convert_masks_to_bbox.py --masks data/raw/fuseg/masks --output data/unified/labels --class-id 0
"""

import cv2
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm


def mask_to_yolo_bbox(mask_path, class_id=0):
    """Convert binary mask to YOLO format bounding box."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    bboxes = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < 100:  # Skip tiny artifacts
            continue

        # Convert to YOLO format (center_x, center_y, width, height) normalized
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        bboxes.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return bboxes


def convert_directory(masks_dir, output_dir, class_id=0):
    """Convert all masks in a directory to YOLO label files."""
    os.makedirs(output_dir, exist_ok=True)

    mask_files = (
        glob.glob(os.path.join(masks_dir, "*.png"))
        + glob.glob(os.path.join(masks_dir, "*.jpg"))
        + glob.glob(os.path.join(masks_dir, "*.jpeg"))
    )

    print(f"Converting {len(mask_files)} masks to YOLO bbox format...")

    converted = 0
    empty = 0

    for mask_path in tqdm(mask_files):
        bboxes = mask_to_yolo_bbox(mask_path, class_id)
        filename = os.path.splitext(os.path.basename(mask_path))[0]
        label_path = os.path.join(output_dir, filename + ".txt")

        if bboxes:
            with open(label_path, "w") as f:
                f.write("\n".join(bboxes) + "\n")
            converted += 1
        else:
            empty += 1

    print(f"\nConversion complete!")
    print(f"Labels created: {converted}")
    print(f"Empty masks (skipped): {empty}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=str, required=True, help="Path to mask images directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output labels directory")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID for all masks")
    args = parser.parse_args()

    convert_directory(args.masks, args.output, args.class_id)
