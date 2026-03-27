"""
Decubitus Detection — Data Augmentation Pipeline
Run locally to expand dataset for YOLOv8 training.
Usage: python scripts/augment_dataset.py --input data/unified --output data/augmented --copies 5
"""

import albumentations as A
import cv2
import os
import glob
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_augmentation_pipeline():
    """
    Combined augmentation pipeline optimized for wound images.
    All transforms preserve bounding box coordinates.
    """
    return A.Compose(
        [
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            # Color and Lighting
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=30,
                        p=1.0,
                    ),
                ],
                p=0.6,
            ),
            # Noise and Blur
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),
            # Occlusion (simulates partial wound coverage)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def read_yolo_labels(label_path):
    """Read YOLO format label file."""
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    bbox = [max(0.0, min(1.0, v)) for v in bbox]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    """Write YOLO format label file."""
    with open(label_path, "w") as f:
        for bbox, cls in zip(bboxes, class_labels):
            line = f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


def augment_dataset(input_dir, output_dir, num_copies=5):
    """
    Main augmentation function.
    input_dir should have images/ and labels/ subdirectories.
    """
    transform = get_augmentation_pipeline()

    img_dir = os.path.join(input_dir, "images", "train")
    lbl_dir = os.path.join(input_dir, "labels", "train")

    out_img_dir = os.path.join(output_dir, "images", "train")
    out_lbl_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Copy val and test unchanged
    for split in ["val", "test"]:
        for subdir in ["images", "labels"]:
            src = os.path.join(input_dir, subdir, split)
            dst = os.path.join(output_dir, subdir, split)
            if os.path.exists(src):
                os.makedirs(dst, exist_ok=True)
                for f in glob.glob(os.path.join(src, "*")):
                    shutil.copy2(f, dst)

    image_files = (
        glob.glob(os.path.join(img_dir, "*.jpg"))
        + glob.glob(os.path.join(img_dir, "*.jpeg"))
        + glob.glob(os.path.join(img_dir, "*.png"))
    )

    print(f"Found {len(image_files)} images in {img_dir}")
    print(f"Generating {num_copies} augmented copies per image...")
    print(f"Expected output: ~{len(image_files) * (num_copies + 1)} images\n")

    total_generated = 0
    total_skipped = 0

    for img_path in tqdm(image_files, desc="Augmenting"):
        filename = Path(img_path).stem
        ext = Path(img_path).suffix

        image = cv2.imread(img_path)
        if image is None:
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(lbl_dir, filename + ".txt")
        bboxes, class_labels = read_yolo_labels(label_path)

        # Copy original
        cv2.imwrite(
            os.path.join(out_img_dir, f"{filename}_orig{ext}"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        write_yolo_labels(
            os.path.join(out_lbl_dir, f"{filename}_orig.txt"),
            bboxes,
            class_labels,
        )

        # Generate augmented copies
        for i in range(num_copies):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )

                aug_image = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]

                aug_filename = f"{filename}_aug{i}"
                cv2.imwrite(
                    os.path.join(out_img_dir, f"{aug_filename}{ext}"),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                )
                write_yolo_labels(
                    os.path.join(out_lbl_dir, f"{aug_filename}.txt"),
                    aug_bboxes,
                    aug_labels,
                )
                total_generated += 1

            except Exception:
                total_skipped += 1
                continue

    print(f"\nAugmentation complete!")
    print(f"Original images: {len(image_files)}")
    print(f"Augmented images generated: {total_generated}")
    print(f"Skipped/failed: {total_skipped}")
    print(f"Total dataset size: {len(image_files) + total_generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to unified dataset directory"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output augmented dataset"
    )
    parser.add_argument(
        "--copies", type=int, default=5, help="Number of augmented copies per image"
    )
    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.copies)
