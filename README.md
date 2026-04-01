================================================================================
     DECUBITUS DETECTION PROJECT — COMPLETE PLAN & DATASET CATALOG
     Student: Ashkan Sadri Ghamshi | HAWK University | NaMeKI Project
     Date: March 2026
================================================================================


────────────────────────────────────────────────────────────────────────────────
SECTION 1: ALL AVAILABLE DATASETS WITH DOWNLOAD LINKS
────────────────────────────────────────────────────────────────────────────────

Note: All datasets listed below are freely downloadable without email requests
or license agreements. A free account on Roboflow/Kaggle may be required.


=== TIER 1: PRESSURE ULCER — BOUNDING BOX ANNOTATIONS (YOLO-READY) ===

1. Pressure Ulcer by stage2 (Roboflow)
   Images: ~2,811
   Classes: stage1, stage2, stage3, stage4
   Annotation: Bounding Boxes (YOLOv8 TXT export)
   License: CC BY 4.0
   Download: https://universe.roboflow.com/stage2-n7xya/pressure-ulcer-sxitf
   Notes: BEST starting point. Used in published YOLOv8 study (mAP@50=90.8%)

2. pressure-ulcer by Project 1 (Roboflow)
   Images: ~2,013
   Classes: stage1, stage2, stage3, stage4
   Annotation: Bounding Boxes (YOLOv8 TXT export)
   License: CC BY 4.0
   Download: https://universe.roboflow.com/project-1-hozvp/pressure-ulcer-fr7kn
   Notes: May have overlap with dataset #1 — deduplicate after download

3. Pressure Injury maskrcnn (Roboflow)
   Images: ~720
   Classes: Stage1, Stage2, Stage3, Stage4, Unstageable, DTI (6 classes)
   Annotation: Instance Segmentation polygons (convertible to bbox)
   License: CC BY 4.0
   Download: https://universe.roboflow.com/pressure-injury/maskrcnn-swd7w/dataset/1
   Notes: Most complete staging — includes Unstageable and Deep Tissue Injury

4. PressureUlcer by fid (Roboflow)
   Images: ~481
   Classes: pressure ulcer (single class)
   Annotation: Bounding Boxes
   License: CC BY 4.0
   Download: https://universe.roboflow.com/fid-fvcc6/pressureulcer-ctn4w
   Notes: Single-class detection — good for binary wound/no-wound training

5. Advances in Wound Care PI Dataset (Roboflow)
   Images: ~313
   Classes: pressure injuries by stage
   Annotation: Bounding Boxes
   License: CC BY 4.0
   Download: https://universe.roboflow.com/pi-iqy1t/advances-in-wound-care-pi-dataset
   Notes: Annotated by wound care clinicians — highest annotation quality

6. PressureUlcer-SciProj-2024 (Roboflow)
   Images: ~148
   Classes: body-MSC2
   Annotation: Bounding Boxes
   License: CC BY 4.0
   Download: https://universe.roboflow.com/pressure-ulcer-pgayk/pressureulcer-sciproj-2024-fmokt
   Notes: Small but recent (2024)

7. Mobile PU Diagnostic (Roboflow)
   Images: ~313
   Classes: injuries
   Annotation: Bounding Boxes
   License: CC BY 4.0
   Download: https://universe.roboflow.com/class-vx0ys/mobile-application-for-diagnostic-of-pressure-ulcer
   Notes: Mobile-captured images — good for real-world variation

8. AZH Wound Localization (GitHub)
   Images: ~1,010
   Classes: Diabetic Foot Ulcer, Pressure Ulcer, Venous Ulcer
   Annotation: Bounding Boxes (LabelImg YOLO format)
   License: Open (academic)
   Download: https://github.com/uwm-bigdata/wound_localization
   Notes: Multi-wound type with bbox — includes PU class


=== TIER 2: PRESSURE ULCER — CLASSIFICATION LABELS (NEED MANUAL BBOX) ===

9. PIID — Pressure Injury Images Dataset (GitHub/Google Drive)
   Images: 1,091
   Classes: Stage-1, Stage-2, Stage-3, Stage-4
   Annotation: Classification labels only (folder-based)
   License: Academic open access
   Download: https://github.com/FU-MedicalAI/PIID
   Notes: Most-cited PU dataset. 299x299px. Needs manual bbox annotation.

10. Medetec Wound Database (GitHub mirror)
    Images: ~594 total (174 pressure ulcer images)
    Classes: 15 wound categories including pressure ulcers
    Annotation: Folder-based classification only
    License: Copyright-free
    Download: https://github.com/mlaradji/deep-learning-for-wound-care
    Original: http://medetec.co.uk/files/medetec-image-databases.html
    Notes: Small but diverse PU images. Good supplementary source.

11. AZH Wound Classification (GitHub)
    Images: 730
    Classes: Diabetic, Pressure, Surgical, Venous
    Annotation: Classification labels + body location
    License: Open (academic)
    Download: https://github.com/uwm-bigdata/wound-classification-using-images-and-locations
    Notes: Cropped wound ROI images. Needs bbox annotation for YOLO.

12. Pressure Ulcer Images on Figshare (Che Wei Chang)
    Images: ~40 files with LabelMe annotations
    Classes: Tissue types within pressure ulcers
    Annotation: Boundary-based labeling (LabelMe format)
    License: CC BY 4.0
    Download: https://figshare.com/articles/dataset/images_of_pressure_ulcer_2_/17206940/1
    Notes: Small but has segmentation masks. May have access issues.

13. Kaggle Pressure Ulcer Stages (sinemgokoz)
    Images: ~1,091 (likely PIID mirror)
    Classes: Stage 1-4
    Annotation: Folder-based classification
    License: Open
    Download: https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages
    Notes: Check for overlap with PIID before using.


=== TIER 3: RELATED WOUND DATASETS (TRANSFER LEARNING) ===

14. Diabetic_ulcers by ssk (Roboflow)
    Images: ~9,881
    Classes: Diabetic ulcers
    Annotation: Instance Segmentation (convertible to bbox)
    License: CC BY 4.0
    Download: https://universe.roboflow.com/ssk-r6ppk/diabetic_ulcers/dataset/1
    Notes: LARGEST freely available wound dataset with annotations

15. Lower Limb and Feet Wound Dataset (Mendeley Data)
    Images: 8,129 total (2,686 wound images)
    Classes: Wound / Normal
    Annotation: Binary segmentation masks
    License: CC BY 4.0
    Download: https://data.mendeley.com/datasets/hsj38fwnvr/3
    Notes: 331x331px. Direct download without login.

16. Wound Segmentation Images (Kaggle)
    Images: ~2,760
    Classes: Mixed chronic wounds
    Annotation: Segmentation masks
    License: Open
    Download: https://www.kaggle.com/datasets/leoscode/wound-segmentation-images
    Notes: Combined from Medetec, FUSeg, and WSNET sources.

17. FUSeg — Foot Ulcer Segmentation (GitHub / MICCAI 2021)
    Images: 1,210 (from 889 patients)
    Classes: Foot ulcer
    Annotation: Expert segmentation masks
    License: Open (academic)
    Download: https://github.com/uwm-bigdata/wound-segmentation
    Notes: Highest annotation quality. MICCAI challenge dataset.

18. Wound Detection by bitirmetezi2 (Roboflow)
    Images: ~2,636
    Classes: Mixed wounds
    Annotation: Bounding Boxes
    License: CC BY 4.0
    Download: https://universe.roboflow.com/bitirmetezi2/wound-detection (search Roboflow)
    Notes: Mixed wound types with bbox.

19. Chronic Wound Database (Poland)
    Images: 188 image sets (multi-modal)
    Classes: Chronic wounds (mostly diabetic)
    Annotation: Wound outlines by surgeon
    License: Open (academic)
    Download: https://chronicwounddatabase.eu/
    Notes: Multi-modal (photo + thermal + stereo + depth)

20. DFU Dataset on Kaggle (laithjj)
    Images: ~2,673
    Classes: Diabetic Foot Ulcer
    Annotation: Classification
    License: Open
    Download: https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu
    Notes: Good for pre-training on wound features.

21. Foot Ulcer Detection (Roboflow)
    Images: ~741
    Classes: Foot ulcer
    Annotation: Bounding Boxes
    License: CC BY 4.0
    Download: https://universe.roboflow.com/ulcer-detection/foot-ulcer-detection
    Notes: Bbox-annotated foot ulcers.

22. Wound Classification (Kaggle — ibrahimfateen)
    Images: Varies
    Classes: Mixed wounds including pressure ulcers
    Annotation: Classification
    License: Open
    Download: https://www.kaggle.com/datasets/ibrahimfateen/wound-classification
    Notes: Check for PU-specific images.

23. Syn3DWound — Synthetic 3D Wound Dataset
    Images: Synthetic (variable count)
    Classes: Wound types
    Annotation: 2D segmentation masks + 3D geometry
    License: CSIRO (may need portal registration)
    Download: https://doi.org/10.25919/5rwz-ts17
    Paper: https://arxiv.org/abs/2311.15836
    Notes: Blender-generated synthetic wounds. ~13 sec/image generation.


=== TIER 4: SKIN LESION DATASETS (BACKBONE PRE-TRAINING) ===

24. ISIC Archive (all datasets)
    Images: 485,000+ public images
    Classes: Various skin lesions
    Annotation: Classification + some segmentation
    License: CC BY-NC (varies by subset)
    Download: https://challenge.isic-archive.com/data/
    Notes: Largest skin image collection. For backbone pre-training only.

25. HAM10000
    Images: 10,015
    Classes: 7 pigmented skin lesion types
    Annotation: Classification + segmentation masks
    License: CC BY-NC-SA 4.0
    Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
    Also: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
    Notes: Most-cited dermatology dataset.

26. SLICE-3D (ISIC 2024 Challenge)
    Images: ~400,000
    Classes: Skin cancer binary (benign/malignant)
    Annotation: Classification + metadata
    License: CC BY-NC
    Download: https://www.kaggle.com/competitions/isic-2024-challenge
    Notes: Massive dataset. Only for pre-training backbone on skin features.


=== SUPERVISOR-SUGGESTED REPOSITORIES (EVALUATED — NOT USEFUL) ===

27. BioImage Archive — https://www.ebi.ac.uk/bioimage-archive/
    Result: NO wound/PU datasets. Focus: microscopy, cell biology.

28. Cancer Imaging Archive — https://www.cancerimagingarchive.net/
    Result: NO wound/PU datasets. Focus: CT/MRI cancer imaging.

29. PIDAR — https://pidar.hpc4ai.unito.it/
    Result: NO wound/PU datasets. 19 datasets, 431 subjects, all preclinical animal studies.

30. IDR Image Data Resource — https://idr.openmicroscopy.org/
    Result: NO wound/PU datasets. Focus: cell biology microscopy.
    "Wound" entries here = cellular scratch assays, NOT clinical wounds.


────────────────────────────────────────────────────────────────────────────────
SECTION 2: PROJECT PHASES — DETAILED BREAKDOWN
────────────────────────────────────────────────────────────────────────────────


================================================================================
PHASE 1 — PPA (PRAXISPROJEKTARBEIT)
================================================================================

Goal: Prepare a training-ready dataset and select + justify the best model
Deliverables: Augmented dataset + model comparison document + process documentation

--- STEP 1.1: DATA COLLECTION (Weeks 1-2) ---

Download all Tier 1 datasets from Roboflow in YOLOv8 format:
  - Go to each Roboflow link
  - Click "Download Dataset"
  - Select format: "YOLOv8"
  - Choose "download zip to computer"

Download Tier 2 datasets from GitHub/Kaggle:
  - git clone each GitHub repository
  - Download Kaggle datasets via kaggle CLI or browser

Create project folder structure:

  decubitus-detection/
  ├── data/
  │   ├── raw/                    # Original downloads, untouched
  │   │   ├── roboflow_stage2/
  │   │   ├── roboflow_project1/
  │   │   ├── roboflow_maskrcnn/
  │   │   ├── roboflow_fid/
  │   │   ├── azh_localization/
  │   │   ├── piid/
  │   │   ├── medetec/
  │   │   └── ...
  │   ├── unified/                # Merged + deduplicated
  │   │   ├── images/
  │   │   │   ├── train/
  │   │   │   ├── val/
  │   │   │   └── test/
  │   │   └── labels/
  │   │       ├── train/
  │   │       ├── val/
  │   │       └── test/
  │   ├── augmented/              # After augmentation pipeline
  │   └── dataset.yaml            # YOLOv8 config file
  ├── models/
  │   ├── yolov8/
  │   ├── densenet/
  │   └── efficientnet/
  ├── scripts/
  │   ├── deduplicate.py
  │   ├── convert_masks_to_bbox.py
  │   ├── augment_dataset.py
  │   ├── train_yolo.py
  │   └── evaluate.py
  ├── docs/
  │   ├── model_comparison.md
  │   ├── data_sources.md
  │   └── augmentation_strategy.md
  └── results/


--- STEP 1.2: DATA CLEANING & UNIFICATION (Weeks 2-3) ---

Task 1: Deduplication
  - Use perceptual hashing (imagehash library) to find duplicate images
  - Remove exact duplicates and near-duplicates across datasets
  - Expected: ~30-40% overlap between Roboflow datasets

Task 2: Label Unification
  - Standardize all class names to: stage1, stage2, stage3, stage4
  - For 6-class datasets: add unstageable, dti classes
  - Convert all annotations to YOLOv8 format:
    <class_id> <center_x> <center_y> <width> <height>
    (all values normalized 0-1)

Task 3: Segmentation Mask to Bounding Box Conversion
  - For datasets with segmentation masks (maskrcnn, FUSeg, etc.)
  - Extract minimum bounding rectangle around each mask
  - Convert to YOLO normalized format

Task 4: Manual Annotation of PIID Images
  - Use LabelImg or Roboflow Annotate (free tier)
  - Annotate bounding boxes on 1,091 PIID images
  - This adds significant high-quality data

Task 5: Dataset Split
  - Train: 70% | Validation: 20% | Test: 10%
  - Stratified split (equal stage distribution in each set)
  - Keep patient-level separation (no same patient in train and test)


--- STEP 1.3: DATA AUGMENTATION PIPELINE (Weeks 3-4) ---

This is the critical step to multiply dataset size.
Run locally on your machine.

=== AUGMENTATION SCRIPT: augment_dataset.py ===

Dependencies to install:
  pip install albumentations opencv-python-headless pillow numpy

Augmentation Strategy (based on published wound detection papers):

Layer 1 — Geometric Transforms:
  - HorizontalFlip (p=0.5)
  - VerticalFlip (p=0.3)
  - RandomRotate90 (p=0.5)
  - ShiftScaleRotate (shift=0.1, scale=0.15, rotate=45, p=0.5)
  - Perspective (scale=0.05, p=0.3)

Layer 2 — Color/Lighting Transforms:
  - RandomBrightnessContrast (brightness=0.3, contrast=0.3, p=0.5)
  - HueSaturationValue (hue=20, sat=30, val=30, p=0.4)
  - CLAHE (clip_limit=4.0, p=0.3)
  - GaussianBlur (blur_limit=7, p=0.2)
  - GaussNoise (var_limit=(10,50), p=0.2)

Layer 3 — Occlusion/Robustness:
  - CoarseDropout (max_holes=8, max_h=32, max_w=32, p=0.3)
  - RandomShadow (p=0.2)

Layer 4 — YOLOv8 Built-in (applied during training, not pre-processing):
  - Mosaic (combines 4 images into 1) — default ON in YOLOv8
  - MixUp (blends 2 images) — enable in training config
  - Copy-Paste (for instance segmentation)
  - HSV augmentation (built-in)

Target: Generate 5 augmented versions per original image
Expected result: 5,000 originals → 25,000-30,000 training images

IMPORTANT: Augmentation must transform BOTH image AND bounding box labels.
Use albumentations with bbox_params=A.BboxParams(format='yolo')


--- STEP 1.4: MODEL SELECTION & COMPARISON (Weeks 4-6) ---

Models to compare (as specified in project brief):

1. YOLOv8 (Ultralytics) — Object Detection
   - Task: Detection + Localization (bbox)
   - Variants: YOLOv8n, YOLOv8s, YOLOv8m
   - Why: Real-time detection, single-stage, proven on wound data
   - Published PU results: mAP@50 = 90.8% (Lau et al. 2024)

2. DenseNet-121 — Classification
   - Task: Image Classification only
   - Why: Dense connections, good for small datasets
   - Published PU results: ~77% accuracy on PIID

3. EfficientNet-B0/B2 — Classification
   - Task: Image Classification only
   - Why: Best accuracy/efficiency tradeoff
   - Published PU results: Used in multiple wound studies

4. Faster R-CNN (bonus comparison)
   - Task: Detection + Localization
   - Why: Two-stage detector, different architecture philosophy
   - Useful for academic comparison with YOLOv8

Comparison criteria:
  - mAP@50, mAP@50:95 (for detection models)
  - Accuracy, Precision, Recall, F1 (for all)
  - Inference speed (FPS)
  - Model size (parameters, MB)
  - Suitability for NaMeKI integration (real-time requirement)

PPA Recommendation: YOLOv8s or YOLOv8m based on literature review
  - Handles detection + localization in single pass
  - Proven on pressure ulcer data
  - Compatible with real-time camera systems (NaMeKI)
  - Strong community and documentation


--- STEP 1.5: PPA DELIVERABLES ---

1. Prepared Dataset:
   - Unified, deduplicated, augmented dataset in YOLOv8 format
   - dataset.yaml configuration file
   - Documentation of all sources, cleaning steps, augmentation parameters

2. Model Selection Report:
   - Architecture comparison table
   - Literature-based justification
   - Preliminary feasibility analysis
   - Recommendation with reasoning

3. Process Documentation:
   - Data pipeline description (collection → cleaning → augmentation)
   - Source attribution and licensing
   - Reproducibility guide

4. Presentation:
   - Current status
   - Challenges encountered
   - Outlook for Bachelor thesis


================================================================================
PHASE 2 — BACHELOR THESIS (FUTURE)
================================================================================

--- STEP 2.1: MODEL TRAINING (Weeks 1-4 of BA) ---

Training Configuration for YOLOv8:
  - Model: yolov8s.pt (pre-trained on COCO)
  - Image size: 640x640
  - Batch size: 16 (adjust based on GPU memory)
  - Epochs: 100-300 (with early stopping, patience=50)
  - Optimizer: AdamW (lr=0.001, weight_decay=0.0005)
  - Augmentation: Mosaic=1.0, MixUp=0.15, HSV_h=0.015, HSV_s=0.7, HSV_v=0.4

Training command:
  yolo detect train data=dataset.yaml model=yolov8s.pt epochs=200 imgsz=640 batch=16

Experiment tracking:
  - Use Weights & Biases (wandb) or TensorBoard
  - Track: loss curves, mAP progression, per-class AP
  - Compare YOLOv8n vs YOLOv8s vs YOLOv8m

Hyperparameter tuning:
  - Learning rate: [0.0001, 0.001, 0.01]
  - Batch size: [8, 16, 32]
  - Image size: [416, 640, 1024]
  - Augmentation intensity variations


--- STEP 2.2: EVALUATION (Weeks 4-6 of BA) ---

Primary metrics:
  - mAP@50 (IoU threshold 0.5)
  - mAP@50:95 (averaged across IoU 0.5 to 0.95)
  - Precision and Recall per class
  - F1-Score
  - Confusion matrix (stage classification accuracy)

Secondary metrics:
  - Inference time (ms per image)
  - FPS on target hardware
  - Model size and memory footprint

Analysis:
  - Per-stage detection performance (Stage 2 should be highest)
  - Error analysis: which stages are confused most often
  - Comparison with published results
  - Effect of augmentation on performance
  - Cross-dataset generalization


--- STEP 2.3: PROTOTYPE & INTEGRATION (Weeks 6-10 of BA) ---

Functional prototype:
  - Python application using Ultralytics library
  - Input: RGB image or video stream
  - Output: Detected PU regions with bounding boxes + stage labels
  - Confidence score display

NaMeKI integration concept:
  - ONNX or TensorRT model export for deployment
  - API design for integration with existing monitoring system
  - Real-time processing pipeline
  - Alert system for detected pressure ulcers

Optional extensions:
  - Segmentation of detected wound area
  - Wound measurement (area estimation)
  - Severity tracking over time


────────────────────────────────────────────────────────────────────────────────
SECTION 3: DATA AUGMENTATION — COMPLETE LOCAL PIPELINE
────────────────────────────────────────────────────────────────────────────────

Below is the complete Python script to run data augmentation locally.
Save as: scripts/augment_dataset.py


=== PREREQUISITES ===

pip install albumentations opencv-python-headless numpy tqdm pillow imagehash


=== SCRIPT: augment_dataset.py ===

"""
Decubitus Detection — Data Augmentation Pipeline
Run locally to expand dataset for YOLOv8 training.
Usage: python augment_dataset.py --input data/unified --output data/augmented --copies 5
"""

import albumentations as A
import cv2
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_augmentation_pipeline():
    """
    Combined augmentation pipeline optimized for wound images.
    All transforms preserve bounding box coordinates.
    """
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.Perspective(scale=(0.02, 0.05), p=0.3),

        # Color and Lighting
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=1.0
            ),
        ], p=0.6),

        # Noise and Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),

        # Occlusion (simulates partial wound coverage)
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.2
        ),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3  # Drop bboxes that become <30% visible
    ))


def read_yolo_labels(label_path):
    """Read YOLO format label file."""
    bboxes = []
    class_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    # Clamp values to [0, 1]
                    bbox = [max(0.0, min(1.0, v)) for v in bbox]
                    bboxes.append(bbox)
                    class_labels.append(class_id)
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    """Write YOLO format label file."""
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            line = f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


def augment_dataset(input_dir, output_dir, num_copies=5):
    """
    Main augmentation function.
    input_dir should have images/ and labels/ subdirectories.
    """
    transform = get_augmentation_pipeline()

    img_dir = os.path.join(input_dir, 'images', 'train')
    lbl_dir = os.path.join(input_dir, 'labels', 'train')

    out_img_dir = os.path.join(output_dir, 'images', 'train')
    out_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Also copy val and test unchanged
    for split in ['val', 'test']:
        for subdir in ['images', 'labels']:
            src = os.path.join(input_dir, subdir, split)
            dst = os.path.join(output_dir, subdir, split)
            if os.path.exists(src):
                os.makedirs(dst, exist_ok=True)
                for f in glob.glob(os.path.join(src, '*')):
                    import shutil
                    shutil.copy2(f, dst)

    image_files = glob.glob(os.path.join(img_dir, '*.jpg')) + \
                  glob.glob(os.path.join(img_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(img_dir, '*.png'))

    print(f"Found {len(image_files)} images in {img_dir}")
    print(f"Generating {num_copies} augmented copies per image...")
    print(f"Expected output: ~{len(image_files) * (num_copies + 1)} images\n")

    total_generated = 0
    total_skipped = 0

    for img_path in tqdm(image_files, desc="Augmenting"):
        filename = Path(img_path).stem
        ext = Path(img_path).suffix

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            total_skipped += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read labels
        label_path = os.path.join(lbl_dir, filename + '.txt')
        bboxes, class_labels = read_yolo_labels(label_path)

        # Copy original
        cv2.imwrite(
            os.path.join(out_img_dir, f"{filename}_orig{ext}"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )
        write_yolo_labels(
            os.path.join(out_lbl_dir, f"{filename}_orig.txt"),
            bboxes, class_labels
        )

        # Generate augmented copies
        for i in range(num_copies):
            try:
                if len(bboxes) > 0:
                    augmented = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                else:
                    # No bboxes — apply image-only transforms
                    augmented = transform(
                        image=image,
                        bboxes=[],
                        class_labels=[]
                    )

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']

                # Save augmented image
                aug_filename = f"{filename}_aug{i}"
                cv2.imwrite(
                    os.path.join(out_img_dir, f"{aug_filename}{ext}"),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )
                write_yolo_labels(
                    os.path.join(out_lbl_dir, f"{aug_filename}.txt"),
                    aug_bboxes, aug_labels
                )
                total_generated += 1

            except Exception as e:
                total_skipped += 1
                continue

    print(f"\nAugmentation complete!")
    print(f"Original images: {len(image_files)}")
    print(f"Augmented images generated: {total_generated}")
    print(f"Skipped/failed: {total_skipped}")
    print(f"Total dataset size: {len(image_files) + total_generated}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Path to unified dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output augmented dataset')
    parser.add_argument('--copies', type=int, default=5,
                        help='Number of augmented copies per image')
    args = parser.parse_args()

    augment_dataset(args.input, args.output, args.copies)


=== SCRIPT: deduplicate.py ===

"""
Remove duplicate images across merged datasets using perceptual hashing.
Usage: python deduplicate.py --input data/raw --output data/unified
"""

import imagehash
from PIL import Image
import os
import glob
import shutil
from tqdm import tqdm
from collections import defaultdict


def find_duplicates(image_dir, hash_size=16, threshold=5):
    """Find duplicate images using perceptual hashing."""
    hashes = {}
    duplicates = []

    image_files = glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True) + \
                  glob.glob(os.path.join(image_dir, '**', '*.jpeg'), recursive=True) + \
                  glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)

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

        except Exception as e:
            continue

    print(f"Found {len(duplicates)} duplicate pairs")
    print(f"Unique images: {len(hashes)}")
    return hashes, duplicates


=== SCRIPT: convert_masks_to_bbox.py ===

"""
Convert segmentation masks to YOLO bounding box format.
Usage: python convert_masks_to_bbox.py --masks data/raw/fuseg/masks --output data/unified/labels
"""

import cv2
import numpy as np
import os
import glob
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


=== dataset.yaml (YOLOv8 configuration) ===

# Decubitus Detection Dataset Configuration
# Place this file in data/ directory

path: ./data/augmented          # Root directory
train: images/train             # Train images relative to path
val: images/val                 # Validation images relative to path
test: images/test               # Test images relative to path

# Number of classes
nc: 4

# Class names (NPUAP/EPUAP staging)
names:
  0: stage1
  1: stage2
  2: stage3
  3: stage4

# For 6-class version, use:
# nc: 6
# names:
#   0: stage1
#   1: stage2
#   2: stage3
#   3: stage4
#   4: unstageable
#   5: dti


────────────────────────────────────────────────────────────────────────────────
SECTION 4: QUICK START — STEP-BY-STEP LOCAL SETUP
────────────────────────────────────────────────────────────────────────────────

1. Create Python environment:
   python -m venv decubitus-env
   source decubitus-env/bin/activate        # Linux/Mac
   decubitus-env\Scripts\activate           # Windows

2. Install dependencies:
   pip install ultralytics albumentations opencv-python-headless
   pip install numpy pillow tqdm imagehash matplotlib seaborn
   pip install wandb tensorboard  # optional: experiment tracking

3. Download datasets:
   # Roboflow (browser or CLI)
   pip install roboflow
   # Then use Roboflow Python API to download each dataset

   # GitHub
   git clone https://github.com/FU-MedicalAI/PIID.git
   git clone https://github.com/uwm-bigdata/wound_localization.git
   git clone https://github.com/uwm-bigdata/wound-segmentation.git
   git clone https://github.com/mlaradji/deep-learning-for-wound-care.git

4. Run deduplication:
   python scripts/deduplicate.py --input data/raw --output data/unified

5. Run augmentation:
   python scripts/augment_dataset.py --input data/unified --output data/augmented --copies 5

6. Verify dataset:
   yolo detect val data=data/dataset.yaml model=yolov8s.pt  # Quick sanity check

7. Train (Phase 2):
   yolo detect train data=data/dataset.yaml model=yolov8s.pt epochs=200 imgsz=640 batch=16


────────────────────────────────────────────────────────────────────────────────
SECTION 5: EXPECTED DATASET SIZE SUMMARY
────────────────────────────────────────────────────────────────────────────────

Category                          | Raw Unique Images | After 5x Augmentation
----------------------------------+-------------------+----------------------
PU with bounding boxes (Tier 1)   | ~3,500-5,000      | ~17,500-25,000
PU classification only (Tier 2)   | ~2,500-3,000      | ~12,500-15,000
Related wounds (Tier 3)           | ~25,000-30,000    | ~125,000-150,000
Skin lesions pre-training (Tier 4)| ~430,000+         | N/A (pre-training)
----------------------------------+-------------------+----------------------
TOTAL USABLE FOR PU DETECTION     | ~5,000-8,000      | ~25,000-40,000

This is sufficient for competitive YOLOv8 performance based on published
benchmarks. The Lau et al. (2024) study achieved 90.8% mAP@50 with only
2,800 images + augmentation.


================================================================================
END OF DOCUMENT
================================================================================