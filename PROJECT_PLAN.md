# DECUBITUS DETECTION — COMPLETE PROJECT PLAN & CHECKLIST
### Student: Ashkan Sadri Ghamshi | HAWK University | NaMeKI Project
### Supervisor: [Your Supervisor Name]
### Date: March 2026

> **Instructions:** Check each box `[x]` as you complete the task.
> This plan covers the full PPA (Praxisprojektarbeit) phase.

---

## PHASE 1 — ENVIRONMENT & PROJECT SETUP

### 1.1 Development Environment
- [ ] 1. Install Python 3.10+ on local machine
- [ ] 2. Create Python virtual environment (`python -m venv venv`)
- [ ] 3. Activate virtual environment and verify Python version
- [ ] 4. Install core ML dependencies (`pip install ultralytics albumentations opencv-python-headless`)
- [ ] 5. Install data processing dependencies (`pip install numpy pillow tqdm imagehash`)
- [ ] 6. Install dataset download tools (`pip install roboflow python-dotenv`)
- [ ] 7. Install visualization libraries (`pip install matplotlib seaborn`)
- [ ] 8. Install experiment tracking tools (`pip install wandb tensorboard`)
- [ ] 9. Freeze dependencies to requirements.txt (`pip freeze > requirements.txt`)
- [ ] 10. Verify all imports work without errors (create test_imports.py)

### 1.2 Project Repository
- [ ] 11. Initialize Git repository with .gitignore
- [ ] 12. Create .env file with ROBOFLOW_API_KEY (excluded from git)
- [ ] 13. Create .env.example template for collaborators
- [ ] 14. Verify .env is properly gitignored (not tracked)
- [ ] 15. Create project folder structure (data/, scripts/, models/, docs/, results/)
- [ ] 16. Add .gitkeep to all empty directories
- [ ] 17. Create initial README.md with project overview
- [ ] 18. Push initial project structure to GitHub

### 1.3 Tool Setup
- [ ] 19. Install and configure LabelImg or CVAT for manual annotation
- [ ] 20. Create Roboflow free account and verify API key works
- [ ] 21. Create Kaggle account and configure kaggle CLI (`pip install kaggle`)
- [ ] 22. Set up Kaggle API credentials (~/.kaggle/kaggle.json)
- [ ] 23. Install LaTeX distribution (TeX Live / MiKTeX) for report writing
- [ ] 24. Verify `pdflatex` or `xelatex` compiles correctly
- [ ] 25. Set up BibTeX for bibliography management
- [ ] 26. Install a LaTeX editor (Overleaf account or VS Code + LaTeX Workshop)

---

## PHASE 2 — DATA COLLECTION

### 2.1 Tier 1: Roboflow Datasets (Bounding Box Annotations — YOLO-Ready)

#### Dataset #1: Pressure Ulcer by stage2 (~2,811 images)
- [ ] 27. Open Roboflow link and verify dataset is accessible
- [ ] 28. Check available versions and select the latest
- [ ] 29. Download dataset in YOLOv8 format to `data/raw/roboflow_stage2/`
- [ ] 30. Verify download integrity (check image count matches ~2,811)
- [ ] 31. Verify label files exist for each image (.txt files)
- [ ] 32. Check class names in data.yaml (should be stage1-stage4)
- [ ] 33. Spot-check 10 random images: open image + label, verify bbox alignment
- [ ] 34. Document dataset statistics (images per class) in spreadsheet

#### Dataset #2: pressure-ulcer by Project 1 (~2,013 images)
- [ ] 35. Open Roboflow link and verify dataset is accessible
- [ ] 36. Download dataset in YOLOv8 format to `data/raw/roboflow_project1/`
- [ ] 37. Verify download integrity (check image count matches ~2,013)
- [ ] 38. Verify label files exist for each image
- [ ] 39. Check class names and IDs match Dataset #1 mapping
- [ ] 40. Spot-check 10 random images for bbox quality
- [ ] 41. Document dataset statistics in spreadsheet
- [ ] 42. Note: flag for deduplication with Dataset #1

#### Dataset #3: Pressure Injury maskrcnn (~720 images, 6 classes)
- [ ] 43. Open Roboflow link and verify dataset is accessible
- [ ] 44. Download dataset in YOLOv8 format to `data/raw/roboflow_maskrcnn/`
- [ ] 45. Verify download integrity (check image count matches ~720)
- [ ] 46. Check class names: Stage1, Stage2, Stage3, Stage4, Unstageable, DTI
- [ ] 47. Note: this has instance segmentation polygons — verify bbox conversion
- [ ] 48. Spot-check 10 random images for annotation quality
- [ ] 49. Document dataset statistics (images per class, including Unstageable/DTI)
- [ ] 50. Create class mapping document (6-class to 4-class conversion rules)

#### Dataset #4: PressureUlcer by fid (~481 images, single class)
- [ ] 51. Open Roboflow link and verify dataset is accessible
- [ ] 52. Download dataset in YOLOv8 format to `data/raw/roboflow_fid/`
- [ ] 53. Verify download integrity (check image count matches ~481)
- [ ] 54. Note: single class "pressure ulcer" — decide on class mapping strategy
- [ ] 55. Spot-check 10 random images for bbox quality
- [ ] 56. Document: these images cannot be staged, only used for wound/no-wound

#### Dataset #5: Advances in Wound Care PI Dataset (~313 images)
- [ ] 57. Open Roboflow link and verify dataset is accessible
- [ ] 58. Download dataset in YOLOv8 format to `data/raw/roboflow_woundcare/`
- [ ] 59. Verify download integrity (check image count matches ~313)
- [ ] 60. Check class names and verify stage annotations
- [ ] 61. Spot-check 10 random images — note: clinician-annotated, highest quality
- [ ] 62. Document dataset statistics in spreadsheet

#### Dataset #6: PressureUlcer-SciProj-2024 (~148 images)
- [ ] 63. Open Roboflow link and verify dataset is accessible
- [ ] 64. Download dataset in YOLOv8 format to `data/raw/roboflow_sciproj/`
- [ ] 65. Verify download integrity (check image count matches ~148)
- [ ] 66. Check class names (body-MSC2 — needs mapping)
- [ ] 67. Spot-check 10 random images for annotation quality
- [ ] 68. Create class mapping from body-MSC2 to stage1-stage4

#### Dataset #7: Mobile PU Diagnostic (~313 images)
- [ ] 69. Open Roboflow link and verify dataset is accessible
- [ ] 70. Download dataset in YOLOv8 format to `data/raw/roboflow_mobile/`
- [ ] 71. Verify download integrity (check image count matches ~313)
- [ ] 72. Check class names ("injuries" — needs mapping)
- [ ] 73. Spot-check 10 random images — note: mobile-captured, varying quality
- [ ] 74. Document image quality characteristics (resolution, blur, lighting)

### 2.2 Tier 1: GitHub Datasets (Bounding Box Annotations)

#### Dataset #8: AZH Wound Localization (~1,010 images)
- [ ] 75. Clone repository: `git clone https://github.com/uwm-bigdata/wound_localization.git`
- [ ] 76. Move to `data/raw/azh_localization/`
- [ ] 77. Verify image count and folder structure
- [ ] 78. Check annotation format (LabelImg YOLO format)
- [ ] 79. Identify which images are Pressure Ulcer (vs Diabetic/Venous)
- [ ] 80. Count PU-specific images separately
- [ ] 81. Verify label class IDs match expected mapping
- [ ] 82. Spot-check 10 random PU images for bbox quality
- [ ] 83. Document: multi-wound dataset, need to filter PU class

### 2.3 Tier 2: Classification-Only Datasets (Need Manual BBox)

#### Dataset #9: PIID — Pressure Injury Images Dataset (1,091 images)
- [ ] 84. Clone repository: `git clone https://github.com/FU-MedicalAI/PIID.git`
- [ ] 85. Move to `data/raw/piid/`
- [ ] 86. Verify folder structure (Stage-1/, Stage-2/, Stage-3/, Stage-4/)
- [ ] 87. Count images per stage folder
- [ ] 88. Verify image dimensions (expected: 299x299px)
- [ ] 89. Note: classification only — no bounding boxes
- [ ] 90. Decide: manual annotation vs full-image bbox vs skip for detection

#### Dataset #10: Medetec Wound Database (~174 PU images)
- [ ] 91. Clone repository: `git clone https://github.com/mlaradji/deep-learning-for-wound-care.git`
- [ ] 92. Move to `data/raw/medetec/`
- [ ] 93. Navigate to pressure ulcer subfolder
- [ ] 94. Count pressure ulcer images specifically
- [ ] 95. Check image quality and dimensions
- [ ] 96. Note: classification only — no bounding boxes

#### Dataset #11: AZH Wound Classification (730 images)
- [ ] 97. Clone repository: `git clone https://github.com/uwm-bigdata/wound-classification-using-images-and-locations.git`
- [ ] 98. Move to `data/raw/azh_classification/`
- [ ] 99. Identify "Pressure" category images
- [ ] 100. Count PU-specific images
- [ ] 101. Check image format (cropped wound ROI)
- [ ] 102. Note: classification only — needs bbox annotation for YOLO

#### Dataset #12: Figshare Pressure Ulcer Images (~40 files)
- [ ] 103. Download from Figshare link to `data/raw/figshare/`
- [ ] 104. Verify LabelMe annotation format
- [ ] 105. Check if annotations can be converted to YOLO bbox
- [ ] 106. Small dataset — evaluate if worth the conversion effort

#### Dataset #13: Kaggle Pressure Ulcer Stages (~1,091 images)
- [ ] 107. Download via Kaggle CLI or browser to `data/raw/kaggle_pu_stages/`
- [ ] 108. Verify folder structure and image count
- [ ] 109. Compare with PIID dataset for overlap (likely a mirror)
- [ ] 110. If duplicate of PIID, mark as skip and document

### 2.4 Tier 3: Related Wound Datasets (Transfer Learning)

#### Dataset #14: Diabetic_ulcers by ssk (~9,881 images)
- [ ] 111. Download from Roboflow to `data/raw/roboflow_diabetic/`
- [ ] 112. Verify image count (~9,881)
- [ ] 113. Check annotation format (instance segmentation → convertible to bbox)
- [ ] 114. Convert segmentation annotations to bounding boxes
- [ ] 115. Document: largest wound dataset, for transfer learning only

#### Dataset #17: FUSeg — Foot Ulcer Segmentation (1,210 images)
- [ ] 116. Clone repository: `git clone https://github.com/uwm-bigdata/wound-segmentation.git`
- [ ] 117. Move to `data/raw/fuseg/`
- [ ] 118. Verify image count and segmentation mask pairs
- [ ] 119. Convert segmentation masks to YOLO bbox using convert_masks_to_bbox.py
- [ ] 120. Verify converted bboxes on 10 random samples

### 2.5 Data Collection Verification
- [ ] 121. Create master spreadsheet of all downloaded datasets
- [ ] 122. Record for each: name, source, image count, classes, annotation type, license
- [ ] 123. Calculate total raw image count across all datasets
- [ ] 124. Verify all Tier 1 datasets have YOLO-format labels
- [ ] 125. List datasets requiring format conversion
- [ ] 126. List datasets requiring manual annotation
- [ ] 127. Back up all raw data to external storage
- [ ] 128. Commit data collection documentation to git

---

## PHASE 3 — DATA CLEANING & UNIFICATION

### 3.1 Label Standardization
- [ ] 129. Create master class mapping document (all source → unified classes)
- [ ] 130. Define unified class IDs: 0=stage1, 1=stage2, 2=stage3, 3=stage4
- [ ] 131. Write label_mapping.py script to remap class IDs across datasets
- [ ] 132. Remap Dataset #1 (roboflow_stage2) labels to unified IDs
- [ ] 133. Remap Dataset #2 (roboflow_project1) labels to unified IDs
- [ ] 134. Remap Dataset #3 (roboflow_maskrcnn) — handle 6→4 class reduction
- [ ] 135. Decide what to do with Unstageable and DTI classes (drop or add)
- [ ] 136. Remap Dataset #4 (roboflow_fid) — single class to unified ID
- [ ] 137. Remap Dataset #5 (roboflow_woundcare) labels to unified IDs
- [ ] 138. Remap Dataset #6 (roboflow_sciproj) — body-MSC2 mapping
- [ ] 139. Remap Dataset #7 (roboflow_mobile) — "injuries" mapping
- [ ] 140. Remap Dataset #8 (azh_localization) — filter PU class only
- [ ] 141. Verify all remapped label files have correct format
- [ ] 142. Spot-check 5 images per dataset after remapping

### 3.2 Annotation Format Conversion
- [ ] 143. Write convert_masks_to_bbox.py (if not already done)
- [ ] 144. Convert Dataset #3 segmentation polygons to YOLO bbox
- [ ] 145. Convert Dataset #14 (diabetic ulcers) segmentation to bbox
- [ ] 146. Convert Dataset #17 (FUSeg) segmentation masks to bbox
- [ ] 147. Convert any LabelMe annotations (Dataset #12) to YOLO format
- [ ] 148. Verify all converted labels: check 10 random samples per dataset
- [ ] 149. Write validation script to check YOLO label format correctness
- [ ] 150. Run validation script on ALL label files — fix any errors

### 3.3 Image Quality Control
- [ ] 151. Write image_quality_check.py script
- [ ] 152. Scan all images for corrupt/unreadable files
- [ ] 153. Remove images that cannot be opened by OpenCV
- [ ] 154. Check for extremely small images (< 50x50 px) — flag or remove
- [ ] 155. Check for extremely large images (> 4000x4000 px) — flag for resize
- [ ] 156. Identify and log grayscale images (should be RGB for wound detection)
- [ ] 157. Identify images with no corresponding label file — decide action
- [ ] 158. Identify label files with no corresponding image — remove orphan labels
- [ ] 159. Check for images with empty label files (no annotations) — decide action
- [ ] 160. Log all quality issues in a report file

### 3.4 Deduplication
- [ ] 161. Write/refine deduplicate.py script with perceptual hashing
- [ ] 162. Configure hash_size=16 and threshold=5 for near-duplicate detection
- [ ] 163. Run deduplication across ALL Tier 1 datasets combined
- [ ] 164. Review duplicate report — check 20 random duplicate pairs visually
- [ ] 165. Adjust threshold if too aggressive or too loose
- [ ] 166. Re-run deduplication with tuned parameters
- [ ] 167. Remove confirmed duplicates, keep the one with better annotation
- [ ] 168. Log deduplication results: total removed, per-dataset breakdown
- [ ] 169. Run deduplication between Tier 1 and Tier 2 datasets
- [ ] 170. Verify PIID vs Kaggle PU Stages overlap (expected: same dataset)
- [ ] 171. Final unique image count after deduplication

### 3.5 Manual Annotation (Tier 2 Datasets)
- [ ] 172. Decide which Tier 2 datasets are worth annotating manually
- [ ] 173. If annotating PIID: set up LabelImg or Roboflow Annotate
- [ ] 174. Create annotation guidelines document (what constitutes each stage bbox)
- [ ] 175. Annotate PIID Stage-1 images with bounding boxes (batch 1 of 4)
- [ ] 176. Annotate PIID Stage-2 images with bounding boxes (batch 2 of 4)
- [ ] 177. Annotate PIID Stage-3 images with bounding boxes (batch 3 of 4)
- [ ] 178. Annotate PIID Stage-4 images with bounding boxes (batch 4 of 4)
- [ ] 179. Export all PIID annotations in YOLO format
- [ ] 180. Quality-check 20 random PIID annotations
- [ ] 181. If annotating Medetec PU images: repeat annotation process
- [ ] 182. If annotating AZH Classification PU images: repeat annotation process

### 3.6 Dataset Merging
- [ ] 183. Create merge_datasets.py script
- [ ] 184. Copy all deduplicated Tier 1 images to `data/unified/images/`
- [ ] 185. Copy corresponding labels to `data/unified/labels/`
- [ ] 186. Add annotated Tier 2 images to unified dataset
- [ ] 187. Rename files with source prefix to avoid name collisions (e.g., `rf1_img001.jpg`)
- [ ] 188. Update label filenames to match renamed images
- [ ] 189. Verify image↔label pairing after rename (count must match)
- [ ] 190. Count total unified dataset size

### 3.7 Dataset Splitting
- [ ] 191. Write split_dataset.py script with stratified splitting
- [ ] 192. Analyze class distribution in unified dataset
- [ ] 193. Perform stratified split: 70% train / 20% val / 10% test
- [ ] 194. Ensure no patient-level leakage (same wound not in train AND test)
- [ ] 195. Move images to train/val/test subdirectories
- [ ] 196. Move labels to corresponding train/val/test subdirectories
- [ ] 197. Verify class balance in train split
- [ ] 198. Verify class balance in val split
- [ ] 199. Verify class balance in test split
- [ ] 200. Log split statistics: images per class per split
- [ ] 201. Create dataset.yaml with correct paths and class names

### 3.8 Data Cleaning Verification
- [ ] 202. Run final validation: every image has a label, every label has an image
- [ ] 203. Run final validation: all labels are valid YOLO format
- [ ] 204. Run final validation: all class IDs are in range [0, nc-1]
- [ ] 205. Run final validation: all bbox values are in range [0, 1]
- [ ] 206. Run YOLOv8 dataset check: `yolo detect val data=dataset.yaml model=yolov8n.pt`
- [ ] 207. Fix any errors from YOLO validation
- [ ] 208. Document total unified dataset: images, classes, distribution
- [ ] 209. Commit all cleaning scripts and documentation

---

## PHASE 4 — DATA AUGMENTATION

### 4.1 Augmentation Pipeline Development
- [ ] 210. Review augment_dataset.py script for correctness
- [ ] 211. Verify albumentations BboxParams format='yolo' is set
- [ ] 212. Verify min_visibility=0.3 threshold is appropriate
- [ ] 213. Test augmentation pipeline on 5 sample images
- [ ] 214. Visually inspect augmented images + bboxes for correctness
- [ ] 215. Verify augmented bbox values stay within [0, 1]
- [ ] 216. Test edge case: image with multiple bboxes
- [ ] 217. Test edge case: image with small bboxes near image border
- [ ] 218. Test edge case: image with no bboxes (background/negative)

### 4.2 Augmentation Strategy Tuning
- [ ] 219. Test geometric transforms individually: HorizontalFlip
- [ ] 220. Test geometric transforms individually: VerticalFlip
- [ ] 221. Test geometric transforms individually: RandomRotate90
- [ ] 222. Test geometric transforms individually: ShiftScaleRotate
- [ ] 223. Test geometric transforms individually: Perspective
- [ ] 224. Test color transforms individually: RandomBrightnessContrast
- [ ] 225. Test color transforms individually: HueSaturationValue
- [ ] 226. Test color transforms individually: CLAHE
- [ ] 227. Test noise transforms individually: GaussianBlur
- [ ] 228. Test noise transforms individually: GaussNoise
- [ ] 229. Test occlusion transforms individually: CoarseDropout
- [ ] 230. Verify combined pipeline produces realistic wound images
- [ ] 231. Adjust probabilities if augmentations look unrealistic
- [ ] 232. Document final augmentation parameters and reasoning

### 4.3 Augmentation Execution
- [ ] 233. Calculate required copies per image to reach 300,000 target
- [ ] 234. Determine copies_per_image based on unified dataset size
- [ ] 235. Run augmentation on train split: `python scripts/augment_dataset.py --input data/unified --output data/augmented --copies N`
- [ ] 236. Monitor progress and check for errors during run
- [ ] 237. Verify augmented train image count matches expected total
- [ ] 238. Verify augmented label count matches image count
- [ ] 239. Verify val and test sets were copied unchanged (NOT augmented)
- [ ] 240. Spot-check 20 random augmented images visually
- [ ] 241. Verify augmented labels have valid YOLO format
- [ ] 242. Check class distribution in augmented dataset
- [ ] 243. Verify total dataset size reaches ≥300,000 images

### 4.4 Augmentation Quality Assurance
- [ ] 244. Write augmentation_qa.py — visual grid of original + augmented pairs
- [ ] 245. Generate QA grids for 20 random images
- [ ] 246. Review QA grids: bboxes align correctly after augmentation?
- [ ] 247. Review QA grids: augmented images look realistic?
- [ ] 248. Review QA grids: no artifacts or corruption?
- [ ] 249. Check for any empty label files in augmented set
- [ ] 250. Run YOLO validation on augmented dataset
- [ ] 251. Fix any issues found during QA
- [ ] 252. Document augmentation results: final counts, parameters, QA findings

### 4.5 Final Dataset Verification
- [ ] 253. Update dataset.yaml to point to augmented directory
- [ ] 254. Run `yolo detect val data=dataset.yaml model=yolov8n.pt` as sanity check
- [ ] 255. Verify YOLO can load and iterate through the entire augmented dataset
- [ ] 256. Document final dataset statistics in data_sources.md
- [ ] 257. Record: total images (train/val/test), images per class, augmentation ratio
- [ ] 258. Back up augmented dataset to external storage
- [ ] 259. Commit augmentation scripts and documentation

---

## PHASE 5 — MODEL COMPARISON & SELECTION (PPA Deliverable)

### 5.1 Literature Review for Model Comparison
- [ ] 260. Search Google Scholar for "YOLOv8 pressure ulcer detection" papers
- [ ] 261. Search Google Scholar for "wound detection deep learning" papers
- [ ] 262. Find Lau et al. 2024 paper (mAP@50=90.8% with YOLOv8)
- [ ] 263. Search for DenseNet wound classification papers
- [ ] 264. Search for EfficientNet wound classification papers
- [ ] 265. Search for Faster R-CNN wound detection papers
- [ ] 266. Compile comparison table from literature: model, dataset, metrics
- [ ] 267. Read at least 10 relevant papers in detail
- [ ] 268. Create BibTeX entries for all referenced papers
- [ ] 269. Document key findings per model architecture

### 5.2 Model Architecture Analysis
- [ ] 270. Document YOLOv8 architecture: layers, parameters, FLOPs
- [ ] 271. Compare YOLOv8 variants: nano, small, medium (size, speed, accuracy)
- [ ] 272. Document DenseNet-121 architecture: layers, parameters, dense blocks
- [ ] 273. Document EfficientNet-B0/B2 architecture: compound scaling
- [ ] 274. Document Faster R-CNN architecture: region proposal + classification
- [ ] 275. Create architecture comparison table (params, FLOPs, input size, output type)

### 5.3 Model Comparison Criteria
- [ ] 276. Define primary metrics: mAP@50, mAP@50:95 (detection models)
- [ ] 277. Define primary metrics: Accuracy, Precision, Recall, F1 (all models)
- [ ] 278. Define secondary metrics: inference speed (FPS), model size (MB)
- [ ] 279. Define practical criteria: real-time capability for NaMeKI integration
- [ ] 280. Define practical criteria: deployment format (ONNX, TensorRT)
- [ ] 281. Create weighted scoring matrix for model selection
- [ ] 282. Document comparison methodology

### 5.4 Preliminary Model Testing (Quick Benchmark)
- [ ] 283. Run YOLOv8n on a small subset (500 images) for feasibility
- [ ] 284. Run YOLOv8s on the same subset
- [ ] 285. Record training time, loss convergence, preliminary mAP
- [ ] 286. Compare quick-benchmark results with literature
- [ ] 287. Document preliminary findings

### 5.5 Model Selection Report
- [ ] 288. Write model_comparison.md with all findings
- [ ] 289. Include architecture diagrams (or cite from papers)
- [ ] 290. Include literature-based performance comparison table
- [ ] 291. Include preliminary benchmark results
- [ ] 292. Write recommendation section: justify YOLOv8s/m selection
- [ ] 293. Explain why YOLO is preferred over classification-only models
- [ ] 294. Explain NaMeKI real-time requirements and how YOLO fits
- [ ] 295. Review and proofread model comparison document

---

## PHASE 6 — LaTeX REPORT (HAWK Standard — PPA Documentation)

### 6.1 LaTeX Project Setup
- [ ] 296. Create `docs/report/` directory for LaTeX project
- [ ] 297. Create main.tex with HAWK-compliant document class
- [ ] 298. Set up document preamble: geometry, fonts, encoding (UTF-8)
- [ ] 299. Configure language settings (German or English — per supervisor preference)
- [ ] 300. Set up chapter-relative numbering (\numberwithin{equation}{section}, etc.)
- [ ] 301. Configure BibTeX with appropriate style (german.bst or english.bst)
- [ ] 302. Create references.bib file for bibliography
- [ ] 303. Set up listings package for Python source code display
- [ ] 304. Configure lstset for Python syntax highlighting
- [ ] 305. Set up graphicx for figure inclusion
- [ ] 306. Configure hyperref for clickable cross-references
- [ ] 307. Set up glossary package for abbreviations table
- [ ] 308. Test compile: main.tex → PDF without errors
- [ ] 309. Set up Makefile or latexmk for automated compilation

### 6.2 Report Structure (per HAWK Abschlussarbeit Guidelines)
- [ ] 310. Create title page (Title, Author, University, Date, Supervisor)
- [ ] 311. Create Zusammenfassung/Abstract page (4-5 sentences)
- [ ] 312. Create Inhaltsverzeichnis (Table of Contents)
- [ ] 313. Create Glossar (Abbreviations: PU, YOLO, mAP, NPUAP, bbox, etc.)
- [ ] 314. Create Abbildungsverzeichnis (List of Figures)
- [ ] 315. Create Tabellenverzeichnis (List of Tables)

### 6.3 Chapter 1 — Einleitung (Introduction)
- [ ] 316. Write: Hintergrund der Arbeit (background on pressure ulcers)
- [ ] 317. Write: Beschreibung des Projektumfeldes (NaMeKI project context)
- [ ] 318. Write: Motivation des Themas (why automated PU detection matters)
- [ ] 319. Write: Zielsetzung (project goals)
- [ ] 320. Write: Aufgabenbeschreibung (task description)
- [ ] 321. Write: Abgrenzung (what is NOT part of this project)
- [ ] 322. Write: Kurze Übersicht der Kapitel (chapter overview)
- [ ] 323. Add proper citations for medical statistics (PU prevalence, costs)
- [ ] 324. Proofread Chapter 1

### 6.4 Chapter 2 — Systemüberblick (System Overview)
- [ ] 325. Write: Description of the detection system architecture
- [ ] 326. Create block diagram of the system (data → model → output)
- [ ] 327. Write: Interface descriptions (input: image, output: bbox + class)
- [ ] 328. Include system diagram as figure with proper caption and label
- [ ] 329. Proofread Chapter 2

### 6.5 Chapter 3 — Stand der Forschung (State of the Art)
- [ ] 330. Write: Literaturrecherche (literature review methodology)
- [ ] 331. Write: Review of existing PU detection approaches
- [ ] 332. Write: Review of YOLO-based wound detection papers
- [ ] 333. Write: Review of DenseNet/EfficientNet classification approaches
- [ ] 334. Write: Comparison table of existing solutions (from Phase 5)
- [ ] 335. Write: Identification of performance criteria and metrics
- [ ] 336. Write: Summary of open problems and gaps
- [ ] 337. Add all BibTeX references for cited papers (minimum 15-20 sources)
- [ ] 338. Verify all citations use proper [1], [2] numbering format
- [ ] 339. Proofread Chapter 3

### 6.6 Chapter 4 — Anforderungsanalyse (Requirements Analysis)
- [ ] 340. Write: Functional requirements (detect PU, classify stage, localize bbox)
- [ ] 341. Write: Non-functional requirements (real-time, accuracy thresholds)
- [ ] 342. Write: Use cases and scenarios (NaMeKI camera system)
- [ ] 343. Write: Methodology description (YOLO object detection approach)
- [ ] 344. Write: Requirements validation
- [ ] 345. Proofread Chapter 4

### 6.7 Chapter 5 — Konzeptentwicklung (Concept Development)
- [ ] 346. Write: Detailed problem analysis (PU staging challenges)
- [ ] 347. Write: Solution approach (YOLOv8 + augmented dataset)
- [ ] 348. Create detailed block diagram of data pipeline
- [ ] 349. Write: Design methodology
- [ ] 350. Write: Tools and design flow (Python, Ultralytics, Albumentations)
- [ ] 351. Proofread Chapter 5

### 6.8 Chapter 6 — Implementierung (Implementation)
- [ ] 352. Write: Data collection process description
- [ ] 353. Write: Data cleaning and deduplication methodology
- [ ] 354. Write: Label unification process
- [ ] 355. Write: Augmentation pipeline design and parameters
- [ ] 356. Include source code listing for key functions (with listings package)
- [ ] 357. Write: Dataset statistics and composition
- [ ] 358. Include figures: class distribution charts, sample images
- [ ] 359. Include table: dataset sources, sizes, and licenses
- [ ] 360. Proofread Chapter 6

### 6.9 Chapter 7 — Test (Testing)
- [ ] 361. Write: Test concept (how dataset quality was validated)
- [ ] 362. Write: Test cases (deduplication accuracy, augmentation integrity)
- [ ] 363. Write: YOLO dataset validation results
- [ ] 364. Write: Visual inspection results
- [ ] 365. Include test result tables and figures
- [ ] 366. Proofread Chapter 7

### 6.10 Chapter 8 — Benchmarking / Model Comparison
- [ ] 367. Write: Model comparison methodology
- [ ] 368. Include model comparison table (from Phase 5)
- [ ] 369. Include preliminary benchmark results (if available)
- [ ] 370. Write: Visualization of comparison results
- [ ] 371. Write: Justification for YOLOv8 selection
- [ ] 372. Proofread Chapter 8

### 6.11 Chapter 9 — Zusammenfassung (Conclusion)
- [ ] 373. Write: Summary of starting point
- [ ] 374. Write: Summary of solution approach
- [ ] 375. Write: Key contributions of this PPA work
- [ ] 376. Write: Assessment of results
- [ ] 377. Write: Challenges encountered and solutions
- [ ] 378. Write: Open points for Bachelor thesis
- [ ] 379. Write: Outlook — model training, evaluation, NaMeKI integration
- [ ] 380. Write: Recommendations for next phase
- [ ] 381. Proofread Chapter 9

### 6.12 Bibliography & Appendix
- [ ] 382. Verify all in-text citations have BibTeX entries
- [ ] 383. Verify all BibTeX entries have: Author, Title, Journal/Conference, Year
- [ ] 384. Add DOI links where available
- [ ] 385. Compile bibliography: run bibtex, check for warnings
- [ ] 386. Create appendix with full dataset catalog table
- [ ] 387. Create appendix with augmentation parameter table
- [ ] 388. Create appendix with class mapping table

### 6.13 Figures & Tables Quality
- [ ] 389. Ensure all figures are ≥600 dpi (prefer vector formats PDF/SVG)
- [ ] 390. Ensure every figure has a caption (\caption{}) and label (\label{fig:})
- [ ] 391. Ensure every table has a caption and label
- [ ] 392. Ensure every figure/table is referenced in the text
- [ ] 393. Ensure diagram axes are labeled and have units
- [ ] 394. Use consistent font sizes across all figures
- [ ] 395. Verify all images have proper source attribution

### 6.14 Final Report Quality Check
- [ ] 396. Verify Table of Contents generates correctly
- [ ] 397. Verify all cross-references resolve (no "??" in output)
- [ ] 398. Verify page numbers are correct
- [ ] 399. Run spell check on entire document
- [ ] 400. Check for consistent terminology throughout
- [ ] 401. Verify "roter Faden" (red thread) — chapters connect logically
- [ ] 402. Verify each chapter starts with motivation and ends with summary
- [ ] 403. Check: no marketing phrases or informal language (HAWK guideline)
- [ ] 404. Check: no unrelated images or humor (HAWK guideline)
- [ ] 405. Verify Selbständigkeitserklärung (independence declaration) is included
- [ ] 406. Generate final PDF and verify all pages render correctly
- [ ] 407. Have someone proofread the final document
- [ ] 408. Submit preliminary version to supervisor (2 weeks before deadline per HAWK rules)

---

## PHASE 7 — DOCUMENTATION & DELIVERABLES

### 7.1 Process Documentation
- [ ] 409. Write docs/data_sources.md — complete list of datasets with metadata
- [ ] 410. Write docs/augmentation_strategy.md — parameters, reasoning, results
- [ ] 411. Write docs/model_comparison.md — architecture comparison + recommendation
- [ ] 412. Write docs/reproducibility_guide.md — step-by-step to reproduce results
- [ ] 413. Document all script usage with examples
- [ ] 414. Document folder structure and data flow

### 7.2 Code Quality
- [ ] 415. Add docstrings to all Python functions
- [ ] 416. Add type hints to function signatures
- [ ] 417. Run all scripts end-to-end on a small test set
- [ ] 418. Verify no hardcoded paths — all configurable via arguments
- [ ] 419. Verify error handling in all scripts (graceful failures)
- [ ] 420. Clean up any debug/test code

### 7.3 Git Repository Cleanup
- [ ] 421. Review all committed files — no secrets, no large binaries
- [ ] 422. Verify .gitignore covers all data/model files
- [ ] 423. Write proper README.md for the repository
- [ ] 424. Add LICENSE file (verify compatibility with dataset licenses)
- [ ] 425. Tag release: `git tag v1.0-ppa`
- [ ] 426. Push all changes and tags to GitHub

### 7.4 PPA Presentation Preparation
- [ ] 427. Create presentation slides (10-13 slides per HAWK guideline)
- [ ] 428. Slide 1: Title slide (topic, author, date)
- [ ] 429. Slide 2: Introduction / Motivation
- [ ] 430. Slide 3: System Overview
- [ ] 431. Slide 4: Problem Definition
- [ ] 432. Slide 5-6: Methodology (data pipeline + model selection)
- [ ] 433. Slide 7-8: Results (dataset statistics, model comparison)
- [ ] 434. Slide 9: Challenges and Solutions
- [ ] 435. Slide 10: Conclusion and Outlook (Bachelor thesis plan)
- [ ] 436. Practice presentation (max 20 minutes talk + 25 min total per HAWK)
- [ ] 437. Get feedback on presentation from peers

---

## PROGRESS TRACKING

| Phase | Description | Tickets | Completed |
|-------|-------------|---------|-----------|
| 1 | Environment & Setup | 1-26 | 0/26 |
| 2 | Data Collection | 27-128 | 0/102 |
| 3 | Data Cleaning & Unification | 129-209 | 0/81 |
| 4 | Data Augmentation | 210-259 | 0/50 |
| 5 | Model Comparison | 260-295 | 0/36 |
| 6 | LaTeX Report | 296-408 | 0/113 |
| 7 | Documentation & Deliverables | 409-437 | 0/29 |
| **TOTAL** | | **437** | **0/437** |

---

*Generated: March 2026*
*Project: Decubitus Detection — NaMeKI*
*Student: Ashkan Sadri Ghamshi*
