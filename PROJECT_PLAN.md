# DECUBITUS DETECTION — COMPLETE PROJECT PLAN & CHECKLIST
### Student: Ashkan Sadri Ghamshi | HAWK University | NaMeKI Project
### Date: March 2026

> **Instructions:** Check each box `[x]` as you complete the task.
> This plan covers the full PPA (Praxisprojektarbeit) phase.
> **Strategy:** PU-only datasets (Tier 1 + Tier 2) with heavy augmentation (~50x) to reach 300K+ images.

---

## PHASE 1 — ENVIRONMENT & PROJECT SETUP (Tickets 1–30)

### 1.1 Development Environment
- [ ] 1. Install Python 3.10+ on local machine
- [ ] 2. Create Python virtual environment (`python -m venv venv`)
- [ ] 3. Activate virtual environment and verify Python version
- [ ] 4. Install core ML dependencies (`pip install ultralytics albumentations opencv-python-headless`)
- [ ] 5. Install data processing dependencies (`pip install numpy pillow tqdm imagehash`)
- [ ] 6. Install dataset download tools (`pip install roboflow python-dotenv`)
- [ ] 7. Install visualization libraries (`pip install matplotlib seaborn`)
- [ ] 8. Install experiment tracking tools (`pip install wandb tensorboard`)
- [ ] 9. Install Kaggle CLI (`pip install kaggle`)
- [ ] 10. Freeze dependencies to requirements.txt (`pip freeze > requirements.txt`)
- [ ] 11. Verify all imports work without errors (create test_imports.py)

### 1.2 Project Repository
- [ ] 12. Initialize Git repository with .gitignore
- [ ] 13. Create .env file with ROBOFLOW_API_KEY (excluded from git)
- [ ] 14. Create .env.example template for collaborators
- [ ] 15. Verify .env is properly gitignored — run `git status` to confirm
- [ ] 16. Create project folder structure (data/, scripts/, models/, docs/, results/)
- [ ] 17. Add .gitkeep to all empty directories
- [ ] 18. Create initial README.md with project overview
- [ ] 19. Push initial project structure to GitHub

### 1.3 Tool Setup
- [ ] 20. Create Roboflow free account at https://app.roboflow.com
- [ ] 21. Get Roboflow API key from Settings → API Key
- [ ] 22. Verify API key works: test download one small dataset
- [ ] 23. Create Kaggle account and configure API credentials (~/.kaggle/kaggle.json)
- [ ] 24. Install and configure LabelImg or CVAT for manual annotation
- [ ] 25. Install LaTeX distribution (TeX Live / MiKTeX) for report writing
- [ ] 26. Verify `pdflatex` or `xelatex` compiles a test document correctly
- [ ] 27. Set up BibTeX for bibliography management
- [ ] 28. Install LaTeX editor (Overleaf account or VS Code + LaTeX Workshop)
- [ ] 29. Create `scripts/test_imports.py` to validate all library installations
- [ ] 30. Run test_imports.py and fix any missing dependencies

---

## PHASE 2 — DATA COLLECTION (Tickets 31–145)

### 2.1 Tier 1: Roboflow Datasets (Bounding Box Annotations — YOLO-Ready)

#### Dataset #1: Pressure Ulcer by stage2 (~2,811 images)
- [ ] 31. Open Roboflow link and verify dataset is accessible
- [ ] 32. Check available versions and select the latest
- [ ] 33. Download dataset in YOLOv8 format to `data/raw/roboflow_stage2/`
- [ ] 34. Verify download integrity (check image count matches ~2,811)
- [ ] 35. Verify label files exist for each image (.txt files)
- [ ] 36. Check class names in data.yaml (should be stage1-stage4)
- [ ] 37. Spot-check 10 random images: open image + label, verify bbox alignment
- [ ] 38. Document dataset statistics (images per class) in spreadsheet

#### Dataset #2: pressure-ulcer by Project 1 (~2,013 images)
- [ ] 39. Open Roboflow link and verify dataset is accessible
- [ ] 40. Download dataset in YOLOv8 format to `data/raw/roboflow_project1/`
- [ ] 41. Verify download integrity (check image count matches ~2,013)
- [ ] 42. Verify label files exist for each image
- [ ] 43. Check class names and IDs match Dataset #1 mapping
- [ ] 44. Spot-check 10 random images for bbox quality
- [ ] 45. Document dataset statistics in spreadsheet
- [ ] 46. Note: flag for deduplication with Dataset #1

#### Dataset #3: Pressure Injury maskrcnn (~720 images, 6 classes)
- [ ] 47. Open Roboflow link and verify dataset is accessible
- [ ] 48. Download dataset in YOLOv8 format to `data/raw/roboflow_maskrcnn/`
- [ ] 49. Verify download integrity (check image count matches ~720)
- [ ] 50. Check class names: Stage1, Stage2, Stage3, Stage4, Unstageable, DTI
- [ ] 51. Note: this has instance segmentation polygons — verify bbox conversion
- [ ] 52. Spot-check 10 random images for annotation quality
- [ ] 53. Document dataset statistics (images per class, including Unstageable/DTI)
- [ ] 54. Create class mapping document (6-class to 4-class conversion rules)

#### Dataset #4: PressureUlcer by fid (~481 images, single class)
- [ ] 55. Open Roboflow link and verify dataset is accessible
- [ ] 56. Download dataset in YOLOv8 format to `data/raw/roboflow_fid/`
- [ ] 57. Verify download integrity (check image count matches ~481)
- [ ] 58. Note: single class "pressure ulcer" — decide on class mapping strategy
- [ ] 59. Spot-check 10 random images for bbox quality
- [ ] 60. Document: these images cannot be staged, only used for wound/no-wound

#### Dataset #5: Advances in Wound Care PI Dataset (~313 images)
- [ ] 61. Open Roboflow link and verify dataset is accessible
- [ ] 62. Download dataset in YOLOv8 format to `data/raw/roboflow_woundcare/`
- [ ] 63. Verify download integrity (check image count matches ~313)
- [ ] 64. Check class names and verify stage annotations
- [ ] 65. Spot-check 10 random images — note: clinician-annotated, highest quality
- [ ] 66. Document dataset statistics in spreadsheet

#### Dataset #6: PressureUlcer-SciProj-2024 (~148 images)
- [ ] 67. Open Roboflow link and verify dataset is accessible
- [ ] 68. Download dataset in YOLOv8 format to `data/raw/roboflow_sciproj/`
- [ ] 69. Verify download integrity (check image count matches ~148)
- [ ] 70. Check class names (body-MSC2 — needs mapping)
- [ ] 71. Spot-check 10 random images for annotation quality
- [ ] 72. Create class mapping from body-MSC2 to stage1-stage4

#### Dataset #7: Mobile PU Diagnostic (~313 images)
- [ ] 73. Open Roboflow link and verify dataset is accessible
- [ ] 74. Download dataset in YOLOv8 format to `data/raw/roboflow_mobile/`
- [ ] 75. Verify download integrity (check image count matches ~313)
- [ ] 76. Check class names ("injuries" — needs mapping)
- [ ] 77. Spot-check 10 random images — note: mobile-captured, varying quality
- [ ] 78. Document image quality characteristics (resolution, blur, lighting)

### 2.2 Tier 1: GitHub Datasets (Bounding Box Annotations)

#### Dataset #8: AZH Wound Localization (~1,010 images)
- [ ] 79. Clone repository: `git clone https://github.com/uwm-bigdata/wound_localization.git`
- [ ] 80. Move to `data/raw/azh_localization/`
- [ ] 81. Verify image count and folder structure
- [ ] 82. Check annotation format (LabelImg YOLO format)
- [ ] 83. Identify which images are Pressure Ulcer (vs Diabetic/Venous)
- [ ] 84. Count PU-specific images separately
- [ ] 85. Verify label class IDs match expected mapping
- [ ] 86. Spot-check 10 random PU images for bbox quality
- [ ] 87. Document: multi-wound dataset, need to filter PU class

### 2.3 Tier 2: Classification-Only Datasets (Need Manual BBox)

#### Dataset #9: PIID — Pressure Injury Images Dataset (1,091 images)
- [ ] 88. Clone repository: `git clone https://github.com/FU-MedicalAI/PIID.git`
- [ ] 89. Move to `data/raw/piid/`
- [ ] 90. Verify folder structure (Stage-1/, Stage-2/, Stage-3/, Stage-4/)
- [ ] 91. Count images per stage folder
- [ ] 92. Verify image dimensions (expected: 299x299px)
- [ ] 93. Note: classification only — no bounding boxes
- [ ] 94. Decide: manual annotation vs full-image bbox vs skip for detection

#### Dataset #10: Medetec Wound Database (~174 PU images)
- [ ] 95. Clone repository: `git clone https://github.com/mlaradji/deep-learning-for-wound-care.git`
- [ ] 96. Move to `data/raw/medetec/`
- [ ] 97. Navigate to pressure ulcer subfolder
- [ ] 98. Count pressure ulcer images specifically
- [ ] 99. Check image quality and dimensions
- [ ] 100. Note: classification only — no bounding boxes

#### Dataset #11: AZH Wound Classification (730 images)
- [ ] 101. Clone repository
- [ ] 102. Move to `data/raw/azh_classification/`
- [ ] 103. Identify "Pressure" category images
- [ ] 104. Count PU-specific images
- [ ] 105. Check image format (cropped wound ROI)
- [ ] 106. Note: classification only — needs bbox annotation for YOLO

#### Dataset #12: Figshare Pressure Ulcer Images (~40 files)
- [ ] 107. Download from Figshare link to `data/raw/figshare/`
- [ ] 108. Verify LabelMe annotation format
- [ ] 109. Check if annotations can be converted to YOLO bbox
- [ ] 110. Small dataset — evaluate if worth the conversion effort

#### Dataset #13: Kaggle Pressure Ulcer Stages (~1,091 images)
- [ ] 111. Download via Kaggle CLI or browser to `data/raw/kaggle_pu_stages/`
- [ ] 112. Verify folder structure and image count
- [ ] 113. Compare with PIID dataset for overlap (likely a mirror)
- [ ] 114. If duplicate of PIID, mark as skip and document

### 2.4 Download Automation Scripts
- [ ] 115. Write `scripts/download_roboflow.py` — automated download of all 7 Roboflow datasets
- [ ] 116. Test download_roboflow.py: run with API key from .env
- [ ] 117. Verify all 7 datasets downloaded correctly
- [ ] 118. Write `scripts/download_github.py` — automated clone of all GitHub repos
- [ ] 119. Test download_github.py: run and verify all repos cloned
- [ ] 120. Write `scripts/download_kaggle.py` — Kaggle dataset download
- [ ] 121. Test download_kaggle.py with Kaggle credentials

### 2.5 Data Collection Verification
- [ ] 122. Create master spreadsheet: `docs/dataset_inventory.csv`
- [ ] 123. Record for each dataset: name, source, URL, image count, classes, annotation type, license
- [ ] 124. Calculate total raw image count across all datasets
- [ ] 125. Verify all Tier 1 datasets have YOLO-format labels
- [ ] 126. List datasets requiring format conversion
- [ ] 127. List datasets requiring manual annotation
- [ ] 128. Take screenshot of each dataset's folder contents as evidence
- [ ] 129. Back up all raw data to external storage (USB/external HDD)
- [ ] 130. Commit data collection documentation to git
- [ ] 131. Update README.md with actual download results vs expected

### 2.6 Download Troubleshooting
- [ ] 132. If Roboflow rate limit hit: wait and retry, or download manually via browser
- [ ] 133. If GitHub clone fails: check URL, try HTTPS vs SSH
- [ ] 134. If Kaggle download fails: verify API token, try browser download
- [ ] 135. If Figshare access denied: try alternative mirror or skip dataset
- [ ] 136. Document any datasets that could not be downloaded and why
- [ ] 137. Verify all dataset licenses allow academic use
- [ ] 138. Create `docs/data_sources.md` with attribution for each dataset
- [ ] 139. Record CC BY 4.0 compliance requirements
- [ ] 140. Check if any dataset requires citing a specific paper
- [ ] 141. Add required citations to references.bib
- [ ] 142. Verify total Tier 1 images: expected ~7,809 before dedup
- [ ] 143. Verify total Tier 2 images: expected ~3,046 before dedup
- [ ] 144. Calculate total raw PU images available
- [ ] 145. Commit and push Phase 2 documentation

---

## PHASE 3 — DATA CLEANING & UNIFICATION (Tickets 146–235)

### 3.1 Label Standardization
- [ ] 146. Create master class mapping document: `docs/class_mapping.md`
- [ ] 147. Define unified class IDs: 0=stage1, 1=stage2, 2=stage3, 3=stage4
- [ ] 148. Document mapping rules for each source dataset
- [ ] 149. Write `scripts/label_mapping.py` — remap class IDs across datasets
- [ ] 150. Remap Dataset #1 (roboflow_stage2) labels to unified IDs
- [ ] 151. Remap Dataset #2 (roboflow_project1) labels to unified IDs
- [ ] 152. Remap Dataset #3 (roboflow_maskrcnn) — handle 6→4 class reduction
- [ ] 153. Decide what to do with Unstageable and DTI classes (drop or keep as extra)
- [ ] 154. Remap Dataset #4 (roboflow_fid) — single class to unified ID
- [ ] 155. Remap Dataset #5 (roboflow_woundcare) labels to unified IDs
- [ ] 156. Remap Dataset #6 (roboflow_sciproj) — body-MSC2 mapping
- [ ] 157. Remap Dataset #7 (roboflow_mobile) — "injuries" mapping
- [ ] 158. Remap Dataset #8 (azh_localization) — filter PU class only, drop others
- [ ] 159. Verify all remapped label files have correct YOLO format
- [ ] 160. Spot-check 5 images per dataset after remapping — open image + label overlay

### 3.2 Annotation Format Conversion
- [ ] 161. Finalize `scripts/convert_masks_to_bbox.py`
- [ ] 162. Convert Dataset #3 segmentation polygons to YOLO bbox
- [ ] 163. Convert any LabelMe annotations (Dataset #12) to YOLO format
- [ ] 164. Write `scripts/validate_labels.py` — check YOLO label format correctness
- [ ] 165. Validate: class_id is integer ≥ 0
- [ ] 166. Validate: center_x, center_y, width, height all in [0.0, 1.0]
- [ ] 167. Validate: exactly 5 values per line
- [ ] 168. Validate: no empty lines or malformed entries
- [ ] 169. Run validate_labels.py on ALL label files
- [ ] 170. Fix any validation errors found
- [ ] 171. Re-run validation to confirm all clean

### 3.3 Image Quality Control
- [ ] 172. Write `scripts/image_quality_check.py`
- [ ] 173. Scan all images: can OpenCV read them? Log corrupt files
- [ ] 174. Remove images that cannot be opened (log filenames)
- [ ] 175. Check image dimensions: flag images < 50×50 px
- [ ] 176. Check image dimensions: flag images > 4000×4000 px
- [ ] 177. Resize oversized images to max 2048px (maintain aspect ratio)
- [ ] 178. Check color mode: flag grayscale images (should be RGB)
- [ ] 179. Convert grayscale to RGB if needed (cv2.cvtColor)
- [ ] 180. Find images with no corresponding label file → decide: remove or create empty label
- [ ] 181. Find label files with no corresponding image → remove orphan labels
- [ ] 182. Find images with empty label files (0 bytes) → decide: keep as negative or remove
- [ ] 183. Check for truncated/partial images (file size < 1KB)
- [ ] 184. Log all quality issues in `results/quality_report.txt`
- [ ] 185. Fix all fixable issues, document unfixable ones

### 3.4 Deduplication
- [ ] 186. Finalize `scripts/deduplicate.py` with perceptual hashing
- [ ] 187. Configure hash_size=16 and threshold=5 for near-duplicate detection
- [ ] 188. Run deduplication on Dataset #1 + #2 first (highest overlap expected)
- [ ] 189. Review 20 random duplicate pairs visually — are they truly duplicates?
- [ ] 190. Adjust threshold if too aggressive (removing different images) or too loose
- [ ] 191. Run deduplication across ALL Tier 1 datasets combined
- [ ] 192. Log results: how many duplicates found per dataset pair
- [ ] 193. Remove confirmed duplicates — keep the one with better annotation quality
- [ ] 194. Run deduplication between Tier 1 and Tier 2 datasets
- [ ] 195. Verify PIID vs Kaggle PU Stages overlap (expected: same dataset)
- [ ] 196. Generate final dedup report: `results/dedup_report.txt`
- [ ] 197. Record: total images before dedup, total after dedup, removal rate per dataset
- [ ] 198. Final unique image count after deduplication

### 3.5 Manual Annotation (Tier 2 Datasets)
- [ ] 199. Decide which Tier 2 datasets are worth annotating manually
- [ ] 200. If annotating PIID: set up LabelImg or Roboflow Annotate (free tier)
- [ ] 201. Create annotation guidelines: `docs/annotation_guidelines.md`
- [ ] 202. Define rules: what constitutes each stage, bbox should cover entire wound area
- [ ] 203. Annotate PIID Stage-1 images with bounding boxes (~273 images)
- [ ] 204. Annotate PIID Stage-2 images with bounding boxes (~273 images)
- [ ] 205. Annotate PIID Stage-3 images with bounding boxes (~273 images)
- [ ] 206. Annotate PIID Stage-4 images with bounding boxes (~273 images)
- [ ] 207. Export all PIID annotations in YOLO format
- [ ] 208. Quality-check 20 random PIID annotations (overlay on image)
- [ ] 209. If annotating Medetec PU images (~174): annotate and export
- [ ] 210. If annotating AZH Classification PU images: annotate and export

### 3.6 Dataset Merging
- [ ] 211. Write `scripts/merge_datasets.py`
- [ ] 212. Copy all deduplicated Tier 1 images to `data/unified/images/`
- [ ] 213. Copy corresponding labels to `data/unified/labels/`
- [ ] 214. Add annotated Tier 2 images to unified dataset
- [ ] 215. Rename files with source prefix to avoid collisions (e.g., `rf1_00001.jpg`)
- [ ] 216. Update label filenames to match renamed images
- [ ] 217. Verify image↔label pairing: count images == count labels
- [ ] 218. Count total unified dataset size and log

### 3.7 Dataset Splitting
- [ ] 219. Write `scripts/split_dataset.py` with stratified splitting
- [ ] 220. Analyze class distribution in unified dataset before split
- [ ] 221. Perform stratified split: 70% train / 20% val / 10% test
- [ ] 222. Ensure no patient-level data leakage (same wound not in train AND test)
- [ ] 223. Move images to `data/unified/images/{train,val,test}/`
- [ ] 224. Move labels to `data/unified/labels/{train,val,test}/`
- [ ] 225. Verify class balance in train split — log distribution
- [ ] 226. Verify class balance in val split — log distribution
- [ ] 227. Verify class balance in test split — log distribution
- [ ] 228. Generate split statistics report: `results/split_report.txt`
- [ ] 229. Update `data/dataset.yaml` with correct paths and class names

### 3.8 Data Cleaning Verification
- [ ] 230. Run final validation: every image has a label, every label has an image
- [ ] 231. Run final validation: all labels pass validate_labels.py
- [ ] 232. Run final validation: all class IDs in range [0, 3]
- [ ] 233. Run YOLOv8 dataset check: `yolo detect val data=dataset.yaml model=yolov8n.pt`
- [ ] 234. Fix any errors from YOLO validation
- [ ] 235. Document total unified dataset: images, classes, distribution per split

---

## PHASE 4 — EXPLORATORY DATA ANALYSIS (Tickets 236–260)

### 4.1 Dataset Statistics
- [ ] 236. Write `scripts/eda.py` — comprehensive dataset analysis script
- [ ] 237. Count total images per class (stage1, stage2, stage3, stage4)
- [ ] 238. Generate bar chart: class distribution (save to `results/class_distribution.png`)
- [ ] 239. Calculate class imbalance ratio (majority class / minority class)
- [ ] 240. Calculate mean and std of image dimensions (width, height)
- [ ] 241. Generate histogram: image width distribution
- [ ] 242. Generate histogram: image height distribution
- [ ] 243. Generate histogram: image aspect ratio distribution

### 4.2 Annotation Statistics
- [ ] 244. Calculate average number of bboxes per image
- [ ] 245. Generate histogram: bbox count per image
- [ ] 246. Calculate average bbox area relative to image area
- [ ] 247. Generate scatter plot: bbox width vs bbox height
- [ ] 248. Generate heatmap: bbox center positions across all images
- [ ] 249. Identify and log outlier bboxes (extremely small or large)

### 4.3 Sample Visualization
- [ ] 250. Write `scripts/visualize_samples.py` — draw bboxes on images
- [ ] 251. Generate sample grid: 5 random images per class with bboxes drawn
- [ ] 252. Save sample grids to `results/samples_stage1.png` ... `results/samples_stage4.png`
- [ ] 253. Visually verify annotations look correct in sample grids

### 4.4 Class Imbalance Strategy
- [ ] 254. If imbalance ratio > 3:1, plan oversampling for minority classes
- [ ] 255. Calculate per-class augmentation multiplier to balance classes
- [ ] 256. Document class imbalance findings in `docs/eda_report.md`
- [ ] 257. Decide: use class weights in training OR oversample minority classes in augmentation
- [ ] 258. If oversampling: update augment_dataset.py to apply more copies to minority classes
- [ ] 259. Generate final EDA report with all charts and statistics
- [ ] 260. Commit EDA scripts and results

---

## PHASE 5 — DATA AUGMENTATION (Tickets 261–310)

### 5.1 Augmentation Pipeline Development
- [ ] 261. Review `scripts/augment_dataset.py` for correctness
- [ ] 262. Verify albumentations BboxParams format='yolo' is set
- [ ] 263. Verify min_visibility=0.3 threshold is appropriate
- [ ] 264. Test augmentation pipeline on 5 sample images
- [ ] 265. Visually inspect augmented images + bboxes for correctness
- [ ] 266. Verify augmented bbox values stay within [0, 1]
- [ ] 267. Test edge case: image with multiple bboxes
- [ ] 268. Test edge case: image with small bboxes near image border
- [ ] 269. Test edge case: image with no bboxes (background/negative)

### 5.2 Augmentation Strategy Tuning
- [ ] 270. Test geometric transforms individually: HorizontalFlip — save sample
- [ ] 271. Test geometric transforms individually: VerticalFlip — save sample
- [ ] 272. Test geometric transforms individually: RandomRotate90 — save sample
- [ ] 273. Test geometric transforms individually: ShiftScaleRotate — save sample
- [ ] 274. Test geometric transforms individually: Perspective — save sample
- [ ] 275. Test color transforms individually: RandomBrightnessContrast — save sample
- [ ] 276. Test color transforms individually: HueSaturationValue — save sample
- [ ] 277. Test color transforms individually: CLAHE — save sample
- [ ] 278. Test noise transforms individually: GaussianBlur — save sample
- [ ] 279. Test noise transforms individually: GaussNoise — save sample
- [ ] 280. Test occlusion transforms individually: CoarseDropout — save sample
- [ ] 281. Verify combined pipeline produces realistic wound images
- [ ] 282. Adjust probabilities if any augmentation looks unrealistic
- [ ] 283. Document final augmentation parameters and reasoning

### 5.3 Multi-Round Augmentation Strategy (to reach 300K)
- [ ] 284. Calculate exact copies needed: 300,000 / (unified_train_count) = N copies
- [ ] 285. If N > 30: split into multiple augmentation rounds with different seeds
- [ ] 286. Round 1: geometric-heavy augmentation (copies = N/3)
- [ ] 287. Round 2: color/lighting-heavy augmentation (copies = N/3)
- [ ] 288. Round 3: mixed augmentation with noise+occlusion (copies = N/3)
- [ ] 289. This ensures maximum diversity despite high copy count
- [ ] 290. Update augment_dataset.py to support `--seed` parameter for different rounds
- [ ] 291. Update augment_dataset.py to support `--strategy` parameter (geometric/color/mixed)

### 5.4 Augmentation Execution
- [ ] 292. Run augmentation Round 1: geometric-heavy
- [ ] 293. Verify Round 1 output count
- [ ] 294. Run augmentation Round 2: color-heavy
- [ ] 295. Verify Round 2 output count
- [ ] 296. Run augmentation Round 3: mixed
- [ ] 297. Verify Round 3 output count
- [ ] 298. Merge all rounds into `data/augmented/images/train/`
- [ ] 299. Verify total augmented train count ≥ 300,000
- [ ] 300. Copy val and test sets unchanged (NOT augmented)
- [ ] 301. Verify val/test image counts match pre-augmentation counts

### 5.5 Augmentation Quality Assurance
- [ ] 302. Write `scripts/augmentation_qa.py` — visual grid of original + augmented pairs
- [ ] 303. Generate QA grids for 30 random images (10 per augmentation round)
- [ ] 304. Review QA grids: bboxes align correctly?
- [ ] 305. Review QA grids: images look realistic?
- [ ] 306. Review QA grids: no artifacts or corruption?
- [ ] 307. Check for empty label files in augmented set
- [ ] 308. Run YOLO validation on augmented dataset: `yolo detect val data=dataset.yaml model=yolov8n.pt`
- [ ] 309. Fix any issues found during QA

### 5.6 Final Dataset Verification
- [ ] 310. Update `data/dataset.yaml` to point to augmented directory
- [ ] 311. Verify YOLO can iterate through entire augmented dataset without errors
- [ ] 312. Generate final dataset statistics: total images, per-class distribution
- [ ] 313. Check class distribution is balanced after class-aware augmentation
- [ ] 314. Document: `docs/augmentation_strategy.md` — parameters, rounds, results
- [ ] 315. Record: total images (train/val/test), augmentation ratio, time taken
- [ ] 316. Back up augmented dataset to external storage
- [ ] 317. Commit augmentation scripts and documentation
- [ ] 318. Push Phase 5 results to GitHub

---

## PHASE 6 — MODEL COMPARISON & SELECTION (Tickets 319–365)

### 6.1 Literature Review for Model Comparison
- [ ] 319. Search Google Scholar: "YOLOv8 pressure ulcer detection"
- [ ] 320. Search Google Scholar: "wound detection deep learning"
- [ ] 321. Search Google Scholar: "pressure injury classification CNN"
- [ ] 322. Find and read Lau et al. 2024 paper (mAP@50=90.8% with YOLOv8)
- [ ] 323. Find DenseNet wound classification papers
- [ ] 324. Find EfficientNet wound classification papers
- [ ] 325. Find Faster R-CNN wound detection papers
- [ ] 326. Read at least 15 relevant papers in detail
- [ ] 327. Compile comparison table from literature: model, dataset size, metrics, year
- [ ] 328. Create BibTeX entries for all referenced papers in references.bib
- [ ] 329. Document key findings per model architecture in notes

### 6.2 Model Architecture Analysis
- [ ] 330. Document YOLOv8 architecture: backbone (CSPDarknet), neck (PANet), head
- [ ] 331. Compare YOLOv8 variants: nano (3.2M params), small (11.2M), medium (25.9M)
- [ ] 332. Document DenseNet-121 architecture: 121 layers, ~8M params, dense blocks
- [ ] 333. Document EfficientNet-B0/B2 architecture: compound scaling method
- [ ] 334. Document Faster R-CNN architecture: RPN + classification head
- [ ] 335. Create comparison table: params, FLOPs, input size, output type, speed

### 6.3 Model Comparison Criteria
- [ ] 336. Define primary metrics: mAP@50, mAP@50:95 (detection models)
- [ ] 337. Define primary metrics: Accuracy, Precision, Recall, F1 (all models)
- [ ] 338. Define secondary metrics: inference speed (FPS), model size (MB)
- [ ] 339. Define practical criteria: real-time capability for NaMeKI
- [ ] 340. Define practical criteria: deployment format (ONNX, TensorRT compatibility)
- [ ] 341. Create weighted scoring matrix for final model selection
- [ ] 342. Document comparison methodology in `docs/model_comparison.md`

### 6.4 Preliminary Model Testing (Quick Benchmark)
- [ ] 343. Extract small subset: 500 train + 100 val images
- [ ] 344. Run YOLOv8n training on subset (20 epochs)
- [ ] 345. Run YOLOv8s training on subset (20 epochs)
- [ ] 346. Run YOLOv8m training on subset (20 epochs)
- [ ] 347. Record: training time, final loss, mAP@50 per variant
- [ ] 348. Compare quick-benchmark results with literature values
- [ ] 349. Identify any training issues (divergence, NaN loss, etc.)

### 6.5 Model Selection Report
- [ ] 350. Write `docs/model_comparison.md` with all findings
- [ ] 351. Section 1: Architecture overview with diagrams (cited from papers)
- [ ] 352. Section 2: Literature-based performance comparison table
- [ ] 353. Section 3: Preliminary benchmark results from Phase 6.4
- [ ] 354. Section 4: Scoring matrix results
- [ ] 355. Section 5: Final recommendation — justify YOLOv8s or YOLOv8m
- [ ] 356. Explain why YOLO preferred over classification-only models for this task
- [ ] 357. Explain NaMeKI real-time requirements and YOLO compatibility
- [ ] 358. Explain dataset size (300K) and how YOLO handles large datasets
- [ ] 359. Review and proofread model comparison document
- [ ] 360. Create summary table for LaTeX report
- [ ] 361. Create comparison bar chart for LaTeX report
- [ ] 362. Save all benchmark plots to `results/`
- [ ] 363. Commit model comparison documentation
- [ ] 364. Push Phase 6 results to GitHub
- [ ] 365. Peer-review model comparison document (ask supervisor or colleague)

---

## PHASE 7 — LaTeX REPORT (HAWK Standard) (Tickets 366–490)

### 7.1 LaTeX Project Setup
- [ ] 366. Create `docs/report/` directory for LaTeX project
- [ ] 367. Create `docs/report/main.tex` with document class (article or report)
- [ ] 368. Set up preamble: geometry (A4, margins), fonts, UTF-8 encoding
- [ ] 369. Configure language settings (German or English — ask supervisor)
- [ ] 370. Set up chapter-relative numbering: `\numberwithin{equation}{section}`
- [ ] 371. Set up chapter-relative numbering: `\numberwithin{figure}{section}`
- [ ] 372. Set up chapter-relative numbering: `\numberwithin{table}{section}`
- [ ] 373. Configure BibTeX with style file (german.bst or english.bst)
- [ ] 374. Create `docs/report/references.bib`
- [ ] 375. Set up `listings` package for Python source code display
- [ ] 376. Configure `lstset` for Python syntax highlighting (colors, font, line numbers)
- [ ] 377. Set up `graphicx` package for figure inclusion
- [ ] 378. Set up `hyperref` for clickable cross-references and URLs
- [ ] 379. Set up `glossaries` or manual glossary table for abbreviations
- [ ] 380. Set up `booktabs` for professional tables
- [ ] 381. Create `docs/report/figures/` directory for all images
- [ ] 382. Test compile: `pdflatex main.tex` → PDF without errors
- [ ] 383. Test compile with BibTeX: `pdflatex → bibtex → pdflatex → pdflatex`
- [ ] 384. Set up Makefile or latexmk for one-command compilation

### 7.2 Report Front Matter (per HAWK Guidelines)
- [ ] 385. Create title page: Title, Author, Matrikelnummer, University, Faculty, Date
- [ ] 386. Add supervisor name and second examiner (if applicable)
- [ ] 387. Create Selbständigkeitserklärung (independence declaration) page
- [ ] 388. Create Zusammenfassung / Abstract (4-5 sentences — write last)
- [ ] 389. Generate Inhaltsverzeichnis (Table of Contents): `\tableofcontents`
- [ ] 390. Create Glossar / Abkürzungsverzeichnis (abbreviation table)
- [ ] 391. Add abbreviations: PU, YOLO, mAP, CNN, NPUAP, EPUAP, bbox, ROI, FPS, etc.
- [ ] 392. Generate Abbildungsverzeichnis: `\listoffigures`
- [ ] 393. Generate Tabellenverzeichnis: `\listoftables`

### 7.3 Chapter 1 — Einleitung (Introduction)
- [ ] 394. Write: Hintergrund — pressure ulcers affect X% of hospital patients [cite]
- [ ] 395. Write: Cost of PU treatment in healthcare systems [cite statistics]
- [ ] 396. Write: Projektumfeld — NaMeKI project at HAWK, camera-based monitoring
- [ ] 397. Write: Motivation — why automated detection saves lives and costs
- [ ] 398. Write: Zielsetzung — build YOLOv8-based PU detection system
- [ ] 399. Write: Aufgabenbeschreibung — collect data, clean, augment, compare models
- [ ] 400. Write: Abgrenzung — PPA covers data+model selection, NOT full training
- [ ] 401. Write: Kapitelübersicht — brief overview of each chapter
- [ ] 402. Add ≥3 citations for medical statistics
- [ ] 403. Proofread Chapter 1 — check factual accuracy

### 7.4 Chapter 2 — Systemüberblick (System Overview)
- [ ] 404. Write: Technical description of detection pipeline
- [ ] 405. Create system block diagram (draw.io or TikZ): Camera → Image → YOLO → BBox + Stage
- [ ] 406. Export diagram as PDF/SVG (vector, ≥600 dpi)
- [ ] 407. Include diagram with `\begin{figure}...\caption{...}\label{fig:system}...\end{figure}`
- [ ] 408. Write: Interface description — input (RGB image), output (bbox + class label + confidence)
- [ ] 409. Proofread Chapter 2

### 7.5 Chapter 3 — Stand der Forschung (State of the Art)
- [ ] 410. Write: Literature review methodology (which databases searched, keywords)
- [ ] 411. Write: Section on NPUAP/EPUAP pressure ulcer staging system [cite]
- [ ] 412. Write: Review of traditional image processing approaches for wound detection
- [ ] 413. Write: Review of CNN-based wound classification (DenseNet, EfficientNet)
- [ ] 414. Write: Review of object detection approaches (YOLO, Faster R-CNN)
- [ ] 415. Write: Review of YOLOv8 specifically — architecture improvements over v5/v7
- [ ] 416. Write: Review of data augmentation strategies for medical imaging
- [ ] 417. Create comparison table: existing solutions (model, dataset, metrics, year)
- [ ] 418. Write: Performance criteria and metrics (mAP, IoU, precision, recall)
- [ ] 419. Write: Summary of gaps — no large-scale unified PU detection dataset exists
- [ ] 420. Add ≥15 BibTeX references for this chapter
- [ ] 421. Verify all citations use [1], [2] numbering (cite package)
- [ ] 422. Proofread Chapter 3

### 7.6 Chapter 4 — Anforderungsanalyse (Requirements Analysis)
- [ ] 423. Write: Functional requirements (FR1: detect PU, FR2: classify stage, FR3: localize)
- [ ] 424. Write: Non-functional requirements (NFR1: real-time ≥15 FPS, NFR2: mAP≥80%)
- [ ] 425. Write: Use case description — nurse uses NaMeKI system for patient monitoring
- [ ] 426. Write: Methodology — YOLO single-stage detection approach
- [ ] 427. Write: Requirements validation criteria
- [ ] 428. Proofread Chapter 4

### 7.7 Chapter 5 — Konzeptentwicklung (Concept Development)
- [ ] 429. Write: Problem analysis — challenges of PU detection (visual similarity, lighting)
- [ ] 430. Write: Solution approach — multi-source dataset + YOLOv8
- [ ] 431. Create data pipeline block diagram: Collection → Cleaning → Dedup → Augmentation → Training
- [ ] 432. Write: Design methodology — iterative data quality improvement
- [ ] 433. Write: Tools and design flow (Python, Ultralytics, Albumentations, OpenCV)
- [ ] 434. Proofread Chapter 5

### 7.8 Chapter 6 — Implementierung (Implementation)
- [ ] 435. Write: Data collection — sources, download process, quantities
- [ ] 436. Write: Data cleaning — deduplication algorithm, quality checks
- [ ] 437. Write: Label unification — class mapping, format conversion
- [ ] 438. Write: Augmentation pipeline — transforms, parameters, multi-round strategy
- [ ] 439. Include Python code listing: augmentation pipeline (using `lstlisting`)
- [ ] 440. Include Python code listing: deduplication function
- [ ] 441. Create figure: class distribution before and after augmentation
- [ ] 442. Create figure: sample augmented images grid
- [ ] 443. Create table: dataset sources with sizes and licenses
- [ ] 444. Create table: augmentation parameters
- [ ] 445. Write: Final dataset composition — total images, split ratios
- [ ] 446. Proofread Chapter 6

### 7.9 Chapter 7 — Test (Testing & Validation)
- [ ] 447. Write: Test concept — how was dataset quality validated?
- [ ] 448. Write: Test case 1 — deduplication accuracy (manual verification of 50 pairs)
- [ ] 449. Write: Test case 2 — augmentation integrity (bbox alignment verification)
- [ ] 450. Write: Test case 3 — YOLO dataset validation pass/fail
- [ ] 451. Write: Test case 4 — label format validation (all files pass validate_labels.py)
- [ ] 452. Include table: test results summary (test case, expected, actual, pass/fail)
- [ ] 453. Include figure: QA grid showing correct augmented bbox overlay
- [ ] 454. Proofread Chapter 7

### 7.10 Chapter 8 — Benchmarking / Model Comparison
- [ ] 455. Write: Comparison methodology — criteria, scoring system
- [ ] 456. Include table: model architecture comparison (params, FLOPs, speed)
- [ ] 457. Include table: literature performance comparison
- [ ] 458. Include table: preliminary benchmark results (if available)
- [ ] 459. Create figure: bar chart comparing model metrics
- [ ] 460. Write: Analysis — why YOLOv8 is the recommended model
- [ ] 461. Write: Discussion of detection vs classification trade-offs
- [ ] 462. Proofread Chapter 8

### 7.11 Chapter 9 — Zusammenfassung und Ausblick (Conclusion & Outlook)
- [ ] 463. Write: Summary of starting point (fragmented datasets, no unified resource)
- [ ] 464. Write: Summary of approach (multi-source collection, cleaning, 300K augmentation)
- [ ] 465. Write: Key contributions (unified dataset, augmentation pipeline, model comparison)
- [ ] 466. Write: Assessment of results — dataset quality, coverage, limitations
- [ ] 467. Write: Challenges encountered and how they were solved
- [ ] 468. Write: Open points for Bachelor thesis (full training, evaluation, deployment)
- [ ] 469. Write: Outlook — model training plan, NaMeKI integration, ONNX export
- [ ] 470. Write: Recommendations for next phase
- [ ] 471. Proofread Chapter 9

### 7.12 Bibliography & Appendix
- [ ] 472. Verify ALL in-text citations have corresponding BibTeX entries
- [ ] 473. Verify each BibTeX entry has: Author, Title, Year, Journal/Conference
- [ ] 474. Add DOI links where available
- [ ] 475. Compile: run `bibtex main` and fix all warnings
- [ ] 476. Create Appendix A: Full dataset catalog table (all 26 datasets)
- [ ] 477. Create Appendix B: Augmentation parameter table
- [ ] 478. Create Appendix C: Complete class mapping table
- [ ] 479. Create Appendix D: EDA charts and full statistics

### 7.13 Figures & Tables Quality (HAWK Guidelines)
- [ ] 480. All figures ≥600 dpi — prefer vector formats (PDF/SVG)
- [ ] 481. Every figure has `\caption{}` and `\label{fig:...}`
- [ ] 482. Every table has `\caption{}` and `\label{tab:...}`
- [ ] 483. Every figure and table is referenced in text (`Figure~\ref{fig:...}`)
- [ ] 484. Diagram axes are labeled with units
- [ ] 485. Consistent font sizes across all figures
- [ ] 486. All external images have source attribution in caption

### 7.14 Final Report Quality Check (HAWK Compliance)
- [ ] 487. Table of Contents generates correctly — no missing chapters
- [ ] 488. All cross-references resolve (no "??" in output PDF)
- [ ] 489. Page numbers are correct and sequential
- [ ] 490. Run spell check on entire document (aspell or IDE spellcheck)
- [ ] 491. Check consistent terminology (same terms throughout)
- [ ] 492. Verify "roter Faden" — each chapter motivates the next
- [ ] 493. Verify each chapter begins with motivation, ends with summary + transition
- [ ] 494. Check: NO marketing phrases or informal language (HAWK rule)
- [ ] 495. Check: NO unrelated images or humor (HAWK rule)
- [ ] 496. Check: sachlich und neutral (factual and neutral tone)
- [ ] 497. Selbständigkeitserklärung is signed and included
- [ ] 498. Generate final PDF — verify all pages render correctly
- [ ] 499. Have someone proofread the final document
- [ ] 500. Submit preliminary version to supervisor (≥2 weeks before deadline, HAWK rule)

---

## PHASE 8 — DOCUMENTATION & DELIVERABLES (Tickets 501–540)

### 8.1 Process Documentation
- [ ] 501. Write `docs/data_sources.md` — complete dataset list with metadata and licenses
- [ ] 502. Write `docs/augmentation_strategy.md` — parameters, multi-round approach, results
- [ ] 503. Write `docs/model_comparison.md` — architecture comparison + recommendation
- [ ] 504. Write `docs/eda_report.md` — dataset statistics, charts, imbalance analysis
- [ ] 505. Write `docs/reproducibility_guide.md` — step-by-step to reproduce all results
- [ ] 506. Document all script usage with CLI examples in each script's docstring
- [ ] 507. Document folder structure and data flow in README.md

### 8.2 Code Quality
- [ ] 508. Add docstrings to all Python functions in all scripts
- [ ] 509. Add type hints to function signatures
- [ ] 510. Run all scripts end-to-end on a small test set (10 images)
- [ ] 511. Verify no hardcoded paths — all configurable via CLI arguments
- [ ] 512. Verify error handling: scripts fail gracefully with helpful messages
- [ ] 513. Remove any debug/test code, print statements
- [ ] 514. Add `--help` descriptions to all argparse arguments
- [ ] 515. Write `scripts/run_pipeline.sh` — single script to run entire pipeline end-to-end

### 8.3 Git Repository Cleanup
- [ ] 516. Review all committed files — NO secrets, NO large binaries
- [ ] 517. Verify .gitignore covers: .env, data/raw/, data/unified/, data/augmented/, *.pt
- [ ] 518. Update README.md: project overview, setup instructions, usage guide
- [ ] 519. Add LICENSE file — verify compatibility with CC BY 4.0 dataset licenses
- [ ] 520. Verify document_pdf.pdf should be in repo (HAWK guidelines doc)
- [ ] 521. Create .github/ directory with issue templates (optional)
- [ ] 522. Tag release: `git tag v1.0-ppa`
- [ ] 523. Push all changes and tags to GitHub
- [ ] 524. Verify GitHub repo looks clean and professional

### 8.4 PPA Presentation Preparation (HAWK Guidelines: max 25 min total)
- [ ] 525. Create presentation slides (10-13 slides, HAWK guideline)
- [ ] 526. Slide 1: Title (topic, author, date, supervisor)
- [ ] 527. Slide 2: Einleitung/Motivation — why PU detection matters
- [ ] 528. Slide 3: Systemüberblick — system block diagram
- [ ] 529. Slide 4: Problemdefinition — fragmented datasets, no unified resource
- [ ] 530. Slide 5: Methodik — data pipeline (collection → cleaning → augmentation)
- [ ] 531. Slide 6: Methodik — model selection approach
- [ ] 532. Slide 7: Ergebnisse — dataset statistics (300K images, 4 classes)
- [ ] 533. Slide 8: Ergebnisse — model comparison table
- [ ] 534. Slide 9: Ergebnisse — EDA charts (class distribution, samples)
- [ ] 535. Slide 10: Zusammenfassung — key contributions, challenges, outlook
- [ ] 536. Ensure readable font sizes — no overloaded slides
- [ ] 537. Ensure correct Rechtschreibung (spelling) on all slides
- [ ] 538. Practice presentation: target ~20 minutes talk time
- [ ] 539. Get feedback from peers or supervisor
- [ ] 540. Prepare for Q&A: anticipate questions about augmentation, model choice

---

## PROGRESS TRACKING

| Phase | Description | Tickets | Completed |
|-------|-------------|---------|-----------|
| 1 | Environment & Setup | 1–30 | 0/30 |
| 2 | Data Collection | 31–145 | 0/115 |
| 3 | Data Cleaning & Unification | 146–235 | 0/90 |
| 4 | Exploratory Data Analysis | 236–260 | 0/25 |
| 5 | Data Augmentation (→300K) | 261–318 | 0/58 |
| 6 | Model Comparison & Selection | 319–365 | 0/47 |
| 7 | LaTeX Report (HAWK Standard) | 366–500 | 0/135 |
| 8 | Documentation & Deliverables | 501–540 | 0/40 |
| **TOTAL** | | **540** | **0/540** |

---

*Generated: March 2026*
*Project: Decubitus Detection — NaMeKI*
*Student: Ashkan Sadri Ghamshi — HAWK University*
*Strategy: PU-only (Tier 1 + Tier 2) with multi-round augmentation to 300K+*
