# Project Context & Key Information

## Student
- **Name:** Ashkan Sadri Ghamshi
- **Matrikelnummer:** 769529
- **University:** HAWK — Hochschule fur angewandte Wissenschaft und Kunst Hildesheim/Holzminden/Gottingen
- **Faculty:** Fakultat Ingenieurwissenschaften und Gesundheit
- **Project:** NaMeKI (Decubitus Detection)

## Supervision
- **Primary Supervisor:** Annika Willnat
  - Role: Wissenschaftliche Mitarbeiterin, Gesundheitscampus Gottingen
  - Email: annika.willnat@hawk.de
- **Co-Supervisor / Scientific Advisor:** Claire Chalopin
  - Involved in project scope decisions and scientific direction

## Thesis Guidelines
- **Guidelines Author:** Prof. Dr.-Ing. Achim Ibenthal
  - Document: `document_pdf.pdf` (Hinweise zu Abschlussarbeiten, August 2024)
  - Ibenthal is NOT the supervisor — only the author of the HAWK thesis formatting guidelines

## Report
- **Language:** English
- **Format:** LaTeX -> PDF
- **Style:** HAWK standard per Ibenthal guidelines
  - Factual and neutral tone (sachlich und neutral)
  - No marketing phrases or informal language
  - Figures >= 600 dpi, prefer vector formats (PDF/SVG)
  - Every figure/table must have \caption{}, \label{}, and be referenced in text
  - Chapter-relative numbering for equations, figures, tables
  - BibTeX with english.bst style
  - Citations in [1], [2] format using cite package
  - Selbstandigkeitserklarung (independence declaration) required
  - Abstract: 4-5 sentences
  - Each chapter: motivation at beginning, summary + transition at end ("roter Faden")
  - Listings package for source code with line numbers
  - Presentation: max 25 min talk, 10-13 slides, max 45 min total with Q&A

## Project Scope — PPA (Praxisprojektarbeit)

### What PPA Covers:
1. **Data Foundation:** Collect, clean, deduplicate, augment publicly available PU datasets
2. **Model Selection:** Literature-based comparison and justified selection of model architecture (YOLOv8 vs DenseNet vs EfficientNet)

### What PPA Does NOT Cover (Outlook for Bachelor Thesis):
3. Model training and evaluation (fine-tuning on prepared dataset)
4. Prototype development and NaMeKI integration

### PPA Deliverables:
1. Prepared, augmented, documented image dataset in YOLOv8 format
2. Written analysis and justification of model selection with architecture comparison
3. Documentation of entire process (data sources, cleaning steps, augmentation strategies, model comparison)
4. Final presentation (current status, challenges, outlook for BA)

## Clinical Focus & Scope Clarifications

### Primary Focus: Decubitus Category II
- Clear visible wound (shallow open ulcer) — easier for AI detection
- NOT early detection / prediction — detection of EXISTING decubitus on RGB images

### Important Clarifications from Supervisor:
- NaMeKI project initially focuses on Grade 1, but Grade 2 is acceptable for this work
- The goal is detection and localization of existing pressure ulcers, NOT prediction
- Dataset should be structured to allow future extension to Category I (non-blanchable erythema)
- Always provide thematic context (background, problem statement) in the report

### Decubitus Categories (NPUAP/EPUAP):
- **Category I:** Non-blanchable erythema (permanent redness, no open wound)
- **Category II:** Partial thickness skin loss — shallow open ulcer (PRIMARY FOCUS)
- **Category III:** Full thickness skin loss — fat visible, no bone/tendon
- **Category IV:** Full thickness tissue loss — bone/tendon/muscle exposed
- **Unstageable:** Depth unknown, covered by slough/eschar
- **DTI (Deep Tissue Injury):** Purple/maroon discolored intact skin

## Technical Environment
- **Machine:** Mac Ultra M2
- **GPU:** Apple Silicon (MPS backend for PyTorch)
- **Power Monitoring:** CRITICAL — monitor power consumption after every heavy operation
- **Storage:** Local only, images NOT pushed to git

## Supervisor-Suggested Repositories (from Claire Chalopin)
These were evaluated and found NOT useful for PU detection:
1. BioImage Archive (https://www.ebi.ac.uk/bioimage-archive/) — microscopy/cell biology focus
2. Cancer Imaging Archive (https://www.cancerimagingarchive.net/) — cancer imaging, irrelevant
3. PIDAR (https://pidar.hpc4ai.unito.it/) — preclinical imaging metadata
4. IDR Image Data Resource (https://idr.openmicroscopy.org/) — published bio-image studies

## Weekly Meeting
- **Day:** Wednesdays (Studierenden-Meeting)
- **Purpose:** Status updates, questions, practice presentations
