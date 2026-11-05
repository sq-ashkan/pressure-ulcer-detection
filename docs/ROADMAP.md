# Bachelorarbeit — Roadmap & Arbeitsprotokoll

Decubitus (pressure ulcer) segmentation with a **self-retraining system** — HAWK, NaMeKI.
Examiners: Prof. Dr. Claire Chalopin (Erstprüferin), Annika Willnat (Zweitprüferin).
This file is the single plan for the work: agreements, rules, features, order of work,
report plan, presentation plan, and (at the end) the running work log.

> **Quality bar — non-negotiable.** This is Ashkan's last chance: one earlier attempt
> already failed. The target is the best possible grade (**1,0**); anything less is not
> acceptable. Quality comes before speed — there is no limit on effort, review rounds
> or time. Every claim must be correct, verified and defensible in the Kolloquium.

## 1. What was agreed (meeting 2026-10-28)

- The contribution is a **system that updates itself when new data arrives**:
  new data → retrain → better model → new report.
- **No fixed accuracy target.** Public data cannot support a clinical-accuracy claim
  (a > 98 % model is not reachable with it). The thesis measures accuracy honestly
  (with confidence intervals) and shows that the system improves as data grows.
- **First work step / first technical chapter:** compare **U-Net** with **YOLOv8-seg**
  on the same mask data, pick the better model, build the rest on it. This answers
  Prof. Chalopin's PPA comments (boxes vs masks; why no U-Net).
- Keep the leakage-free split by photograph and the negative examples.

## 2. Operating rules for autonomous work

1. Telegram update (`scripts/notify_telegram.py`) at every step: start/finish, results,
   errors, pause/resume.
2. 5-hour usage window: at ~95 % stop cleanly, schedule a one-shot timer for just after
   the reset, continue then. Right before such a pause, start long GPU jobs so the
   GPU works while tokens rest; keep the Mac awake with `caffeinate`.
3. Weekly limit: warn on Telegram at ~80 %, stop at ~95 %, report the reset time.
4. Context: no manual compaction; the automatic one at 97 % is enough.

## 3. Software: self-retraining segmentation system (Python package `selftrain`)

**Language: Python.** The data pipeline, PyTorch, Ultralytics, segmentation-models-pytorch
and the Apple-GPU (MPS) backend are all Python; the web UI uses Gradio (already used
by `app/`). Node.js would add a second stack without benefit.

| # | Feature | What it does |
|---|---|---|
| F1 | Ingest | Accept a new data batch in `inbox/` (folder or zip): YOLO polygons (Roboflow), COCO segmentation, or PNG masks; validate images and labels |
| F2 | Preprocess | Map source classes to stage 1–4 (per-source YAML), polygons ↔ masks, quality checks (corrupt images, tiny/invalid masks) |
| F3 | Deduplicate / anti-leakage | Photograph identity (filename stem + pHash + dHash + dihedral correlation + ORB/RANSAC); duplicates join the existing photograph; SIFT audit |
| F4 | Versioned dataset | Each cycle writes dataset vN with a manifest; the **test set is frozen** and never trained on, so every version is measured on the same images |
| F5 | Retrain | Retrain the chosen architecture on the Apple GPU with a fixed configuration and seed |
| F6 | Evaluate | Dice and IoU (binary and per stage), image-level sensitivity, false-positive rate on negatives, bootstrap 95 % CIs |
| F7 | Gate / model registry | Promote the new model only if it beats the current one on the frozen test; keep history, allow rollback |
| F8 | Report | Automatic PDF/HTML report per cycle (data growth, metrics over versions, example overlays); sent on Telegram |
| F9 | Interfaces | CLI (`selftrain run / status / report`), watch mode (service-like: polls `inbox/`), Gradio dashboard |
| F10 | Reproducibility | Config file, seeds, pinned environment, every run logged |
| F11 | Label quality gate | Rejects implausible labels automatically: whole-frame masks (the PPA Fig. 6.4 defect), tiny regions, out-of-range coordinates, unreadable images |
| F12 | Model card | Per version: intended use (research prototype, not a medical device), data sources + CC BY 4.0 attribution, metrics with CIs, limitations |
| F13 | External test | One source never seen in training (rf_dekubitus) measures generalisation to a "new clinic" |
| F14 | Active learning | Ranks unlabeled photos by model uncertainty so experts label the most informative first |
| F15 | One-command reproduction | `python -m selftrain reproduce` rebuilds the dataset and re-runs every thesis experiment |
| F16 | Drift check | Warns when a new batch differs from the existing data (colour, sharpness, resolution; KS tests) |

Full module contract: `selftrain/system/SPEC.md`.

**Pre-registered evaluation protocol** (fixed before any test result):
- Operating threshold chosen on **val only** (grid 0.05–0.95, maximise mean per-image
  binary Dice over all val images; empty-vs-empty = 1); test evaluated once.
- **Primary metric:** mean per-image binary Dice over ulcer-containing **test** images.
- Secondary: IoU, pooled per-stage Dice (macro), dominant-stage accuracy, sensitivity,
  false-positive rate on ulcer-free photos (test, extra AZH negatives per wound type,
  external), inference time. Bootstrap (2000) and Wilson 95 % CIs.
- **Model choice:** paired bootstrap of the test Dice difference; if the 95 % CI excludes
  0 the higher model wins; otherwise tie broken by macro stage Dice, then FP rate, then
  speed. Both models' checkpoints are selected by the same rule (best val binary Dice).
- **Gate (system):** promote a new version iff mean Dice difference > 0 and the lower
  95 % CI bound > −0.02 (non-inferiority margin) on the frozen test set.

```
 inbox/ ─▶ preprocess ─▶ dedupe ─▶ dataset vN (test frozen) ─▶ retrain ─▶ evaluate
   ▲                                                                        │
   └──────── wait for new data ◀── report + Telegram ◀── gate (better? → model v+1)
```

## 4. Phases and work packages, in order

Each phase ends with a check; the next phase starts only when the check passes.

**Phase 1 — Software (complete)**
- [x] **WP0 Setup** — Python 3.12 venv, all data sources, pretrained weights, LaTeX,
      GitHub, Telegram.
- [ ] **WP1 Segmentation dataset** — `scripts/build_seg_dataset.py`: one image set, two
      label formats (YOLOv8-seg polygons + PNG masks), split by photograph, negatives
      (AZH N/S/D/V/BG, FUSeg), SIFT audit. Sources: maskrcnn, woundcare, fid, mobile,
      petchs_seg, dekubitus. Excluded: thai_sores (boxes only), PISeg (same project as
      woundcare). Images with DTI/Unstageable are dropped (outside stage 1–4 scope).
- [ ] **WP2 Model comparison (pre-registered)** — U-Net (ResNet-34 encoder, Dice + CE
      loss) vs YOLOv8s-seg; same split, 640 px, comparable budget, early stopping on val.
      *Primary metric:* binary Dice (ulcer vs background) on the frozen test set.
      *Secondary:* per-stage Dice, IoU, false-positive rate on negatives, inference time.
      Bootstrap 95 % CIs; paired bootstrap for the difference. Winner → WP3.
- [ ] **WP3 Self-retraining system** — features F1–F10 around the winner, CLI + watch mode.
- [ ] **WP3b Dashboard** — see §6.
- [ ] **WP4 Data-growth demonstration** — training data fed to the system in four batches
      (25/50/75/100 % of photographs); each batch is one real cycle → versions v1–v4,
      gate decisions, learning curve.

**Phase 2 — Full software review and testing**
- [ ] Unit tests per feature, end-to-end test of a full cycle, dashboard test in a
      browser, leakage audit, reproducibility check (re-run gives the same numbers).
- [ ] Code review of the whole package; fix everything found.

**Phase 3 — Report**
- [ ] **WP5 Writing rules** — read `docs/private/Anleitung_Abschlussarbeiten.pdf`
      completely and extract every rule into `docs/thesis_guidelines.md` first.
- [ ] **WP6 Thesis** — LaTeX from the PPA template (see §5).

**Phase 4 — Report vs. reality**
- [ ] Check every number, table, figure and statement in the thesis against the result
      files, the code and the work log. Nothing unverified stays in.

**Phase 5 — Ibenthal compliance**
- [ ] Go through every rule in `docs/thesis_guidelines.md` against the compiled PDF;
      fix; recompile; re-run the tests.

**Phase 6 — Independent multi-agent review (loop)**
- [ ] **Three independent agents in parallel** review the whole project 0–100 against a
      written rubric (agreements with the examiners, Ibenthal rules, features F1–F10,
      correctness of code and results, writing quality) and each gives a grade with
      findings.
- [ ] Every finding is fixed; **every reference and link is checked one by one** (the
      source exists, says what the text claims, citation data correct).
- [ ] Then **three new agents** review again in parallel. Repeat until no substantive
      finding remains.

**Phase 7 — Presentation and delivery**
- [ ] **WP7** 10–13 slides / 25 min, demo rehearsal, German Q&A preparation.
- [ ] **WP8** PDF on Telegram; commit and push code + text (never data or `docs/private/`).

## 5. How the report is written

1. WP5 first: the Ibenthal rules decide structure and form; chapter order below is
   confirmed against the PDF before writing.
2. Draft structure: Introduction → State of the art → Requirements (features F1–F10 as
   functional requirements) → Concept: model comparison U-Net vs YOLOv8-seg (the agreed
   first technical part) and system design → Implementation → Test and validation
   (system tests, data-growth cycles) → Conclusion and outlook. Every chapter opens with
   its motivation and closes with a summary and a transition.
3. Every number comes from tonight's result files (`results/*.json`); tables are
   generated from them, so text and results cannot drift apart. No PPA figures reused.
4. Figures from matplotlib as vector PDF (≥ 600 dpi for raster); caption + label +
   reference for each; chapter-relative numbering.
5. Title page with both examiners, Selbständigkeitserklärung, 4–5 sentence abstract.
   Check whether HAWK requires a declaration of AI-tool use; a placeholder is left.
6. Compile with latexmk, send the PDF on Telegram for review.

## 6. Presentation / demo — dashboard requirements

One Python program, three ways to run it:

- **CLI** — one cycle per command; used for experiments and tests.
- **Watch mode** — behaves like a service: checks `inbox/` and runs a cycle when data arrives.
- **Dashboard** (for the Kolloquium) — Python backend (FastAPI) serving one HTML page;
  no Node.js, no build step.
  - Opens on a local port while the program runs (e.g. `http://localhost:8080`);
    **no password**.
  - **Drop zone:** drag photos or a data batch onto the page; they go to `inbox/`.
  - **Live pipeline schematic:** the cycle drawn as connected stages
    (ingest → preprocess → deduplicate → dataset → retrain → evaluate → gate → report);
    each stage **lights up** while running and turns green when finished, with its
    key number (e.g. "12 new photographs, 3 duplicates", "Dice 0.71").
  - Version history chart (metric per model version), latest report, and a
    "try the model" panel: upload a photo, see the predicted mask overlay.
  - Must look clean and attractive on a projector / Zoom screen share.
- A full retrain is too slow for a live talk, so the demo has a **demo mode** (few
  epochs on a small batch, ~2–3 min) that runs the real cycle end-to-end, and the
  dashboard shows the real overnight results.
## 7. Arbeitsprotokoll (work log)

| Date / time | Step | Result |
|---|---|---|
| 2026-11-05 | WP0 setup | Environment, 16 data sources (2.2 GB), weights, LaTeX, GitHub, Telegram ready |
| 2026-11-05 | Data review | 473 independent photographs in the four original mask sources (unify v3, SIFT audit: 4 twins); PISeg = woundcare duplicate; thai_sores boxes only |
| 2026-11-05 | WP1 | `build_seg_dataset.py` written, not yet run — paused to agree the plan |
| 2026-11-05 | Plan | Phases 1–7, dashboard requirements and the 1,0 quality bar agreed with Ashkan |
