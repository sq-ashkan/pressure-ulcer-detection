# selftrain.system — specification (contract for all modules)

The self-retraining system wraps the experiment library (`selftrain.unet`,
`selftrain.yoloseg`, `selftrain.evaluate`, `selftrain.metrics`) in a cycle:

```
ingest → quality → dedupe → drift → dataset → train → evaluate → gate → report → notify
```

Python 3.12, no new heavy dependencies (available: numpy, opencv, PIL, imagehash,
torch, ultralytics, segmentation_models_pytorch, scipy, pandas, matplotlib, jinja2,
fastapi, uvicorn, pydantic, pyyaml, tqdm, statsmodels, scikit-learn). Code style:
type hints, docstrings that state the method (they are quoted in the thesis),
no global state except the workspace on disk.

## 1. Workspace layout (`selftrain.system.workspace.Workspace`, root default `system_ws/`)

```
system_ws/
  inbox/                 drop zone: each batch is a folder or .zip (see §2)
  inbox/unlabeled/       photos without labels -> active-learning ranking only
  inbox/processed/       batches moved here after ingest (with ingest_report.json)
  inbox/rejected/        batches that failed completely
  pool/images|labels|masks/   every accepted image ever (name = unique id)
  pool/manifest.csv      name, source, photograph, split, batch, license, added_cycle
  datasets/v{N}/         images|labels|masks/{train,val,test}/ SYMLINKS into pool,
                         dataset.yaml (Ultralytics), manifest.csv (snapshot)
  models/v{N}/           weights (best.pt or selected.pt), train run dir, evaluation.json,
                         model_card.md
  registry.json          {"current": int|null, "versions": [VersionRecord...]}
  reports/cycle_{N}.pdf  + cycle_{N}.html + figures/
  events.jsonl           one JSON object per line (see §4)
  status.json            latest snapshot for the dashboard
  config.yaml            system configuration (architecture, epochs, gate margin, ...)
```

`Workspace(root: Path)` methods: `ensure()`, `next_dataset_version() -> int`,
`dataset_dir(v) -> Path`, `model_dir(v) -> Path`, `read_manifest() -> pandas.DataFrame`,
`write_manifest(df)`, `load_config() -> SystemConfig`, `save_config(cfg)`.

`SystemConfig` (pydantic or dataclass, stored as YAML): architecture ("unet"|"yoloseg"),
epochs, patience, demo_epochs (default 3), demo_max_train_images (default 120),
val_fraction_new (0.15), gate_margin (0.02), bootstrap (2000), min_component_frac,
quality thresholds (min_side_px 64, max_mask_frac 0.95, min_mask_frac 0.0005),
drift_alpha (0.01), telegram (bool), seed.

## 2. Batch formats accepted by ingest (`selftrain.system.ingest`)

A batch folder may contain any of:
* `images/` + `labels/` — YOLO segmentation txt (class x1 y1 x2 y2 ... normalised)
* `images/` + `masks/` — PNG masks with values 0 = background, 1-4 = stage 1-4
* `images/` + `annotations.json` — COCO instance segmentation (polygons)
* Roboflow export layout `{train,valid,test}/images|labels` (any subset)
* optional `source.yaml`: `name`, `license`, `class_map` {source_class: 1-4 | "background" | "drop"}
  (default class map: identity 0→stage1 … 3→stage4 for YOLO labels)
* images without any label file are treated as ulcer-free (negative) photos ONLY if
  `source.yaml` says `negatives: true`; otherwise they go to inbox/unlabeled/.

`ingest.scan_inbox(ws) -> list[Batch]`; `ingest.load_batch(batch) -> list[Record]` where
`Record = dataclass(path: Path, polygons: list[tuple[int, np.ndarray]], source: str,
license: str, batch: str)` — polygons: (stage 0-3, Nx2 normalised float32).

## 3. Cycle steps (`selftrain.system.steps`) — each `step(ctx: CycleContext) -> StepResult`

`CycleContext`: ws, cfg, cycle id, demo flag, records (list[Record]), accepted,
rejected (list[(Record, reason)]), groups, dataset_version, model_version, metrics,
artifacts dict, `emit(step, status, message, **data)`.
`StepResult`: ok: bool, summary: str (one line for the dashboard), data: dict.

1. **ingest** — load all batches; record count per batch.
2. **quality** (feature F11) — reject with reason: unreadable/too small image, polygon
   coordinates outside [0,1] by > 1 %, mask covering > max_mask_frac of the image
   (whole-frame labels, the PPA Fig. 6.4 defect), regions < min_mask_frac, unknown
   classes. Writes `inbox/processed/<batch>/quality_report.json`.
3. **dedupe** (F3) — photograph identity of new records against the pool AND within the
   batch using `scripts/unify_dataset_v3.build_groups` logic (stem rule only inside a
   batch; pHash + dHash + dihedral correlation + ORB/RANSAC across). A new image that
   matches an existing photograph inherits that photograph's split; if that split is
   `test`, the image is NOT added (frozen test protection) and is reported.
4. **drift** (F16) — compare new accepted photos with the pool on per-image features
   (mean L*, a*, b* in CIELAB, colourfulness, Laplacian sharpness, resolution) using
   two-sample Kolmogorov–Smirnov tests with Bonferroni correction (alpha = drift_alpha);
   report which features shifted. Drift is a warning, never a blocker.
5. **dataset** (F4) — new photographs split train/val (val_fraction_new, by photograph,
   seeded); test never changes. Writes pool files, manifest, `datasets/v{N}` symlinks.
6. **train** (F5) — trains `cfg.architecture` on `datasets/v{N}` via
   `selftrain.unet.train` / `selftrain.yoloseg.train` + `select_checkpoint`.
   Demo mode: `demo_epochs`, at most `demo_max_train_images` train images.
7. **evaluate** (F6) — `selftrain.evaluate.full_evaluation` (threshold on val, frozen test,
   extra negatives if available).
8. **gate** (F7) — paired bootstrap of per-image test Dice, candidate − current
   (`selftrain.metrics.paired_bootstrap`). Promote iff no current model, or
   mean_diff > 0 AND lower CI bound > −gate_margin (better on average, not plausibly
   worse by more than the margin). Registry keeps every version; `rollback(v)` allowed.
9. **report** (F8) — HTML + PDF (matplotlib figures, PDF via LaTeX `latexmk` with a
   jinja2 template; fall back to matplotlib PdfPages if LaTeX fails): data added,
   rejected with reasons, duplicates, drift, metrics with CIs, version history chart,
   gate decision, example overlays, model card (F12).
10. **notify** — Telegram via `scripts/notify_telegram.py` functions (silent), PDF attached.

`selftrain.system.cycle.run_cycle(ws, demo=False, emit=None) -> dict` runs the steps in
order, stops at the first failing step (status `failed`), always writes status.json.

## 4. Events (dashboard contract)

Each line of `events.jsonl`:
`{"ts": iso8601, "cycle": int, "step": one of STEP_NAMES, "status":
"started"|"done"|"failed"|"skipped", "summary": str, "data": {...}}`
`STEP_NAMES = ["ingest","quality","dedupe","drift","dataset","train","evaluate","gate","report","notify"]`
`status.json`: `{"cycle": int, "running": bool, "steps": {name: {"status","summary"}},
"current_model": int|null, "versions": [...]}`

## 5. Registry (`selftrain.system.registry`)

`VersionRecord`: version, created (iso), dataset_version, architecture, train_images,
train_photographs, metrics (test dice mean + ci, fpr, stage dice macro), gate
{"promoted": bool, "mean_diff", "ci95", "reason"}, weights (path), model_card (path).
Functions: `load(ws)`, `save(ws, reg)`, `add_version(ws, rec)`, `promote(ws, v)`,
`rollback(ws, v)`, `current(ws) -> VersionRecord|None`.

## 6. Model card (F12) — `selftrain.system.modelcard.write(ws, version) -> Path`
Intended use (research prototype; NOT a medical device, no diagnosis), data sources with
licenses and counts, training configuration, metrics with CIs on the frozen test,
false-positive rates per wound type, known limitations (public data ceiling, skin-tone
coverage unknown, stage-1 scarcity), version lineage.

## 7. Active learning (F14) — `selftrain.system.active.rank_unlabeled(ws, top_k=20)`
For every photo in `inbox/unlabeled/`: run the current model; uncertainty =
mean over the predicted-or-candidate region of (1 − |2p − 1|) where p is the foreground
confidence (`predictor.fg_confidence`), plus 0 for empty predictions with max p < 0.1.
Writes `active/ranking.csv` (path, uncertainty, predicted_area, predicted_stage) and
thumbnails with overlays for the top_k. Purpose: label the most informative photos first.

## 8. Interfaces (F9)
* CLI `python -m selftrain` (argparse): `init`, `run [--demo]`, `watch [--interval S]`,
  `status`, `report --version N`, `rank-unlabeled`, `rollback --to N`, `dashboard [--port]`,
  `reproduce` (F15: re-runs dataset build, comparison, data-growth; prints the numbers).
* `init --from data/seg --architecture A --weights W` creates the workspace from the
  thesis dataset (pool = data/seg, version 0 = that split) and registers W as model v0.
* Dashboard (`selftrain.system.dashboard`): FastAPI on 127.0.0.1:8080, no auth.
  `GET /` single HTML page (inline CSS/JS/SVG, no external CDN — must work offline);
  `GET /api/status`; `GET /api/events` (Server-Sent Events, tails events.jsonl);
  `POST /api/upload` (multipart: images and/or a zip -> new inbox batch);
  `POST /api/run?demo=1` (starts a cycle in a background thread; one at a time);
  `GET /api/versions`; `GET /api/report/{n}`; `POST /api/predict` (image -> PNG overlay
  + JSON: area, stage, confidence). Page: drop zone; the cycle drawn as an SVG pipeline
  of the 10 steps — grey (waiting), pulsing amber (running), green (done), red (failed)
  — each with its one-line summary; version-history chart (test Dice with CI per
  version, promoted versions highlighted); latest report link; "try the model" panel.
