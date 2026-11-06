"""The ten steps of one self-retraining cycle (SPEC §3).

Every step is a function `step(ctx: CycleContext) -> StepResult`. The steps share one
mutable `CycleContext` and run strictly in the order of STEP_NAMES:

    ingest -> quality -> dedupe -> drift -> dataset -> train -> evaluate -> gate
           -> report -> notify

A step reports its outcome as a StepResult: `ok`, a one-line `summary` for the
dashboard and JSON-ready `data` (stored with the step's final event in events.jsonl).
A step whose input is empty -- e.g. `train` when no new image survived quality control
and deduplication -- returns a *skipped* result instead of failing, so a batch that
adds nothing still completes the cycle. Exceptions are not handled here: the cycle
runner (`selftrain.system.cycle`) records a raising step as failed and stops the cycle.
During a long step, progress lines are emitted as events with status "started".

Steps 1-5 and 9 delegate their method to the feature modules (ingest, quality, dedupe,
drift, dataset, report, modelcard). Steps 6-8 and 10 are implemented here on top of the
experiment library, so the system trains, evaluates and compares models with exactly
the code and protocol of the thesis experiments.

Interfaces of the feature modules used here (SPEC §2, §3, §6)
    ingest.scan_inbox(ws) -> list[Batch]          Batch has `.name`
    ingest.load_batch(batch) -> list[Record]      Record: path, polygons, source,
                                                  license, batch, issue (SPEC §2)
    quality.check_all(records, cfg)
        -> (accepted: list[Record], rejected: list[(Record, reason: str)])
    quality.write_reports(ws, accepted, rejected, cfg) -> {batch: Path}
    dedupe.dedupe_records(ws, records, log) -> DedupeResult
    drift.pool_reference(ws) -> list[Path]; drift.drift_report(new_paths,
        reference_paths, alpha, log) -> dict with "status", "summary", "shifted", ...
    dataset.add_records(ws, records, dedupe_result, cycle, val_fraction, seed, log)
        -> AddResult;  dataset.create_version(ws, log) -> int
    modelcard.write(ws, version) -> Path
    report.write_report(ws, summary, emit) -> {"html": Path, "pdf": Path | None, ...}
        where `summary` is the cycle summary of `cycle_summary` (reports/cycle_{N}.json)
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import math
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from ..config import ROOT, UNetConfig, YOLOConfig
from . import STEP_NAMES, STEP_STATUSES
from . import gate as gate_rule
from .workspace import ARCHITECTURES, atomic_write_json, utc_now_iso

if TYPE_CHECKING:            # imported for type hints only: no import-time coupling
    from .registry import VersionRecord
    from .workspace import SystemConfig, Workspace

log = logging.getLogger(__name__)

WEIGHTS_FILE = {"unet": "best.pt", "yoloseg": "selected.pt"}
# Operating threshold used when a version has no evaluation yet: 0.5 on the U-Net
# foreground probability; 0.25 on the YOLO confidence, the value select_checkpoint uses.
DEFAULT_THRESHOLD = {"unet": 0.5, "yoloseg": 0.25}
NOTIFY_SCRIPT = ROOT / "scripts" / "notify_telegram.py"

STARTED_MESSAGES = {
    "ingest": "reading new batches from inbox/",
    "quality": "checking images and labels",
    "dedupe": "matching photographs against the pool",
    "drift": "testing new photographs for distribution drift",
    "dataset": "writing the next dataset version",
    "train": "training",
    "evaluate": "evaluating on the frozen test split",
    "gate": "comparing with the current model",
    "report": "writing report and model card",
    "notify": "sending Telegram notification",
}

Emit = Callable[..., None]
Log = Callable[[str], None]


def _no_emit(step: str, status: str, message: str, /, **data: Any) -> None:
    """Default event sink of a context that is not run by the cycle runner. Like every
    event sink it takes the event fields positional-only, so `data` may use any key."""


@dataclass
class StepResult:
    """Outcome of one step: `ok`, a one-line `summary` for the dashboard and JSON-ready
    `data`. `skipped` marks a step that had nothing to do (it is still ok)."""
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    def __post_init__(self) -> None:
        if self.skipped and not self.ok:
            raise ValueError("a skipped step must be ok")

    @property
    def status(self) -> str:
        """Final event status of the step: "skipped", "done" or "failed"."""
        return "skipped" if self.skipped else "done" if self.ok else "failed"


def skipped(summary: str, **data: Any) -> StepResult:
    """Result of a step that had nothing to do."""
    return StepResult(True, summary, dict(data), skipped=True)


@dataclass
class CycleContext:
    """State shared by the steps of one cycle (SPEC §3).

    batches: batches found in inbox/; records: labelled images loaded by ingest;
    accepted: images that passed quality control (the input of dedupe and dataset);
    rejected: (record, reason) pairs of quality control; groups: photograph identity
    of the accepted images (DedupeResult); dataset_version / model_version: versions
    created by this cycle; metrics: frozen-test summary of the new model; artifacts:
    paths and decisions produced along the way (weights, evaluation, gate, report);
    results: the StepResult of every finished step. `emit(step, status, message, /,
    **data)` records an event: steps use status "started" to report progress, the
    runner emits the final status of every step."""
    ws: Workspace
    cfg: SystemConfig
    cycle: int
    demo: bool = False
    emit: Emit = _no_emit
    batches: list[Any] = field(default_factory=list)
    records: list[Any] = field(default_factory=list)
    accepted: list[Any] = field(default_factory=list)
    rejected: list[tuple[Any, str]] = field(default_factory=list)
    groups: Any = None
    dataset_version: int | None = None
    model_version: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StepResult] = field(default_factory=dict)


# --------------------------------------------------------------------- helpers

def to_jsonable(obj: Any) -> Any:
    """Lenient conversion to strict JSON types for event and status data: numpy
    scalars and arrays become Python numbers and lists, paths strings, dataclasses
    and pydantic models dicts, sets sorted lists, non-finite floats (NaN, inf) None.
    Unlike workspace.json_safe, any other value is stored as its str() instead of
    raising, because a display problem must never end a cycle."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=str)]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "model_dump"):                      # pydantic v2 model
        return to_jsonable(obj.model_dump())
    return str(obj)


def check_architecture(architecture: str) -> str:
    """Validate an architecture name ("unet" or "yoloseg")."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}; expected one of {ARCHITECTURES}")
    return architecture


def workspace_relative(ws: Workspace, path: Path) -> str:
    """`path` relative to the workspace root when it lies inside it (the registry
    then stays valid when the workspace is moved), else the absolute path."""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(ws.root))
    except ValueError:
        return str(path)


def batch_name(batch: Any) -> str:
    """Name of an ingest batch (its `.name`; the inbox folder or zip name)."""
    return str(getattr(batch, "name", batch))


def reason_category(reason: str) -> str:
    """Rejection reasons are counted by their text up to the first colon
    ("whole-frame label: mask covers 98 % ..." -> "whole-frame label")."""
    return str(reason).split(":", 1)[0].strip() or str(reason)


def load_predictor(architecture: str, weights: Path) -> Any:
    """Inference wrapper of the experiment library for one checkpoint
    (UNetPredictor or YOLOPredictor; both expose infer/to_semantic/fg_confidence)."""
    if check_architecture(architecture) == "unet":
        from ..unet import UNetPredictor
        return UNetPredictor(Path(weights))
    from ..yoloseg import YOLOPredictor
    return YOLOPredictor(Path(weights))


def operating_threshold(ws: Workspace, version: int, architecture: str) -> float:
    """Threshold a version operates at: the one chosen on its validation split
    (models/v{N}/evaluation.json), else DEFAULT_THRESHOLD for its architecture."""
    path = ws.model_dir(version) / "evaluation.json"
    if path.exists():
        thr = json.loads(path.read_text()).get("threshold")
        if thr is not None:
            return float(thr)
    return DEFAULT_THRESHOLD[check_architecture(architecture)]


def train_names(data_root: Path) -> list[str]:
    """Stems of the training images of a dataset version. The experiment library
    reads *.jpg only (SegDataset, split_items), so the same rule applies here."""
    return sorted(p.stem for p in (Path(data_root) / "images" / "train").glob("*.jpg"))


def demo_subset(names: Sequence[str], max_images: int, seed: int) -> list[str]:
    """Demo mode trains on at most `max_images` training images: a uniform random
    subset drawn without replacement with numpy default_rng(seed), returned sorted."""
    if max_images <= 0:
        raise ValueError("demo_max_train_images must be positive")
    if len(names) <= max_images:
        return list(names)
    pick = np.random.default_rng(seed).choice(len(names), size=max_images, replace=False)
    return sorted(names[i] for i in pick)


def count_photographs(data_root: Path, names: Sequence[str]) -> int:
    """Number of distinct photographs among the images `names`, from the manifest
    snapshot of the dataset version (column `photograph`)."""
    manifest = Path(data_root) / "manifest.csv"
    import pandas as pd
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    if not {"name", "photograph"} <= set(df.columns):
        raise ValueError(f"{manifest} lacks the columns name/photograph")
    return int(df.loc[df["name"].isin(set(names)), "photograph"].nunique())


def make_unet_config(cfg: SystemConfig, demo: bool, n_train: int) -> UNetConfig:
    """U-Net hyper-parameters: library defaults (selftrain.config) with the system's
    epochs, patience and seed; demo mode uses demo_epochs. The batch never exceeds
    the number of training images, because the loader drops incomplete batches."""
    base = UNetConfig()
    return dataclasses.replace(base, epochs=cfg.demo_epochs if demo else cfg.epochs,
                               patience=cfg.patience, seed=cfg.seed,
                               batch=max(1, min(base.batch, n_train)))


def make_yolo_config(cfg: SystemConfig, demo: bool) -> YOLOConfig:
    """YOLOv8-seg hyper-parameters: library defaults with the system's epochs,
    patience and seed; demo mode uses demo_epochs."""
    return dataclasses.replace(YOLOConfig(), epochs=cfg.demo_epochs if demo else cfg.epochs,
                               patience=cfg.patience, seed=cfg.seed)


def write_train_yaml(data_root: Path, run_dir: Path, names: Sequence[str] | None) -> Path:
    """Ultralytics data file for one training run: the version's dataset.yaml with an
    absolute `path` (independent of the working directory); with `names` the train
    entry becomes a list file of exactly those images (demo subset)."""
    data_root = Path(data_root).absolute()
    spec = yaml.safe_load((data_root / "dataset.yaml").read_text())
    spec["path"] = str(data_root)
    if names is not None:
        listing = Path(run_dir) / "train_subset.txt"
        listing.write_text("".join(f"{data_root / 'images' / 'train' / f'{n}.jpg'}\n" for n in names))
        spec["train"] = str(listing)
    out = Path(run_dir) / "train_data.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False))
    return out


def evaluate_version(ws: Workspace, version: int, architecture: str, weights: Path,
                     data_root: Path, dataset_version: int | None = None, subdir: str = "eval",
                     write_summary: bool = True, log_fn: Log = log.info) -> dict[str, Any]:
    """Evaluate one model version with the pre-registered protocol
    (`selftrain.evaluate.full_evaluation`: threshold chosen on val, frozen test,
    external split and extra negatives when present). Library outputs (per-image
    CSVs) go to models/v{N}/<subdir>/; the returned dict is the library result plus
    provenance and the per-image CSV paths relative to models/v{N}. With
    `write_summary` it is also stored as models/v{N}/evaluation.json (strict JSON:
    NaN becomes null)."""
    from ..evaluate import full_evaluation
    model_dir = ws.model_dir(version)
    out_dir = model_dir / subdir
    label = f"v{version}"
    res = full_evaluation(load_predictor(architecture, weights), Path(data_root), out_dir,
                          label, log=log_fn)
    if "test" not in res:
        raise RuntimeError(f"dataset {data_root} has no test images: the frozen test split is missing")
    per_image = {split: str((out_dir / f"{label}_{split}_per_image.csv").relative_to(model_dir))
                 for split in ("val", "test", "external") if split in res}
    res.update({"version": version, "dataset_version": dataset_version,
                "architecture": architecture, "weights": str(weights), "data_root": str(data_root),
                "per_image_csv": per_image, "evaluated": utc_now_iso()})
    if write_summary:
        atomic_write_json(model_dir / "evaluation.json", res)
    return res


def load_notifier() -> ModuleType:
    """scripts/notify_telegram.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("notify_telegram", NOTIFY_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {NOTIFY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt(x: Any, spec: str = ".3f") -> str:
    """Format a metric that may be missing or NaN."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return format(x, spec)


def progress(ctx: CycleContext, step: str) -> Log:
    """Log sink for library code: every line goes to the Python log and, as a
    progress event (status "started"), to the dashboard."""
    def report(message: str) -> None:
        text = str(message).strip()
        log.info(text)
        if text:
            ctx.emit(step, "started", text.splitlines()[0][:200])
    return report


# ------------------------------------------------------------------ steps 1-5

def step_ingest(ctx: CycleContext) -> StepResult:
    """1. ingest (F1): read every batch waiting in inbox/.

    `ingest.scan_inbox` lists the batches (unless the runner already put them into
    ctx.batches) and `ingest.load_batch` converts each into Records -- image path,
    polygons with stage 0-3, source, license, batch (SPEC §2); the ingest module
    moves a batch to inbox/processed/ (or inbox/rejected/) and photos without labels
    to inbox/unlabeled/. A batch whose loading raises is reported and the remaining
    batches are still read; the step fails only if no batch could be read at all."""
    from . import ingest
    if not ctx.batches:
        ctx.batches = list(ingest.scan_inbox(ctx.ws))
    if not ctx.batches:
        return skipped("no batch waiting in inbox/")
    per_batch: dict[str, int] = {}
    failed: dict[str, str] = {}
    for batch in ctx.batches:
        name = batch_name(batch)
        ctx.emit("ingest", "started", f"reading batch {name}")
        try:
            records = list(ingest.load_batch(batch))
        except Exception as exc:                 # one bad batch must not block the others
            log.exception("batch %s cannot be read", name)
            failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        ctx.records.extend(records)
        per_batch[name] = len(records)
    data = {"batches": per_batch, "failed_batches": failed, "images": len(ctx.records)}
    if not per_batch:
        return StepResult(False, f"no batch could be read ({len(failed)} failed: "
                                 f"{'; '.join(f'{k}: {v}' for k, v in failed.items())})", data)
    summary = f"{len(ctx.records)} labelled images from {len(per_batch)} batch(es)"
    if failed:
        summary += f"; {len(failed)} batch(es) unreadable ({', '.join(failed)})"
    return StepResult(True, summary, data)


def step_quality(ctx: CycleContext) -> StepResult:
    """2. quality (F11): label quality gate.

    `quality.check_all` applies the checks of `selftrain.system.quality` in a fixed
    order -- unreadable image, shorter side below min_side_px, unknown class, invalid
    or degenerate polygon, coordinates outside [0, 1] by more than 1 %, label covering
    more than max_mask_frac of the image (whole-frame labels, the PPA Fig. 6.4 defect),
    region smaller than min_mask_frac -- and rejects a record with the reason
    "<code>: <message>" of the first failing check. Every loaded record must end up
    either accepted or rejected. `quality.write_reports` stores the outcome of each
    batch as inbox/processed/<batch>/quality_report.json."""
    if not ctx.records:
        return skipped("no labelled images to check")
    from . import quality
    accepted, rejected = quality.check_all(ctx.records, ctx.cfg)
    ctx.accepted = list(accepted)
    ctx.rejected = [(rec, str(reason)) for rec, reason in rejected]
    if len(ctx.accepted) + len(ctx.rejected) != len(ctx.records):
        raise RuntimeError(f"quality control returned {len(ctx.accepted)} accepted + "
                           f"{len(ctx.rejected)} rejected images for {len(ctx.records)} checked")
    reports = quality.write_reports(ctx.ws, ctx.accepted, ctx.rejected, ctx.cfg)
    reasons = Counter(reason_category(r) for _, r in ctx.rejected)
    summary = f"{len(ctx.accepted)} of {len(ctx.records)} images accepted"
    if reasons:
        summary += "; rejected: " + ", ".join(f"{n} {why}" for why, n in reasons.most_common(3))
    return StepResult(True, summary, {
        "checked": len(ctx.records), "accepted": len(ctx.accepted),
        "rejected": [{"batch": str(rec.batch), "file": str(rec.path), "reason": why}
                     for rec, why in ctx.rejected],
        "reasons": dict(reasons), "reports": reports})


def step_dedupe(ctx: CycleContext) -> StepResult:
    """3. dedupe (F3): photograph identity of the accepted images.

    `dedupe.dedupe_records` matches them against every pool image and the extra
    negatives, and against each other (stem rule only inside a batch; pHash, dHash,
    dihedral correlation and ORB/RANSAC across). An image matching an existing train
    or val photograph inherits its photograph and split; an image matching a frozen
    evaluation photograph (test, external, extra negatives) is not added and is
    reported (frozen-test protection)."""
    if not ctx.accepted:
        return skipped("no accepted images to match")
    from . import dedupe
    result = dedupe.dedupe_records(ctx.ws, ctx.accepted, log=progress(ctx, "dedupe"))
    ctx.groups = result
    return StepResult(True, result.summary(), result.to_dict(ctx.accepted))


def step_drift(ctx: CycleContext) -> StepResult:
    """4. drift (F16): do the new photographs differ from the pool?

    One image per new photograph (the first of each new photograph group; further
    images of the same photograph are not independent samples) is compared by
    `drift.drift_report` with one image per photograph of the pool's train, val and
    test splits (`drift.pool_reference`), i.e. with the pool before this cycle adds
    to it: two-sample Kolmogorov-Smirnov tests on per-image features (mean L*, a*,
    b* in CIELAB, colourfulness, Laplacian sharpness, resolution) with Bonferroni
    correction at family-wise level drift_alpha. With fewer than drift.MIN_IMAGES
    readable photographs on either side the test is skipped with a note. Drift is a
    warning, never a blocker: the step is ok whatever the outcome."""
    if ctx.groups is None or not ctx.groups.new_groups:
        return skipped("no new photographs to compare")
    from . import drift
    sample = [Path(ctx.accepted[group[0]].path) for group in ctx.groups.new_groups]
    result = drift.drift_report(sample, drift.pool_reference(ctx.ws), alpha=ctx.cfg.drift_alpha,
                                log=progress(ctx, "drift"))
    data = {**to_jsonable(result), "photographs_tested": len(sample)}
    if data.get("status") == "skipped":
        return skipped(str(data["summary"]), **data)
    ctx.artifacts["drift"] = data
    shifted = [str(f) for f in data.get("shifted") or []]
    return StepResult(True, f"warning: {data['summary']}" if shifted else str(data["summary"]), data)


def step_dataset(ctx: CycleContext) -> StepResult:
    """5. dataset (F4): write the accepted images to the pool and the next dataset
    version.

    `dataset.add_records` follows the photograph assignment of dedupe: new
    photographs are split train/val by photograph (val_fraction_new, seeded with
    seed and the cycle number), images of known photographs inherit their split,
    the test split never changes. If at least one image was added,
    `dataset.create_version` writes datasets/v{N} (symlinks into the pool, manifest
    snapshot, frozen test verified against version 0); otherwise the dataset is
    unchanged and the training steps are skipped."""
    if ctx.groups is None:
        return skipped("no accepted images: dataset unchanged")
    from . import dataset
    sink = progress(ctx, "dataset")
    added = dataset.add_records(ctx.ws, ctx.accepted, ctx.groups, cycle=ctx.cycle,
                                val_fraction=ctx.cfg.val_fraction_new, seed=ctx.cfg.seed,
                                log=sink)
    ctx.artifacts["added"] = added
    if not added.rows:
        return skipped(f"dataset unchanged: {added.summary()}", **added.to_dict())
    version = dataset.create_version(ctx.ws, log=sink)
    ctx.dataset_version = version
    splits = dataset.split_counts(ctx.ws.read_manifest())
    return StepResult(True, f"dataset v{version}: {added.summary()}",
                      {"version": version, **added.to_dict(), "splits": splits})


# ----------------------------------------------------------------- steps 6-8

def step_train(ctx: CycleContext) -> StepResult:
    """6. train (F5): train cfg.architecture on datasets/v{N}; the model becomes
    version N (model vN is trained on dataset vN).

    U-Net: `selftrain.unet.train` (early stopping on validation binary Dice);
    YOLOv8-seg: `selftrain.yoloseg.train`, then `select_checkpoint` picks the epoch
    with the best validation binary Dice -- the same selection rule for both. The
    chosen weights are moved to models/v{N}/best.pt (U-Net) or selected.pt (YOLO);
    the run directory is models/v{N}/train/ (a leftover of an interrupted attempt is
    removed first). YOLO's per-epoch checkpoints are deleted after the selection
    (selection.json keeps every epoch's score; a full run would otherwise leave
    several GB per version). Demo mode trains demo_epochs epochs on at most
    demo_max_train_images training images (seeded random subset)."""
    if ctx.dataset_version is None:
        return skipped("no new dataset version: nothing to train")
    from . import registry
    arch = check_architecture(ctx.cfg.architecture)
    version = ctx.dataset_version
    if any(rec.version == version for rec in registry.load(ctx.ws)["versions"]):
        raise RuntimeError(f"model version {version} is already registered")
    data_root = ctx.ws.dataset_dir(version)
    model_dir = ctx.ws.model_dir(version)
    run_dir = model_dir / "train"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    names = train_names(data_root)
    if not names:
        raise RuntimeError(f"dataset v{version} has no training images")
    if ctx.demo:
        names = demo_subset(names, ctx.cfg.demo_max_train_images, ctx.cfg.seed)
    sink = progress(ctx, "train")
    sink(f"{arch} on {len(names)} training images" + (" (demo)" if ctx.demo else ""))
    weights = model_dir / WEIGHTS_FILE[arch]
    lib_cfg: UNetConfig | YOLOConfig
    if arch == "unet":
        from .. import unet
        lib_cfg = make_unet_config(ctx.cfg, ctx.demo, len(names))
        best = unet.train(data_root, run_dir, lib_cfg, train_names=names, log=sink)
        os.replace(best, weights)
        summary = json.loads((run_dir / "summary.json").read_text())
        val_dice, epochs_run = summary["best_val_dice"], summary["epochs_run"]
    else:
        from .. import yoloseg
        lib_cfg = make_yolo_config(ctx.cfg, ctx.demo)
        data_yaml = write_train_yaml(data_root, run_dir, names if ctx.demo else None)
        yoloseg.train(data_yaml, run_dir, lib_cfg, log=sink)
        selected, _, val_dice = yoloseg.select_checkpoint(run_dir, data_root, log=sink)
        os.replace(selected, weights)
        for checkpoint in (run_dir / "weights").glob("epoch*.pt"):
            checkpoint.unlink()
        epochs_run = len(json.loads((run_dir / "selection.json").read_text())["scores"])
    photographs = count_photographs(data_root, names)
    ctx.model_version = version
    ctx.artifacts.update({"weights": weights, "train_images": len(names),
                          "train_photographs": photographs})
    return StepResult(True, f"{arch} v{version}: {len(names)} images, {epochs_run} epochs, "
                            f"val Dice {fmt(val_dice)}",
                      {"architecture": arch, "version": version, "train_images": len(names),
                       "train_photographs": photographs, "epochs_run": epochs_run,
                       "max_epochs": lib_cfg.epochs, "val_dice": val_dice,
                       "weights": weights, "demo": ctx.demo})


def summary_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen-test summary of a `full_evaluation` result as the cycle stores it
    (ctx.metrics, the evaluate event, VersionRecord.metrics): exactly
    `registry.metrics_from_evaluation`, so the cycle, the registry, the model card,
    the report and the dashboard read one set of keys -- threshold, test_dice_mean,
    test_dice_ci95, test_iou_mean, test_fpr, test_fpr_ci95, stage_dice_macro,
    n_test_positive, n_test_negative, inference_ms_per_image and, when evaluated,
    extra_negatives_fpr(_ci95) and external_dice_mean(_ci95). Strict JSON values."""
    from . import registry
    return registry.metrics_from_evaluation(evaluation)


def step_evaluate(ctx: CycleContext) -> StepResult:
    """7. evaluate (F6): `selftrain.evaluate.full_evaluation` of the new model on
    datasets/v{N} -- threshold chosen on val only, frozen test evaluated once,
    external split and extra negatives when the dataset provides them. The result
    is stored as models/v{N}/evaluation.json; its frozen-test summary
    (`summary_metrics`) becomes ctx.metrics."""
    if ctx.model_version is None:
        return skipped("no new model: nothing to evaluate")
    version = ctx.model_version
    res = evaluate_version(ctx.ws, version, check_architecture(ctx.cfg.architecture),
                           Path(ctx.artifacts["weights"]), ctx.ws.dataset_dir(ctx.dataset_version),
                           dataset_version=ctx.dataset_version, log_fn=progress(ctx, "evaluate"))
    ctx.artifacts["evaluation"] = {"path": ctx.ws.model_dir(version) / "evaluation.json",
                                   "per_image_csv": res["per_image_csv"]}
    ctx.metrics = summary_metrics(res)
    m = ctx.metrics
    lo, hi = m["test_dice_ci95"] or (None, None)
    return StepResult(True, f"test Dice {fmt(m['test_dice_mean'])} [{fmt(lo)}, {fmt(hi)}], "
                            f"FPR {fmt(m['test_fpr'], '.2f')}, "
                            f"threshold {fmt(m['threshold'], '.2f')}",
                      dict(m))


def current_test_dice(ctx: CycleContext, current: VersionRecord, names: set[str]) -> dict[str, float]:
    """Per-image test Dice of the deployed version on exactly the candidate's
    ulcer-containing test images. Its stored evaluation is used when it covers the
    same images (the test split is frozen, so normally it does); otherwise -- e.g. a
    version registered without evaluation -- the deployed model is evaluated on the
    candidate's dataset (same frozen test; threshold re-chosen on that dataset's val)
    and the result is kept in models/v{C}/eval_on_dataset_v{N}/."""
    model_dir = ctx.ws.model_dir(current.version)
    stored = model_dir / "evaluation.json"
    if stored.exists():
        rel = json.loads(stored.read_text()).get("per_image_csv", {}).get("test")
        if rel and (model_dir / rel).exists():
            dice = gate_rule.load_test_dice(model_dir / rel)
            if set(dice) == names:
                return dice
    res = evaluate_version(ctx.ws, current.version, current.architecture,
                           current.weights_path(ctx.ws), ctx.ws.dataset_dir(ctx.dataset_version),
                           dataset_version=ctx.dataset_version,
                           subdir=f"eval_on_dataset_v{ctx.dataset_version}", write_summary=False,
                           log_fn=progress(ctx, "gate"))
    return gate_rule.load_test_dice(model_dir / res["per_image_csv"]["test"])


def gate_summary(version: int, decision: gate_rule.GateDecision) -> str:
    """One dashboard line for a gate decision."""
    if decision.mean_diff is None or decision.ci95 is None:
        verdict = "promoted" if decision.promoted else "not promoted"
        return f"v{version} {verdict}: {decision.reason}"
    lo, hi = decision.ci95
    verdict = f"v{version} promoted" if decision.promoted else f"v{decision.compared_to} kept"
    return (f"{verdict}: Dice difference {decision.mean_diff:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"vs v{decision.compared_to}")


def step_gate(ctx: CycleContext) -> StepResult:
    """8. gate (F7): paired bootstrap of per-image test Dice, candidate - current
    (rule: selftrain.system.gate). The candidate is registered in any case (the
    registry keeps every version) and becomes current iff the gate promotes it; both
    happen in one locked registry write."""
    if ctx.model_version is None or "evaluation" not in ctx.artifacts:
        return skipped("no new model: nothing to compare")
    from . import registry
    version, cfg = ctx.model_version, ctx.cfg
    model_dir = ctx.ws.model_dir(version)
    candidate = gate_rule.load_test_dice(model_dir / ctx.artifacts["evaluation"]["per_image_csv"]["test"])
    current = registry.current(ctx.ws)
    if current is None:
        decision = gate_rule.decide(list(candidate.values()), None, margin=cfg.gate_margin)
    else:
        reference = current_test_dice(ctx, current, set(candidate))
        _, cand, cur = gate_rule.pair_by_name(candidate, reference)
        decision = gate_rule.decide(cand, cur, margin=cfg.gate_margin, n_boot=cfg.bootstrap,
                                    seed=cfg.seed, current_version=current.version)
    gate = {**decision.to_dict(), "version": version, "demo": ctx.demo}
    record = registry.VersionRecord(
        version=version, created=utc_now_iso(), dataset_version=ctx.dataset_version,
        architecture=cfg.architecture, train_images=ctx.artifacts["train_images"],
        train_photographs=ctx.artifacts["train_photographs"], metrics=dict(ctx.metrics),
        gate=gate, weights=workspace_relative(ctx.ws, ctx.artifacts["weights"]))
    registry.add_version(ctx.ws, record, make_current=decision.promoted)
    ctx.artifacts["gate"] = gate
    return StepResult(True, gate_summary(version, decision), gate)


# ----------------------------------------------------------------- steps 9-10

def cycle_summary(ctx: CycleContext) -> dict[str, Any]:
    """JSON-ready record of the cycle for the report (stored as
    reports/cycle_{N}.json): configuration, batches, quality rejections with
    reasons, photograph assignment, dataset and model version, frozen-test metrics,
    gate decision, current model, model card, every finished step's result, and
    `report_context` -- the transient cycle results in the form the report reads
    them (`report.context_from_cycle`: labelled images per source, rejections and
    duplicates with image names). Storing the latter makes the report a function of
    this file and the workspace, so `python -m selftrain report --version N`
    re-renders the report of a past cycle with that cycle's results."""
    from . import registry, report
    current = registry.current(ctx.ws)
    return to_jsonable({
        "cycle": ctx.cycle, "demo": ctx.demo, "created": utc_now_iso(),
        "config": ctx.cfg.to_dict(),
        "batches": [batch_name(b) for b in ctx.batches],
        "records": len(ctx.records), "accepted": len(ctx.accepted),
        "rejected": [{"batch": str(rec.batch), "file": str(rec.path), "reason": why}
                     for rec, why in ctx.rejected],
        "dedupe": ctx.groups.to_dict(ctx.accepted) if ctx.groups is not None else None,
        "drift": ctx.artifacts.get("drift"),
        "dataset_version": ctx.dataset_version, "model_version": ctx.model_version,
        "metrics": ctx.metrics, "gate": ctx.artifacts.get("gate"),
        "current_model": None if current is None else current.version,
        "model_card": ctx.artifacts.get("model_card"),
        "steps": {name: {"status": res.status, "summary": res.summary, "data": res.data}
                  for name, res in ctx.results.items()},
        "report_context": report.context_from_cycle(ctx),
    })


def step_report(ctx: CycleContext) -> StepResult:
    """9. report (F8, F12): model card of the new version (`modelcard.write`,
    registered as the version's model_card), then the cycle summary
    (reports/cycle_{N}.json) and the HTML + PDF report rendered from it by
    `report.write_report` (data added, rejections with reasons, duplicates, drift,
    metrics with CIs, version history, gate decision, example overlays, model card;
    PDF typeset with LaTeX, or by the matplotlib fallback if LaTeX fails). The
    report's progress lines are emitted as events of this step. Skipped when the
    cycle ingested nothing."""
    if not ctx.batches or not ctx.records:
        return skipped("nothing ingested: no report")
    from . import modelcard, registry, report
    if ctx.model_version is not None and "gate" in ctx.artifacts:     # registered this cycle
        card = Path(modelcard.write(ctx.ws, ctx.model_version))
        registry.update_version(ctx.ws, ctx.model_version, model_card=workspace_relative(ctx.ws, card))
        ctx.artifacts["model_card"] = card
    summary = cycle_summary(ctx)
    summary_path = ctx.ws.reports / f"cycle_{ctx.cycle}.json"
    atomic_write_json(summary_path, summary)
    paths = report.write_report(ctx.ws, summary, emit=ctx.emit)
    html, pdf = paths.get("html"), paths.get("pdf")
    ctx.artifacts.update(report_html=html, report_pdf=pdf, cycle_summary=summary_path)
    return StepResult(True, f"report written: {paths.get('summary') or 'no files'}",
                      {"html": html, "pdf": pdf, "summary": summary_path,
                       "engine": paths.get("engine"), "pages": paths.get("pages"),
                       "pdf_error": paths.get("pdf_error") or "",
                       "model_card": ctx.artifacts.get("model_card")})


def cycle_message(ctx: CycleContext) -> str:
    """Plain-text cycle summary for Telegram: one line per finished step."""
    head = f"Pressure-ulcer system: cycle {ctx.cycle}" + (" (demo)" if ctx.demo else "")
    lines = [head] + [f"{name}: {res.summary}" for name, res in ctx.results.items()
                      if name != "notify"]
    return "\n".join(lines)


def step_notify(ctx: CycleContext) -> StepResult:
    """10. notify: silent Telegram message via scripts/notify_telegram.py -- the PDF
    report as a document with the cycle summary as caption, or the summary alone
    when no report was written. Skipped when cfg.telegram is false; fails when the
    bot is not configured (.env) or Telegram cannot be reached."""
    if not ctx.cfg.telegram:
        return skipped("Telegram disabled in config")
    text = cycle_message(ctx)
    pdf = ctx.artifacts.get("report_pdf")
    notifier = load_notifier()
    try:
        if pdf and Path(pdf).exists():
            notifier.send_document(str(pdf), caption=text, silent=True)
            sent = f"report {Path(pdf).name} sent to Telegram"
        else:
            notifier.send_message(text, silent=True)
            sent = "summary sent to Telegram"
    except SystemExit as exc:           # notify_telegram exits when .env lacks token/chat id
        return StepResult(False, f"Telegram not configured: {exc}")
    return StepResult(True, sent, {"document": pdf})


_FUNCTIONS: dict[str, Callable[[CycleContext], StepResult]] = {
    "ingest": step_ingest, "quality": step_quality, "dedupe": step_dedupe,
    "drift": step_drift, "dataset": step_dataset, "train": step_train,
    "evaluate": step_evaluate, "gate": step_gate, "report": step_report,
    "notify": step_notify,
}
STEPS: tuple[tuple[str, Callable[[CycleContext], StepResult]], ...] = tuple(
    (name, _FUNCTIONS[name]) for name in STEP_NAMES)
"""(name, function) of the ten steps in execution order."""

__all__ = ["STEP_NAMES", "STEP_STATUSES", "STEPS", "CycleContext", "StepResult", "skipped",
           "to_jsonable", "load_predictor", "operating_threshold", "evaluate_version",
           "summary_metrics", "cycle_summary", "cycle_message"] + [f"step_{name}" for name in STEP_NAMES]
