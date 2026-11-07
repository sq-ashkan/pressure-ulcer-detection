"""Model card (feature F12): a Markdown description of one registered model version.

write(ws, version) renders models/v{N}/model_card.md (SPEC section 6) and records its
path in the registry. The card is assembled only from what the system has already
stored -- nothing is trained or evaluated again -- so it can be regenerated at any time
and always agrees with the registry and the evaluation files. Its fixed text is plain
ASCII, so the card can also be embedded in the LaTeX cycle report.

Method (section: where its content comes from)
----------------------------------------------
* Model details: the registry record (architecture, dataset version, gate decision,
  whether the version is the current model when the card is generated) and the SHA-256
  digest of the weights file, which identifies the exact checkpoint.
* Intended use: fixed text -- research prototype of the thesis; NOT a medical device;
  not for diagnosis, staging, triage or treatment of patients.
* Data: the manifest snapshot datasets/v{D}/manifest.csv, i.e. the pool manifest as it
  was when the model's dataset version D was built (the current pool manifest if the
  snapshot is missing). Per source: images per split, distinct photographs, the licence
  recorded at ingest, and title and URL of the thesis sources (workspace.KNOWN_SOURCES),
  since CC BY 4.0 requires attribution. Sources are the original public collections:
  an image that arrived in an inbox batch (source e.g. growth_batch2) is counted under
  the known source named in its image name (pool names "<batch>__<source>__..."; the
  growth batches are images of data/seg, whose names start with "<source>__"), with
  that source's licence (original_sources). Labelled regions per stage are counted in the
  YOLO label files of datasets/v{D}: one polygon line is one region, class c is stage
  c + 1, and a label file without polygons belongs to an ulcer-free photograph.
* Training configuration: the training run directory -- models/v{N}/train, or for a
  version registered with external weights the directory of the weights file or, in
  the Ultralytics layout <run>/weights/, its parent; the first that holds config.json.
  config.json of the experiment library gives the hyper-parameters, summary.json
  (U-Net) or selection.json (YOLOv8-seg) the outcome of checkpoint selection.
* Metrics: models/v{N}/evaluation.json (selftrain.evaluate.full_evaluation): the
  operating threshold chosen on the validation split, then the frozen test split
  evaluated once. Dice and IoU are means of per-photograph binary scores over the
  ulcer-containing test photographs with percentile-bootstrap 95 % CIs; sensitivity,
  false-positive rate and dominant-stage accuracy are proportions with Wilson 95 % CIs;
  per-stage Dice is computed from the pixel confusion matrix pooled over all test
  photographs. Without evaluation.json the registry's metric summary is shown.
* False-positive rates per wound type: evaluation.json["extra_negatives"] -- the share
  of ulcer-free photographs of each wound type (classes of the AZH wound-classification
  dataset) in which the model predicts an ulcer, with Wilson 95 % CIs -- next to the
  rate on the ulcer-free photographs of the frozen test split.
* Gate decision: the record's gate (paired bootstrap against the parent version).
* Known limitations: fixed statements (public-data ceiling, unknown skin-tone coverage,
  stage-1 scarcity, missing clinical context, operating point and drift) with this
  version's numbers filled in.
* Version lineage: the parent links of the registry -- each version's parent is the
  model that was current when it was registered, i.e. the baseline of its gate -- from
  this version back to the first model, and the registry's promote/rollback history.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from ..config import BOOTSTRAP, MIN_COMPONENT_FRAC, NUM_STAGES, THRESHOLD_GRID
from . import registry
from .registry import Registry, VersionRecord
from .workspace import (KNOWN_SOURCES, SPLITS, UNSTATED_LICENSE, Workspace, atomic_write_text,
                        check_version, utc_now_iso, validate_manifest)

__all__ = [
    "CARD_NAME", "NA", "WOUND_TYPES", "card_path", "dataset_manifest", "file_sha256",
    "find_train_dir", "fmt_ci", "fmt_estimate", "fmt_value", "label_statistics", "lineage",
    "load_evaluation", "markdown_table", "original_source", "original_sources", "render",
    "source_rows", "write",
]

CARD_NAME = "model_card.md"
EVALUATION_NAME = "evaluation.json"
NA = "n/a"
"""Shown for a metric that is missing or undefined (NaN), e.g. a rate without photographs."""

ARCHITECTURE_NAMES: Mapping[str, str] = MappingProxyType({
    "unet": "U-Net (segmentation_models_pytorch): semantic segmentation into background "
            "and stages 1-4",
    "yoloseg": "YOLOv8-seg (Ultralytics): instance segmentation with one class per stage, "
               "merged into a semantic stage mask",
})
SELECTION_RULES: Mapping[str, str] = MappingProxyType({
    "unet": "the epoch with the highest mean binary Dice on the validation split (early "
            "stopping after `patience` epochs without improvement)",
    "yoloseg": "the epoch checkpoint with the highest mean binary Dice on the validation "
               "split (every epoch is checkpointed and scored)",
})
WOUND_TYPES: Mapping[str, str] = MappingProxyType({
    "BG": "background", "N": "normal skin", "D": "diabetic ulcer", "P": "pressure ulcer",
    "S": "surgical wound", "V": "venous ulcer",
})
"""Classes of the AZH wound-classification dataset (column azh_class of negatives_extra.csv)."""
LICENSE_URLS: Mapping[str, str] = MappingProxyType({
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
})
TASK = ("pixel-wise segmentation of pressure ulcers in colour photographs into background "
        "and stages 1-4; the stage of a photograph is the stage covering most predicted "
        "ulcer pixels")
BOOTSTRAP_CI = f"percentile bootstrap, {BOOTSTRAP} resamples"
WILSON_CI = "Wilson score"

INTENDED_USE = """## Intended use

* **Research prototype.** The model belongs to a bachelor thesis on the automatic
  segmentation and staging of pressure ulcers in colour photographs and on a system that
  retrains it when new labelled photographs arrive. It is intended for research and
  teaching: reproducing and extending the experiments of the thesis on public data.
* **NOT a medical device.** The model has not been clinically validated, has undergone
  no conformity assessment and carries no CE marking; it is not a medical device in the
  sense of Regulation (EU) 2017/745 (MDR). It must not be used to diagnose, stage, triage
  or treat patients, nor to replace the assessment of a qualified clinician. Its outputs
  -- a segmentation mask, a stage and a confidence -- are not diagnoses.
* **Out of scope.** Any clinical or care use; wound types other than pressure ulcers
  (the model can mark them as pressure ulcers, see the false-positive rates); deep tissue
  injuries and unstageable pressure ulcers, which are not among the modelled classes
  (stages 1-4); images other than colour photographs of skin."""


# --------------------------------------------------------------------------- formatting

def _real(x: Any) -> float | None:
    """`x` as a float if it is a finite real number (booleans excluded), else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def fmt_value(x: Any, digits: int = 3, signed: bool = False) -> str:
    """`x` with `digits` decimals (with a leading sign if `signed`); NA if `x` is missing,
    not a number, NaN or infinite. The evaluation stores NaN for undefined metrics, e.g.
    a false-positive rate on a split without ulcer-free photographs."""
    value = _real(x)
    if value is None:
        return NA
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_ci(ci: Any, digits: int = 3, signed: bool = False) -> str:
    """A confidence interval as "[lo, hi]"; NA unless `ci` is a pair of finite numbers."""
    if isinstance(ci, (str, bytes)) or not isinstance(ci, Sequence) or len(ci) != 2:
        return NA
    lo, hi = fmt_value(ci[0], digits, signed), fmt_value(ci[1], digits, signed)
    return NA if NA in (lo, hi) else f"[{lo}, {hi}]"


def fmt_estimate(x: Any, ci: Any, digits: int = 3) -> str:
    """"estimate [lo, hi]", or the estimate alone if there is no interval."""
    value, interval = fmt_value(x, digits), fmt_ci(ci, digits)
    return value if interval == NA else f"{value} {interval}"


def _fmt_count(x: Any) -> str:
    """A count as an integer; NA if missing."""
    value = _real(x)
    return NA if value is None else str(int(value))


def _fmt_param(value: Any) -> str:
    """A hyper-parameter: containers as compact JSON, floats in shortest form."""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _code(text: Any) -> str:
    """Markdown inline code (a backtick inside would end the span; it becomes ')."""
    return "`" + str(text).replace("`", "'") + "`"


def _cell(value: Any) -> str:
    """A table cell: whitespace (also line breaks) collapsed, "|" escaped."""
    return " ".join(str(value).split()).replace("|", "\\|")


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]],
                   align: str | None = None) -> str:
    """A GitHub-flavoured Markdown table.

    `align` holds one character per column: "l" (left, the default), "r" (right) or "c"
    (centre). Cells are converted with str; line breaks become spaces and "|" is escaped,
    so no value can break the table. ValueError if a row or `align` has the wrong length."""
    width = len(header)
    align = "l" * width if align is None else align
    if len(align) != width or set(align) - set("lrc"):
        raise ValueError(f"align must hold one of l/r/c per column ({width}), got {align!r}")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"table row has {len(row)} cells, the header {width}: {row!r}")
    rule = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(_cell(h) for h in header) + " |",
             "| " + " | ".join(rule[a] for a in align) + " |"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _shown_path(ws: Workspace, path: str | Path) -> str:
    """`path` relative to the workspace root if it lies inside it, else unchanged."""
    p = Path(path)
    if p.is_absolute():
        try:
            return p.relative_to(ws.root).as_posix()
        except ValueError:
            return str(p)
    return p.as_posix()


# --------------------------------------------------------------------------- inputs

def card_path(ws: Workspace, version: int) -> Path:
    """models/v{version}/model_card.md."""
    return ws.model_dir(version) / CARD_NAME


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hexadecimal SHA-256 digest of a file, read in chunks of `chunk_size` bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    """A JSON file (NaN and Infinity, which json.dumps writes, are accepted); ValueError
    naming the file if it is not valid JSON."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """A JSON object from `path`; None if the file does not exist or holds another type."""
    if not Path(path).is_file():
        return None
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def load_evaluation(ws: Workspace, version: int) -> dict[str, Any] | None:
    """models/v{version}/evaluation.json -- written by the evaluate step: the result of
    selftrain.evaluate.full_evaluation plus provenance -- or None if it does not exist.
    ValueError if it is not a JSON object with a "test" section."""
    path = ws.model_dir(version) / EVALUATION_NAME
    if not path.is_file():
        return None
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("test"), dict):
        raise ValueError(f"{path}: not an evaluation result with a 'test' section")
    return data


def dataset_manifest(ws: Workspace, dataset_version: int) -> tuple[pd.DataFrame, str]:
    """The manifest describing dataset version `dataset_version` and its origin as text.

    datasets/v{D}/manifest.csv is the snapshot of the pool manifest taken when version D
    was built, so it lists exactly the images of that version (validated like the pool
    manifest). If the snapshot is missing, the current pool manifest is returned and the
    origin says so."""
    snapshot = ws.dataset_dir(dataset_version) / "manifest.csv"
    if snapshot.is_file():
        try:
            df = validate_manifest(pd.read_csv(snapshot, dtype=str, keep_default_na=False))
        except ValueError as exc:
            raise ValueError(f"{snapshot}: {exc}") from exc
        return df, (f"{_code(_shown_path(ws, snapshot))}, the snapshot of the pool manifest "
                    f"taken when dataset v{dataset_version} was built")
    return ws.read_manifest(), (f"{_code('pool/manifest.csv')}, the current pool (the "
                                f"snapshot of dataset v{dataset_version} is missing)")


def original_source(name: str, source: str) -> str:
    """The original public source of an image: `source` itself if it is a known thesis
    source (workspace.KNOWN_SOURCES), otherwise the first "__"-separated token of the
    image `name` that is a known source (an inbox batch such as growth_batch2 holds
    images of data/seg, whose names start with "<source>__"); `source` if none is."""
    if source in KNOWN_SOURCES:
        return source
    for token in str(name).split("__"):
        if token in KNOWN_SOURCES:
            return token
    return source


def original_sources(manifest: pd.DataFrame) -> pd.DataFrame:
    """Copy of `manifest` whose ``source`` column holds the original source of every image
    (original_source); a re-attributed image gets the licence of its known source."""
    if manifest.empty:
        return manifest
    out = manifest.copy()
    resolved = [original_source(n, s) for n, s in zip(out["name"], out["source"].astype(str))]
    changed = [r != s for r, s in zip(resolved, out["source"].astype(str))]
    out["source"] = resolved
    if "license" in out.columns:
        out["license"] = [KNOWN_SOURCES[r].license if c else lic
                          for r, c, lic in zip(resolved, changed, out["license"])]
    return out


def source_rows(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """One entry per source of a manifest, sorted by source name: ``title`` and ``url``
    (workspace.KNOWN_SOURCES; the source name and "" for any other source), ``license``
    (the distinct licences recorded in the manifest, joined by "; "; "unknown" if none),
    ``images`` (images per split, every split of SPLITS) and ``photographs`` (distinct
    photograph ids among the source's images)."""
    rows = []
    for source, part in manifest.groupby("source", sort=True):
        info = KNOWN_SOURCES.get(str(source))
        licences = sorted({str(x) for x in part["license"] if str(x)})
        rows.append({
            "source": str(source),
            "title": info.title if info else str(source),
            "url": info.url if info else "",
            "license": "; ".join(licences) if licences else "unknown",
            "images": {s: int((part["split"] == s).sum()) for s in SPLITS},
            "photographs": int(part["photograph"].nunique()),
        })
    return rows


def _stage_index(token: str) -> int | None:
    """Stage index 0-3 of a YOLO class token, None if it is not an integer in range."""
    try:
        value = int(token)
    except ValueError:
        return None
    return value if 0 <= value < NUM_STAGES else None


def label_statistics(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    """Label counts per split of a dataset version, from labels/<split>/*.txt.

    For every split with a label directory: ``images`` (label files), ``ulcer_free``
    (files without a polygon), ``regions`` (polygons per stage; class c of a YOLO line
    is stage c + 1) and ``invalid`` (lines whose class is not an integer 0-3, plus files
    that cannot be read, e.g. dangling symlinks). Splits without a directory are omitted."""
    stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        folder = Path(dataset_dir) / "labels" / split
        if not folder.is_dir():
            continue
        regions = [0] * NUM_STAGES
        images = ulcer_free = invalid = 0
        for path in sorted(folder.glob("*.txt")):
            images += 1
            try:
                lines = [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()
                         if ln.strip()]
            except (OSError, UnicodeDecodeError):
                invalid += 1
                continue
            if not lines:
                ulcer_free += 1
            for tokens in lines:
                stage = _stage_index(tokens[0])
                if stage is None:
                    invalid += 1
                else:
                    regions[stage] += 1
        stats[split] = {"images": images, "ulcer_free": ulcer_free, "regions": regions,
                        "invalid": invalid}
    return stats


def find_train_dir(ws: Workspace, rec: VersionRecord) -> Path | None:
    """The training run directory of a version: the first of models/v{N}/train (written
    by the train step), the directory of the weights file and its parent (Ultralytics
    keeps weights in <run>/weights/) that contains config.json; None if none does."""
    weights = rec.weights_path(ws)
    for candidate in (ws.model_dir(rec.version) / "train", weights.parent, weights.parent.parent):
        if (candidate / "config.json").is_file():
            return candidate
    return None


def lineage(reg: Registry, version: int) -> list[VersionRecord]:
    """The record of `version`, its parent, the parent's parent, ... up to the first
    model. The walk stops at a version without (registered) parent and never visits a
    version twice, so a corrupted registry with a parent cycle cannot loop forever.
    KeyError if `version` is not registered."""
    by_version = {r.version: r for r in reg["versions"]}
    if version not in by_version:
        raise KeyError(f"model version {version} is not registered "
                       f"(registered: {sorted(by_version)})")
    chain: list[VersionRecord] = []
    step: int | None = version
    while step is not None and step in by_version and all(r.version != step for r in chain):
        chain.append(by_version[step])
        step = by_version[step].parent
    return chain


# --------------------------------------------------------------------------- sections

def _status(reg: Registry, rec: VersionRecord) -> str:
    if reg["current"] == rec.version:
        return "current model (serves predictions)"
    if reg["current"] is None:
        return "not current (no model is current)"
    return f"not current (the current model is v{reg['current']})"


def _section_details(ws: Workspace, reg: Registry, rec: VersionRecord, generated: str) -> str:
    weights = rec.weights_path(ws)
    digest = _code(file_sha256(weights)) if weights.is_file() else "weights file not found"
    rows = [
        ("Version", f"v{rec.version}"),
        ("Registered (UTC)", rec.created),
        ("Architecture", ARCHITECTURE_NAMES.get(rec.architecture, rec.architecture)),
        ("Task", TASK),
        ("Dataset version", f"v{rec.dataset_version}"),
        ("Gate decision", "promoted" if rec.promoted else "not promoted"),
        ("Status when this card was generated", _status(reg, rec)),
        ("Weights", _code(_shown_path(ws, weights))),
        ("Weights SHA-256", digest),
    ]
    return "\n\n".join([
        f"# Model card: pressure-ulcer segmentation model v{rec.version}",
        f"Generated {generated} by `selftrain.system.modelcard` from the model registry and "
        f"the files of the workspace. The card is regenerated from these sources: change "
        f"them, not this file.",
        "## Model details",
        markdown_table(("Field", "Value"), rows),
    ])


def _licence_notes(licences: set[str]) -> list[str]:
    notes = [f"{lic} ({url}) requires attribution: the sources are credited above by title "
             f"and link." for lic, url in LICENSE_URLS.items() if lic in licences]
    if UNSTATED_LICENSE in licences:
        notes.append(f"Sources marked \"{UNSTATED_LICENSE}\" published no licence terms.")
    return notes


def _stage_table(labels: Mapping[str, Mapping[str, Any]]) -> str:
    header = ["Split", "Images", "Ulcer-free", *(f"Stage {k}" for k in range(1, NUM_STAGES + 1))]
    rows = [[split, s["images"], s["ulcer_free"], *s["regions"]] for split, s in labels.items()]
    return markdown_table(header, rows, align="l" + "r" * (len(header) - 1))


def _section_data(rec: VersionRecord, manifest: pd.DataFrame, origin: str,
                  labels: Mapping[str, Mapping[str, Any]]) -> str:
    parts = ["## Data"]
    if manifest.empty:
        parts.append(f"The manifest ({origin}) lists no images, so the data of this version "
                     f"are not recorded in the workspace.")
        return "\n\n".join(parts)
    splits = [s for s in SPLITS if s != "external" or (manifest["split"] == "external").any()]
    sources = source_rows(manifest)
    rows: list[list[Any]] = []
    for row in sources:
        title = f"[{row['title']}]({row['url']})" if row["url"] else row["title"]
        rows.append([_code(row["source"]), title, row["license"],
                     *(row["images"][s] for s in splits), row["photographs"]])
    rows.append(["**total**", "", "", *(int((manifest["split"] == s).sum()) for s in splits),
                 int(manifest["photograph"].nunique())])
    parts += [
        f"Images per split and distinct photographs per source, from {origin}. A photograph "
        f"is one physical photograph: copies and near-duplicates, also across sources and "
        f"resolutions, share one photograph id, and every photograph lies in exactly one "
        f"split, so the splits share no photograph (the total counts each photograph once). "
        f"The test split is frozen: it is the same in every dataset version.",
        markdown_table(["Source", "Dataset", "Licence", *splits, "Photographs"], rows,
                       align="lll" + "r" * (len(splits) + 1)),
        *_licence_notes({lic for row in sources for lic in row["license"].split("; ")}),
    ]
    n_train = int((manifest["split"] == "train").sum())
    if rec.train_images < n_train:
        parts.append(f"This version was trained on a subset: {rec.train_images} of the "
                     f"{n_train} training images (demo mode trains on at most "
                     f"demo_max_train_images randomly chosen images).")
    if labels:
        parts += [f"Labelled regions per stage, counted in the label files of "
                  f"`datasets/v{rec.dataset_version}` (one polygon = one region):",
                  _stage_table(labels)]
        invalid = sum(int(s["invalid"]) for s in labels.values())
        if invalid:
            parts.append(f"{invalid} label line(s) or file(s) could not be read as a stage 1-4 "
                         f"polygon and are not counted.")
    else:
        parts.append(f"The label files of `datasets/v{rec.dataset_version}` are not available, "
                     f"so labelled regions per stage cannot be counted.")
    return "\n\n".join(parts)


def _training_outcome(train_dir: Path) -> str | None:
    """Outcome of checkpoint selection from summary.json (selftrain.unet.train) or
    selection.json (selftrain.yoloseg.select_checkpoint); None if neither exists."""
    summary = _read_json_object(train_dir / "summary.json")
    if summary is not None:
        return (f"Training outcome (`summary.json`): best validation binary Dice "
                f"{fmt_value(summary.get('best_val_dice'))} at epoch "
                f"{_fmt_count(summary.get('best_epoch'))}; "
                f"{_fmt_count(summary.get('epochs_run'))} epoch(s) run.")
    selection = _read_json_object(train_dir / "selection.json")
    if selection is not None:
        scores = selection.get("scores")
        scores = scores if isinstance(scores, Mapping) else {}
        chosen = selection.get("selected")
        return (f"Training outcome (`selection.json`): checkpoint {_code(chosen)} selected "
                f"among {len(scores)} epoch checkpoint(s) by "
                f"{selection.get('criterion', 'validation Dice')} "
                f"({fmt_value(scores.get(chosen))}).")
    return None


def _section_training(ws: Workspace, rec: VersionRecord, train_dir: Path | None,
                      evaluation: Mapping[str, Any] | None) -> str:
    threshold = (evaluation or {}).get("threshold", rec.metrics.get("threshold"))
    grid = (f"{THRESHOLD_GRID[0]:.2f}, {THRESHOLD_GRID[1]:.2f}, ..., "
            f"{THRESHOLD_GRID[-1]:.2f}")
    threshold_text = (
        f"{fmt_value(threshold, 2)}: the value of the grid {grid} with the highest mean "
        f"per-photograph binary Dice on the validation split (ties: the value closest to 0.5)"
        if _real(threshold) is not None else "not chosen yet (the version has no evaluation)")
    rows = [
        ("Architecture", ARCHITECTURE_NAMES.get(rec.architecture, rec.architecture)),
        ("Training data", f"{rec.train_images} images ({rec.train_photographs} photographs) "
                          f"of the train split of dataset v{rec.dataset_version}"),
        ("Checkpoint selection", SELECTION_RULES.get(rec.architecture, NA)),
        ("Operating threshold", threshold_text),
        ("Post-processing", f"predicted regions smaller than {MIN_COMPONENT_FRAC * 100:g} % "
                            f"of the image are removed"),
        ("Run directory", _code(_shown_path(ws, train_dir)) if train_dir else "not found"),
    ]
    parts = ["## Training configuration", markdown_table(("Setting", "Value"), rows)]
    if train_dir is None:
        parts.append(f"No training run directory with a config.json was found (expected "
                     f"`models/v{rec.version}/train/`), so the hyper-parameters are not "
                     f"recorded in this card.")
        return "\n\n".join(parts)
    config = _read_json_object(train_dir / "config.json")
    if config:
        parts += [f"Hyper-parameters (`{_shown_path(ws, train_dir / 'config.json')}`, written "
                  f"by the experiment library):",
                  markdown_table(("Parameter", "Value"),
                                 [(key, _fmt_param(value)) for key, value in config.items()])]
    outcome = _training_outcome(train_dir)
    if outcome:
        parts.append(outcome)
    return "\n\n".join(parts)


def _staged_photographs(test: Mapping[str, Any]) -> int | None:
    """Ulcer-containing photographs with a predicted stage -- the denominator of the
    dominant-stage accuracy -- from the stage confusion matrix (rows: labelled stage 1-4,
    columns: predicted stage 0-4); None if the matrix is missing or malformed."""
    matrix = test.get("stage_confusion_gt_rows_pred_cols0to4")
    try:
        return int(sum(sum(row[1:]) for row in matrix))
    except (TypeError, ValueError):
        return None


def _metrics_from_registry(rec: VersionRecord) -> list[str]:
    m = rec.metrics
    if "test_dice_mean" not in m:
        return ["This version has not been evaluated on the frozen test split; no metrics "
                "are available."]
    rows = [
        ("Dice, binary: mean per photograph", fmt_value(m.get("test_dice_mean")),
         fmt_ci(m.get("test_dice_ci95")), _fmt_count(m.get("n_test_positive"))),
        ("IoU, binary: mean per photograph", fmt_value(m.get("test_iou_mean")), NA,
         _fmt_count(m.get("n_test_positive"))),
        ("False-positive rate: ulcer-free photographs with a predicted ulcer",
         fmt_value(m.get("test_fpr")), fmt_ci(m.get("test_fpr_ci95")),
         _fmt_count(m.get("n_test_negative"))),
        ("Stage Dice: macro mean over stages 1-4", fmt_value(m.get("stage_dice_macro")), NA, NA),
    ]
    if "external_dice_mean" in m:
        rows.append(("External split: Dice, binary", fmt_value(m.get("external_dice_mean")),
                     fmt_ci(m.get("external_dice_ci95")), NA))
    return [f"`models/v{rec.version}/{EVALUATION_NAME}` was not found; the values are the "
            f"registry's summary of this version.",
            markdown_table(("Metric", "Estimate", "95 % CI", "n"), rows, align="lrrr")]


def _section_metrics(rec: VersionRecord, evaluation: Mapping[str, Any] | None) -> str:
    parts = ["## Metrics on the frozen test split"]
    if evaluation is None:
        parts += _metrics_from_registry(rec)
        return "\n\n".join(parts)
    test = evaluation["test"]
    n_pos = _fmt_count(test.get("n_positive"))
    staged = _staged_photographs(test)
    rows = [
        ("Dice, binary: mean per photograph", fmt_value(test.get("dice_mean")),
         fmt_ci(test.get("dice_ci95")), n_pos, BOOTSTRAP_CI),
        ("IoU, binary: mean per photograph", fmt_value(test.get("iou_mean")),
         fmt_ci(test.get("iou_ci95")), n_pos, BOOTSTRAP_CI),
        ("Dice, binary: median per photograph", fmt_value(test.get("dice_median")), NA, n_pos,
         "-"),
        ("Sensitivity: ulcer photographs with a predicted ulcer",
         fmt_value(test.get("sensitivity")), fmt_ci(test.get("sensitivity_ci95")), n_pos,
         WILSON_CI),
        ("False-positive rate: ulcer-free photographs with a predicted ulcer",
         fmt_value(test.get("false_positive_rate")),
         fmt_ci(test.get("false_positive_rate_ci95")), _fmt_count(test.get("n_negative")),
         WILSON_CI),
        ("Dominant-stage accuracy: predicted stage = labelled stage",
         fmt_value(test.get("dominant_stage_accuracy")),
         fmt_ci(test.get("dominant_stage_accuracy_ci95")), _fmt_count(staged), WILSON_CI),
    ]
    stage_dice = test.get("stage_dice")
    stage_dice = stage_dice if isinstance(stage_dice, Mapping) else {}
    stage_rows = [(f"Stage {k}", fmt_value(stage_dice.get(f"stage{k}")))
                  for k in range(1, NUM_STAGES + 1)]
    stage_rows.append(("Macro mean", fmt_value(test.get("stage_dice_macro"))))
    parts += [
        f"The operating threshold was chosen on the validation split; the frozen test split, "
        f"never used for training, checkpoint selection or the threshold, was then evaluated "
        f"once. Test photographs: {_fmt_count(test.get('n_images'))} ({n_pos} with a pressure "
        f"ulcer, {_fmt_count(test.get('n_negative'))} ulcer-free). Dice and IoU are averaged "
        f"over the photographs with an ulcer; a photograph counts as positive if a predicted "
        f"region of at least {MIN_COMPONENT_FRAC * 100:g} % of the image remains.",
        markdown_table(("Metric", "Estimate", "95 % CI", "n", "Interval"), rows, align="lrrrl"),
        "Dice per stage, from the pixel confusion matrix pooled over all test photographs "
        "(n/a: the stage occurs neither in the labels nor in the predictions):",
        markdown_table(("Stage", "Dice"), stage_rows, align="lr"),
    ]
    external = evaluation.get("external")
    if isinstance(external, Mapping):
        parts.append(f"External split (a held-out source, not used for training or model "
                     f"selection): Dice "
                     f"{fmt_estimate(external.get('dice_mean'), external.get('dice_ci95'))}, "
                     f"n = {_fmt_count(external.get('n_positive'))} photographs with an ulcer.")
    if _real(evaluation.get("inference_ms_per_image")) is not None:
        parts.append(f"Mean inference time: {fmt_value(evaluation['inference_ms_per_image'], 1)} "
                     f"ms per photograph (hardware of the evaluation run).")
    return "\n\n".join(parts)


def _fpr_row(label: str, n: Any, rate: Any, ci: Any) -> tuple[str, str, str, str]:
    return label, _fmt_count(n), fmt_value(rate), fmt_ci(ci)


def _section_false_positives(rec: VersionRecord, evaluation: Mapping[str, Any] | None) -> str:
    parts = ["## False-positive rates per wound type"]
    intro = (f"Share of ulcer-free photographs in which the model predicts a pressure ulcer "
             f"(a region of at least {MIN_COMPONENT_FRAC * 100:g} % of the image), with "
             f"Wilson 95 % CIs.")
    header, align = ("Photographs", "n", "False-positive rate", "95 % CI"), "lrrr"
    test_label = "Ulcer-free photographs of the frozen test split"
    if evaluation is None:
        m = rec.metrics
        rows = []
        if "test_fpr" in m:
            rows.append(_fpr_row(test_label, m.get("n_test_negative"), m.get("test_fpr"),
                                 m.get("test_fpr_ci95")))
        if "extra_negatives_fpr" in m:
            rows.append(_fpr_row("All extra ulcer-free photographs", None,
                                 m.get("extra_negatives_fpr"), m.get("extra_negatives_fpr_ci95")))
        if not rows:
            parts.append("No evaluation is available for this version.")
            return "\n\n".join(parts)
        parts += [intro, markdown_table(header, rows, align=align),
                  f"Rates per wound type are stored only in `{EVALUATION_NAME}`, which was not "
                  f"found for this version."]
        return "\n\n".join(parts)
    test = evaluation["test"]
    rows = [_fpr_row(test_label, test.get("n_negative"), test.get("false_positive_rate"),
                     test.get("false_positive_rate_ci95"))]
    extra = evaluation.get("extra_negatives")
    if isinstance(extra, Mapping) and extra:
        for code in sorted(str(k) for k in extra if k != "all"):
            entry = extra[code]
            if isinstance(entry, Mapping):
                rows.append(_fpr_row(f"Extra: {WOUND_TYPES.get(code, code)} ({code})",
                                     entry.get("n"), entry.get("false_positive_rate"),
                                     entry.get("ci95")))
        if isinstance(extra.get("all"), Mapping):
            rows.append(_fpr_row("All extra ulcer-free photographs", extra["all"].get("n"),
                                 extra["all"].get("false_positive_rate"),
                                 extra["all"].get("ci95")))
        note = ("Extra ulcer-free photographs: `negatives_extra.csv` of the dataset -- "
                "photographs of other wound types and of skin without a wound (classes of the "
                "AZH wound-classification dataset) that belong to no split and were therefore "
                "never used for training.")
    else:
        note = ("The dataset has no extra ulcer-free photographs (`negatives_extra.csv`), so "
                "rates per wound type could not be measured; only the ulcer-free photographs "
                "of the frozen test split are reported.")
    parts += [intro, markdown_table(header, rows, align=align), note]
    return "\n\n".join(parts)


def _section_gate(rec: VersionRecord) -> str:
    gate = rec.gate
    baseline = gate.get("compared_to", rec.parent)
    rows: list[tuple[str, str]] = [
        ("Decision", "promoted" if rec.promoted else "not promoted"),
        ("Compared with", f"v{baseline}" if baseline is not None else "no model (none was current)"),
    ]
    if gate.get("mean_diff") is not None:
        rows += [("Mean Dice difference (candidate - baseline)",
                  fmt_value(gate["mean_diff"], signed=True)),
                 ("95 % CI (paired bootstrap)", fmt_ci(gate.get("ci95"), signed=True))]
    if gate.get("p_value") is not None:
        rows.append(("p-value (two-sided)", fmt_value(gate["p_value"])))
    if gate.get("n"):
        rows.append(("Paired test photographs", _fmt_count(gate["n"])))
    if gate.get("margin") is not None:
        rows.append(("Margin", fmt_value(gate["margin"])))
    rows.append(("Reason", gate.get("reason") or NA))
    return "\n\n".join([
        "## Gate decision",
        "Rule: a candidate is promoted if no model is current, or if the mean paired "
        "difference of per-photograph test Dice (candidate minus current model) is positive "
        "and the lower bound of its 95 % paired-bootstrap CI is above -margin.",
        markdown_table(("Field", "Value"), rows),
    ])


def _section_limitations(rec: VersionRecord, manifest: pd.DataFrame,
                         labels: Mapping[str, Mapping[str, Any]],
                         evaluation: Mapping[str, Any] | None) -> str:
    sources = sorted(set(manifest["source"])) if not manifest.empty else []
    public = sum(s in KNOWN_SOURCES for s in sources)
    if not sources:
        origin = "The labelled photographs come from public collections"
    elif public == len(sources):
        origin = f"The labelled photographs come from {len(sources)} public collections"
    else:
        origin = (f"The labelled photographs come from {len(sources)} sources, {public} of "
                  f"them public collections")
    train = labels.get("train")
    total = sum(train["regions"]) if train else 0
    if total:
        scarcity = (f"Stage 1 accounts for {train['regions'][0]} of the {total} labelled "
                    f"regions ({100 * train['regions'][0] / total:.1f} %) in the training split "
                    f"of dataset v{rec.dataset_version}")
    else:
        scarcity = "Stage 1 is under-represented in the public training data"
    test = (evaluation or {}).get("test") or {}
    stage_dice = test.get("stage_dice") if isinstance(test.get("stage_dice"), Mapping) else {}
    if _real(stage_dice.get("stage1")) is not None and _real(test.get("dice_mean")) is not None:
        scarcity += (f"; on the frozen test split its Dice is {fmt_value(stage_dice['stage1'])} "
                     f"against {fmt_value(test['dice_mean'])} for the binary ulcer/background "
                     f"segmentation")
    bullets = [
        f"**Public-data ceiling.** {origin}, taken with unknown cameras and protocols and "
        f"labelled by different annotators following different conventions. The labels are "
        f"therefore a noisy reference that limits the attainable agreement (Dice), and since "
        f"the frozen test split comes from the same sources, the metrics describe photographs "
        f"like these, not photographs from other clinics, cameras or patient populations. No "
        f"prospective or clinical validation has been performed.",
        "**Skin-tone coverage unknown.** Neither the sources nor the pool manifest record "
        "Fitzpatrick skin type or other demographic attributes, so it is unknown which skin "
        "tones the training data cover, and performance across skin tones could not be "
        "measured. Erythema, the sign of stage 1, is harder to see on darkly pigmented skin, "
        "so errors may be unevenly distributed.",
        f"**Stage-1 scarcity.** {scarcity}. Stage-1 predictions should be treated with "
        f"particular caution.",
        "**One photograph, no clinical context.** Staging uses only the visible surface in a "
        "single photograph; palpation, the blanching test, wound depth and patient history, "
        "which clinicians use, are not available to the model.",
        "**Operating point and drift.** The threshold was chosen on validation photographs "
        "from the same sources. The drift check of the system only warns when new "
        "photographs differ from the pool; it neither blocks nor adapts the model.",
    ]
    return "## Known limitations\n\n" + "\n".join(f"* {b}" for b in bullets)


def _gate_short(rec: VersionRecord) -> str:
    decision = "promoted" if rec.promoted else "not promoted"
    diff, baseline = rec.gate.get("mean_diff"), rec.gate.get("compared_to", rec.parent)
    if _real(diff) is None or baseline is None:
        return decision
    return f"{decision} (Dice {fmt_value(diff, signed=True)} vs v{baseline})"


def _section_lineage(reg: Registry, rec: VersionRecord) -> str:
    chain = lineage(reg, rec.version)
    rows = [(f"**v{r.version}** (this card)" if r.version == rec.version else f"v{r.version}",
             r.created, f"v{r.dataset_version}", r.architecture, r.train_images,
             fmt_estimate(r.metrics.get("test_dice_mean"), r.metrics.get("test_dice_ci95")),
             _gate_short(r)) for r in chain]
    parts = [
        "## Version lineage",
        f"{' <- '.join(f'v{r.version}' for r in chain)}. The parent of a version is the model "
        f"that was current when the version was registered, i.e. the baseline of its gate "
        f"decision.",
        markdown_table(("Version", "Registered (UTC)", "Dataset", "Architecture",
                        "Train images", "Test Dice [95 % CI]", "Gate"), rows, align="llllrrl"),
    ]
    history = [(h.get("ts", NA), h.get("action", NA), f"v{h.get('version')}",
                f"v{h['previous']}" if h.get("previous") is not None else "-")
               for h in reg["history"]]
    if history:
        parts += ["Changes of the current model (registry history, oldest first):",
                  markdown_table(("Time (UTC)", "Action", "Version", "Previous"), history)]
    else:
        parts.append("No model has been made current yet.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- public API

def _record(reg: Registry, version: int) -> VersionRecord:
    version = check_version(version)
    for rec in reg["versions"]:
        if rec.version == version:
            return rec
    raise KeyError(f"model version {version} is not registered "
                   f"(registered: {[r.version for r in reg['versions']]})")


def render(ws: Workspace, version: int, *, generated: str | None = None) -> str:
    """The model card of the registered `version` as Markdown (sections and their sources:
    module docstring). Only reads the workspace. `generated` is the time stamp printed in
    the header (default: now, UTC). KeyError if the version is not registered."""
    reg = registry.load(ws)
    rec = _record(reg, version)
    evaluation = load_evaluation(ws, rec.version)
    manifest, origin = dataset_manifest(ws, rec.dataset_version)
    manifest = original_sources(manifest)
    labels = label_statistics(ws.dataset_dir(rec.dataset_version))
    sections = [
        _section_details(ws, reg, rec, generated or utc_now_iso()),
        INTENDED_USE,
        _section_data(rec, manifest, origin, labels),
        _section_training(ws, rec, find_train_dir(ws, rec), evaluation),
        _section_metrics(rec, evaluation),
        _section_false_positives(rec, evaluation),
        _section_gate(rec),
        _section_limitations(rec, manifest, labels, evaluation),
        _section_lineage(reg, rec),
    ]
    return "\n\n".join(sections) + "\n"


def write(ws: Workspace, version: int) -> Path:
    """Render the card of the registered `version` (render), write it atomically to
    models/v{N}/model_card.md and return that path. If the registry entry of the version
    does not point to this file yet (e.g. "" for a version registered by init), its
    model_card field is updated. KeyError if the version is not registered."""
    text = render(ws, version)
    path = card_path(ws, version)
    atomic_write_text(path, text)
    recorded = registry.get_version(ws, version).model_card
    if not recorded or _resolve(ws, recorded) != path:
        registry.update_version(ws, version, model_card=str(path))
    return path


def _resolve(ws: Workspace, path: str) -> Path:
    """A path stored in the registry: absolute, or relative to the workspace root."""
    p = Path(path)
    return p if p.is_absolute() else ws.root / p
