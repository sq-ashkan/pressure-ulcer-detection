"""Workspace of the self-retraining system: directory layout, pool manifest, configuration.

The system keeps all of its state in one directory (default ``system_ws/``, SPEC §1).
There is no other global state, so a workspace can be inspected, copied or deleted as a
unit::

    inbox/                     drop zone for new batches (folder or .zip)
    inbox/unlabeled/           photos without labels (active-learning ranking only)
    inbox/processed/           batches after ingest, with their ingest/quality reports
    inbox/rejected/            batches that failed completely
    pool/images|labels|masks/  every accepted image ever; file stem = unique name
    pool/manifest.csv          name, source, photograph, split, batch, license, added_cycle
    datasets/v{N}/             dataset version N: symlinks into the pool, dataset.yaml,
                               manifest.csv (snapshot)
    models/v{N}/               model version N: weights, training run, evaluation, model card
    registry.json              model versions and the current model (registry module)
    reports/                   cycle_{N}.pdf / cycle_{N}.html and figures/
    active/                    active-learning ranking of unlabeled photos (SPEC §7)
    events.jsonl, status.json  dashboard contract (SPEC §4)
    config.yaml                SystemConfig

Method
------
* The root is resolved to an absolute path once. Every derived path -- including paths
  stored in the registry and the targets of dataset symlinks -- is therefore independent
  of the working directory of the process that uses it.
* Version numbers are never reused: the next number is one more than the largest number
  in use, where "in use" means a canonical ``v{N}`` directory on disk *or* a version
  recorded in the registry (0 for an empty workspace).
* Files that another process may read while they are being written (manifest, config,
  registry, status) are replaced atomically: the content is written to a uniquely named
  temporary file in the same directory, flushed to disk with fsync and renamed over the
  target with ``os.replace``, which POSIX guarantees to be atomic. A reader sees either
  the old or the new file, never a partially written one.
* The pool manifest is validated on every read and write: required columns, non-empty
  and unique names, known splits, integer photograph ids and cycle numbers, and the two
  data-integrity invariants of the thesis -- every photograph lies in exactly one split
  (no leakage between splits) and, once written, the set of test images never changes
  (frozen test set).
* SystemConfig is a dataclass stored as commented YAML. Missing keys take their
  default; unknown keys, wrong types and out-of-range values are rejected, so a typo in
  config.yaml cannot silently change an experiment.
"""
from __future__ import annotations

import json
import math
import operator
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..config import BOOTSTRAP, MIN_COMPONENT_FRAC, SEED

__all__ = [
    "ARCHITECTURES", "DEFAULT_ROOT", "INBOX_RESERVED", "KNOWN_SOURCES", "MANIFEST_COLUMNS",
    "SPLITS", "UNSTATED_LICENSE", "SourceInfo", "SystemConfig", "Workspace",
    "atomic_write_json", "atomic_write_text", "check_version", "empty_manifest",
    "json_safe", "license_for_source", "utc_now_iso", "validate_manifest", "version_dirs",
]

DEFAULT_ROOT = Path("system_ws")
ARCHITECTURES: tuple[str, ...] = ("unet", "yoloseg")
SPLITS: tuple[str, ...] = ("train", "val", "test", "external")
MANIFEST_COLUMNS: tuple[str, ...] = (
    "name", "source", "photograph", "split", "batch", "license", "added_cycle")
INBOX_RESERVED: tuple[str, ...] = ("unlabeled", "processed", "rejected")

_TEXT_COLUMNS = ("name", "source", "split", "batch", "license")
_NONEMPTY_COLUMNS = ("name", "source", "split")
_INT_COLUMNS = ("photograph", "added_cycle")
_VERSION_DIR = re.compile(r"v(0|[1-9][0-9]*)")
_INT_TEXT = re.compile(r"[+-]?[0-9]+")


# --------------------------------------------------------------------------- sources

@dataclass(frozen=True)
class SourceInfo:
    """Attribution of one data source: dataset title, where it was obtained, licence."""
    title: str
    url: str
    license: str


UNSTATED_LICENSE = "not stated by the source"

# The data sources of the thesis dataset (data/seg). Titles, URLs and licences are taken
# from the README.dataset.txt of each Roboflow export (all CC BY 4.0) and from the git
# remotes of the two UWM BigData Lab repositories, which state no licence.
KNOWN_SOURCES: Mapping[str, SourceInfo] = MappingProxyType({
    "roboflow_maskrcnn": SourceInfo(
        "maskrcnn (export pistagingaugmented), Roboflow Universe",
        "https://universe.roboflow.com/pressure-injury/maskrcnn-swd7w", "CC BY 4.0"),
    "roboflow_woundcare": SourceInfo(
        "PI Staging TSYL, Roboflow Universe",
        "https://universe.roboflow.com/pi-iqy1t/pi-staging-tsyl", "CC BY 4.0"),
    "roboflow_fid": SourceInfo(
        "PressureUlcer, Roboflow Universe",
        "https://universe.roboflow.com/fid-fvcc6/pressureulcer-ctn4w", "CC BY 4.0"),
    "roboflow_mobile": SourceInfo(
        "Mobile application for diagnostic of pressure ulcer, Roboflow Universe",
        "https://universe.roboflow.com/class-vx0ys/mobile-application-for-diagnostic-of-pressure-ulcer",
        "CC BY 4.0"),
    "rf_petchs_seg": SourceInfo(
        "Pressure-Ulcer-Segmentation (version 9), Roboflow Universe",
        "https://universe.roboflow.com/petchs-workspace-v8pvc/pressure-ulcer-segmentation",
        "CC BY 4.0"),
    "rf_dekubitus": SourceInfo(
        "Dekubitus, Roboflow Universe",
        "https://universe.roboflow.com/mweber1-rheinahrcampus-de/dekubitus", "CC BY 4.0"),
    "neg_azh": SourceInfo(
        "AZH wound classification dataset (UWM BigData Lab), ulcer-free classes only",
        "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations",
        UNSTATED_LICENSE),
    "neg_fuseg": SourceInfo(
        "FUSeg foot ulcer segmentation challenge (UWM BigData Lab), used as "
        "non-pressure-ulcer photographs",
        "https://github.com/uwm-bigdata/wound-segmentation", UNSTATED_LICENSE),
})


def license_for_source(source: str) -> str | None:
    """Licence of a known thesis source (KNOWN_SOURCES), None for any other source.

    New batches declare their licence in source.yaml; this lookup only fills the
    manifest's ``license`` column for the sources of the initial thesis dataset."""
    info = KNOWN_SOURCES.get(source)
    return info.license if info is not None else None


# --------------------------------------------------------------------------- file helpers

def utc_now_iso() -> str:
    """Current time as ISO-8601 in UTC with seconds precision (registry and events)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_safe(obj: Any) -> Any:
    """Convert `obj` into values that strict JSON (RFC 8259) can represent.

    NaN and +-inf become None (JSON has no such numbers; browsers and Starlette's
    JSONResponse reject them), numpy scalars and arrays become Python numbers and
    lists, paths become strings, tuples become lists, dataclasses become dicts and
    mapping keys become strings. Any other type raises TypeError, so nothing is
    silently stringified."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(asdict(obj))
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    raise TypeError(f"value of type {type(obj).__name__} cannot be stored as JSON: {obj!r}")


def atomic_write_text(path: Path, text: str) -> None:
    """Replace `path` atomically with `text` (UTF-8).

    The text goes to a uniquely named temporary file in the target directory (same file
    system, created with the process umask), is flushed and fsync'ed, and is then renamed
    over the target with os.replace. On any error the temporary file is removed and the
    previous content of `path` is left untouched."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Write `obj` as strict JSON (see json_safe) atomically (see atomic_write_text)."""
    text = json.dumps(json_safe(obj), indent=indent, ensure_ascii=False, allow_nan=False)
    atomic_write_text(path, text + "\n")


def check_version(version: Any) -> int:
    """Validate a version number: a non-negative integer (numpy integers accepted,
    booleans rejected). Returns it as a Python int."""
    if isinstance(version, (bool, np.bool_)):
        raise TypeError(f"version must be an integer, got {version!r}")
    try:
        number = operator.index(version)
    except TypeError:
        raise TypeError(f"version must be an integer, got {version!r}") from None
    if number < 0:
        raise ValueError(f"version must be >= 0, got {number}")
    return number


def version_dirs(parent: Path) -> list[int]:
    """Version numbers N of the canonical ``v{N}`` sub-directories of `parent`, sorted.

    Only exact names count (``v0``, ``v12``); ``v01``, ``v1_tmp`` or files are ignored."""
    if not parent.is_dir():
        return []
    found = []
    for entry in parent.iterdir():
        match = _VERSION_DIR.fullmatch(entry.name)
        if match and entry.is_dir():
            found.append(int(match.group(1)))
    return sorted(found)


# --------------------------------------------------------------------------- manifest

def empty_manifest() -> pd.DataFrame:
    """A pool manifest without rows, with the column order and dtypes of read_manifest."""
    return pd.DataFrame({col: pd.Series(dtype="int64" if col in _INT_COLUMNS else "str")
                         for col in MANIFEST_COLUMNS})


def _as_int64(series: pd.Series, column: str) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    finite = numbers.notna() & np.isfinite(numbers.astype(float))
    integral = finite & (numbers.where(finite, 0) % 1 == 0)
    if not integral.all():
        rows = series.index[~integral].tolist()[:5]
        raise ValueError(f"manifest column {column!r} must hold integers; offending row(s) "
                         f"{rows}: {series[~integral].tolist()[:5]}")
    return numbers.astype("int64")


def validate_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Check a pool manifest and return a normalised copy; raise ValueError if invalid.

    Normalisation: required columns first in MANIFEST_COLUMNS order, extra columns kept
    after them, text columns as strings (missing values become ""), ``photograph`` and
    ``added_cycle`` as int64, a fresh 0..n-1 index. Rules: ``name``, ``source`` and
    ``split`` are non-empty; names are unique (a name is the image's id in the pool);
    splits are one of SPLITS; ``added_cycle`` >= 0; and every photograph lies in exactly
    one split -- the photograph-disjoint split of the thesis is enforced on every write."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"manifest must be a pandas DataFrame, got {type(df).__name__}")
    if df.columns.duplicated().any():
        raise ValueError(f"manifest has duplicate column names: {list(df.columns)}")
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest lacks required column(s) {missing}; "
                         f"required: {list(MANIFEST_COLUMNS)}")
    extras = [c for c in df.columns if c not in MANIFEST_COLUMNS]
    out = df.loc[:, [*MANIFEST_COLUMNS, *extras]].reset_index(drop=True)
    for col in _TEXT_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    for col in _NONEMPTY_COLUMNS:
        empty = out.index[out[col] == ""].tolist()
        if empty:
            raise ValueError(f"manifest column {col!r} is empty in row(s) {empty[:5]}")
    duplicated = sorted(set(out.loc[out["name"].duplicated(), "name"]))
    if duplicated:
        raise ValueError(f"manifest names must be unique; duplicated: {duplicated[:5]}")
    unknown = sorted(set(out["split"]) - set(SPLITS))
    if unknown:
        raise ValueError(f"unknown split(s) {unknown}; allowed: {list(SPLITS)}")
    for col in _INT_COLUMNS:
        out[col] = _as_int64(out[col], col)
    negative = out.index[out["added_cycle"] < 0].tolist()
    if negative:
        raise ValueError(f"added_cycle must be >= 0; offending row(s) {negative[:5]}")
    splits_per_photo = out.groupby("photograph")["split"].nunique()
    leaking = splits_per_photo.index[splits_per_photo > 1].tolist()
    if leaking:
        raise ValueError(f"photograph(s) {leaking[:5]} lie in more than one split; a "
                         f"photograph must stay in one split (no leakage)")
    return out


# --------------------------------------------------------------------------- configuration

def _doc(text: str) -> dict[str, str]:
    return {"doc": text}


def _coerce(name: str, value: Any, kind: type) -> Any:
    """Convert `value` to the type of the field's default, or raise ValueError.

    bool fields accept only booleans; int fields accept integers, integral floats and
    integer strings (not booleans); float fields accept finite numbers and numeric
    strings such as "5e-4" (which YAML 1.1 loads as a string); str fields accept strings."""
    if isinstance(value, np.generic):
        value = value.item()
    if kind is bool:
        if isinstance(value, bool):
            return value
    elif kind is int:
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            return value
        elif isinstance(value, float) and value.is_integer():
            return int(value)
        elif isinstance(value, str) and _INT_TEXT.fullmatch(value.strip()):
            return int(value.strip())
    elif kind is float:
        number: float | None = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
        elif isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                number = None
        if number is not None and math.isfinite(number):
            return number
    elif kind is str:
        if isinstance(value, str):
            return value
    raise ValueError(f"config value {name!r} must be of type {kind.__name__}, got {value!r}")


@dataclass
class SystemConfig:
    """Configuration of the self-retraining system (SPEC §1), stored as config.yaml.

    Defaults follow the pre-registered protocol (docs/ROADMAP.md) and selftrain.config:
    epochs/patience as UNetConfig and YOLOConfig, bootstrap = BOOTSTRAP,
    min_component_frac = MIN_COMPONENT_FRAC, seed = SEED. Values are coerced to the type
    of their default and range-checked on construction (also by dataclasses.replace);
    an invalid configuration raises ValueError naming every offending key."""

    architecture: str = field(default="unet", metadata=_doc(
        'model trained by every cycle: "unet" or "yoloseg"'))
    epochs: int = field(default=80, metadata=_doc(
        "maximum training epochs of a full (non-demo) cycle"))
    patience: int = field(default=15, metadata=_doc(
        "early stopping: epochs without a better validation Dice before training stops"))
    demo_epochs: int = field(default=3, metadata=_doc(
        "training epochs in demo mode (run --demo)"))
    demo_max_train_images: int = field(default=120, metadata=_doc(
        "demo mode trains on at most this many training images"))
    val_fraction_new: float = field(default=0.15, metadata=_doc(
        "fraction of NEW photographs assigned to val (by photograph, seeded); "
        "the test split never changes"))
    gate_margin: float = field(default=0.02, metadata=_doc(
        "gate non-inferiority margin: promote iff mean test-Dice difference > 0 and "
        "lower 95 % CI bound > -gate_margin"))
    bootstrap: int = field(default=BOOTSTRAP, metadata=_doc(
        "bootstrap resamples for confidence intervals and the gate"))
    min_component_frac: float = field(default=MIN_COMPONENT_FRAC, metadata=_doc(
        "predicted regions smaller than this fraction of the image are removed as specks"))
    min_side_px: int = field(default=64, metadata=_doc(
        "quality gate: minimum image side length in pixels; smaller images are rejected"))
    max_mask_frac: float = field(default=0.95, metadata=_doc(
        "quality gate: labels covering more than this fraction of the image are rejected "
        "(whole-frame labels)"))
    min_mask_frac: float = field(default=0.0005, metadata=_doc(
        "quality gate: labelled regions smaller than this fraction of the image are rejected"))
    drift_alpha: float = field(default=0.01, metadata=_doc(
        "family-wise significance level of the drift KS tests (Bonferroni-corrected)"))
    telegram: bool = field(default=False, metadata=_doc(
        "send the cycle report to Telegram (scripts/notify_telegram.py); off by default"))
    seed: int = field(default=SEED, metadata=_doc(
        "random seed for splitting new photographs and for training"))

    def __post_init__(self) -> None:
        errors = []
        for f in fields(self):
            try:
                setattr(self, f.name, _coerce(f.name, getattr(self, f.name), type(f.default)))
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            raise ValueError("invalid configuration: " + "; ".join(errors))
        self._check_ranges()

    def _check_ranges(self) -> None:
        rules = [
            (self.architecture in ARCHITECTURES,
             f"architecture must be one of {list(ARCHITECTURES)}, got {self.architecture!r}"),
            (self.epochs >= 1, f"epochs must be >= 1, got {self.epochs}"),
            (self.patience >= 1, f"patience must be >= 1, got {self.patience}"),
            (self.demo_epochs >= 1, f"demo_epochs must be >= 1, got {self.demo_epochs}"),
            (self.demo_max_train_images >= 1,
             f"demo_max_train_images must be >= 1, got {self.demo_max_train_images}"),
            (0.0 <= self.val_fraction_new < 1.0,
             f"val_fraction_new must lie in [0, 1), got {self.val_fraction_new}"),
            (0.0 <= self.gate_margin < 1.0,
             f"gate_margin must lie in [0, 1), got {self.gate_margin}"),
            (self.bootstrap >= 1, f"bootstrap must be >= 1, got {self.bootstrap}"),
            (0.0 <= self.min_component_frac < 1.0,
             f"min_component_frac must lie in [0, 1), got {self.min_component_frac}"),
            (self.min_side_px >= 1, f"min_side_px must be >= 1, got {self.min_side_px}"),
            (0.0 < self.max_mask_frac <= 1.0,
             f"max_mask_frac must lie in (0, 1], got {self.max_mask_frac}"),
            (0.0 <= self.min_mask_frac < self.max_mask_frac,
             f"min_mask_frac must lie in [0, max_mask_frac), got {self.min_mask_frac}"),
            (0.0 < self.drift_alpha < 1.0,
             f"drift_alpha must lie in (0, 1), got {self.drift_alpha}"),
            (self.seed >= 0, f"seed must be >= 0, got {self.seed}"),
        ]
        errors = [message for ok, message in rules if not ok]
        if errors:
            raise ValueError("invalid configuration: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of all settings, in declaration order."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SystemConfig:
        """Build from a mapping; missing keys take defaults, unknown keys raise ValueError."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(repr(k) for k in data if k not in known)
        if unknown:
            raise ValueError(f"unknown configuration key(s) {', '.join(unknown)}; "
                             f"known keys: {sorted(known)}")
        return cls(**dict(data))

    def to_yaml(self) -> str:
        """YAML text with one explanatory comment per key (the file users edit)."""
        lines = ["# selftrain system configuration (selftrain/system/SPEC.md, section 1).",
                 "# Missing keys take their default; unknown keys and out-of-range values "
                 "are rejected."]
        for f in fields(self):
            lines += ["", f"# {f.metadata['doc']}",
                      yaml.safe_dump({f.name: getattr(self, f.name)}, sort_keys=False).rstrip("\n")]
        return "\n".join(lines) + "\n"

    @classmethod
    def from_yaml(cls, text: str) -> SystemConfig:
        """Parse config.yaml text (an empty document gives the defaults)."""
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"configuration is not valid YAML: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"configuration must be a YAML mapping of key: value, "
                             f"got {type(data).__name__}")
        return cls.from_dict(data)


# --------------------------------------------------------------------------- workspace

@dataclass(frozen=True)
class Workspace:
    """One self-retraining system on disk (layout: module docstring and SPEC §1).

    The object only derives paths; nothing is created until ensure() or a write method
    is called. `root` is stored as an absolute, symlink-resolved path."""

    root: Path = DEFAULT_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    # ---- layout ------------------------------------------------------------------
    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def unlabeled(self) -> Path:
        return self.inbox / "unlabeled"

    @property
    def processed(self) -> Path:
        return self.inbox / "processed"

    @property
    def rejected(self) -> Path:
        return self.inbox / "rejected"

    @property
    def pool(self) -> Path:
        return self.root / "pool"

    @property
    def pool_images(self) -> Path:
        return self.pool / "images"

    @property
    def pool_labels(self) -> Path:
        return self.pool / "labels"

    @property
    def pool_masks(self) -> Path:
        return self.pool / "masks"

    @property
    def manifest_path(self) -> Path:
        return self.pool / "manifest.csv"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def active(self) -> Path:
        return self.root / "active"

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.yaml"

    def directories(self) -> tuple[Path, ...]:
        """Every directory of the layout, parents before children."""
        return (self.root, self.inbox, *(self.inbox / name for name in INBOX_RESERVED),
                self.pool, self.pool_images, self.pool_labels, self.pool_masks,
                self.datasets, self.models, self.reports, self.figures, self.active)

    def ensure(self) -> Workspace:
        """Create the directory layout and, if absent, a config.yaml with the defaults.

        Idempotent: existing files (an edited config.yaml in particular) are never
        overwritten. Returns the workspace, so ``Workspace(p).ensure()`` can be chained."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self.save_config(SystemConfig())
        return self

    # ---- versions ----------------------------------------------------------------
    def dataset_dir(self, version: int) -> Path:
        """datasets/v{version} (not created)."""
        return self.datasets / f"v{check_version(version)}"

    def model_dir(self, version: int) -> Path:
        """models/v{version} (not created)."""
        return self.models / f"v{check_version(version)}"

    def next_dataset_version(self) -> int:
        """1 + the largest dataset version in use (a datasets/v{N} directory or a
        dataset_version referenced in the registry); 0 for an empty workspace."""
        from . import registry   # imported here because registry builds on this module
        used = set(version_dirs(self.datasets))
        used.update(rec.dataset_version for rec in registry.load(self)["versions"])
        return max(used, default=-1) + 1

    def next_model_version(self) -> int:
        """1 + the largest model version in use (a models/v{N} directory or a registered
        version); 0 for an empty workspace. Same rule as registry.next_version."""
        from . import registry   # imported here because registry builds on this module
        return registry.next_version(self)

    # ---- pool manifest -----------------------------------------------------------
    def read_manifest(self) -> pd.DataFrame:
        """The validated pool manifest (see validate_manifest); without a manifest file,
        an empty one with the required columns."""
        if not self.manifest_path.exists():
            return empty_manifest()
        df = pd.read_csv(self.manifest_path, dtype={c: str for c in _TEXT_COLUMNS},
                         keep_default_na=False)
        try:
            return validate_manifest(df)
        except ValueError as exc:
            raise ValueError(f"{self.manifest_path}: {exc}") from exc

    def write_manifest(self, df: pd.DataFrame, *, allow_test_change: bool = False) -> None:
        """Validate `df` (validate_manifest) and replace pool/manifest.csv atomically.

        Frozen-test protection: if a manifest exists, the set of names in the test split
        must stay exactly the same, otherwise ValueError. `allow_test_change=True` is
        meant only for (re-)initialising a workspace from the thesis dataset."""
        new = validate_manifest(df)
        if not allow_test_change and self.manifest_path.exists():
            old = self.read_manifest()
            old_test = set(old.loc[old["split"] == "test", "name"])
            new_test = set(new.loc[new["split"] == "test", "name"])
            if old_test != new_test:
                added, removed = sorted(new_test - old_test), sorted(old_test - new_test)
                raise ValueError(
                    f"the frozen test split would change ({len(added)} image(s) added, e.g. "
                    f"{added[:3]}; {len(removed)} removed, e.g. {removed[:3]}); the test set "
                    f"is fixed once written")
        atomic_write_text(self.manifest_path, new.to_csv(index=False, lineterminator="\n"))

    # ---- configuration -----------------------------------------------------------
    def load_config(self) -> SystemConfig:
        """config.yaml as SystemConfig; the defaults if the file does not exist."""
        if not self.config_path.exists():
            return SystemConfig()
        try:
            return SystemConfig.from_yaml(self.config_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"{self.config_path}: {exc}") from exc

    def save_config(self, cfg: SystemConfig) -> None:
        """Re-validate `cfg` (attributes may have been assigned after construction) and
        write config.yaml atomically."""
        if not isinstance(cfg, SystemConfig):
            raise TypeError(f"expected SystemConfig, got {type(cfg).__name__}")
        checked = SystemConfig.from_dict(cfg.to_dict())
        atomic_write_text(self.config_path, checked.to_yaml())
