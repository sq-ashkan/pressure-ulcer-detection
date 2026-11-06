"""Model registry (feature F7): every trained version, the current model, promote, rollback.

registry.json (SPEC §5)::

    {"current":  int | null,                 # the version that serves predictions
     "versions": [VersionRecord, ...],       # every version ever registered, ascending
     "history":  [{"ts", "action", "version", "previous"}, ...]}   # changes of "current"

Method
------
* Every version is kept, also those the gate did not promote, so the development of
  the system can be reported and any earlier model restored.
* "current" changes only through promote() -- the gate's decision -- or rollback() --
  a manual override that may restore any registered version. Each change is appended to
  "history" (UTC time, action, new and previous version), so it stays auditable which
  model served when. A version can only become current if its weights file exists.
* add_version() stores the version that is current at registration time as the new
  record's ``parent``: the model the gate compared the candidate with. Parent links give
  the version lineage shown in the model card.
* Every read-modify-write (add, promote, rollback, update) holds an exclusive advisory
  lock (fcntl.flock on registry.json.lock), so a cycle running in the dashboard's
  background thread and a CLI command cannot lose each other's change. The file is
  replaced atomically (temporary file, fsync, os.replace), so readers -- which take no
  lock -- always see a complete registry.
* Parent links must point to registered versions, so the lineage of every version can be
  followed back to the first model.
* registry.json is strict JSON: NaN/inf metrics are stored as null and numpy scalars as
  plain numbers, so a browser (JSON.parse) and Starlette's JSONResponse accept it as is.
* VersionRecord.metrics holds the headline numbers of the frozen-test evaluation under the
  keys of metrics_from_evaluation (the same keys the cycle's evaluate step stores).
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

from .workspace import (ARCHITECTURES, Workspace, atomic_write_json, check_version, json_safe,
                        utc_now_iso, version_dirs)

__all__ = [
    "ACTIONS", "Registry", "VersionRecord", "add_version", "current", "empty_registry",
    "get_version", "load", "metrics_from_evaluation", "next_version", "promote", "rollback",
    "save", "update_version",
]

LOCK_NAME = "registry.json.lock"
ACTIONS: tuple[str, ...] = ("promote", "rollback")
GATE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"promoted": False, "mean_diff": None, "ci95": None, "reason": ""})


def _count(value: Any, name: str) -> int:
    try:
        return check_version(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from None


def _path_text(value: Any, name: str, allow_empty: bool) -> str:
    text = os.fspath(value) if isinstance(value, (str, PathLike)) else None
    if not isinstance(text, str):
        raise ValueError(f"{name} must be a path, got {value!r}")
    if not text:
        if allow_empty:
            return ""
        raise ValueError(f"{name} must not be empty")
    return str(Path(text))


def _normalise_gate(gate: Any) -> dict[str, Any]:
    """SPEC gate keys first (promoted, mean_diff, ci95, reason) with their defaults, then
    any further keys of the gate step (e.g. p_value, n, margin, compared_to)."""
    if not isinstance(gate, Mapping):
        raise ValueError(f"gate must be a mapping, got {gate!r}")
    out = json_safe({**GATE_DEFAULTS, **gate})
    if not isinstance(out["promoted"], bool):
        raise ValueError(f"gate['promoted'] must be a boolean, got {gate['promoted']!r}")
    if not isinstance(out["reason"], str):
        raise ValueError(f"gate['reason'] must be a string, got {gate['reason']!r}")
    return out


@dataclass(kw_only=True)
class VersionRecord:
    """One trained model version (SPEC §5).

    version            model version N (weights and evaluation live in models/v{N})
    created            registration time, ISO-8601 UTC
    dataset_version    dataset version the model was trained on (datasets/v{D})
    architecture       "unet" or "yoloseg"
    train_images       images the model was trained on
    train_photographs  distinct photographs among those images
    metrics            frozen-test summary (keys of metrics_from_evaluation): test_dice_mean,
                       test_dice_ci95, test_fpr(_ci95), stage_dice_macro, ...
    gate               {"promoted": bool, "mean_diff", "ci95", "reason", ...} decision of
                       the gate at registration (candidate - parent)
    weights            weights file (absolute, or relative to the workspace root)
    model_card         model_card.md of this version ("" until written)
    parent             version that was current at registration (the gate's baseline);
                       None for the first model

    Construction validates every field and normalises metrics/gate to strict JSON values
    (NaN -> None), so a record equals itself after a save/load round trip."""

    version: int
    created: str = field(default_factory=utc_now_iso)
    dataset_version: int
    architecture: str
    train_images: int
    train_photographs: int
    metrics: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    weights: str
    model_card: str = ""
    parent: int | None = None

    def __post_init__(self) -> None:
        self.version = _count(self.version, "version")
        self.dataset_version = _count(self.dataset_version, "dataset_version")
        self.train_images = _count(self.train_images, "train_images")
        self.train_photographs = _count(self.train_photographs, "train_photographs")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {list(ARCHITECTURES)}, "
                             f"got {self.architecture!r}")
        if not isinstance(self.created, str):
            raise ValueError(f"created must be an ISO-8601 string, got {self.created!r}")
        try:
            datetime.fromisoformat(self.created)
        except ValueError:
            raise ValueError(f"created must be an ISO-8601 time, got {self.created!r}") from None
        if not isinstance(self.metrics, Mapping):
            raise ValueError(f"metrics must be a mapping, got {self.metrics!r}")
        self.metrics = json_safe(self.metrics)
        self.gate = _normalise_gate(self.gate)
        self.weights = _path_text(self.weights, "weights", allow_empty=False)
        self.model_card = _path_text(self.model_card, "model_card", allow_empty=True)
        if self.parent is not None:
            self.parent = _count(self.parent, "parent")
            if self.parent == self.version:
                raise ValueError(f"version {self.version} cannot be its own parent")

    @property
    def promoted(self) -> bool:
        """The gate's decision for this version (not whether it is current now)."""
        return bool(self.gate["promoted"])

    def weights_path(self, ws: Workspace) -> Path:
        """The weights file as an absolute path (relative paths are taken from ws.root)."""
        path = Path(self.weights)
        return path if path.is_absolute() else ws.root / path

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in field order."""
        return json_safe({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VersionRecord:
        """Inverse of to_dict; unknown or missing keys raise ValueError."""
        if not isinstance(data, Mapping):
            raise ValueError(f"a version record must be a mapping, got {data!r}")
        names = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls)
                    if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
        unknown = sorted(str(k) for k in data if k not in names)
        missing = sorted(required - set(data))
        if unknown or missing:
            raise ValueError(f"invalid version record: unknown key(s) {unknown}, "
                             f"missing key(s) {missing}")
        return cls(**dict(data))


class Registry(TypedDict):
    """In-memory form of registry.json (versions as VersionRecord, ascending)."""
    current: int | None
    versions: list[VersionRecord]
    history: list[dict[str, Any]]


def empty_registry() -> Registry:
    """A registry without versions and without a current model."""
    return {"current": None, "versions": [], "history": []}


def _validated(reg: Mapping[str, Any]) -> Registry:
    """Check a registry and return a normalised copy (records parsed, versions sorted)."""
    if not isinstance(reg, Mapping):
        raise ValueError(f"registry must be a mapping, got {type(reg).__name__}")
    unknown = sorted(str(k) for k in reg if k not in ("current", "versions", "history"))
    if unknown:
        raise ValueError(f"unknown registry key(s) {unknown}")
    raw_versions = reg.get("versions", [])
    if not isinstance(raw_versions, list):
        raise ValueError("registry 'versions' must be a list")
    records = [r if isinstance(r, VersionRecord) else VersionRecord.from_dict(r)
               for r in raw_versions]
    records.sort(key=lambda r: r.version)
    numbers = [r.version for r in records]
    duplicated = sorted({n for n in numbers if numbers.count(n) > 1})
    if duplicated:
        raise ValueError(f"registry lists version(s) {duplicated} more than once")
    cur = reg.get("current")
    if cur is not None:
        cur = _count(cur, "current")
        if cur not in numbers:
            raise ValueError(f"current version {cur} is not registered (versions: {numbers})")
    dangling = [(r.version, r.parent) for r in records
                if r.parent is not None and r.parent not in numbers]
    if dangling:
        raise ValueError(f"parent version(s) not registered (version, parent): {dangling}")
    history = reg.get("history", [])
    if not isinstance(history, list) or not all(isinstance(h, Mapping) for h in history):
        raise ValueError("registry 'history' must be a list of mappings")
    for entry in history:
        if entry.get("action") not in ACTIONS:
            raise ValueError(f"history entry has unknown action: {dict(entry)}")
    return {"current": cur, "versions": records, "history": [json_safe(h) for h in history]}


def _to_json(reg: Registry) -> dict[str, Any]:
    return {"current": reg["current"], "versions": [r.to_dict() for r in reg["versions"]],
            "history": reg["history"]}


def load(ws: Workspace) -> Registry:
    """Read registry.json (validated); an empty registry if the file does not exist."""
    path = ws.registry_path
    if not path.exists():
        return empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _validated(data)
    except (ValueError, TypeError) as exc:     # json.JSONDecodeError is a ValueError
        raise ValueError(f"{path}: invalid registry: {exc}") from exc


def _write(ws: Workspace, reg: Mapping[str, Any]) -> Registry:
    checked = _validated(reg)
    atomic_write_json(ws.registry_path, _to_json(checked))
    return checked


@contextmanager
def _locked(ws: Workspace) -> Iterator[None]:
    """Exclusive advisory lock for one read-modify-write of registry.json.

    flock locks belong to an open file description, so the lock also serialises threads
    of one process (each call opens the lock file anew). Not re-entrant: functions that
    hold it never call each other."""
    ws.root.mkdir(parents=True, exist_ok=True)
    with open(ws.root / LOCK_NAME, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _mutate(ws: Workspace, change: Callable[[Registry], bool]) -> Registry:
    """Load, apply `change` (returns True if it modified the registry), save if modified;
    all under the registry lock."""
    with _locked(ws):
        reg = load(ws)
        if change(reg):
            reg = _write(ws, reg)
        return reg


def save(ws: Workspace, reg: Mapping[str, Any]) -> None:
    """Validate and write a whole registry (versions may be VersionRecord or dicts).

    Prefer add_version/promote/rollback/update_version: a caller-side load + modify +
    save is not protected against a concurrent change made in between."""
    with _locked(ws):
        _write(ws, reg)


def _find(reg: Registry, version: int) -> VersionRecord:
    version = check_version(version)
    for rec in reg["versions"]:
        if rec.version == version:
            return rec
    raise KeyError(f"model version {version} is not registered "
                   f"(registered: {[r.version for r in reg['versions']]})")


def _set_current(ws: Workspace, reg: Registry, version: int, action: str) -> bool:
    rec = _find(reg, version)
    weights = rec.weights_path(ws)
    if not weights.is_file():
        raise FileNotFoundError(f"cannot make v{rec.version} current: weights file "
                                f"{weights} does not exist")
    if reg["current"] == rec.version:
        return False
    reg["history"].append({"ts": utc_now_iso(), "action": action, "version": rec.version,
                           "previous": reg["current"]})
    reg["current"] = rec.version
    return True


def add_version(ws: Workspace, rec: VersionRecord, *, make_current: bool = False) -> Registry:
    """Register a new version. Its number must not be registered yet. If rec.parent is
    None it is set to the version current at this moment (the gate's baseline). With
    `make_current` the version is also promoted, in the same locked write."""
    if not isinstance(rec, VersionRecord):
        raise TypeError(f"expected VersionRecord, got {type(rec).__name__}")

    def change(reg: Registry) -> bool:
        if any(r.version == rec.version for r in reg["versions"]):
            raise ValueError(f"model version {rec.version} is already registered")
        new = rec
        if new.parent is None and reg["current"] is not None:
            new = dataclasses.replace(rec, parent=reg["current"])
        reg["versions"].append(new)
        if make_current:
            _set_current(ws, reg, new.version, "promote")
        return True

    return _mutate(ws, change)


def promote(ws: Workspace, version: int) -> Registry:
    """Make `version` current because the gate accepted it (recorded as "promote").
    Promoting the current version changes nothing."""
    return _mutate(ws, lambda reg: _set_current(ws, reg, version, "promote"))


def rollback(ws: Workspace, version: int) -> Registry:
    """Manual override: make any registered `version` current again (recorded as
    "rollback"), e.g. after a promoted model turned out to be worse in practice.
    Rolling back to the current version changes nothing."""
    return _mutate(ws, lambda reg: _set_current(ws, reg, version, "rollback"))


def current(ws: Workspace) -> VersionRecord | None:
    """The record of the current (serving) model, or None if no model is current."""
    reg = load(ws)
    return None if reg["current"] is None else _find(reg, reg["current"])


def get_version(ws: Workspace, version: int) -> VersionRecord:
    """The record of `version`; KeyError if it is not registered."""
    return _find(load(ws), version)


def update_version(ws: Workspace, version: int, /, **changes: Any) -> VersionRecord:
    """Replace fields of a registered record (validated like a new record), e.g.
    ``model_card=...`` once the card is written. The version number cannot change:
    `ws` and `version` are positional-only, so ``version=...`` arrives among the
    changes and is refused with ValueError."""
    if "version" in changes:
        raise ValueError("the version number of a registered record cannot change")
    updated: list[VersionRecord] = []

    def change(reg: Registry) -> bool:
        old = _find(reg, version)
        new = dataclasses.replace(old, **changes)
        reg["versions"] = [new if r.version == old.version else r for r in reg["versions"]]
        updated.append(new)
        return new != old

    _mutate(ws, change)
    return updated[0]


def next_version(ws: Workspace) -> int:
    """1 + the largest model version in use (registered, or a models/v{N} directory);
    0 for an empty workspace. Numbers of failed or deleted versions are not reused."""
    used = set(version_dirs(ws.models))
    used.update(r.version for r in load(ws)["versions"])
    return max(used, default=-1) + 1


def metrics_from_evaluation(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Headline metrics of a selftrain.evaluate.full_evaluation result, in the form stored
    as VersionRecord.metrics. The keys are those of the cycle's evaluate step
    (steps.summary_metrics), so every record carries one format:

    threshold                         operating threshold, chosen on the validation split
    test_dice_mean, test_dice_ci95    mean per-photograph binary Dice over the
                                      ulcer-containing frozen-test photographs and its
                                      percentile-bootstrap 95 % CI
    test_iou_mean                     mean per-photograph binary IoU, same photographs
    test_fpr, test_fpr_ci95           share of ulcer-free test photographs with a predicted
                                      ulcer and its Wilson 95 % CI
    stage_dice_macro                  macro mean of the per-stage Dice (pooled pixels)
    n_test_positive, n_test_negative  ulcer-containing / ulcer-free test photographs
    inference_ms_per_image            mean inference time per test photograph
    extra_negatives_fpr(_ci95)        false-positive rate on all extra ulcer-free
                                      photographs (only if they were evaluated)
    external_dice_mean(_ci95)         Dice on the external split (only if it exists)

    Non-finite numbers become None and tuples lists (strict JSON, see json_safe)."""
    test = evaluation.get("test")
    if not isinstance(test, Mapping):
        raise ValueError("evaluation has no 'test' results; the frozen test split is required")
    out: dict[str, Any] = {
        "threshold": evaluation.get("threshold"),
        "test_dice_mean": test.get("dice_mean"), "test_dice_ci95": test.get("dice_ci95"),
        "test_iou_mean": test.get("iou_mean"),
        "test_fpr": test.get("false_positive_rate"),
        "test_fpr_ci95": test.get("false_positive_rate_ci95"),
        "stage_dice_macro": test.get("stage_dice_macro"),
        "n_test_positive": test.get("n_positive"), "n_test_negative": test.get("n_negative"),
        "inference_ms_per_image": evaluation.get("inference_ms_per_image"),
    }
    extra = evaluation.get("extra_negatives")
    if isinstance(extra, Mapping) and isinstance(extra.get("all"), Mapping):
        out["extra_negatives_fpr"] = extra["all"].get("false_positive_rate")
        out["extra_negatives_fpr_ci95"] = extra["all"].get("ci95")
    external = evaluation.get("external")
    if isinstance(external, Mapping):
        out["external_dice_mean"] = external.get("dice_mean")
        out["external_dice_ci95"] = external.get("dice_ci95")
    return json_safe(out)
