"""Ingest of new data batches (SPEC §2, feature F1).

A batch is one entry of ``inbox/``: a folder or a ``.zip`` archive. Ingest reads it into
Records -- image path, stage polygons, source, licence, batch id -- for quality control,
deduplication and the dataset step, and files the batch away::

    inbox/processed/<batch>/   the batch content and ingest_report.json
    inbox/unlabeled/<batch>/   photos without a label (active learning, SPEC §7)
    inbox/rejected/<batch>/    a batch that cannot be read as a whole, with the reason

Method
------
Batches (scan_inbox). The top-level entries of inbox/ in name order, except the
reserved folders (processed, rejected, unlabeled), hidden entries (e.g. the dashboard's
".upload-*" staging) and folders without a visible file -- the rule of
cycle.inbox_signature, so every entry that starts a cycle is consumed by it. The batch
id is the folder name, or the archive name without ".zip", with every run of
characters outside [A-Za-z0-9._-] replaced by "_"; it is made unique, ignoring case,
against processed/, rejected/, unlabeled/ and the earlier batches of the same scan by
appending _2, _3, ... The id names the batch's folder below processed/ (or rejected/)
and is Record.batch.

Archives. A .zip is rejected as a whole, before anything is extracted, if a member path
is absolute (leading "/" or a drive letter), has a ".." component or a control
character, if a member is a symbolic link or encrypted, if two members name the same
file ignoring case, or if it has more than MAX_ZIP_MEMBERS members or unpacks to more
than MAX_ZIP_BYTES. Hidden members and "__MACOSX" metadata are skipped. The members are
written into a fresh hidden staging folder, each file created exclusively (never
through or over an existing file), so nothing can be written outside that folder.

Layout. Wrapper folders are skipped: while a folder holds exactly one visible entry and
that entry is a folder not named like a layout folder (images, labels, masks, train,
valid, val, test), ingest descends into it. The folder reached is the batch root, where
source.yaml and data.yaml are read. The root and each of its split folders train/,
valid/, val/, test/ (a Roboflow export, any subset) is a *unit* if it has an images/
folder (images below it, recursively) or image files directly in it (flat layout,
labels beside the images). The label of an image is the first one found of
  1. YOLO segmentation  labels/<path relative to images/>.txt (flat layout: beside the
                        image); one region per line "class x1 y1 x2 y2 ...", normalised.
  2. COCO               annotations.json or _annotations.coco.json in the unit: the image
                        entry whose file_name is the image's path relative to images/
                        (or to the unit); polygon segmentations in pixels, divided by
                        the entry's width and height.
  3. mask               masks/<path relative to images/>.png with 0 = background and
                        1-4 = stage 1-4. Every connected region of one value becomes one
                        polygon: its outer contour (cv2.findContours, RETR_EXTERNAL)
                        simplified by approxPolyDP with a tolerance of 1 px; contours
                        under 25 px^2 are removed as specks (both as
                        build_seg_dataset.collect_dekubitus). Holes are filled -- YOLO
                        polygons, the label format of the pool, cannot represent them --
                        while a nested region of another value keeps its stage because
                        build_seg_dataset.rasterise draws larger outlines first.
An empty label (empty .txt, COCO image entry without annotations, all-zero mask) marks
an ulcer-free photograph. An image without any label is an ulcer-free photograph only
if source.yaml says ``negatives: true``; otherwise it is moved to
inbox/unlabeled/<batch>/ (its path below the batch root kept) and yields no Record.
Symbolic links inside a batch are never followed (they are reported and ignored), so a
batch cannot make the system read files outside it.

Classes. source.yaml (or source.yml) at the batch root, every key optional, unknown keys
rejected: ``name`` (the source; default: the batch id), ``license`` (default: the licence
of a known thesis source, workspace.KNOWN_SOURCES, else "not stated by the source"),
``negatives`` (true/false, default false) and ``class_map`` {source class: 1-4 |
background | drop}. A source class is a YOLO class id, a mask value or a COCO category
id (integer keys) or a COCO category name (text keys). Stage s becomes the stage index
s - 1 of the Record; a "background" region is ignored (it is background); "drop"
excludes the whole image, as build_seg_dataset does for DTI and unstageable wounds.
Without class_map the conventions of the formats apply: YOLO class c is stage c + 1
(SPEC §2), mask value v is stage v, a COCO category whose name reads "stage s" is stage
s (stage_from_name). Because a wrong class map silently corrupts every label, a batch
with YOLO labels whose data.yaml names classes 0-3 other than "stage 1" ... "stage 4" is
rejected unless source.yaml has a class_map -- e.g. a Roboflow export with the classes
NonPU, Stage1, ..., Stage4.

Per-image problems never reject the batch. A region whose class the map does not cover
is kept with the stage code quality.unknown_stage(c) = -1 - c, so quality control
rejects the image as unknown_class naming c. An image whose label cannot be used is
returned with Record.issue, a reason of quality's vocabulary (quality.INGEST_CODES):
malformed_label (unparsable line, odd number of coordinates, RLE or missing
segmentation, invalid COCO image entry), unreadable_mask, size_mismatch (mask or COCO
size differs from the decoded image), dropped_class, box_label (a stage region given as
a detection box -- four YOLO numbers or a COCO bbox without segmentation;
build_seg_dataset turned such stray boxes into rectangles, the system rejects them).
Quality control rejects such a record with exactly that reason, so the quality and the
cycle report explain every image. Coordinates are passed on unclipped: whether they lie
within the 1 % tolerance is quality's decision.

Filing (load_batch). A folder is read in place, an archive in its staging folder;
nothing is changed until the whole batch has been read. The batch is then renamed to
inbox/processed/<id>/ (an archive is deleted once its content is there; its name, size
and SHA-256 are kept), photos without label are moved to inbox/unlabeled/<id>/, and
ingest_report.json lists every image with its label file, format, number of regions,
outcome and issue, and every file that was ignored. A batch that cannot be read as a
whole -- not a folder or .zip, an unsafe or unreadable archive, an invalid source.yaml,
annotations.json or data.yaml, no image, a class map contradicting the batch -- is
moved unchanged to inbox/rejected/<id>/ next to an ingest_report.json stating the
reason, and load_batch raises IngestError.
"""
from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml

from ..config import NUM_STAGES
from .quality import make_reason, unknown_stage
from .workspace import (INBOX_RESERVED, UNSTATED_LICENSE, Workspace, atomic_write_json,
                        license_for_source, utc_now_iso)

__all__ = [
    "BACKGROUND", "COCO_FILES", "DROP", "IMAGE_SUFFIXES", "INGEST_REPORT", "OUTCOMES",
    "SOURCE_KEYS", "Batch", "BatchPlan", "CocoFile", "ImageEntry", "IngestError", "Label",
    "Record", "SourceSpec", "Unit", "batch_id", "class_target", "coco_label",
    "declared_names", "default_target", "extract_zip", "find_root", "find_units",
    "ingest_report", "load_batch", "load_coco", "mask_contours", "mask_label",
    "parse_source", "plan_batch", "read_source", "scan_inbox", "stage_from_name",
    "unit_at", "yolo_label", "zip_members",
]

_LOG = logging.getLogger(__name__)
Log = Callable[[str], None]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
"""Image files of a batch (the upload types of the dashboard; all decoded by OpenCV)."""
SPLIT_DIRS: tuple[str, ...] = ("train", "valid", "val", "test")
"""Split folders of a Roboflow / Ultralytics export; each may be a unit."""
LAYOUT_DIRS = frozenset({"images", "labels", "masks", *SPLIT_DIRS})
SOURCE_FILES: tuple[str, ...] = ("source.yaml", "source.yml")
COCO_FILES: tuple[str, ...] = ("annotations.json", "_annotations.coco.json")
NAMES_FILE = "data.yaml"
INGEST_REPORT = "ingest_report.json"
SOURCE_KEYS: tuple[str, ...] = ("name", "license", "negatives", "class_map")
BACKGROUND = "background"
DROP = "drop"
OUTCOMES: tuple[str, ...] = ("labelled", "ulcer_free", "issue", "unlabeled")
"""Outcome of an image: a Record with regions, a Record without regions (ulcer-free),
a Record with Record.issue, or no Record (moved to inbox/unlabeled/)."""
MIN_CONTOUR_AREA = 25.0
"""Mask contours with a smaller area (px^2) are specks (build_seg_dataset.collect_dekubitus)."""
CONTOUR_EPSILON = 1.0
"""approxPolyDP tolerance in px for mask contours (build_seg_dataset.collect_dekubitus)."""
MAX_ZIP_MEMBERS = 200_000
MAX_ZIP_BYTES = 16 << 30
"""Zip-bomb guard: total uncompressed size of an archive (the dashboard's upload limit)."""
MAX_ID_LENGTH = 80
COPY_CHUNK = 1 << 20

_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_STAGE_NAME = re.compile(r"stage([1-4])")
_NOT_ALNUM = re.compile(r"[^a-z0-9]")
_SYMLINK = "symbolic link: not followed"

Target = int | str
"""Target of a source class: a stage index 0-3, BACKGROUND or DROP."""


class IngestError(RuntimeError):
    """A batch cannot be ingested; the message is the reason. load_batch raises it after
    moving the batch to inbox/rejected/, or leaves the batch in inbox/ if it could not
    even be claimed (it vanished, or its id was taken meanwhile)."""


# ------------------------------------------------------------------ data types

@dataclass(eq=False)
class Record:
    """One image of a batch, as quality control, dedupe, drift and the dataset step read
    it (SPEC §2).

    path      the image file (absolute; below inbox/processed/<batch>/)
    polygons  regions as (stage, vertices): stage 0-3 (= stage 1-4), or the negative code
              quality.unknown_stage(c) of a source class c that the class map does not
              cover; vertices an (N, 2) float32 array of normalised (x, y), not clipped.
              Empty for an ulcer-free photograph.
    source    source name (source.yaml ``name``)
    license   licence of the source (source.yaml ``license``)
    batch     batch id = the folder name below inbox/processed/
    issue     None for a usable label, else the reason ("<code>: <message>", a code of
              quality.INGEST_CODES) why ingest found the label unusable; quality control
              rejects the record with exactly this reason.
    Equality is identity (eq=False): records hold numpy arrays, and two reads of one
    file are two records."""
    path: Path
    polygons: list[tuple[int, np.ndarray]]
    source: str
    license: str
    batch: str
    issue: str | None = None


@dataclass(frozen=True)
class Batch:
    """One entry of inbox/ waiting to be ingested (scan_inbox).

    name  batch id (unique in the workspace; the folder below processed/ or rejected/)
    path  the inbox entry
    kind  "folder", "zip" or "other" (any other entry; load_batch rejects it)
    ws    the workspace whose inbox holds the entry"""
    name: str
    path: Path
    kind: str
    ws: Workspace = field(repr=False, compare=False)


@dataclass(frozen=True)
class SourceSpec:
    """source.yaml of a batch, validated, with the defaults of the module docstring.

    class_map is None when source.yaml has none (the formats' defaults apply); else it
    maps integer keys (YOLO class id, mask value, COCO category id) and text keys (COCO
    category names) to a stage index 0-3, BACKGROUND or DROP. `file` is the name of the
    file read, None without source.yaml."""
    name: str
    license: str
    negatives: bool = False
    class_map: Mapping[int | str, Target] | None = None
    file: str | None = None


@dataclass
class Label:
    """The label of one image as read: format ("yolo", "coco" or "mask"), regions
    (stage index or unknown_stage code, (N, 2) float32 normalised vertices), the issue
    that makes it unusable (None if usable) and, for masks, the number of contours
    removed as specks."""
    format: str
    polygons: list[tuple[int, np.ndarray]] = field(default_factory=list)
    issue: str | None = None
    specks: int = 0


@dataclass(frozen=True)
class Unit:
    """A folder of images with their labels (module docstring, Layout). `images` is the
    images/ folder, or `folder` itself in the flat layout; `labels` the YOLO label folder
    (`folder` itself in the flat layout without labels/)."""
    folder: Path
    images: Path
    flat: bool
    labels: Path | None
    masks: Path | None
    coco: Path | None


@dataclass
class CocoImage:
    """One image entry of a COCO file and its annotations; `issue` if the entry itself
    is unusable (the image is listed twice)."""
    file_name: str
    width: Any
    height: Any
    annotations: list[Any] = field(default_factory=list)
    issue: str | None = None


@dataclass
class CocoFile:
    """A COCO file of one unit: image entries by normalised file_name, categories
    (id -> name), their class-map targets (None = not covered) and the entries that
    were ignored as (what, reason)."""
    path: Path
    name: str
    images: dict[str, CocoImage]
    categories: dict[int, str]
    targets: dict[int, Target | None]
    ignored: list[tuple[str, str]] = field(default_factory=list)

    def find(self, *names: str) -> str | None:
        """The first of `names` (normalised file names) that has an image entry."""
        return next((n for n in names if n in self.images), None)


@dataclass
class ImageEntry:
    """What ingest decided for one image: its path and label file relative to the batch
    folder, the label read (None without label) and the outcome (OUTCOMES)."""
    rel: PurePosixPath
    label_file: PurePosixPath | None
    label: Label | None
    outcome: str


@dataclass
class BatchPlan:
    """A batch as read by plan_batch, before anything was moved. Paths are relative to
    the batch folder; `units` lists (unit folder, number of images); `ignored` lists
    (file, reason) of files that are not used."""
    root: PurePosixPath
    source: SourceSpec
    units: list[tuple[PurePosixPath, int]]
    images: list[ImageEntry]
    ignored: list[tuple[str, str]]


@dataclass
class _UnitIndex:
    """A unit with its image files and label files keyed by relative path (no suffix)."""
    unit: Unit
    images: list[Path]
    yolo: dict[str, Path]
    masks: dict[str, Path]
    coco: CocoFile | None


# ------------------------------------------------------------------ small helpers

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _one_line(value: Any, limit: int = 300) -> str:
    """str(value) on one line, at most `limit` characters."""
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _numbers(items: Sequence[Any]) -> str:
    return ", ".join(str(i) for i in items)


def _visible(name: str) -> bool:
    """False for hidden names (".DS_Store", ".git", staging folders) and "__MACOSX"."""
    return not name.startswith(".") and name != "__MACOSX"


def _entries(folder: Path) -> list[Path]:
    """Visible children of `folder`, sorted by name."""
    return sorted((p for p in folder.iterdir() if _visible(p.name)), key=lambda p: p.name)


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def _child(folder: Path, name: str, *, is_dir: bool) -> Path | None:
    """The visible child of `folder` called `name` -- an exact match preferred, else one
    differing only in case -- that is a real folder (is_dir) or file, never a link."""
    found = [p for p in _entries(folder) if p.name.lower() == name.lower()
             and not p.is_symlink() and (p.is_dir() if is_dir else p.is_file())]
    found.sort(key=lambda p: p.name != name)
    return found[0] if found else None


def _note_link(path: Path, ignored: dict[Any, str]) -> bool:
    """True, and `path` recorded in `ignored`, if `path` is a symbolic link."""
    if path.is_symlink():
        ignored.setdefault(path, _SYMLINK)
        return True
    return False


def _files(top: Path, recursive: bool, ignored: dict[Any, str]) -> list[Path]:
    """Visible regular files below `top` (recursive) or directly in it, sorted by their
    path relative to `top`. Symbolic links are not followed but recorded in `ignored`."""
    found: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(top):
            base = Path(dirpath)
            dirnames[:] = [d for d in dirnames if _visible(d) and not _note_link(base / d, ignored)]
            found += [base / f for f in filenames
                      if _visible(f) and not _note_link(base / f, ignored) and (base / f).is_file()]
    else:
        found = [p for p in _entries(top) if not _note_link(p, ignored) and p.is_file()]
    return sorted(found, key=lambda p: p.relative_to(top).as_posix())


def _keyed(top: Path, suffix: str, recursive: bool, ignored: dict[Any, str]) -> dict[str, Path]:
    """Files with `suffix` (any case) below `top`, keyed by their path relative to `top`
    without the suffix (posix) -- the key of the image they label."""
    return {p.relative_to(top).with_suffix("").as_posix(): p
            for p in _files(top, recursive, ignored) if p.suffix.lower() == suffix}


def _rel(path: Path, folder: Path) -> PurePosixPath:
    """`path` relative to the batch folder, as a posix path ("." for the folder)."""
    return PurePosixPath(*Path(path).relative_to(folder).parts)


def _move(src: Path, dst: Path) -> None:
    """Rename a file, link or folder to `dst`, which must not exist (never overwrites);
    across file systems shutil.move copies and deletes."""
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, "target exists", str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(src), str(dst))


def _file_digest(path: Path) -> dict[str, Any]:
    """Size in bytes and SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _decoded_hw(image: Path) -> tuple[int, int] | None:
    """(height, width) of an image as OpenCV decodes it for the pool (EXIF orientation
    applied); None if it cannot be decoded (quality control rejects it later)."""
    img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    return None if img is None else (int(img.shape[0]), int(img.shape[1]))


# ------------------------------------------------------------------ classes

def stage_from_name(name: Any) -> int | None:
    """Stage 1-4 that a class name denotes, else None: lower-cased and stripped of every
    character outside a-z and 0-9, the name must read "stage1" ... "stage4" ("Stage 1",
    "stage-1" and "STAGE_1" qualify; "NonPU", "DTI" and "State1" do not)."""
    match = _STAGE_NAME.fullmatch(_NOT_ALNUM.sub("", str(name).lower()))
    return int(match.group(1)) if match else None


def default_target(fmt: str, class_id: int, name: str | None = None) -> Target | None:
    """Target of a source class without class_map (the formats' conventions): YOLO
    class c -> stage index c (stage c + 1, SPEC §2); mask value v -> v - 1 (stage v);
    COCO category named "stage s" -> s - 1. None: not covered."""
    if fmt == "yolo":
        return class_id if 0 <= class_id < NUM_STAGES else None
    if fmt == "mask":
        return class_id - 1 if 1 <= class_id <= NUM_STAGES else None
    if fmt == "coco":
        stage = stage_from_name(name) if name is not None else None
        return None if stage is None else stage - 1
    raise ValueError(f"unknown label format {fmt!r}")


def _target_text(target: Target) -> str:
    return f"stage {target + 1}" if isinstance(target, int) else repr(target)


def class_target(spec: SourceSpec, fmt: str, class_id: int,
                 name: str | None = None) -> Target | None:
    """Target of source class `class_id` (with its `name`, COCO only) under the batch's
    class map, or the format default without one. A class matched by id and by name
    must have one target (IngestError otherwise). None: not covered."""
    if spec.class_map is None:
        return default_target(fmt, class_id, name)
    by_id = spec.class_map.get(class_id)
    by_name = spec.class_map.get(name) if name is not None else None
    if by_id is not None and by_name is not None and by_id != by_name:
        raise IngestError(f"class_map maps class {class_id} by its id to {_target_text(by_id)} "
                          f"and by its name {name!r} to {_target_text(by_name)}")
    return by_name if by_name is not None else by_id


def _stage(target: Target | None, class_id: int) -> int:
    """Stage code of a region: the stage index, or unknown_stage(c) if not covered."""
    return int(target) if target is not None else unknown_stage(class_id)


def _dropped_reason(file: str, classes: set[int]) -> str:
    return make_reason("dropped_class", f"{file}: class {_numbers(sorted(classes))} is mapped to "
                                        f"'drop' (outside the four-stage scope); the image is excluded")


# ------------------------------------------------------------------ source.yaml

def _text_value(value: Any, key: str, where: str) -> str | None:
    """A text field of source.yaml on one line; None if absent or empty."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IngestError(f"{where}: {key} must be text, got {value!r} (quote it)")
    text = " ".join(value.split())
    if _CONTROL.search(text):
        raise IngestError(f"{where}: {key} contains a control character")
    return text or None


def _class_key(key: Any, where: str) -> int | str:
    """A class_map key: a class id >= 0 (an integer or a text of digits) or a class name."""
    if isinstance(key, str):
        text = key.strip()
        if text.isascii() and text.isdigit():
            return int(text)
        if text:
            return text
    elif _is_int(key) and key >= 0:
        return int(key)
    raise IngestError(f"{where}: class_map key {key!r} must be a class id (an integer >= 0) "
                      f"or a class name")


def _class_value(value: Any, key: Any, where: str) -> Target:
    """A class_map value: stage 1-4 -> stage index 0-3; "background" or "drop"."""
    if _is_int(value) and 1 <= value <= NUM_STAGES:
        return int(value) - 1
    if isinstance(value, str) and value.strip().lower() in (BACKGROUND, DROP):
        return value.strip().lower()
    raise IngestError(f"{where}: class_map maps {key!r} to {value!r}; allowed: a stage "
                      f"1-{NUM_STAGES}, '{BACKGROUND}' or '{DROP}'")


def _class_map(raw: Any, where: str) -> dict[int | str, Target] | None:
    """The validated class_map of source.yaml; None if absent."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise IngestError(f"{where}: class_map must be a non-empty mapping "
                          f"{{source class: 1-{NUM_STAGES} | {BACKGROUND} | {DROP}}}")
    table: dict[int | str, Target] = {}
    for key, value in raw.items():
        k = _class_key(key, where)
        if k in table:
            raise IngestError(f"{where}: class_map lists the source class {k!r} twice")
        table[k] = _class_value(value, key, where)
    return table


def parse_source(data: Any, default_name: str, file: str | None = None) -> SourceSpec:
    """Validate the content of source.yaml (`data` as loaded by yaml.safe_load; None
    without the file) and fill in the defaults (module docstring, Classes)."""
    where = file or "source.yaml"
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise IngestError(f"{where} must be a mapping with the keys {', '.join(SOURCE_KEYS)}")
    unknown = sorted(str(k) for k in data if k not in SOURCE_KEYS)
    if unknown:
        raise IngestError(f"{where} has unknown key(s) {unknown}; allowed: {list(SOURCE_KEYS)}")
    name = _text_value(data.get("name"), "name", where) or default_name
    licence = (_text_value(data.get("license"), "license", where)
               or license_for_source(name) or UNSTATED_LICENSE)
    negatives = data.get("negatives")
    if negatives is not None and not isinstance(negatives, bool):
        raise IngestError(f"{where}: negatives must be true or false, got {negatives!r}")
    return SourceSpec(name, licence, bool(negatives), _class_map(data.get("class_map"), where), file)


def read_source(root: Path, default_name: str) -> SourceSpec:
    """source.yaml (or source.yml) at the batch root, validated (parse_source)."""
    found = [p for p in (_child(root, n, is_dir=False) for n in SOURCE_FILES) if p is not None]
    if len(found) > 1:
        raise IngestError("the batch has both source.yaml and source.yml; keep one")
    if not found:
        return parse_source(None, default_name)
    try:
        data = yaml.safe_load(found[0].read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise IngestError(f"{found[0].name} cannot be read: {_one_line(exc)}") from None
    return parse_source(data, default_name, found[0].name)


def declared_names(path: Path) -> dict[int, str]:
    """Class names declared by an Ultralytics / Roboflow data.yaml (`names` as a list or
    an {id: name} mapping); {} if it declares none."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise IngestError(f"{path.name} cannot be read: {_one_line(exc)}") from None
    names = data.get("names") if isinstance(data, Mapping) else None
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, Mapping):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            raise IngestError(f"{path.name}: names must map class ids to names") from None
    return {}


# ------------------------------------------------------------------ label formats

def _yolo_line(tokens: list[str]) -> tuple[int, list[float]] | str:
    """(class, coordinates) of one YOLO line, or why the line is malformed."""
    try:
        head = float(tokens[0])
        values = [float(t) for t in tokens[1:]]
    except ValueError:
        return "expected a class id followed by numbers"
    if not head.is_integer() or head < 0:
        return f"class {tokens[0]!r} is not an integer >= 0"
    if not values or len(values) % 2:
        return f"{len(values)} coordinate(s), expected x y pairs"
    return int(head), values


def yolo_label(text: str, file: str, spec: SourceSpec) -> Label:
    """Regions of one YOLO segmentation label (module docstring, Layout 1 and Classes).

    Blank lines are skipped. A line is "class x1 y1 ... xn yn": an integer class >= 0
    (an integral number such as "1.0" is accepted, as Ultralytics does) and an even,
    non-zero number of coordinates. The first malformed line makes the label
    malformed_label; otherwise a region of a class mapped to "drop" makes it
    dropped_class, and a line of four numbers (a detection box) of a class not mapped to
    background makes it box_label. Lines of background classes are ignored."""
    regions: list[tuple[int, np.ndarray]] = []
    dropped: set[int] = set()
    boxes: list[int] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        parsed = _yolo_line(tokens)
        if isinstance(parsed, str):
            return Label("yolo", issue=make_reason("malformed_label", f"{file} line {lineno}: {parsed}"))
        cls, values = parsed
        target = class_target(spec, "yolo", cls)
        if target == BACKGROUND:
            continue
        if target == DROP:
            dropped.add(cls)
        elif len(values) == 4:
            boxes.append(lineno)
        else:
            regions.append((_stage(target, cls), np.asarray(values, np.float32).reshape(-1, 2)))
    if dropped:
        return Label("yolo", issue=_dropped_reason(file, dropped))
    if boxes:
        return Label("yolo", issue=make_reason(
            "box_label", f"{file} line(s) {_numbers(boxes)}: four numbers form a detection box, "
                         f"not an outline; export instance-segmentation polygons"))
    return Label("yolo", regions)


def _single_channel(raw: np.ndarray) -> np.ndarray | None:
    """A mask as one channel: grey as is; colour (alpha ignored) only if all colour
    channels are equal; None otherwise."""
    if raw.ndim == 2:
        return raw
    if raw.ndim == 3 and raw.shape[2] in (3, 4):
        colour = raw[..., :3]
        if (colour == colour[..., :1]).all():
            return np.ascontiguousarray(colour[..., 0])
    return None


def mask_contours(binary: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Polygons (pixel coordinates, float32 (N, 2)) of the connected regions of a binary
    mask and the number of specks removed (module docstring, Layout 3)."""
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    outlines: list[np.ndarray] = []
    specks = 0
    for contour in contours:
        pts = cv2.approxPolyDP(contour, CONTOUR_EPSILON, True).reshape(-1, 2).astype(np.float32)
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA or len(pts) < 3:
            specks += 1
        else:
            outlines.append(pts)
    return outlines, specks


def mask_label(path: Path, file: str, image_hw: tuple[int, int] | None,
               spec: SourceSpec) -> Label:
    """Regions of one PNG mask (module docstring, Layout 3 and Classes).

    unreadable_mask if the file cannot be decoded or is a colour image; size_mismatch if
    its size differs from the decoded image (`image_hw`, None if the image cannot be
    decoded: then quality control rejects the image); dropped_class if a value is
    mapped to "drop". Value 0 is always background. Vertices are normalised by the mask
    size: x / width, y / height."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return Label("mask", issue=make_reason("unreadable_mask", f"{file} cannot be decoded"))
    mask = _single_channel(raw)
    if mask is None:
        return Label("mask", issue=make_reason(
            "unreadable_mask", f"{file} is a colour image {tuple(raw.shape)}; expected one "
                               f"channel with the values 0-{NUM_STAGES}"))
    height, width = mask.shape
    if image_hw is not None and image_hw != (height, width):
        return Label("mask", issue=make_reason(
            "size_mismatch", f"{file} is {width} x {height} px, the image "
                             f"{image_hw[1]} x {image_hw[0]} px"))
    values = [int(v) for v in np.unique(mask) if v != 0]
    targets = {v: class_target(spec, "mask", v) for v in values}
    dropped = {v for v, t in targets.items() if t == DROP}
    if dropped:
        return Label("mask", issue=_dropped_reason(file, dropped))
    scale = np.array([width, height], np.float32)
    regions: list[tuple[int, np.ndarray]] = []
    specks = 0
    for value in values:
        if targets[value] == BACKGROUND:
            continue
        outlines, removed = mask_contours(mask == value)
        specks += removed
        regions += [(_stage(targets[value], value), (pts / scale).astype(np.float32))
                    for pts in outlines]
    return Label("mask", regions, specks=specks)


def _coco_key(file_name: str) -> str | None:
    """A COCO file_name as a normalised relative posix path; None if it is absolute or
    has a ".." component (it could name a file outside the batch)."""
    name = file_name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if name.startswith("/") or not parts or ".." in parts or ":" in parts[0]:
        return None
    return "/".join(parts)


def _coco_categories(items: list[Any], file: str) -> dict[int, str]:
    """Category id -> name; every category needs a unique integer id >= 0 and a name."""
    categories: dict[int, str] = {}
    for item in items:
        if (not isinstance(item, dict) or not _is_int(item.get("id")) or item["id"] < 0
                or not isinstance(item.get("name"), str)):
            raise IngestError(f"{file}: every category needs an integer id >= 0 and a name, "
                              f"got {_one_line(item, 80)}")
        if item["id"] in categories:
            raise IngestError(f"{file}: category id {item['id']} is used twice")
        categories[item["id"]] = item["name"]
    return categories


def load_coco(path: Path, file: str, spec: SourceSpec) -> CocoFile:
    """Read a COCO instance-segmentation file of one unit.

    The file as a whole must be valid -- a JSON object whose images, annotations and
    categories are lists; categories with unique integer ids >= 0 and names; image
    entries with unique integer ids and a file_name -- otherwise IngestError (the batch
    is rejected). Problems of single entries concern only their image: an image listed
    twice gets a malformed_label issue, an unsafe file_name and an annotation of an
    unknown image are ignored and reported."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise IngestError(f"{file} is not valid JSON: {_one_line(exc)}") from None
    if not isinstance(data, dict):
        raise IngestError(f"{file} must be a JSON object with images, annotations and categories")
    lists: dict[str, list[Any]] = {}
    for key in ("images", "annotations", "categories"):
        value = data.get(key, [])
        if not isinstance(value, list):
            raise IngestError(f"{file}: {key!r} must be a list")
        lists[key] = value
    categories = _coco_categories(lists["categories"], file)
    targets = {cid: class_target(spec, "coco", cid, name) for cid, name in categories.items()}
    coco = CocoFile(path, file, {}, categories, targets)
    by_id: dict[int, CocoImage | None] = {}
    for item in lists["images"]:
        if not isinstance(item, dict) or not _is_int(item.get("id")) \
                or not isinstance(item.get("file_name"), str):
            raise IngestError(f"{file}: every image entry needs an integer id and a file_name, "
                              f"got {_one_line(item, 80)}")
        if item["id"] in by_id:
            raise IngestError(f"{file}: image id {item['id']} is used twice")
        key = _coco_key(item["file_name"])
        if key is None:
            coco.ignored.append((f"{file}: {item['file_name']}", "unsafe file_name (absolute or "
                                                                 "with '..'): not read"))
            by_id[item["id"]] = None
        elif key in coco.images:
            entry = coco.images[key]
            entry.issue = make_reason("malformed_label", f"{file} lists {key} twice")
            by_id[item["id"]] = entry
        else:
            entry = CocoImage(item["file_name"], item.get("width"), item.get("height"))
            coco.images[key] = by_id[item["id"]] = entry
    for ann in lists["annotations"]:
        image_id = ann.get("image_id") if isinstance(ann, dict) else None
        if not _is_int(image_id) or image_id not in by_id:
            what = ann.get("id") if isinstance(ann, dict) else _one_line(ann, 40)
            coco.ignored.append((f"{file}: annotation {what}", "refers to no image entry"))
            continue
        target_entry = by_id[image_id]
        if target_entry is not None:
            target_entry.annotations.append(ann)
    return coco


def _coco_polygons(segmentation: Any) -> list[np.ndarray] | None:
    """Polygons of a COCO segmentation [[x1, y1, x2, y2, ...], ...] as float64 (N, 2)
    pixel arrays; None if it is not a list of non-empty lists of an even count of numbers."""
    if not isinstance(segmentation, list):
        return None
    polygons = []
    for poly in segmentation:
        if not isinstance(poly, list) or not poly or len(poly) % 2 \
                or not all(_is_number(v) for v in poly):
            return None
        polygons.append(np.asarray(poly, np.float64).reshape(-1, 2))
    return polygons


def coco_label(entry: CocoImage, coco: CocoFile, image_hw: tuple[int, int] | None,
               spec: SourceSpec) -> Label:
    """Regions of one COCO image entry (module docstring, Layout 2 and Classes).

    malformed_label if the entry lacks a positive integer width and height, or an
    annotation has no valid category_id, an RLE segmentation (iscrowd), or neither
    segmentation nor bbox; size_mismatch if width x height differs from the decoded
    image; dropped_class for a category mapped to "drop"; box_label for an annotation
    of a non-background category with a bbox but no segmentation. Vertices are
    normalised as x / width, y / height of the entry."""
    file = coco.name

    def malformed(message: str) -> Label:
        """A label with a malformed_label issue naming the COCO file."""
        return Label("coco", issue=make_reason("malformed_label", f"{file}: {message}"))

    if entry.issue is not None:
        return Label("coco", issue=entry.issue)
    width, height = entry.width, entry.height
    if not (_is_int(width) and _is_int(height) and width > 0 and height > 0):
        return malformed(f"the entry of {entry.file_name} needs integer width and height > 0")
    if image_hw is not None and image_hw != (height, width):
        return Label("coco", issue=make_reason(
            "size_mismatch", f"{file} gives {entry.file_name} as {width} x {height} px, the "
                             f"image is {image_hw[1]} x {image_hw[0]} px"))
    scale = np.array([width, height], np.float64)
    regions: list[tuple[int, np.ndarray]] = []
    dropped: set[int] = set()
    boxes: list[Any] = []
    for ann in entry.annotations:
        cid, ann_id = ann.get("category_id"), ann.get("id")
        if not _is_int(cid) or cid < 0:
            return malformed(f"annotation {ann_id} has no valid category_id")
        target = coco.targets[cid] if cid in coco.targets else class_target(spec, "coco", cid)
        if target == BACKGROUND:
            continue
        if target == DROP:
            dropped.add(cid)
            continue
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, dict):
            return malformed(f"annotation {ann_id} is run-length encoded (RLE); export polygons")
        if segmentation is None or segmentation == []:
            if ann.get("bbox") is None:
                return malformed(f"annotation {ann_id} has no segmentation")
            boxes.append(ann_id)
            continue
        polygons = _coco_polygons(segmentation)
        if polygons is None:
            return malformed(f"annotation {ann_id}: segmentation must be a list of polygons "
                             f"[x1, y1, x2, y2, ...] with an even number of coordinates")
        regions += [(_stage(target, cid), (p / scale).astype(np.float32)) for p in polygons]
    if dropped:
        return Label("coco", issue=_dropped_reason(file, dropped))
    if boxes:
        return Label("coco", issue=make_reason(
            "box_label", f"{file}: annotation(s) {_numbers(boxes)} of {entry.file_name} give "
                         f"only a bounding box, not an outline"))
    return Label("coco", regions)


# ------------------------------------------------------------------ archives

def zip_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """The members of an archive to extract, with their safe relative paths; IngestError
    if the archive is unsafe (module docstring, Archives). Folders, hidden members and
    "__MACOSX" metadata are left out."""
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise IngestError(f"the archive has {len(infos)} members; at most {MAX_ZIP_MEMBERS} "
                          f"are accepted")
    members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: dict[str, str] = {}
    total = 0
    for info in infos:
        raw = info.filename
        name = raw.replace("\\", "/")
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if _CONTROL.search(name):
            raise IngestError(f"zip member {raw!r} has a control character in its name")
        if name.startswith("/") or (parts and ":" in parts[0]):
            raise IngestError(f"zip member {raw!r} has an absolute path")
        if ".." in parts:
            raise IngestError(f"zip member {raw!r} has a '..' component (path traversal)")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise IngestError(f"zip member {raw!r} is a symbolic link")
        if info.flag_bits & 0x1:
            raise IngestError(f"zip member {raw!r} is encrypted")
        if info.is_dir() or not parts or not all(_visible(p) for p in parts):
            continue
        key = "/".join(parts).casefold()
        if key in seen:
            raise IngestError(f"zip members {seen[key]!r} and {raw!r} name the same file")
        seen[key] = raw
        total += info.file_size
        members.append((info, PurePosixPath(*parts)))
    if total > MAX_ZIP_BYTES:
        raise IngestError(f"the archive unpacks to {total} bytes; at most {MAX_ZIP_BYTES} "
                          f"are accepted")
    return members


def extract_zip(path: Path, target: Path) -> dict[str, int]:
    """Extract the archive `path` into the empty folder `target` after zip_members has
    checked every member; each file is created exclusively below `target`. Returns
    {members, unpacked_bytes}; IngestError if the archive is unsafe or unreadable."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = zip_members(archive)
            for info, rel in members:
                dest = target.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, open(dest, "xb") as dst:
                    shutil.copyfileobj(src, dst, COPY_CHUNK)
    except IngestError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError,
            EOFError, OSError, ValueError) as exc:
        raise IngestError(f"the archive cannot be extracted ({type(exc).__name__}: "
                          f"{_one_line(exc)})") from None
    return {"members": len(members), "unpacked_bytes": sum(i.file_size for i, _ in members)}


# ------------------------------------------------------------------ layout

def find_root(folder: Path) -> Path:
    """The batch root: descend while the folder holds exactly one visible entry that is
    a folder (not a link) whose name is not a layout folder (wrapper folders)."""
    root = folder
    while True:
        entries = _entries(root)
        if len(entries) != 1:
            return root
        only = entries[0]
        if only.is_symlink() or not only.is_dir() or only.name.lower() in LAYOUT_DIRS:
            return root
        root = only


def unit_at(folder: Path) -> Unit | None:
    """The unit rooted at `folder` (module docstring, Layout), or None if it has neither
    an images/ folder nor image files of its own."""
    images = _child(folder, "images", is_dir=True)
    labels = _child(folder, "labels", is_dir=True)
    masks = _child(folder, "masks", is_dir=True)
    coco = next((p for p in (_child(folder, n, is_dir=False) for n in COCO_FILES) if p), None)
    if images is not None:
        return Unit(folder, images, False, labels, masks, coco)
    if any(_is_image_name(p.name) and p.is_file() and not p.is_symlink() for p in _entries(folder)):
        return Unit(folder, folder, True, labels or folder, masks, coco)
    return None


def find_units(root: Path) -> list[Unit]:
    """The units of a batch: the root itself and its split folders train/, valid/,
    val/, test/, in this order, where they are units."""
    candidates = [root] + [d for d in (_child(root, s, is_dir=True) for s in SPLIT_DIRS) if d]
    return [u for u in (unit_at(c) for c in candidates) if u is not None]


def _index_unit(unit: Unit, folder: Path, spec: SourceSpec, ignored: dict[Any, str]) -> _UnitIndex:
    """List the images and label files of a unit and read its COCO file."""
    files = _files(unit.images, not unit.flat, ignored)
    images = [p for p in files if _is_image_name(p.name)]
    if not unit.flat:
        for p in files:
            if not _is_image_name(p.name):
                ignored.setdefault(p, "not an image file")
    yolo = _keyed(unit.labels, ".txt", unit.labels != unit.folder, ignored) if unit.labels else {}
    masks = _keyed(unit.masks, ".png", True, ignored) if unit.masks else {}
    coco = load_coco(unit.coco, _rel(unit.coco, folder).as_posix(), spec) if unit.coco else None
    return _UnitIndex(unit, images, yolo, masks, coco)


def _image_key(idx: _UnitIndex, image: Path) -> str:
    return image.relative_to(idx.unit.images).with_suffix("").as_posix()


def _check_name_keys(spec: SourceSpec, indexes: Sequence[_UnitIndex]) -> None:
    """Text keys of class_map must name a COCO category of the batch (typo guard)."""
    if spec.class_map is None:
        return
    names = {n for idx in indexes if idx.coco for n in idx.coco.categories.values()}
    unknown = sorted(k for k in spec.class_map if isinstance(k, str) and k not in names)
    if unknown:
        raise IngestError(f"class_map key(s) {unknown} name no COCO category of the batch "
                          f"(categories: {sorted(names) or 'none'}); YOLO classes and mask "
                          f"values are given as integers")


def _check_declared_names(root: Path, spec: SourceSpec, indexes: Sequence[_UnitIndex]) -> None:
    """Without class_map, YOLO class c is read as stage c + 1; refuse the batch if its
    data.yaml names one of the classes 0-3 differently (module docstring, Classes)."""
    if spec.class_map is not None:
        return
    if not any(_image_key(idx, im) in idx.yolo for idx in indexes for im in idx.images):
        return
    path = _child(root, NAMES_FILE, is_dir=False)
    if path is None:
        return
    bad = [(c, n) for c, n in sorted(declared_names(path).items())
           if c < NUM_STAGES and stage_from_name(n) != c + 1]
    if bad:
        listed = ", ".join(f"{c} as {n!r}" for c, n in bad)
        raise IngestError(f"{NAMES_FILE} names class {listed}, but without a class_map YOLO "
                          f"class c is read as stage c + 1; add a class_map to source.yaml "
                          f"(every class -> 1-{NUM_STAGES}, {BACKGROUND} or {DROP})")


def _outcome(label: Label | None, negatives: bool) -> str:
    if label is None:
        return "ulcer_free" if negatives else "unlabeled"
    if label.issue is not None:
        return "issue"
    return "labelled" if label.polygons else "ulcer_free"


def _plan_unit(idx: _UnitIndex, folder: Path, spec: SourceSpec,
               ignored: dict[Any, str]) -> list[ImageEntry]:
    """Read the label of every image of a unit (first of YOLO, COCO, mask) and record
    the label files that were not used."""
    unit, coco = idx.unit, idx.coco
    entries: list[ImageEntry] = []
    used: set[Any] = set()
    image_keys: set[str] = set()
    matched_coco: set[str] = set()
    for image in idx.images:
        key = _image_key(idx, image)
        image_keys.add(key)
        coco_key = None
        if coco is not None:
            coco_key = coco.find(image.relative_to(unit.images).as_posix(),
                                 image.relative_to(unit.folder).as_posix())
            if coco_key is not None:
                matched_coco.add(coco_key)
        label: Label | None = None
        label_file: Path | None = None
        if key in idx.yolo:
            label_file = idx.yolo[key]
            label = _read_yolo(label_file, folder, spec)
        elif coco is not None and coco_key is not None:
            label_file = coco.path
            label = coco_label(coco.images[coco_key], coco, _decoded_hw(image), spec)
            used.add(("coco", coco_key))
        elif key in idx.masks:
            label_file = idx.masks[key]
            label = mask_label(label_file, _rel(label_file, folder).as_posix(),
                               _decoded_hw(image), spec)
        used.add(label_file)
        entries.append(ImageEntry(_rel(image, folder),
                                  _rel(label_file, folder) if label_file else None,
                                  label, _outcome(label, spec.negatives)))
    if unit.labels is not None and unit.labels != unit.folder:
        for path in idx.yolo.values():
            if path not in used:
                ignored.setdefault(path, "label file without image")
    for key, path in idx.masks.items():
        if path not in used:
            ignored.setdefault(path, "superseded by the YOLO or COCO label of the image"
                               if key in image_keys else "mask without image")
    if coco is not None:
        for key in coco.images:
            if ("coco", key) not in used:
                ignored.setdefault(f"{coco.name}: {key}", "superseded by the YOLO label of "
                                   "the image" if key in matched_coco else "no such image")
        for what, why in coco.ignored:
            ignored.setdefault(what, why)
    return entries


def _read_yolo(path: Path, folder: Path, spec: SourceSpec) -> Label:
    file = _rel(path, folder).as_posix()
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        return Label("yolo", issue=make_reason("malformed_label", f"{file} is not UTF-8 text"))
    return yolo_label(text, file, spec)


def plan_batch(folder: Path, default_name: str) -> BatchPlan:
    """Read a batch folder without changing it (module docstring: Layout, Classes).

    Finds the batch root and its units, validates source.yaml (default source name:
    `default_name`, the batch id), reads every label and decides the outcome of every
    image. IngestError if the batch cannot be read as a whole."""
    root = find_root(folder)
    spec = read_source(root, default_name)
    ignored: dict[Any, str] = {}
    indexes = [_index_unit(u, folder, spec, ignored) for u in find_units(root)]
    if not any(idx.images for idx in indexes):
        raise IngestError("no image found: a batch needs images/ (with labels/, masks/ or "
                          "annotations.json), a Roboflow export (train/, valid/, test/) or "
                          "image files")
    _check_name_keys(spec, indexes)
    _check_declared_names(root, spec, indexes)
    entries: list[ImageEntry] = []
    for idx in indexes:
        entries += _plan_unit(idx, folder, spec, ignored)
    return BatchPlan(
        root=_rel(root, folder), source=spec,
        units=[(_rel(idx.unit.folder, folder), len(idx.images)) for idx in indexes],
        images=entries,
        ignored=[(_rel(k, folder).as_posix() if isinstance(k, Path) else str(k), why)
                 for k, why in ignored.items()])


# ------------------------------------------------------------------ inbox

def batch_id(entry_name: str, kind: str) -> str:
    """Batch id of an inbox entry before uniqueness: the folder name, or the archive name
    without ".zip", with every run of characters outside [A-Za-z0-9._-] replaced by
    "_", leading/trailing ".", "_", "-" removed, at most MAX_ID_LENGTH characters;
    "batch" if nothing is left."""
    stem = entry_name[:-4] if kind == "zip" and entry_name.lower().endswith(".zip") else entry_name
    text = _ID_UNSAFE.sub("_", stem).strip("._-")[:MAX_ID_LENGTH].strip("._-")
    return text or "batch"


def _unique(base: str, taken: set[str]) -> str:
    """`base`, or base_2, base_3, ...: the first not in `taken` (casefolded names)."""
    name, k = base, 2
    while name.casefold() in taken:
        name, k = f"{base}_{k}", k + 1
    return name


def _entry_kind(path: Path) -> str:
    if path.is_symlink():
        return "other"
    if path.is_dir():
        return "folder"
    return "zip" if path.is_file() and path.suffix.lower() == ".zip" else "other"


def _has_visible_file(folder: Path) -> bool:
    """Whether a file whose path below `folder` has no hidden component exists
    (cycle.inbox_signature counts exactly these files)."""
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if any(not f.startswith(".") and os.path.isfile(os.path.join(dirpath, f)) for f in filenames):
            return True
    return False


def _taken_ids(ws: Workspace) -> set[str]:
    """Casefolded names in use below processed/, rejected/ and unlabeled/."""
    return {p.name.casefold() for d in (ws.processed, ws.rejected, ws.unlabeled)
            if d.is_dir() for p in d.iterdir()}


def scan_inbox(ws: Workspace) -> list[Batch]:
    """The batches waiting in inbox/, in name order, with unique ids (module docstring,
    Batches). Reads the directory only; nothing is moved."""
    inbox = ws.inbox
    if not inbox.is_dir():
        return []
    taken = _taken_ids(ws)
    batches = []
    for entry in sorted(inbox.iterdir(), key=lambda p: p.name):
        if entry.name in INBOX_RESERVED or entry.name.startswith("."):
            continue
        kind = _entry_kind(entry)
        if kind == "folder" and not _has_visible_file(entry):
            continue
        name = _unique(batch_id(entry.name, kind), taken)
        taken.add(name.casefold())
        batches.append(Batch(name, entry, kind, ws))
    return batches


def _class_map_view(spec: SourceSpec) -> Any:
    """The class map as stated in source.yaml terms (stages 1-4), or the default rule."""
    if spec.class_map is None:
        return ("default: YOLO class c -> stage c+1, mask value v -> stage v, "
                "COCO category 'stage s' -> stage s")
    return {str(k): (v + 1 if isinstance(v, int) else v) for k, v in spec.class_map.items()}


def ingest_report(name: str, plan: BatchPlan, origin: Mapping[str, Any],
                  unlabeled: Sequence[str]) -> dict[str, Any]:
    """JSON-ready ingest report of a processed batch (inbox/processed/<batch>/
    ingest_report.json): origin, source, units, counts per outcome and label format,
    every image (path and label file relative to the batch folder) and every ignored file."""
    outcomes = Counter(e.outcome for e in plan.images)
    formats = Counter(e.label.format for e in plan.images if e.label is not None)
    spec = plan.source
    return {
        "batch": name,
        "status": "processed",
        "created": utc_now_iso(),
        "origin": dict(origin),
        "root": plan.root.as_posix(),
        "source": {"file": spec.file, "name": spec.name, "license": spec.license,
                   "negatives": spec.negatives, "class_map": _class_map_view(spec)},
        "units": [{"folder": f.as_posix(), "images": n} for f, n in plan.units],
        "counts": {"images": len(plan.images),
                   "records": len(plan.images) - outcomes["unlabeled"],
                   **{o: outcomes[o] for o in OUTCOMES},
                   "mask_specks_removed": sum(e.label.specks for e in plan.images if e.label)},
        "label_formats": dict(sorted(formats.items())),
        "unlabeled_folder": f"inbox/unlabeled/{name}" if unlabeled else None,
        "images": [{"image": e.rel.as_posix(),
                    "label_file": e.label_file.as_posix() if e.label_file else None,
                    "format": e.label.format if e.label else None,
                    "regions": len(e.label.polygons) if e.label else 0,
                    "outcome": e.outcome,
                    "issue": e.label.issue if e.label else None} for e in plan.images],
        "ignored": [{"file": f, "reason": why} for f, why in plan.ignored],
    }


def _reject(batch: Batch, reason: str, origin: Mapping[str, Any]) -> Path:
    """Move a batch unchanged to inbox/rejected/<id>/ (a folder becomes that folder, any
    other entry is placed inside it) and write the rejection report there."""
    dest = batch.ws.rejected / batch.name
    if batch.kind == "folder" and batch.path.is_dir() and not batch.path.is_symlink():
        _move(batch.path, dest)
    else:
        dest.mkdir(parents=True)
        if os.path.lexists(batch.path):
            _move(batch.path, dest / batch.path.name)
    atomic_write_json(dest / INGEST_REPORT, {"batch": batch.name, "status": "rejected",
                                             "reason": reason, "origin": dict(origin),
                                             "created": utc_now_iso()})
    return dest


def _move_unlabeled(plan: BatchPlan, target: Path, dest: Path) -> list[str]:
    """Move the images without label from the processed batch to `dest`, keeping their
    path below the batch root; returns these paths."""
    moved = []
    for entry in plan.images:
        if entry.outcome != "unlabeled":
            continue
        below_root = entry.rel.relative_to(plan.root)
        _move(target.joinpath(*entry.rel.parts), dest.joinpath(*below_root.parts))
        moved.append(below_root.as_posix())
    return moved


def _summary(name: str, plan: BatchPlan) -> str:
    """One log line per batch."""
    n = Counter(e.outcome for e in plan.images)
    text = (f"batch {name}: {len(plan.images)} images -> {len(plan.images) - n['unlabeled']} "
            f"records ({n['labelled']} labelled, {n['ulcer_free']} ulcer-free, {n['issue']} with "
            f"an unusable label)")
    if n["unlabeled"]:
        text += f"; {n['unlabeled']} without label moved to inbox/unlabeled/{name}/"
    return text


def load_batch(batch: Batch, *, log: Log | None = None) -> list[Record]:
    """Ingest one batch (module docstring, Filing): read it completely, then file it
    below inbox/processed/<id>/ and return its Records -- every image except those moved
    to inbox/unlabeled/<id>/, in unit and path order.

    A batch that cannot be read as a whole is moved to inbox/rejected/<id>/ with an
    ingest_report.json stating the reason, and IngestError is raised. IngestError
    without any move if the entry has vanished from the inbox or its id has been taken
    since scan_inbox. An I/O error after the batch was renamed into processed/<id>/
    (removing the archive, moving unlabeled photos, writing the report) propagates
    unchanged; the batch content then stays in processed/<id>/. `log` receives one
    summary line (default: the module logger)."""
    log = log or _LOG.info
    ws = batch.ws
    for folder in (ws.processed, ws.rejected, ws.unlabeled):
        folder.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(batch.path):
        raise IngestError(f"batch {batch.name}: {batch.path.name} is no longer in inbox/")
    if batch.name.casefold() in _taken_ids(ws):
        raise IngestError(f"batch id {batch.name} is already used below inbox/; the batch was "
                          f"left in inbox/ for the next scan")
    origin: dict[str, Any] = {"entry": batch.path.name, "kind": batch.kind}
    target = ws.processed / batch.name
    staging: Path | None = None
    try:
        content = batch.path
        if batch.kind == "zip":
            origin.update(_file_digest(batch.path))
            staging = ws.processed / f".staging-{batch.name}-{uuid.uuid4().hex[:12]}"
            staging.mkdir()
            origin.update(extract_zip(batch.path, staging))
            content = staging
        elif batch.kind != "folder":
            raise IngestError("not a batch: an inbox entry must be a folder or a .zip archive")
        plan = plan_batch(content, batch.name)
        _move(content, target)
    except Exception as exc:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        reason = str(exc) if isinstance(exc, IngestError) else f"{type(exc).__name__}: {_one_line(exc)}"
        _reject(batch, reason, origin)
        log(f"batch {batch.name} rejected: {reason}")
        raise IngestError(f"batch {batch.name} rejected: {reason}") from exc
    if batch.kind == "zip":
        batch.path.unlink()
    unlabeled = _move_unlabeled(plan, target, ws.unlabeled / batch.name)
    atomic_write_json(target / INGEST_REPORT, ingest_report(batch.name, plan, origin, unlabeled))
    spec = plan.source
    records = [Record(target.joinpath(*e.rel.parts), list(e.label.polygons) if e.label else [],
                      spec.name, spec.license, batch.name, e.label.issue if e.label else None)
               for e in plan.images if e.outcome != "unlabeled"]
    log(_summary(batch.name, plan))
    return records
