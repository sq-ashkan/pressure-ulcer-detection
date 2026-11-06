"""Image pool and versioned datasets (SPEC §1; SPEC §3 step 5, feature F4).

Layout inside the workspace root:

    pool/images/<name>.jpg     every image ever accepted (JPEG, quality 95)
    pool/labels/<name>.txt     YOLOv8-seg polygons: class 0-3 = stage 1-4, normalised
    pool/masks/<name>.png      uint8 mask: 0 background, 1-4 stage 1-4
    pool/manifest.csv          name, source, photograph, split, batch, license, added_cycle
    pool/negatives_extra.csv   extra ulcer-free photographs for false-positive rates
                               (copied from the thesis dataset; evaluation only)
    pool/init.json             provenance of the initial pool (source directory, counts)
    datasets/v{N}/{images,labels,masks}/{train,val,test[,external]}/
                               relative symlinks into pool/
    datasets/v{N}/dataset.yaml       Ultralytics data file with absolute paths
    datasets/v{N}/manifest.csv       snapshot of the pool manifest
    datasets/v{N}/test_frozen.txt    sorted names of the test images
    datasets/v{N}/frozen.sha256      SHA-256 of every test/external image, label and
                                     mask ("shasum -a 256 -c frozen.sha256" re-checks it)
    datasets/v{N}/negatives_extra.csv  link to pool/negatives_extra.csv (if present)

Initial pool (init_from_seg). The pool is the thesis dataset data/seg: its image,
label and mask files are linked (or copied) under their names, with their photograph
ids and splits; batch "init", added_cycle 0 (cycles count from 1). Version 0 therefore
trains and evaluates on exactly the thesis split, which is checked file by file.

Adding the records of a cycle (add_records), given the photograph assignment of
selftrain.system.dedupe:
  * the pool must be initialised (init_from_seg): without version 0 there is no
    frozen test split to protect;
  * copies of frozen photographs and split conflicts are never written;
  * a record that joins an existing photograph keeps that photograph's id and split.
    Train keeps every image of a photograph; val keeps ONE image per photograph, as
    in the thesis dataset (unify_dataset_v3, build_seg_dataset), so that the number
    of val images equals the number of independent photographs -- a further copy of
    a val photograph is skipped and reported;
  * new photographs get fresh integer ids (one more than the largest id in use) and
    are split between train and val BY PHOTOGRAPH. The number q of new photographs
    placed in val keeps the share of val among all photographs added since
    initialisation as close as possible to f = val_fraction (rounded half up):
        q = clip(floor(f * (P + n) + 1/2) - V, 0, n),
    with n new photographs now, P photographs added in earlier cycles and V of
    those in val; this avoids rounding every small batch down to zero val
    photographs. WHICH photographs go to val is a seeded random permutation of the
    new photographs (numpy default_rng([seed, cycle])); a new val photograph keeps
    its first image only;
  * the test and external splits never receive an image;
  * files are written exactly as build_seg_dataset writes data/seg: the image
    decoded by OpenCV (EXIF orientation applied) and re-encoded as JPEG quality 95,
    the YOLO-seg label with coordinates clipped to [0, 1] and six decimals, the mask
    rasterised by build_seg_dataset.rasterise at the image size (larger outlines
    first, so a nested smaller outline keeps its own stage);
  * pool names follow data/seg, "<source>__<file stem[:80]>__<6 hex digits>", except
    that the digits are the MD5 of the file CONTENT (data/seg hashes the file path),
    so a name does not depend on where the inbox lives; names are unique ignoring
    case, because the default macOS file system is case-insensitive;
  * order of writing: the extended manifest is validated first (unique names, one
    split per photograph, ...), then the files are written, then the manifest is
    replaced atomically -- a failing call leaves at most unreferenced files in
    pool/, never a manifest row without its files.

Dataset versions (create_version). datasets/v{N} lists every pool image under its
split. It is assembled in datasets/.v{N}.partial and published by an atomic rename
only after these checks pass: (1) the manifest is valid (Workspace.read_manifest:
unique names, known splits, every photograph in exactly one split); (2) every pool
file exists; (3) every split directory lists exactly the manifest's names of that
split; (4) frozen splits (N > 0): test_frozen.txt and frozen.sha256 are
byte-identical to those of version 0 -- the same test and external images, labels
and masks under the same names with the same content. A violation raises and leaves
no version behind.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

from ..config import NUM_STAGES, ROOT, STAGE_NAMES
from .workspace import (MANIFEST_COLUMNS, SPLITS, atomic_write_json, license_for_source,
                        utc_now_iso, validate_manifest)

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:      # scripts/ is not a package (same idiom as build_seg_dataset)
    sys.path.insert(0, str(_SCRIPTS))
import build_seg_dataset as bsd  # noqa: E402  (rasterise and norm_poly of data/seg)

if TYPE_CHECKING:                      # type hints only: dedupe imports this module
    from .dedupe import DedupeResult, RecordLike
    from .workspace import Workspace

_LOG = logging.getLogger(__name__)
Log = Callable[[str], None]

FROZEN_SPLITS: tuple[str, ...] = ("test", "external")     # evaluation only, never change
KINDS: dict[str, str] = {"images": ".jpg", "labels": ".txt", "masks": ".png"}
INIT_BATCH = "init"
INIT_CYCLE = 0
JPEG_QUALITY = 95
UNKNOWN_LICENSE = "unknown"
FROZEN_LIST = "test_frozen.txt"
FROZEN_DIGESTS = "frozen.sha256"
EXTRA_NEGATIVES_CSV = "negatives_extra.csv"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class FrozenSplitError(RuntimeError):
    """A dataset version would differ from version 0 in a frozen split."""


# ----------------------------------------------------------------- pool layout

def pool_dir(ws: Workspace) -> Path:
    """pool/ of a workspace (SPEC §1)."""
    return Path(ws.root) / "pool"


def pool_file(ws: Workspace, kind: str, name: str) -> Path:
    """pool/<kind>/<name><ext> for kind in images, labels, masks."""
    return pool_dir(ws) / kind / f"{name}{KINDS[kind]}"


def extra_negative_paths(ws: Workspace) -> list[Path]:
    """Extra ulcer-free photographs (column `path` of pool/negatives_extra.csv)."""
    csv_path = pool_dir(ws) / EXTRA_NEGATIVES_CSV
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    return [Path(p) for p in df["path"]] if "path" in df.columns else []


def split_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Images and distinct photographs per split present in a manifest."""
    return {s: {"images": int((df["split"] == s).sum()),
                "photographs": int(df.loc[df["split"] == s, "photograph"].nunique())}
            for s in SPLITS if (df["split"] == s).any()}


def missing_pool_files(ws: Workspace, df: pd.DataFrame) -> list[Path]:
    """Image, label or mask files of manifest rows that do not exist (a symlink
    whose target vanished counts as missing)."""
    return [pool_file(ws, kind, n) for n in df["name"] for kind in KINDS
            if not pool_file(ws, kind, n).exists()]


# ----------------------------------------------------------------------- init

def read_seg_manifest(seg_root: Path) -> pd.DataFrame:
    """name, source, photograph, split of a build_seg_dataset output."""
    df = pd.read_csv(Path(seg_root) / "manifest.csv", dtype=str, keep_default_na=False)
    need = ["name", "source", "photograph", "split"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{seg_root}/manifest.csv lacks columns {missing}")
    return df[need].reset_index(drop=True)


def seg_split_names(root: Path) -> dict[str, list[str]]:
    """Sorted image names per split as the training code sees a dataset directory
    (images/<split>/*.jpg; selftrain.data.SegDataset and evaluate.split_items)."""
    return {s: sorted(p.stem for p in (Path(root) / "images" / s).glob("*.jpg")) for s in SPLITS}


def initial_manifest(seg: pd.DataFrame) -> pd.DataFrame:
    """Pool manifest of the thesis dataset: its names, sources, photograph ids and
    splits; batch "init", added_cycle 0, licence of each source from
    workspace.KNOWN_SOURCES ("unknown" for a source not listed there). Validated
    by workspace.validate_manifest (unique names, known splits, integer
    photograph ids, every photograph in exactly one split)."""
    pool = seg.assign(batch=INIT_BATCH,
                      license=[license_for_source(s) or UNKNOWN_LICENSE for s in seg["source"]],
                      added_cycle=INIT_CYCLE)
    return validate_manifest(pool[list(MANIFEST_COLUMNS)])


def check_seg_files(seg_root: Path, pool: pd.DataFrame) -> None:
    """The thesis dataset must be self-consistent before it becomes version 0: per
    split, the manifest rows equal the image files; every image has a label and a
    mask; test_frozen.txt (if present) equals the test rows."""
    listed = seg_split_names(seg_root)
    for s in SPLITS:
        rows = sorted(pool.loc[pool["split"] == s, "name"])
        if rows != listed[s]:
            raise ValueError(f"{seg_root}: images/{s} holds {len(listed[s])} images but "
                             f"manifest.csv lists {len(rows)}")
    missing = [Path(seg_root) / kind / s / f"{n}{ext}"
               for n, s in zip(pool["name"], pool["split"]) for kind, ext in KINDS.items()
               if not (Path(seg_root) / kind / s / f"{n}{ext}").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files missing in {seg_root}, e.g. {missing[:3]}")
    frozen = Path(seg_root) / FROZEN_LIST
    if frozen.exists() and sorted(frozen.read_text().split()) != listed["test"]:
        raise ValueError(f"{frozen} does not list the images of {seg_root}/images/test")


def _place(src: Path, dst: Path, copy: bool) -> None:
    """Put `src` at `dst`: an absolute symlink to the resolved source, or a copy."""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(Path(src).resolve())


def init_from_seg(ws: Workspace, seg_root: Path, copy: bool = False,
                  log: Log | None = None) -> int:
    """Create the pool from a build_seg_dataset output (data/seg) and dataset
    version 0 with exactly its split; returns 0.

    Method: build and validate the initial manifest (initial_manifest), check the
    files (check_seg_files); link every image, label and mask of seg_root into
    pool/ under its name (copy=True: copy them, making the workspace independent
    of seg_root); write the manifest; copy negatives_extra.csv; record the
    provenance in pool/init.json; create version 0 and verify that each of its
    splits lists the same images as seg_root. Refuses a workspace whose pool
    manifest has rows or that already has dataset versions."""
    log = log or _LOG.info
    seg_root = Path(seg_root).resolve()
    pool = initial_manifest(read_seg_manifest(seg_root))
    check_seg_files(seg_root, pool)
    if len(ws.read_manifest()):
        raise FileExistsError(f"the pool of {ws.root} is already initialised")
    if ws.next_dataset_version() != 0:
        raise FileExistsError(f"{ws.root} already has dataset versions")
    for kind in KINDS:
        (pool_dir(ws) / kind).mkdir(parents=True, exist_ok=True)
    for name, split in zip(pool["name"], pool["split"]):
        for kind, ext in KINDS.items():
            _place(seg_root / kind / split / f"{name}{ext}", pool_file(ws, kind, name), copy)
    # allow_test_change: this write DEFINES the frozen test split (Workspace contract)
    ws.write_manifest(pool, allow_test_change=True)
    if (seg_root / EXTRA_NEGATIVES_CSV).exists():
        shutil.copy2(seg_root / EXTRA_NEGATIVES_CSV, pool_dir(ws) / EXTRA_NEGATIVES_CSV)
    atomic_write_json(pool_dir(ws) / "init.json", {
        "seg_root": str(seg_root), "mode": "copy" if copy else "symlink",
        "created": utc_now_iso(), "images": len(pool), "splits": split_counts(pool)})
    version = create_version(ws, log=log)
    if version != 0:
        raise RuntimeError(f"the initial dataset became version {version}, expected 0")
    if seg_split_names(ws.dataset_dir(0)) != seg_split_names(seg_root):
        raise RuntimeError(f"version 0 does not reproduce the split of {seg_root}")
    log(f"[dataset] pool initialised from {seg_root}: {len(pool)} images; version 0 created")
    return version


# ------------------------------------------------------------------- add records

@dataclass(frozen=True)
class AddResult:
    """What add_records wrote. `rows`: the manifest rows appended to the pool;
    `added`: record index -> pool name; `skipped`: (record index, reason) for every
    record not written; `new_photographs`: photograph id -> split of the new
    photographs of this call."""
    rows: list[dict[str, Any]]
    added: dict[int, str]
    skipped: list[tuple[int, str]]
    new_photographs: dict[int, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One line for the dashboard."""
        n_train = sum(r["split"] == "train" for r in self.rows)
        n_val = sum(r["split"] == "val" for r in self.rows)
        n_new_val = sum(s == "val" for s in self.new_photographs.values())
        return (f"added {len(self.rows)} images ({n_train} train, {n_val} val); "
                f"{len(self.new_photographs)} new photographs ({n_new_val} to val); "
                f"{len(self.skipped)} not added")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"rows": self.rows, "added": {str(k): v for k, v in self.added.items()},
                "skipped": [[i, why] for i, why in self.skipped],
                "new_photographs": {str(k): v for k, v in self.new_photographs.items()},
                "summary": self.summary()}


def val_quota(n_new: int, n_prev: int, n_prev_val: int, val_fraction: float) -> int:
    """New photographs to place in val: q = floor(f (P + n) + 1/2) - V clipped to
    [0, n], so that the val share among all photographs added since
    initialisation stays as close as possible to f (see the module docstring)."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in [0, 1), got {val_fraction}")
    target = math.floor(val_fraction * (n_prev + n_new) + 0.5)
    return max(0, min(n_new, target - n_prev_val))


def choose_val_groups(n_groups: int, quota: int, seed: int, cycle: int) -> set[int]:
    """Indices of the new photographs placed in val: the first `quota` entries of a
    random permutation drawn with numpy default_rng([seed, cycle])."""
    rng = np.random.default_rng([seed, cycle])
    return {int(k) for k in rng.permutation(n_groups)[:quota]}


def added_photographs(pool: pd.DataFrame) -> tuple[int, int]:
    """(photographs added after initialisation, how many of them are in val).
    Initial photographs are those with a row of batch "init" and added_cycle 0."""
    initial = set(pool.loc[(pool["batch"] == INIT_BATCH) & (pool["added_cycle"] == INIT_CYCLE),
                           "photograph"])
    added = pool[~pool["photograph"].isin(initial)].groupby("photograph")["split"].first()
    return int(len(added)), int((added == "val").sum())


def next_photograph_ids(existing: Iterable[int], k: int) -> list[int]:
    """k fresh photograph ids: consecutive integers after the largest id in use."""
    used = [int(p) for p in existing]
    start = max(used) + 1 if used else 0
    return [start + i for i in range(k)]


def pool_name(source: str, path: Path, digest: str, taken: set[str]) -> str:
    """Pool name in the style of data/seg: "<source>__<stem[:80]>__<digest[:6]>"
    with every run of characters outside [A-Za-z0-9._-] replaced by "_"; "_2",
    "_3", ... is appended while the name is in `taken`, a set of casefolded names
    (unique ignoring case, see the module docstring)."""
    base = _UNSAFE.sub("_", f"{source}__{Path(path).stem[:80]}__{digest[:6]}")
    name, k = base, 2
    while name.casefold() in taken:
        name, k = f"{base}_{k}", k + 1
    return name


def clean_polygons(polygons: Iterable[tuple[int, Any]]) -> list[tuple[int, np.ndarray]]:
    """Stage polygons as (stage 0-3, (N, 2) float32 clipped to [0, 1]) --
    build_seg_dataset.norm_poly; a stage outside 0-3 or fewer than three vertices
    violates the contract of the quality step and raises ValueError."""
    out = []
    for stage, pts in polygons:
        s = int(stage)
        if not 0 <= s < NUM_STAGES:
            raise ValueError(f"stage {stage} outside 0..{NUM_STAGES - 1}")
        p = bsd.norm_poly(np.asarray(pts, dtype=np.float32).ravel())
        if len(p) < 3:
            raise ValueError("polygon with fewer than three vertices")
        out.append((s, p))
    return out


def yolo_label_text(polygons: Sequence[tuple[int, np.ndarray]]) -> str:
    """YOLOv8-seg label: one line "stage x1 y1 x2 y2 ..." per polygon, six decimals
    (the format of data/seg/labels)."""
    return "".join(f"{s} " + " ".join(f"{v:.6f}" for v in pts.ravel()) + "\n"
                   for s, pts in polygons)


def _imwrite(path: Path, img: np.ndarray, params: Sequence[int] = ()) -> None:
    """cv2.imwrite that raises instead of returning False."""
    if not cv2.imwrite(str(path), img, list(params)):
        raise OSError(f"cannot write {path}")


def write_pool_files(ws: Workspace, name: str, record: RecordLike) -> None:
    """Write the image (decoded by OpenCV, JPEG quality 95), the YOLO-seg label and
    the mask (build_seg_dataset.rasterise at the image size) of one record. A link
    already at a destination is removed first, so a write never goes through a
    link into the thesis dataset."""
    img = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {record.path}")
    polys = clean_polygons(record.polygons)
    h, w = img.shape[:2]
    for kind in KINDS:
        if pool_file(ws, kind, name).is_symlink():
            pool_file(ws, kind, name).unlink()
    _imwrite(pool_file(ws, "images", name), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    pool_file(ws, "labels", name).write_text(yolo_label_text(polys))
    _imwrite(pool_file(ws, "masks", name), bsd.rasterise(polys, h, w))


def plan_additions(result: DedupeResult, new_ids: Sequence[int], val_groups: set[int]
                   ) -> tuple[dict[int, tuple[int, str]], list[tuple[int, str]]]:
    """Where each record goes: {record: (photograph, split)}, and the records that
    are not written with their reason (policy in the module docstring)."""
    plan: dict[int, tuple[int, str]] = {}
    skipped: list[tuple[int, str]] = []
    for k, (group, pid) in enumerate(zip(result.new_groups, new_ids)):
        split = "val" if k in val_groups else "train"
        keep = group if split == "train" else group[:1]
        plan.update({i: (pid, split) for i in keep})
        skipped += [(i, f"further copy of new val photograph {pid} (val keeps one image "
                        f"per photograph)") for i in group[len(keep):]]
    for m in result.inherited:
        if m.split == "train":
            plan[m.record] = (int(m.photograph), "train")
        elif m.split == "val":
            skipped.append((m.record, f"copy of val photograph {m.photograph} (val keeps "
                                      f"one image per photograph)"))
        else:
            raise ValueError(f"record {m.record} cannot inherit split {m.split!r}")
    skipped += [(m.record, f"copy of a frozen {m.split} photograph ({m.method} match "
                           f"with {m.pool_name})") for m in result.dropped_test_matches]
    skipped += [(c.record, f"matches photographs of several splits {c.splits}")
                for c in result.conflicts]
    return plan, sorted(skipped)


def add_records(ws: Workspace, records: Sequence[RecordLike], dedupe_result: DedupeResult,
                cycle: int, val_fraction: float, seed: int, log: Log | None = None) -> AddResult:
    """SPEC §3 step 5 (pool part): write the records of cycle `cycle` (>= 1) to the
    pool and append them to the manifest, following `dedupe_result`, the photograph
    assignment of exactly these records (policy and order of writing in the module
    docstring); `val_fraction` and `seed` drive the train/val split of the new
    photographs (val_quota, choose_val_groups)."""
    log = log or _LOG.info
    if dedupe_result.n_records != len(records):
        raise ValueError(f"the dedupe result covers {dedupe_result.n_records} records, "
                         f"{len(records)} were given")
    if cycle < 1:
        raise ValueError(f"cycle must be >= 1 (added_cycle 0 marks the initial data), got {cycle}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    pool = ws.read_manifest()
    if pool.empty:
        raise ValueError("the pool is empty: initialise the workspace first (init_from_seg)")
    n_new = len(dedupe_result.new_groups)
    n_prev, n_prev_val = added_photographs(pool)
    quota = val_quota(n_new, n_prev, n_prev_val, val_fraction)
    new_ids = next_photograph_ids(pool["photograph"], n_new)
    val_groups = choose_val_groups(n_new, quota, seed, cycle)
    plan, skipped = plan_additions(dedupe_result, new_ids, val_groups)
    taken = {str(n).casefold() for n in pool["name"]}
    rows: list[dict[str, Any]] = []
    added: dict[int, str] = {}
    for i in sorted(plan):
        rec = records[i]
        photo, split = plan[i]
        digest = hashlib.md5(Path(rec.path).read_bytes()).hexdigest()
        name = pool_name(str(rec.source), Path(rec.path), digest, taken)
        taken.add(name.casefold())
        rows.append({"name": name, "source": str(rec.source), "photograph": int(photo),
                     "split": split, "batch": str(rec.batch), "license": str(rec.license),
                     "added_cycle": int(cycle)})
        added[i] = name
    extended = pool if not rows else validate_manifest(pd.concat(
        [pool, pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))], ignore_index=True))
    for kind in KINDS:
        (pool_dir(ws) / kind).mkdir(parents=True, exist_ok=True)
    for i, name in added.items():
        write_pool_files(ws, name, records[i])
    if rows:
        ws.write_manifest(extended)
    result = AddResult(rows, added, skipped,
                       {pid: "val" if k in val_groups else "train" for k, pid in enumerate(new_ids)})
    log(f"[dataset] cycle {cycle}: {result.summary()}")
    return result


# --------------------------------------------------------------------- versions

def frozen_list_bytes(df: pd.DataFrame) -> bytes:
    """Content of test_frozen.txt: sorted test image names, one per line."""
    return "".join(f"{n}\n" for n in sorted(df.loc[df["split"] == "test", "name"])).encode()


def frozen_digests(vdir: Path) -> bytes:
    """Content of frozen.sha256: one line "<sha256>  <kind>/<split>/<file>" (the
    format of `shasum -a 256`) for every image, label and mask of the frozen splits
    of a version, ordered by kind, split and file name. Links are followed, so the
    digests describe the bytes the training and evaluation code reads."""
    lines = []
    for kind in KINDS:
        for split in FROZEN_SPLITS:
            d = Path(vdir) / kind / split
            for f in (sorted(os.listdir(d)) if d.is_dir() else []):
                with open(d / f, "rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                lines.append(f"{digest}  {kind}/{split}/{f}\n")
    return "".join(lines).encode()


def verify_frozen(v0: Path, candidate: Path) -> None:
    """Frozen-test protection: `candidate` must carry a test_frozen.txt and a
    frozen.sha256 byte-identical to those of version 0 `v0`, i.e. the same test and
    external images, labels and masks with the same names and the same content;
    FrozenSplitError otherwise."""
    v0, candidate = Path(v0), Path(candidate)
    for fname in (FROZEN_LIST, FROZEN_DIGESTS):
        if not (v0 / fname).exists():
            raise FrozenSplitError(f"{v0 / fname} is missing: the frozen splits cannot be verified")
        ref, new = (v0 / fname).read_bytes(), (candidate / fname).read_bytes()
        if ref != new:
            raise FrozenSplitError(f"{fname} differs from version 0: "
                                   f"{_first_difference(ref, new)}")


def _first_difference(ref: bytes, new: bytes) -> str:
    """First line that differs between two text files (for error messages)."""
    a, b = ref.decode().splitlines(), new.decode().splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i + 1}: {x!r} in version 0, {y!r} now"
    return f"{len(a)} lines in version 0, {len(b)} now"


def link_relative(target: Path, link: Path) -> None:
    """Symlink `link` -> `target` by a relative path between the real parent
    directories, so the workspace can be moved as a whole."""
    target, link = Path(target), Path(link)
    link.symlink_to(os.path.relpath(target.parent.resolve() / target.name, link.parent.resolve()))


def dataset_yaml(final_dir: Path) -> dict[str, Any]:
    """Ultralytics data description of a version with absolute paths."""
    root = Path(final_dir).resolve()
    return {"path": str(root), "train": str(root / "images" / "train"),
            "val": str(root / "images" / "val"), "test": str(root / "images" / "test"),
            "nc": NUM_STAGES, "names": list(STAGE_NAMES)}


def populate_version(ws: Workspace, pool: pd.DataFrame, vdir: Path) -> None:
    """Write the split directories (relative symlinks into pool/; train, val, test
    always, external if the pool has external images), the manifest snapshot,
    test_frozen.txt, frozen.sha256 and the negatives_extra.csv link into `vdir`."""
    splits = [s for s in SPLITS if s != "external" or (pool["split"] == "external").any()]
    for kind in KINDS:
        for s in splits:
            (vdir / kind / s).mkdir(parents=True, exist_ok=True)
    for name, split in zip(pool["name"], pool["split"]):
        for kind, ext in KINDS.items():
            link_relative(pool_file(ws, kind, name), vdir / kind / split / f"{name}{ext}")
    pool.to_csv(vdir / "manifest.csv", index=False, lineterminator="\n")
    (vdir / FROZEN_LIST).write_bytes(frozen_list_bytes(pool))
    (vdir / FROZEN_DIGESTS).write_bytes(frozen_digests(vdir))
    extra = pool_dir(ws) / EXTRA_NEGATIVES_CSV
    if extra.exists():
        link_relative(extra, vdir / EXTRA_NEGATIVES_CSV)


def check_listing(vdir: Path, pool: pd.DataFrame) -> None:
    """Every images/labels/masks split directory of a version must list exactly the
    manifest's names of that split (with the extension of the kind); RuntimeError
    otherwise."""
    for kind, ext in KINDS.items():
        for s in SPLITS:
            d = Path(vdir) / kind / s
            have = sorted(os.listdir(d)) if d.is_dir() else []
            want = sorted(f"{n}{ext}" for n in pool.loc[pool["split"] == s, "name"])
            if have != want:
                raise RuntimeError(f"{kind}/{s} of {vdir} lists {len(have)} files, the "
                                   f"manifest {len(want)}")


def create_version(ws: Workspace, log: Log | None = None) -> int:
    """SPEC §3 step 5 (version part): write datasets/v{N} for the current pool,
    N = ws.next_dataset_version(), and return N. The version is assembled in
    datasets/.v{N}.partial, checked (pool files present, check_listing, and
    verify_frozen against version 0 for N > 0) and published by renaming; on failure
    nothing is published and the partial directory is removed."""
    log = log or _LOG.info
    pool = ws.read_manifest()
    if pool.empty:
        raise ValueError("the pool is empty: initialise the workspace first (init_from_seg)")
    missing = missing_pool_files(ws, pool)
    if missing:
        raise FileNotFoundError(f"{len(missing)} pool files missing, e.g. {[str(p) for p in missing[:3]]}")
    version = ws.next_dataset_version()
    final = Path(ws.dataset_dir(version))
    if final.exists():
        raise FileExistsError(f"{final} already exists")
    tmp = final.with_name(f".{final.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        populate_version(ws, pool, tmp)
        check_listing(tmp, pool)
        (tmp / "dataset.yaml").write_text(
            yaml.safe_dump(dataset_yaml(final), sort_keys=False, default_flow_style=None))
        if version > 0:
            verify_frozen(Path(ws.dataset_dir(0)), tmp)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    counts = ", ".join(f"{s} {c['images']} ({c['photographs']} photographs)"
                       for s, c in split_counts(pool).items())
    log(f"[dataset] version {version}: {counts}")
    return version
