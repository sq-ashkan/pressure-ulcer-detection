"""Quality control of ingested records (SPEC §3 step 2, feature F11).

Every labelled image produced by ingest passes the checks below before it may enter the
pool. The checks run in a fixed order and the FIRST failing check is the reason of the
rejection, written as ``"<code>: <message>"``: a short, stable code (REASONS) by which
reports count and filter rejections, followed by a human-readable message that names the
measured value and the threshold. Thresholds come from SystemConfig (config.yaml).

Checks, in order
  1. unreadable_image          the file cannot be decoded completely. PIL must load every
                               pixel (this detects truncated files, which OpenCV decodes
                               silently with a grey remainder) and OpenCV -- the decoder of
                               the pool and of both models -- must decode it (the decode
                               test of unify_dataset_v3.image_ok).
  2. image_too_small           the shorter side of the decoded image (EXIF orientation
                               applied, i.e. the pixels the pool stores) is below
                               min_side_px (default 64 px).
  3. unknown_class             a region whose stage is not 0-3. Ingest keeps a region of a
                               source class that the batch's class map does not cover with
                               the negative stage code unknown_stage(c) = -1 - c, so that
                               the message can name the class c.
  4. invalid_polygon           the vertices of a region are not an (N, 2) array of finite
                               numbers.
  5. degenerate_polygon        a region has fewer than three vertices, i.e. no area.
  6. coordinates_out_of_range  a vertex lies outside [0, 1] by more than COORD_TOLERANCE =
                               0.01 (1 % of the image side). Smaller excursions are rounding
                               of annotation tools; the dataset step clips them to [0, 1]
                               when it writes the pool files (dataset.clean_polygons).
  7. whole_frame_label         the label mask -- the polygons clipped to [0, 1] and
                               rasterised exactly like the pool mask
                               (build_seg_dataset.rasterise) -- covers more than
                               max_mask_frac (default 95 %) of the image. A label that
                               covers the whole photograph marks the photograph, not the
                               wound (the whole-frame defect of PPA Fig. 6.4); a
                               segmentation model would learn "everything is ulcer".
  8. region_too_small          a single region (one polygon rasterised alone at the image
                               resolution, with the rounding of the pool mask) covers less
                               than min_mask_frac (default 0.05 %, about 200 px of a
                               640 x 640 photograph) of the image: a stray click rather
                               than the outline of a wound.
An ulcer-free record (no polygon) is subject to checks 1 and 2 only.

Ingest finds some labels unusable before these checks, with codes of the same vocabulary
(malformed_label, unreadable_mask, size_mismatch, box_label, dropped_class). It returns
such an image as a Record whose ``issue`` holds the reason, and check_record rejects it
with exactly that reason before check 1 -- an unusable label must never pass as an
ulcer-free photograph. REASONS documents every code, so the report of a cycle can
explain every rejection, and write_reports stores them per batch in
``inbox/processed/<batch>/quality_report.json``.
"""
from __future__ import annotations

import logging
import operator
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from ..config import NUM_STAGES, ROOT
from .workspace import atomic_write_json, utc_now_iso

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:      # scripts/ is not a package (same idiom as dataset.py)
    sys.path.insert(0, str(_SCRIPTS))
import build_seg_dataset as bsd  # noqa: E402  (rasterise / norm_poly of the pool masks)

if TYPE_CHECKING:                      # type hints only
    from .ingest import Record
    from .workspace import SystemConfig, Workspace

__all__ = [
    "COORD_TOLERANCE", "INGEST_CODES", "QUALITY_CODES", "QUALITY_REPORT", "REASONS",
    "QualityResult", "UnreadableImageError", "check_all", "check_record", "decoded_shape",
    "label_coverage", "make_reason", "quality_report", "reason_code", "reason_message",
    "region_pixels", "unknown_source_class", "unknown_stage", "write_reports",
]

_LOG = logging.getLogger(__name__)

COORD_TOLERANCE = 0.01
"""Vertices may lie outside [0, 1] by at most this much (1 % of the image side, SPEC §3)."""

QUALITY_REPORT = "quality_report.json"

REASONS: Mapping[str, str] = MappingProxyType({
    # ingest (selftrain.system.ingest): the label cannot be used as delivered
    "malformed_label": "the label cannot be parsed (file and line are named in the message)",
    "unreadable_mask": "the mask PNG cannot be decoded",
    "size_mismatch": "the label refers to an image size other than the image's own",
    "box_label": "a stage region is a detection box, not a segmentation outline",
    "dropped_class": "the image shows a class mapped to 'drop' (outside the four-stage scope)",
    # quality (check_record), in the order in which the checks are applied
    "unreadable_image": "the image cannot be decoded completely",
    "image_too_small": "the shorter image side is below min_side_px",
    "unknown_class": "a region has a class that the batch's class map does not cover",
    "invalid_polygon": "the vertices of a region are not an (N, 2) array of finite numbers",
    "degenerate_polygon": "a region has fewer than three vertices",
    "coordinates_out_of_range": "a vertex lies outside [0, 1] by more than 1 % of the image side",
    "whole_frame_label": "the label covers more than max_mask_frac of the image (whole-frame label)",
    "region_too_small": "a region covers less than min_mask_frac of the image",
})
"""Every rejection code of the system with a one-line explanation (report legend)."""

INGEST_CODES: tuple[str, ...] = (
    "malformed_label", "unreadable_mask", "size_mismatch", "box_label", "dropped_class")
QUALITY_CODES: tuple[str, ...] = tuple(c for c in REASONS if c not in INGEST_CODES)

_CODE = re.compile(r"[a-z][a-z0-9_]*")


# ------------------------------------------------------------------ reasons

def make_reason(code: str, message: str) -> str:
    """Rejection reason ``"<code>: <message>"``; `code` must be one of REASONS."""
    if code not in REASONS:
        raise ValueError(f"unknown rejection code {code!r}; known: {sorted(REASONS)}")
    text = " ".join(str(message).split())
    if not text:
        raise ValueError("a rejection reason needs a message")
    return f"{code}: {text}"


def reason_code(reason: str) -> str:
    """The code of a reason made by make_reason (the text before the first ": ")."""
    code = str(reason).split(": ", 1)[0]
    if not _CODE.fullmatch(code):
        raise ValueError(f"not a rejection reason: {reason!r}")
    return code


def reason_message(reason: str) -> str:
    """The human-readable part of a reason made by make_reason."""
    parts = str(reason).split(": ", 1)
    return parts[1] if len(parts) == 2 else ""


def unknown_stage(source_class: int) -> int:
    """Stage code of a region whose source class `c` the class map does not cover:
    -1 - c. It is negative, so it can never be mistaken for a stage (0-3), and it is
    invertible (unknown_source_class), so the rejection can name the class."""
    c = operator.index(source_class)
    if c < 0:
        raise ValueError(f"source classes are non-negative integers, got {c}")
    return -1 - c


def unknown_source_class(stage: int) -> int | None:
    """Inverse of unknown_stage: the source class encoded by a negative stage code,
    None for a stage code >= 0."""
    s = operator.index(stage)
    return -1 - s if s < 0 else None


# ------------------------------------------------------------------ measurements

class UnreadableImageError(Exception):
    """The image file cannot be decoded completely."""


def decoded_shape(path: Path) -> tuple[int, int]:
    """(height, width) of an image as stored in the pool, or UnreadableImageError.

    PIL loads every pixel (a truncated or corrupt file raises), then OpenCV decodes the
    file with EXIF orientation applied, exactly as dataset.write_pool_files reads it."""
    path = Path(path)
    try:
        with Image.open(path) as im:
            im.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UnreadableImageError(f"the file is missing, truncated or corrupt ({exc})") from None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.ndim < 3:
        raise UnreadableImageError("OpenCV cannot decode the file")
    h, w = img.shape[:2]
    return int(h), int(w)


def _pixel_vertices(points: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalised vertices -> integer pixel vertices, rounded as build_seg_dataset.rasterise."""
    return np.round(np.asarray(points, np.float64) * [width, height]).astype(np.int32)


def region_pixels(points: np.ndarray, height: int, width: int) -> int:
    """Pixels of one region: the polygon (normalised (N, 2), inside [0, 1]) filled alone
    on a height x width image with the rounding of the pool mask. Only the polygon's
    bounding box is rasterised (equal result, independent of the image size)."""
    px = _pixel_vertices(points, height, width)
    x0, y0 = max(int(px[:, 0].min()), 0), max(int(px[:, 1].min()), 0)
    x1, y1 = min(int(px[:, 0].max()), width - 1), min(int(px[:, 1].max()), height - 1)
    if x1 < x0 or y1 < y0:
        return 0
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(canvas, [(px - [x0, y0]).astype(np.int32)], 1)
    return int(np.count_nonzero(canvas))


def label_coverage(polygons: Sequence[tuple[int, np.ndarray]], height: int, width: int) -> float:
    """Fraction of the image covered by the label mask: polygons (stage 0-3, normalised)
    clipped to [0, 1] (build_seg_dataset.norm_poly) and rasterised with
    build_seg_dataset.rasterise -- the mask the pool would store."""
    if not polygons:
        return 0.0
    clipped = [(int(s), bsd.norm_poly(np.asarray(p, np.float32).ravel())) for s, p in polygons]
    mask = bsd.rasterise(clipped, height, width)
    return float(np.count_nonzero(mask)) / float(height * width)


# ------------------------------------------------------------------ the checks

def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """"1 region", "2 regions"; `plural` for irregular nouns."""
    return f"{n} {singular if n == 1 else plural or singular + 's'}"


def _unknown_class_reason(stages: Iterable[int]) -> str | None:
    """unknown_class reason for the stage codes outside 0-3, None if there is none."""
    bad = Counter(s for s in stages if not 0 <= s < NUM_STAGES)
    if not bad:
        return None
    parts = []
    for stage, n in sorted(bad.items(), key=lambda item: (item[0] >= 0, abs(item[0]))):
        source = unknown_source_class(stage)
        what = (f"source class {source}" if source is not None
                else f"stage index {stage} (stages are 0-{NUM_STAGES - 1})")
        parts.append(f"{what} ({_plural(n, 'region')})")
    return make_reason("unknown_class", "; ".join(parts) + " not covered by the class map of "
                       "the batch; map every class in source.yaml (1-4, background or drop)")


def _vertices(polygons: Any) -> tuple[list[tuple[int, np.ndarray]], str | None]:
    """Validate the structure of Record.polygons (checks 3-5): returns the regions as
    (stage, float64 (N, 2) array) and None, or [] and the reason of the first failure."""
    try:
        items = list(polygons)
    except TypeError:
        return [], make_reason("invalid_polygon", f"polygons must be a list of (stage, "
                                                  f"vertices) pairs, got {type(polygons).__name__}")
    stages: list[int] = []
    for k, item in enumerate(items, 1):
        try:
            stage, _ = item
            if isinstance(stage, (bool, np.bool_)):
                raise TypeError
            stages.append(operator.index(stage))
        except (TypeError, ValueError):
            return [], make_reason("invalid_polygon", f"region {k} is not a (stage, vertices) "
                                                      f"pair with an integer stage")
    reason = _unknown_class_reason(stages)
    if reason:
        return [], reason
    regions: list[tuple[int, np.ndarray]] = []
    for k, (stage, (_, pts)) in enumerate(zip(stages, items), 1):
        try:
            arr = np.asarray(pts, dtype=np.float64)
        except (TypeError, ValueError):
            return [], make_reason("invalid_polygon", f"region {k}: the vertices are not numbers")
        if arr.ndim != 2 or arr.shape[1] != 2:
            return [], make_reason("invalid_polygon", f"region {k}: vertices must have shape "
                                                      f"(N, 2), got {arr.shape}")
        if not np.isfinite(arr).all():
            return [], make_reason("invalid_polygon", f"region {k} has non-finite coordinates")
        regions.append((stage, arr))
    for k, (_, arr) in enumerate(regions, 1):
        if len(arr) < 3:
            return [], make_reason("degenerate_polygon",
                                   f"region {k} has {_plural(len(arr), 'vertex', 'vertices')}; an outline "
                                   f"needs at least 3 (a four-number YOLO line is a box)")
    return regions, None


def _range_reason(regions: Sequence[tuple[int, np.ndarray]]) -> str | None:
    """coordinates_out_of_range reason (check 6) for the largest excursion, or None."""
    worst: tuple[float, int, str, float] | None = None
    for k, (_, arr) in enumerate(regions, 1):
        for axis, column in (("x", arr[:, 0]), ("y", arr[:, 1])):
            for value in (float(column.min()), float(column.max())):
                excess = max(-value, value - 1.0)
                if excess > COORD_TOLERANCE and (worst is None or excess > worst[0]):
                    worst = (excess, k, axis, value)
    if worst is None:
        return None
    _, k, axis, value = worst
    return make_reason("coordinates_out_of_range",
                       f"region {k} has a vertex at {axis} = {value:.4f}, outside [0, 1] by more "
                       f"than {COORD_TOLERANCE:.0%} of the image side")


def check_record(record: Record, cfg: SystemConfig) -> tuple[bool, str | None]:
    """Apply the quality checks of the module docstring to one record.

    Returns (True, None) for an accepted record and (False, reason) otherwise, where
    reason = "<code>: <message>" of the first failing check (codes: REASONS). A record
    whose label ingest found unusable (``record.issue``, a reason made by make_reason,
    normally with an ingest code) is rejected with that reason before any check; an
    issue without a code of REASONS is a programming error (ValueError). Uses
    cfg.min_side_px, cfg.max_mask_frac and cfg.min_mask_frac."""
    issue = getattr(record, "issue", None)
    if issue is not None:
        if reason_code(issue) not in REASONS:
            raise ValueError(f"record issue {issue!r} has no rejection code of REASONS")
        return False, str(issue)
    path = Path(record.path)
    try:
        height, width = decoded_shape(path)
    except UnreadableImageError as exc:
        return False, make_reason("unreadable_image", f"{path.name}: {exc}")
    if min(height, width) < cfg.min_side_px:
        return False, make_reason("image_too_small", f"{width} x {height} px; the shorter side "
                                                     f"must be at least {cfg.min_side_px} px")
    regions, reason = _vertices(record.polygons)
    if reason is None:
        reason = _range_reason(regions)
    if reason is not None:
        return False, reason
    if not regions:
        return True, None
    coverage = label_coverage(regions, height, width)
    if coverage > cfg.max_mask_frac:
        return False, make_reason("whole_frame_label",
                                  f"the label covers {coverage:.1%} of the image (limit "
                                  f"{cfg.max_mask_frac:.0%}): a whole-frame label marks the "
                                  f"photograph, not the wound")
    area = float(height * width)
    for k, (_, arr) in enumerate(regions, 1):
        pixels = region_pixels(np.clip(arr, 0.0, 1.0), height, width)
        if pixels / area < cfg.min_mask_frac:
            return False, make_reason("region_too_small",
                                      f"region {k} covers {pixels / area:.4%} of the image "
                                      f"({pixels} px), less than min_mask_frac = "
                                      f"{cfg.min_mask_frac:.4%}")
    return True, None


class QualityResult(NamedTuple):
    """Outcome of check_all: accepted records and (record, reason) pairs, both in input
    order. Unpacks as ``accepted, rejected = check_all(records, cfg)``."""
    accepted: list[Any]
    rejected: list[tuple[Any, str]]

    def counts(self) -> dict[str, int]:
        """Number of rejections per code, sorted by code."""
        return dict(sorted(Counter(reason_code(r) for _, r in self.rejected).items()))

    def summary(self) -> str:
        """One line for the dashboard, e.g. "38 of 40 images passed; rejected: 2 whole_frame_label"."""
        n = len(self.accepted) + len(self.rejected)
        if n == 0:
            return "no images to check"
        if not self.rejected:
            return f"all {n} images passed"
        parts = ", ".join(f"{k} {code}" for code, k in self.counts().items())
        return f"{len(self.accepted)} of {n} images passed; rejected: {parts}"


def check_all(records: Iterable[Record], cfg: SystemConfig) -> QualityResult:
    """check_record for every record: (accepted records, [(record, reason)])."""
    accepted: list[Any] = []
    rejected: list[tuple[Any, str]] = []
    for record in records:
        ok, reason = check_record(record, cfg)
        if ok:
            accepted.append(record)
        else:
            assert reason is not None
            rejected.append((record, reason))
            _LOG.debug("rejected %s: %s", record.path, reason)
    return QualityResult(accepted, rejected)


# ------------------------------------------------------------------ reports

def _check_batch_name(batch: str) -> str:
    """A batch id is one plain path component (ingest makes it so); anything else could
    write a report outside inbox/processed/."""
    name = str(batch)
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"invalid batch id {batch!r}")
    return name


def _relative(path: Path, base: Path) -> str:
    try:
        return Path(path).relative_to(base).as_posix()
    except ValueError:
        return str(path)


def quality_report(batch: str, accepted: Sequence[Any], rejected: Sequence[tuple[Any, str]],
                   cfg: SystemConfig, batch_dir: Path) -> dict[str, Any]:
    """JSON-ready quality report of one batch: counts, thresholds, every rejection
    (image path relative to `batch_dir`, code, message) and the legend of the codes used."""
    by_code = Counter(reason_code(reason) for _, reason in rejected)
    return {
        "batch": batch,
        "images": len(accepted) + len(rejected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_by_code": dict(sorted(by_code.items())),
        "thresholds": {"min_side_px": cfg.min_side_px, "max_mask_frac": cfg.max_mask_frac,
                       "min_mask_frac": cfg.min_mask_frac,
                       "coordinate_tolerance": COORD_TOLERANCE},
        "rejections": [{"image": _relative(rec.path, batch_dir), "code": reason_code(reason),
                        "message": reason_message(reason)} for rec, reason in rejected],
        "codes": {code: REASONS.get(code, "") for code in sorted(by_code)},
        "written": utc_now_iso(),
    }


def write_reports(ws: Workspace, accepted: Iterable[Any], rejected: Iterable[tuple[Any, str]],
                  cfg: SystemConfig) -> dict[str, Path]:
    """Write ``inbox/processed/<batch>/quality_report.json`` for every batch that has a
    record in `accepted` or `rejected` (pass all rejections of the cycle, ingest and
    quality, so that the report explains every image). Returns {batch: report path}."""
    groups: dict[str, tuple[list[Any], list[tuple[Any, str]]]] = defaultdict(lambda: ([], []))
    for rec in accepted:
        groups[_check_batch_name(rec.batch)][0].append(rec)
    for rec, reason in rejected:
        groups[_check_batch_name(rec.batch)][1].append((rec, reason))
    written: dict[str, Path] = {}
    for batch in sorted(groups):
        batch_dir = Path(ws.processed) / batch
        path = batch_dir / QUALITY_REPORT
        atomic_write_json(path, quality_report(batch, *groups[batch], cfg, batch_dir))
        written[batch] = path
    return written
