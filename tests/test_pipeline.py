"""
Regression tests for the data pipeline and the detector.

These encode the defects found during development, so that a future change that
reintroduces one of them fails here instead of silently corrupting a dataset or
inflating a metric:

  * polygon annotations must be converted by the extent of the polygon, not by
    reading its first two vertices as a box;
  * no photograph may appear in more than one split;
  * annotations that fall outside the four-stage scheme must be dropped, not
    remapped;
  * every written box must lie inside the frame.

Run with:  venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

UNIFIED = PROJECT_ROOT / "data" / "unified"
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

requires_dataset = pytest.mark.skipif(
    not (UNIFIED / "dataset.yaml").exists(),
    reason="unified dataset not built; run scripts/unify_dataset_v3.py",
)


# --------------------------------------------------------------- annotations

def test_polygon_to_box_uses_full_extent():
    """The bug that produced boxes wrong by a median factor of 3.6."""
    import unify_dataset_v3 as U

    # A square polygon spanning x 0.2-0.6, y 0.3-0.7.
    poly = [0.2, 0.3, 0.6, 0.3, 0.6, 0.7, 0.2, 0.7]
    cx, cy, w, h = U.polygon_to_box(poly)
    assert (cx, cy, w, h) == pytest.approx((0.4, 0.5, 0.4, 0.4))

    # Reading the first two vertices as (cx, cy, w, h) would give (0.2, 0.3,
    # 0.6, 0.3): a different centre and more than twice the area.
    naive_area = poly[2] * poly[3]
    assert abs(naive_area - w * h) > 1e-6


def test_read_annotations_distinguishes_boxes_from_polygons(tmp_path):
    import unify_dataset_v3 as U

    f = tmp_path / "mixed.txt"
    f.write_text(
        "0 0.5 0.5 0.2 0.2\n"                                  # a box
        "2 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n"                  # a polygon
        "1 0.1 0.1 0.9\n"                                      # malformed
    )
    rows = U.read_annotations(f)
    assert len(rows) == 2, "the malformed line must be dropped, not guessed at"
    assert rows[0] == (0, 0.5, 0.5, 0.2, 0.2)
    assert rows[1][0] == 2
    assert rows[1][3] == pytest.approx(0.4)


def test_remap_drops_out_of_scheme_classes():
    import unify_dataset_v3 as U

    # roboflow_maskrcnn: [DTI, Unstageable, stage-1..4]. Ids 0 and 1 are outside
    # the four-stage scheme and must disappear entirely.
    entries = [(0, .5, .5, .2, .2), (1, .5, .5, .2, .2),
               (2, .5, .5, .2, .2), (5, .5, .5, .2, .2)]
    kept = U.remap("roboflow_maskrcnn", entries)
    assert [c for c, *_ in kept] == [0, 3]


def test_remap_clips_boxes_into_the_frame():
    import unify_dataset_v3 as U

    # A box hanging over the left and top edges.
    kept = U.remap("roboflow_mobile", [(1, 0.05, 0.05, 0.4, 0.4)])
    assert len(kept) == 1
    _, cx, cy, w, h = kept[0]
    assert cx - w / 2 >= -1e-9 and cy - h / 2 >= -1e-9
    assert cx + w / 2 <= 1 + 1e-9 and cy + h / 2 <= 1 + 1e-9


def test_remap_discards_degenerate_boxes():
    import unify_dataset_v3 as U
    assert U.remap("roboflow_mobile", [(1, 0.5, 0.5, 1e-6, 0.4)]) == []


# ------------------------------------------------------------------- dataset

@requires_dataset
def test_every_image_has_a_label_and_vice_versa():
    for split in SPLITS:
        imgs = {p.stem for p in (UNIFIED / "images" / split).iterdir()
                if p.suffix.lower() in IMG_EXT}
        lbls = {p.stem for p in (UNIFIED / "labels" / split).iterdir()
                if p.suffix == ".txt"}
        assert imgs == lbls, f"{split}: {len(imgs ^ lbls)} unpaired files"


@requires_dataset
def test_all_labels_are_wellformed_and_inside_the_frame():
    bad = []
    for split in SPLITS:
        for f in (UNIFIED / "labels" / split).iterdir():
            if f.suffix != ".txt":
                continue
            for n, line in enumerate(f.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad.append(f"{f.name}:{n} has {len(parts)} fields")
                    continue
                cls = int(parts[0])
                cx, cy, w, h = (float(v) for v in parts[1:])
                if cls not in (0, 1, 2, 3):
                    bad.append(f"{f.name}:{n} class {cls}")
                if not (0 < w <= 1 and 0 < h <= 1):
                    bad.append(f"{f.name}:{n} size {w}x{h}")
                if cx - w / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6:
                    bad.append(f"{f.name}:{n} x outside frame")
                if cy - h / 2 < -1e-6 or cy + h / 2 > 1 + 1e-6:
                    bad.append(f"{f.name}:{n} y outside frame")
    assert not bad, f"{len(bad)} malformed annotations, first: {bad[:3]}"


@requires_dataset
def test_no_filename_appears_in_two_splits():
    seen = defaultdict(set)
    for split in SPLITS:
        for p in (UNIFIED / "images" / split).iterdir():
            if p.suffix.lower() in IMG_EXT:
                seen[p.name].add(split)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"{len(dupes)} filenames span splits"


@requires_dataset
def test_no_source_photograph_spans_two_splits():
    """Roboflow ships several augmented copies of each photograph."""
    import unify_dataset_v3 as U

    groups = defaultdict(set)
    for split in SPLITS:
        for p in (UNIFIED / "images" / split).iterdir():
            if p.suffix.lower() in IMG_EXT:
                groups[U.source_stem(p.name)].add(split)
    leaked = {k: v for k, v in groups.items() if len(v) > 1}
    assert not leaked, (f"{len(leaked)} photographs span splits, "
                        f"e.g. {list(leaked.items())[:3]}")


@requires_dataset
def test_no_near_duplicate_image_crosses_a_split():
    """Perceptual identity, at the threshold calibrated for a 64-bit hash."""
    imagehash = pytest.importorskip("imagehash")
    from PIL import Image

    entries = []
    for split in SPLITS:
        for p in sorted((UNIFIED / "images" / split).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                entries.append((split, p.name, imagehash.phash(Image.open(p), hash_size=8)))

    cross = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][0] == entries[j][0]:
                continue
            if entries[i][2] - entries[j][2] <= 5:
                cross.append((entries[i][1], entries[j][1]))
    assert not cross, f"{len(cross)} cross-split near-duplicates, e.g. {cross[:2]}"


@requires_dataset
def test_evaluation_splits_hold_one_image_per_photograph():
    import unify_dataset_v3 as U

    for split in ("val", "test"):
        names = [p.name for p in (UNIFIED / "images" / split).iterdir()
                 if p.suffix.lower() in IMG_EXT]
        stems = {U.source_stem(n) for n in names}
        assert len(stems) == len(names), (
            f"{split} holds {len(names)} images from only {len(stems)} photographs; "
            "its size would overstate the number of independent observations")


@requires_dataset
def test_all_four_stages_are_present_in_every_split():
    for split in SPLITS:
        present = set()
        for f in (UNIFIED / "labels" / split).iterdir():
            if f.suffix == ".txt":
                for line in f.read_text().splitlines():
                    if line.strip():
                        present.add(int(line.split()[0]))
        assert present == {0, 1, 2, 3}, f"{split} is missing stages {set(range(4)) - present}"


# ------------------------------------------------------------------ detector

@pytest.fixture(scope="module")
def detector():
    from detector import DEFAULT_WEIGHTS, PressureUlcerDetector
    if not Path(DEFAULT_WEIGHTS).exists():
        pytest.skip("no trained weights yet")
    return PressureUlcerDetector()


def test_detector_returns_wellformed_detections(detector):
    img = np.full((640, 640, 3), 180, dtype=np.uint8)
    dets, elapsed = detector.predict(img, conf=0.05)
    assert elapsed > 0
    for d in dets:
        assert d.stage_id in (0, 1, 2, 3)
        assert 0.0 <= d.confidence <= 1.0
        x1, y1, x2, y2 = d.box_xyxy
        assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 640
        assert 0.0 < d.area_fraction <= 1.0


def test_draw_preserves_image_shape(detector):
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    dets, _ = detector.predict(img, conf=0.05)
    out = detector.draw(img, dets)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert out is not img, "draw must not modify its input in place"


def test_describe_handles_the_empty_case(detector):
    text = detector.describe([], 12.0)
    assert "No pressure ulcer detected" in text
    assert "not a clinical exclusion" in text


@requires_dataset
def test_detector_runs_on_a_real_test_photograph(detector):
    import cv2
    files = sorted(p for p in (UNIFIED / "images" / "test").iterdir()
                   if p.suffix.lower() in IMG_EXT)
    if not files:
        pytest.skip("empty test split")
    rgb = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
    dets, elapsed = detector.predict(rgb, conf=0.05)
    assert elapsed < 5000, "inference should be well under five seconds"
    detector.draw(rgb, dets)
    assert isinstance(detector.describe(dets, elapsed), str)
