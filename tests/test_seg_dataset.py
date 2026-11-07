"""Integrity of the thesis segmentation dataset data/seg (built by
scripts/build_seg_dataset.py). Skipped when data/seg is not present (fresh clone).

  SELFTRAIN_DEVICE=cpu .venv/bin/python -m pytest tests/test_seg_dataset.py -q

Checks: no photograph id in two splits of manifest.csv; exactly one image per photograph
in val and test; all four stages (YOLO classes 0-3) labelled in train; test_frozen.txt
lists exactly the test images, and the frozen-test digest file of the system's dataset v0
(system_ws/datasets/v0/frozen.sha256, if present) matches the bytes of these images.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "data" / "seg"
DIGESTS = ROOT / "system_ws" / "datasets" / "v0" / "frozen.sha256"

pytestmark = pytest.mark.skipif(not (SEG / "manifest.csv").is_file(),
                                reason="data/seg is missing")


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with open(SEG / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_no_photograph_in_two_splits(rows):
    splits = defaultdict(set)
    for r in rows:
        splits[r["photograph"]].add(r["split"])
    shared = {p: s for p, s in splits.items() if len(s) > 1}
    assert not shared, f"{len(shared)} photographs in several splits, e.g. {list(shared.items())[:3]}"


@pytest.mark.parametrize("split", ["val", "test"])
def test_one_image_per_photograph(rows, split):
    counts = Counter(r["photograph"] for r in rows if r["split"] == split)
    assert counts, f"no {split} images"
    assert max(counts.values()) == 1, [p for p, n in counts.items() if n > 1][:5]


def test_all_four_stages_in_train(rows):
    classes = set()
    for r in rows:
        if r["split"] == "train":
            label = SEG / "labels" / "train" / f"{r['name']}.txt"
            if label.is_file():
                classes |= {int(line.split()[0]) for line in label.read_text().splitlines()
                            if line.strip()}
    assert {0, 1, 2, 3} <= classes, f"train stages (classes) present: {sorted(classes)}"


def test_frozen_test_list_and_digests(rows):
    test_names = sorted(r["name"] for r in rows if r["split"] == "test")
    frozen = sorted((SEG / "test_frozen.txt").read_text().split())
    assert frozen == test_names
    images = {p.stem: p for p in (SEG / "images" / "test").iterdir() if not p.name.startswith(".")}
    assert sorted(images) == test_names
    if not DIGESTS.is_file():
        pytest.skip("system_ws/datasets/v0/frozen.sha256 is missing")
    digests = {}
    for line in DIGESTS.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if path.startswith("images/test/"):
            digests[Path(path).stem] = digest
    assert sorted(digests) == test_names
    bad = [n for n in test_names
           if hashlib.sha256(images[n].read_bytes()).hexdigest() != digests[n]]
    assert not bad, f"{len(bad)} test images differ from the frozen digests, e.g. {bad[:3]}"
