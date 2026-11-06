"""Tests of photograph identity, the image pool and dataset versions of the
self-retraining system: selftrain.system.dedupe and selftrain.system.dataset
(SPEC §1 and §3 steps 3 and 5).

The integration tests initialise workspaces (always below pytest's tmp_path) from
data/seg_smoke, the small copy of the thesis dataset, and add one batch built from
it and from the raw Roboflow export data/raw/roboflow_woundcare:

    0  copy_of_test.jpg       byte copy of a test image            -> dropped (frozen test)
    1  flipped_train.jpg      mirrored train image                 -> inherits train
    2  101_jpg.rf.134c...     photograph not in the pool           \\
    3  101_jpg.rf.8c24...     other export of the same photograph   > one new photograph
    4  102_jpg.rf.a4c9...     photograph not in the pool, negative -> new photograph
    5  rotated_101.png        record 2 rotated by 90 degrees       /  (joins 2 and 3)
    6  flipped_val.jpg        upside-down val image                -> inherits val (not added)
    7  external_q80.jpg       external image re-encoded (JPEG q80) -> dropped (external)
    8  extra_negative.jpg     byte copy of an extra negative       -> dropped (extra negatives)

data/seg_smoke and data/raw are only read; nothing is written outside tmp_path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from selftrain.config import NUM_STAGES, ROOT, STAGE_NAMES
from selftrain.system import dataset, dedupe
from selftrain.system.dataset import FROZEN_SPLITS, KINDS, FrozenSplitError, pool_file
from selftrain.system.dedupe import EXTRA_NEGATIVES, DedupeResult, PoolMatch, Reference
from selftrain.system.workspace import Workspace

u = dedupe.u            # scripts/unify_dataset_v3 (thresholds of the thesis dataset)
bsd = dataset.bsd       # scripts/build_seg_dataset (rasterise)

SMOKE = ROOT / "data" / "seg_smoke"
RAW = ROOT / "data" / "raw" / "roboflow_woundcare" / "train"
NEW_101 = ("101_jpg.rf.134c36e7a9bca8c9a3a4f4606c4ecd74", "101_jpg.rf.8c247b76634b87f51bb7772c78daed46")
NEW_102 = "102_jpg.rf.a4c94d7401fbe48d44fbe1357113a2b6"
WOUNDCARE_STAGES = {2: 0, 3: 1, 4: 2, 5: 3}     # export classes stage-1..stage-4 -> 0..3
N_BATCH = 9


def _data_available() -> bool:
    if not (SMOKE / "manifest.csv").exists():
        return False
    raw = [RAW / "images" / f"{stem}.jpg" for stem in (*NEW_101, NEW_102)]
    extra = pd.read_csv(SMOKE / "negatives_extra.csv", dtype=str)["path"]
    return all(p.exists() for p in raw) and len(extra) > 0 and Path(extra.iloc[0]).exists()


requires_data = pytest.mark.skipif(not _data_available(),
                                   reason="data/seg_smoke or data/raw/roboflow_woundcare missing")


@dataclass
class Rec:
    """Stand-in for selftrain.system.ingest.Record (SPEC §2)."""
    path: Path
    polygons: list[tuple[int, np.ndarray]] = field(default_factory=list)
    source: str = "woundcare_new"
    license: str = "CC BY 4.0"
    batch: str = "batch1"


def quiet(_message: str) -> None:
    """Log sink of the tests."""


def read_label(path: Path, table: dict[int, int] | None = None) -> list[tuple[int, np.ndarray]]:
    """Polygons of a YOLO-seg label file; `table` maps export classes to stages 0-3
    (classes missing from it are skipped), None keeps the classes."""
    polys = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        stage = int(parts[0]) if table is None else table.get(int(parts[0]))
        if stage is not None:
            polys.append((stage, np.array(parts[1:], np.float32).reshape(-1, 2)))
    return polys


def init_ws(root: Path, copy: bool = False) -> Workspace:
    """A workspace whose pool and version 0 are data/seg_smoke."""
    ws = Workspace(root)
    assert dataset.init_from_seg(ws, SMOKE, copy=copy, log=quiet) == 0
    return ws


def smoke_file(kind: str, split: str, name: str) -> Path:
    return SMOKE / kind / split / f"{name}{KINDS[kind]}"


@dataclass(frozen=True)
class Picks:
    """Images of data/seg_smoke the batch is built from (first Roboflow image of a
    split in name order; Roboflow photographs are not among the extra negatives)."""
    test: str
    train: str
    train_photo: int
    val: str
    val_photo: int
    external: str
    external_photo: int
    extra_negative: Path


@pytest.fixture(scope="module")
def picks() -> Picks:
    if not _data_available():
        pytest.skip("test data missing")
    man = pd.read_csv(SMOKE / "manifest.csv", dtype=str)

    def first(split: str) -> tuple[str, int]:
        rows = man[(man["split"] == split) & man["source"].str.startswith(("roboflow", "rf_"))]
        name = sorted(rows["name"])[0]
        return name, int(rows.set_index("name").loc[name, "photograph"])

    test, _ = first("test")
    train, train_photo = first("train")
    val, val_photo = first("val")
    external, external_photo = first("external")
    extra = Path(pd.read_csv(SMOKE / "negatives_extra.csv", dtype=str)["path"].iloc[0])
    return Picks(test, train, train_photo, val, val_photo, external, external_photo, extra)


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory, picks: Picks) -> list[Rec]:
    """The nine records of the module docstring."""
    d = tmp_path_factory.mktemp("batch")
    recs: list[Rec] = []
    shutil.copyfile(smoke_file("images", "test", picks.test), d / "copy_of_test.jpg")
    recs.append(Rec(d / "copy_of_test.jpg", read_label(smoke_file("labels", "test", picks.test))))
    img = cv2.imread(str(smoke_file("images", "train", picks.train)))
    cv2.imwrite(str(d / "flipped_train.jpg"), np.ascontiguousarray(img[:, ::-1]))
    recs.append(Rec(d / "flipped_train.jpg",
                    [(s, np.c_[1 - p[:, 0], p[:, 1]])
                     for s, p in read_label(smoke_file("labels", "train", picks.train))]))
    for stem in NEW_101:
        shutil.copyfile(RAW / "images" / f"{stem}.jpg", d / f"{stem}.jpg")
        recs.append(Rec(d / f"{stem}.jpg", read_label(RAW / "labels" / f"{stem}.txt", WOUNDCARE_STAGES)))
    shutil.copyfile(RAW / "images" / f"{NEW_102}.jpg", d / f"{NEW_102}.jpg")
    recs.append(Rec(d / f"{NEW_102}.jpg", []))                  # used as an ulcer-free photo
    cv2.imwrite(str(d / "rotated_101.png"), np.ascontiguousarray(np.rot90(cv2.imread(str(recs[2].path)))))
    recs.append(Rec(d / "rotated_101.png", [(s, np.c_[p[:, 1], 1 - p[:, 0]]) for s, p in recs[2].polygons]))
    img = cv2.imread(str(smoke_file("images", "val", picks.val)))
    cv2.imwrite(str(d / "flipped_val.jpg"), np.ascontiguousarray(img[::-1]))
    recs.append(Rec(d / "flipped_val.jpg", []))
    img = cv2.imread(str(smoke_file("images", "external", picks.external)))
    cv2.imwrite(str(d / "external_q80.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    recs.append(Rec(d / "external_q80.jpg", []))
    shutil.copyfile(picks.extra_negative, d / "extra_negative.jpg")
    recs.append(Rec(d / "extra_negative.jpg", []))
    assert len(recs) == N_BATCH
    return recs


@pytest.fixture(scope="module")
def dedupe_result(tmp_path_factory: pytest.TempPathFactory, batch: list[Rec]) -> DedupeResult:
    """Photograph assignment of the batch against a freshly initialised pool. Every
    workspace made by init_ws has the same pool, so the result applies to each."""
    ws = init_ws(tmp_path_factory.mktemp("reference") / "ws")
    return dedupe.dedupe_records(ws, batch, log=quiet)


# ============================================================ dedupe: building blocks

def smooth_image(seed: int, shape: tuple[int, int] = (120, 90)) -> np.ndarray:
    """Grey uint8 image of blurred noise (photograph-like spatial correlation)."""
    noise = np.random.default_rng(seed).random(shape).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), 6)
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def textured_image(seed: int, side: int = 240) -> np.ndarray:
    """Grey uint8 image of 60 random filled rectangles: many corners for ORB."""
    rng = np.random.default_rng(seed)
    img = np.full((side, side), 128, np.uint8)
    for _ in range(60):
        x0, y0 = (int(v) for v in rng.integers(0, side - 20, 2))
        w, h = (int(v) for v in rng.integers(8, 60, 2))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), int(rng.integers(0, 256)), -1)
    return img


def test_hamming_matches_bit_count() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.random((3, 64)) > 0.5, rng.random((4, 64)) > 0.5
    expected = (a[:, None, :] != b[None, :, :]).sum(axis=2)
    assert np.array_equal(dedupe.hamming(a, b), expected)


def test_dihedral_correlation_sees_through_rotations_and_mirrors() -> None:
    grey = smooth_image(1)
    base = dedupe.unit_grid(grey)[None]
    for k in range(4):
        for mirror in (False, True):
            view = np.rot90(grey, k)
            view = np.ascontiguousarray(view[:, ::-1] if mirror else view)
            assert dedupe.dihedral_correlation(base, dedupe.unit_grid(view)[None])[0, 0] > 0.99
    other = dedupe.unit_grid(smooth_image(2))[None]
    assert dedupe.dihedral_correlation(base, other)[0, 0] < u.GRID_CORR
    flat = dedupe.unit_grid(np.full((40, 40), 7, np.uint8))
    assert not flat.any()                                    # a flat image matches nothing
    assert dedupe.dihedral_correlation(base, flat[None])[0, 0] == 0.0


def test_signature_of_unreadable_file_is_none(tmp_path: Path) -> None:
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    assert dedupe.read_signature(bad) is None
    with pytest.raises(ValueError, match="quality control"):
        dedupe.find_duplicates([Rec(bad)], [], log=quiet)


def test_find_duplicates_without_records() -> None:
    result = dedupe.find_duplicates([], [], log=quiet)
    assert result == DedupeResult([], [], [], 0)
    assert result.n_records == 0 and result.kept() == []


def test_orb_matcher_memo_is_bounded(tmp_path: Path) -> None:
    paths = []
    for seed in range(3):
        paths.append(tmp_path / f"{seed}.png")
        cv2.imwrite(str(paths[-1]), textured_image(seed))
    orb = dedupe.OrbMatcher(max_cached=2)
    assert orb.inliers(paths[0], paths[0]) >= u.ORB_MIN_INLIERS    # an image matches itself
    assert orb.inliers(paths[1], paths[2]) < u.ORB_MIN_INLIERS     # unrelated images do not
    assert list(orb._features) == paths[1:]                          # least recently used left
    with pytest.raises(ValueError):
        dedupe.OrbMatcher(max_cached=1)


@requires_data
def test_stem_rule_joins_records_of_one_batch_only(tmp_path: Path) -> None:
    man = pd.read_csv(SMOKE / "manifest.csv", dtype=str)
    train = man[man["split"] == "train"].drop_duplicates("photograph").sort_values("name")
    p1, p2, p3 = (smoke_file("images", "train", n) for n in train["name"].iloc[:3])
    files = [(p1, "b1", "x_jpg.rf.aaa.jpg"), (p2, "b1", "x_jpg.rf.bbb.jpg"),
             (p3, "b2", "x_jpg.rf.ccc.jpg"), (p3, "b1", "unrelated_name.jpg")]
    recs = []
    for src, batch_name, fname in files:
        (tmp_path / batch_name).mkdir(exist_ok=True)
        shutil.copyfile(src, tmp_path / batch_name / fname)
        recs.append(Rec(tmp_path / batch_name / fname, batch=batch_name))
    images = dedupe.ImageSet.of([r.path for r in recs], dedupe.record_signatures(recs))
    groups = dedupe.group_records([dedupe.stem_key(r) for r in recs], images, dedupe.OrbMatcher())
    # 0, 1: different photographs, same Roboflow stem in the same batch -> joined by name;
    # 2: same stem in another batch -> not joined by name; 3: same pixels as 2 -> joined
    assert groups == [[0, 1], [2, 3]]


# ============================================================ dedupe: decision rules

P = Path("unused.jpg")
REFS = [Reference("test_a", "10", "test", P),                       # 0
        Reference("train_a", "3", "train", P),                      # 1
        Reference("train_b", "4", "train", P),                      # 2
        Reference("val_a", "7", "val", P),                          # 3
        Reference("/neg.jpg", "/neg.jpg", EXTRA_NEGATIVES, P)]      # 4


def test_decide_applies_frozen_conflict_inherit_and_new_rules() -> None:
    groups = [[0, 1], [2], [3], [4], [5]]
    evidence = [{}, {0: ("pixel", 0.95)},                    # group [0, 1] hits a test photo
                {1: ("phash", 2.0), 2: ("orb", 30.0)},       # two train photographs
                {1: ("dhash", 4.0), 3: ("pixel", 0.93)},     # train and val photographs
                {},                                          # nothing
                {4: ("phash", 0.0)}]                         # an extra negative
    result = dedupe.decide(groups, evidence, REFS)
    assert result.new_groups == [[4]]
    assert result.inherited == [PoolMatch(2, "3", "train", "train_a", "phash", 2.0)]
    assert result.dropped_test_matches == [
        PoolMatch(0, "10", "test", "test_a", "pixel", 0.95, via=1),
        PoolMatch(1, "10", "test", "test_a", "pixel", 0.95),
        PoolMatch(5, "/neg.jpg", EXTRA_NEGATIVES, "/neg.jpg", "phash", 0.0)]
    assert [c.record for c in result.conflicts] == [3]
    assert result.conflicts[0].splits == ["train", "val"]
    assert result.n_duplicates == 5 and result.kept() == [2, 4]
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()


def test_frozen_evidence_wins_over_train_evidence() -> None:
    evidence = [{1: ("phash", 0.0)}, {0: ("orb", 26.0)}]
    result = dedupe.decide([[0, 1]], evidence, REFS)
    assert [m.split for m in result.dropped_test_matches] == ["test", "test"]
    assert result.dropped_test_matches[0] == PoolMatch(0, "10", "test", "test_a", "orb", 26.0, via=1)


def test_choose_photograph_majority_then_smallest_id() -> None:
    refs = [Reference("a", "9", "train", P), Reference("b", "10", "train", P),
            Reference("c", "10", "train", P)]
    tie = [(0, 0, "pixel", 0.95), (0, 1, "pixel", 0.95)]
    assert dedupe.choose_photograph(tie, refs) == "9"            # numeric, not string order
    majority = tie + [(1, 2, "orb", 40.0)]
    assert dedupe.choose_photograph(majority, refs) == "10"
    assert dedupe.photograph_order("/a.jpg") > dedupe.photograph_order("12")


def test_dedupe_result_must_partition_the_records() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        DedupeResult([[0], [2]], [], [], 1)
    with pytest.raises(ValueError, match="n_duplicates"):
        DedupeResult([[0, 1]], [], [], 0)
    with pytest.raises(ValueError, match="empty"):
        DedupeResult([[0], []], [], [], 0)
    with pytest.raises(ValueError, match="frozen"):
        DedupeResult([[0]], [PoolMatch(1, "10", "test", "test_a", "phash", 0.0)], [], 1)
    with pytest.raises(ValueError, match="frozen"):
        DedupeResult([[0]], [], [PoolMatch(1, "3", "train", "train_a", "phash", 0.0)], 1)


# ============================================================ dedupe: real images

@requires_data
def test_dedupe_assigns_every_record_to_a_photograph(dedupe_result: DedupeResult,
                                                     batch: list[Rec], picks: Picks) -> None:
    r = dedupe_result
    assert r.n_records == N_BATCH
    assert r.new_groups == [[2, 3, 5], [4]]
    assert r.n_duplicates == N_BATCH - 2
    assert r.conflicts == []
    inherited = {m.record: m for m in r.inherited}
    assert set(inherited) == {1, 6}
    flipped = inherited[1]
    assert (flipped.split, flipped.photograph, flipped.pool_name) == \
        ("train", str(picks.train_photo), picks.train)
    assert flipped.method == "pixel" and flipped.score >= u.GRID_CORR   # no hash survives a mirror
    assert (inherited[6].split, inherited[6].photograph) == ("val", str(picks.val_photo))
    dropped = {m.record: m for m in r.dropped_test_matches}
    assert set(dropped) == {0, 7, 8}
    assert (dropped[0].split, dropped[0].pool_name, dropped[0].method, dropped[0].score) == \
        ("test", picks.test, "phash", 0.0)
    assert (dropped[7].split, dropped[7].photograph) == ("external", str(picks.external_photo))
    assert (dropped[8].split, dropped[8].pool_name) == (EXTRA_NEGATIVES, str(picks.extra_negative))
    assert r.kept() == [1, 2, 3, 4, 5, 6]
    report = json.loads(json.dumps(r.to_dict(batch)))
    assert report["dropped_test_matches"][0]["file"] == str(batch[0].path)
    assert r.summary() == ("9 images -> 2 new photographs; 7 duplicates: 2 within the cycle, "
                           "2 of train/val photographs, 3 of frozen evaluation photographs "
                           "(dropped), 0 split conflicts (dropped)")


@requires_data
def test_reference_set_covers_pool_and_extra_negatives(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws")
    refs = dedupe.reference_images(ws)
    pool = ws.read_manifest()
    n_extra = len(pd.read_csv(SMOKE / "negatives_extra.csv"))
    assert len(refs) == len(pool) + n_extra
    assert [r.name for r in refs[:len(pool)]] == list(pool["name"])
    assert {r.split for r in refs[len(pool):]} == {EXTRA_NEGATIVES}
    missing = [Reference("gone", "1", "train", tmp_path / "gone.jpg")]
    with pytest.raises(FileNotFoundError):
        dedupe.reference_set(missing, quiet)
    lines: list[str] = []
    kept, images = dedupe.reference_set(
        [Reference("gone", "gone", EXTRA_NEGATIVES, tmp_path / "gone.jpg")], lines.append)
    assert kept == [] and len(images) == 0 and "1 unreadable" in lines[0]


# ============================================================ dataset: building blocks

@pytest.mark.parametrize("n_new, n_prev, n_prev_val, f, expected", [
    (3, 0, 0, 0.15, 0),       # 0.45 rounds to 0
    (3, 3, 0, 0.15, 1),       # 0.9 rounds to 1
    (2, 0, 0, 0.5, 1),
    (5, 10, 5, 0.15, 0),      # val already above its share
    (1, 0, 0, 0.6, 1),
    (4, 0, 0, 0.0, 0),
    (0, 7, 1, 0.15, 0),
])
def test_val_quota(n_new: int, n_prev: int, n_prev_val: int, f: float, expected: int) -> None:
    assert dataset.val_quota(n_new, n_prev, n_prev_val, f) == expected


@pytest.mark.parametrize("f", [-0.1, 1.0])
def test_val_quota_rejects_fraction_outside_unit_interval(f: float) -> None:
    with pytest.raises(ValueError):
        dataset.val_quota(3, 0, 0, f)


def test_choose_val_groups_is_seeded() -> None:
    first = dataset.choose_val_groups(20, 5, seed=42, cycle=3)
    assert first == dataset.choose_val_groups(20, 5, seed=42, cycle=3)
    assert len(first) == 5 and first <= set(range(20))
    assert dataset.choose_val_groups(4, 0, seed=1, cycle=1) == set()
    assert dataset.choose_val_groups(0, 0, seed=1, cycle=1) == set()


def test_next_photograph_ids() -> None:
    assert dataset.next_photograph_ids([3, 7, 5], 2) == [8, 9]
    assert dataset.next_photograph_ids([], 2) == [0, 1]
    assert dataset.next_photograph_ids([4], 0) == []


def test_pool_name_is_sanitised_and_unique_ignoring_case() -> None:
    name = dataset.pool_name("my source", Path("/in/IMG 01 (copy).JPG"), "abcdef123", set())
    assert name == "my_source__IMG_01_copy___abcdef"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name)
    taken = {name.casefold(), f"{name}_2".casefold()}
    # equal to `name` ignoring case, and "_2" is taken as well -> "_3"
    assert dataset.pool_name("my source", Path("img 01 (copy).jpg"), "ABCDEF99", taken) == \
        "my_source__img_01_copy___ABCDEF_3"
    long_stem = "x" * 200
    assert dataset.pool_name("s", Path(f"{long_stem}.jpg"), "0" * 32, set()) == f"s__{'x' * 80}__000000"


def test_clean_polygons_and_label_text() -> None:
    polys = dataset.clean_polygons([(1, [[-0.1, 0.5], [0.5, 1.2], [0.9, 0.9]])])
    assert polys[0][0] == 1 and polys[0][1].dtype == np.float32
    assert np.allclose(polys[0][1], [[0.0, 0.5], [0.5, 1.0], [0.9, 0.9]])
    assert dataset.yolo_label_text(polys) == "1 0.000000 0.500000 0.500000 1.000000 0.900000 0.900000\n"
    assert dataset.yolo_label_text([]) == ""
    with pytest.raises(ValueError, match="stage"):
        dataset.clean_polygons([(NUM_STAGES, [[0, 0], [1, 0], [1, 1]])])
    with pytest.raises(ValueError, match="three"):
        dataset.clean_polygons([(0, [[0, 0], [1, 1]])])


def test_plan_additions_policy() -> None:
    result = DedupeResult(
        new_groups=[[0, 1], [2, 3]],
        inherited=[PoolMatch(4, "3", "train", "train_a", "pixel", 0.97),
                   PoolMatch(5, "7", "val", "val_a", "phash", 1.0)],
        dropped_test_matches=[PoolMatch(6, "10", "test", "test_a", "phash", 0.0)],
        n_duplicates=6,                                  # 8 records, 2 new photographs
        conflicts=[dedupe.SplitConflict(7, (PoolMatch(7, "3", "train", "train_a", "orb", 30.0),
                                            PoolMatch(7, "7", "val", "val_a", "orb", 31.0)))])
    plan, skipped = dataset.plan_additions(result, [100, 101], val_groups={1})
    assert plan == {0: (100, "train"), 1: (100, "train"), 2: (101, "val"), 4: (3, "train")}
    reasons = dict(skipped)
    assert sorted(reasons) == [3, 5, 6, 7]
    assert "new val photograph 101" in reasons[3]
    assert "val photograph 7" in reasons[5]
    assert "frozen test" in reasons[6]
    assert "several splits" in reasons[7]


def test_added_photographs_counts_photographs_after_init() -> None:
    pool = pd.DataFrame({
        "name": ["a", "b", "c", "d", "e", "f"], "photograph": [1, 1, 2, 3, 3, 4],
        "split": ["train", "train", "val", "train", "train", "val"],
        "batch": ["init", "b1", "init", "b1", "b1", "b2"], "added_cycle": [0, 1, 0, 1, 1, 2]})
    assert dataset.added_photographs(pool) == (2, 1)      # photographs 3 (train) and 4 (val)


def test_write_pool_files_never_writes_through_a_link(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    for kind in KINDS:
        (dataset.pool_dir(ws) / kind).mkdir(parents=True)
    outside = tmp_path / "thesis_label.txt"
    outside.write_text("keep")
    pool_file(ws, "labels", "x").symlink_to(outside)
    img = tmp_path / "in.png"
    cv2.imwrite(str(img), smooth_image(3, (40, 60)))
    tri = [(2, np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], np.float32))]
    dataset.write_pool_files(ws, "x", Rec(img, tri))
    assert outside.read_text() == "keep"
    assert not pool_file(ws, "labels", "x").is_symlink()
    assert pool_file(ws, "images", "x").read_bytes()[:2] == b"\xff\xd8"      # JPEG
    mask = cv2.imread(str(pool_file(ws, "masks", "x")), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (40, 60) and set(np.unique(mask)) == {0, 3}
    assert np.array_equal(mask, bsd.rasterise(tri, 40, 60))


def test_verify_frozen_needs_version_0(tmp_path: Path) -> None:
    (tmp_path / "cand").mkdir()
    with pytest.raises(FrozenSplitError, match="missing"):
        dataset.verify_frozen(tmp_path / "v0", tmp_path / "cand")


# ============================================================ dataset: workspaces

@requires_data
def test_init_from_seg_reproduces_the_thesis_split(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws")
    pool = ws.read_manifest()
    seg = pd.read_csv(SMOKE / "manifest.csv", dtype=str)
    assert list(pool["name"]) == list(seg["name"])
    assert list(pool["split"]) == list(seg["split"])
    assert list(pool["photograph"]) == [int(p) for p in seg["photograph"]]
    assert set(pool["batch"]) == {"init"} and set(pool["added_cycle"]) == {0}
    assert set(pool.loc[pool["source"].str.startswith("roboflow"), "license"]) == {"CC BY 4.0"}
    for name, split in zip(pool["name"], pool["split"]):
        for kind in KINDS:
            linked = pool_file(ws, kind, name)
            assert linked.is_symlink() and linked.resolve() == smoke_file(kind, split, name).resolve()
    v0 = ws.dataset_dir(0)
    assert dataset.seg_split_names(v0) == dataset.seg_split_names(SMOKE)
    for kind in KINDS:
        for link in (v0 / kind).rglob(f"*{KINDS[kind]}"):
            assert link.is_symlink() and not os.path.isabs(os.readlink(link))
            assert link.resolve() == smoke_file(kind, link.parent.name, link.stem).resolve()
    test_names = sorted(pool.loc[pool["split"] == "test", "name"])
    assert (v0 / "test_frozen.txt").read_text() == "".join(f"{n}\n" for n in test_names)
    assert test_names == sorted((SMOKE / "test_frozen.txt").read_text().split())
    digests = (v0 / "frozen.sha256").read_text().splitlines()
    n_frozen = int(pool["split"].isin(FROZEN_SPLITS).sum())
    assert len(digests) == len(KINDS) * n_frozen
    assert all(re.fullmatch(r"[0-9a-f]{64}  (images|labels|masks)/(test|external)/\S+", line)
               for line in digests)
    spec = yaml.safe_load((v0 / "dataset.yaml").read_text())
    assert spec["path"] == str(v0) and spec["nc"] == NUM_STAGES and spec["names"] == STAGE_NAMES
    for split in ("train", "val", "test"):
        assert spec[split] == str(v0 / "images" / split) and Path(spec[split]).is_absolute()
    snapshot = pd.read_csv(v0 / "manifest.csv", dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(snapshot, pd.read_csv(ws.manifest_path, dtype=str,
                                                        keep_default_na=False))
    assert (v0 / "negatives_extra.csv").read_text() == (SMOKE / "negatives_extra.csv").read_text()
    assert json.loads((dataset.pool_dir(ws) / "init.json").read_text())["images"] == len(pool)
    assert not list(ws.datasets.glob(".*"))                     # no partial version left


@requires_data
def test_init_copy_mode_and_refusal_to_reinitialise(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws", copy=True)
    pool = ws.read_manifest()
    name, split = pool["name"].iloc[0], pool["split"].iloc[0]
    copied = pool_file(ws, "images", name)
    assert not copied.is_symlink()
    assert copied.read_bytes() == smoke_file("images", split, name).read_bytes()
    with pytest.raises(FileExistsError):
        dataset.init_from_seg(ws, SMOKE, log=quiet)


@requires_data
def test_init_rejects_an_inconsistent_thesis_dataset(tmp_path: Path) -> None:
    seg = tmp_path / "seg"
    shutil.copytree(SMOKE, seg)
    mask = Path("masks") / "test" / f"{(seg / 'test_frozen.txt').read_text().split()[0]}.png"
    (seg / mask).unlink()
    with pytest.raises(FileNotFoundError):
        dataset.init_from_seg(Workspace(tmp_path / "ws1"), seg, log=quiet)
    shutil.copyfile(SMOKE / mask, seg / mask)
    (seg / "test_frozen.txt").write_text("someone_else\n")
    with pytest.raises(ValueError, match="test_frozen"):
        dataset.init_from_seg(Workspace(tmp_path / "ws2"), seg, log=quiet)
    assert not (tmp_path / "ws1" / "pool" / "manifest.csv").exists()


@requires_data
def test_add_records_and_create_version(tmp_path: Path, batch: list[Rec],
                                        dedupe_result: DedupeResult, picks: Picks) -> None:
    ws = init_ws(tmp_path / "ws")
    before = ws.read_manifest()
    res = dataset.add_records(ws, batch, dedupe_result, cycle=1, val_fraction=0.5, seed=7, log=quiet)

    # two new photographs with fresh ids; f (P + n) = 0.5 * 2 = 1 of them in val
    first_id = int(before["photograph"].max()) + 1
    assert sorted(res.new_photographs) == [first_id, first_id + 1]
    assert sorted(res.new_photographs.values()) == ["train", "val"]
    pool = ws.read_manifest()
    pd.testing.assert_frame_equal(pool.iloc[:len(before)].reset_index(drop=True), before)
    new = pool.iloc[len(before):].set_index("name")
    assert len(new) == len(res.rows) == len(res.added)
    assert set(new["added_cycle"]) == {1} and set(new["batch"]) == {"batch1"}
    assert set(new["source"]) == {"woundcare_new"} and set(new["license"]) == {"CC BY 4.0"}
    assert set(new["split"]) <= {"train", "val"}

    # inheritance: the mirrored train image joins its photograph in train
    assert (new.loc[res.added[1], "split"], new.loc[res.added[1], "photograph"]) == \
        ("train", picks.train_photo)
    # train keeps every image of a new photograph, val its first image only
    for pid, group in zip(sorted(res.new_photographs), dedupe_result.new_groups):
        split = res.new_photographs[pid]
        kept = [i for i in group if i in res.added]
        assert kept == (group if split == "train" else group[:1])
        assert all((new.loc[res.added[i], "split"], new.loc[res.added[i], "photograph"]) == (split, pid)
                   for i in kept)
    reasons = dict(res.skipped)
    assert {0, 6, 7, 8} <= set(reasons) and set(reasons).isdisjoint(res.added)
    assert set(reasons) | set(res.added) == set(range(N_BATCH))
    assert "frozen test" in reasons[0] and "frozen external" in reasons[7]
    assert f"frozen {EXTRA_NEGATIVES}" in reasons[8] and "val photograph" in reasons[6]
    # frozen test: same test images in the pool
    assert sorted(pool.loc[pool["split"] == "test", "name"]) == \
        sorted(before.loc[before["split"] == "test", "name"])

    # pool files: JPEG image (also for the PNG input), label text and rasterised mask
    for i, name in res.added.items():
        src = cv2.imread(str(batch[i].path))
        image = pool_file(ws, "images", name)
        assert image.read_bytes()[:2] == b"\xff\xd8" and not image.is_symlink()
        assert cv2.imread(str(image)).shape == src.shape
        polys = dataset.clean_polygons(batch[i].polygons)
        assert pool_file(ws, "labels", name).read_text() == dataset.yolo_label_text(polys)
        mask = cv2.imread(str(pool_file(ws, "masks", name)), cv2.IMREAD_UNCHANGED)
        assert mask.dtype == np.uint8
        assert np.array_equal(mask, bsd.rasterise(polys, *src.shape[:2]))
    assert pool_file(ws, "labels", res.added[4]).read_text() == ""          # negative photo
    assert not cv2.imread(str(pool_file(ws, "masks", res.added[4])), cv2.IMREAD_UNCHANGED).any()
    assert any(dataset.clean_polygons(batch[i].polygons) for i in res.added)  # masks were drawn

    # version 1: same frozen splits as version 0, every pool image under its split
    assert dataset.create_version(ws, log=quiet) == 1
    v0, v1 = ws.dataset_dir(0), ws.dataset_dir(1)
    for fname in ("test_frozen.txt", "frozen.sha256"):
        assert (v1 / fname).read_bytes() == (v0 / fname).read_bytes()
    for kind in KINDS:
        for split in FROZEN_SPLITS:
            assert sorted(os.listdir(v1 / kind / split)) == sorted(os.listdir(v0 / kind / split))
            assert all((v1 / kind / split / f).resolve() == (v0 / kind / split / f).resolve()
                       for f in os.listdir(v0 / kind / split))
    listed = dataset.seg_split_names(v1)
    for split in ("train", "val", "test", "external"):
        assert listed[split] == sorted(pool.loc[pool["split"] == split, "name"])
    for i, name in res.added.items():
        split = new.loc[name, "split"]
        for kind, ext in KINDS.items():
            link = v1 / kind / split / f"{name}{ext}"
            assert link.is_symlink() and not os.path.isabs(os.readlink(link))
            assert link.resolve() == pool_file(ws, kind, name).resolve()
    spec = yaml.safe_load((v1 / "dataset.yaml").read_text())
    assert all(Path(spec[k]).is_absolute() and Path(spec[k]).is_dir()
               for k in ("path", "train", "val", "test"))
    assert spec["train"] == str(v1 / "images" / "train")
    pd.testing.assert_frame_equal(pd.read_csv(v1 / "manifest.csv", dtype=str, keep_default_na=False),
                                  pd.read_csv(ws.manifest_path, dtype=str, keep_default_na=False))
    assert not list(ws.datasets.glob(".*"))


@requires_data
def test_version_is_readable_by_the_training_code(tmp_path: Path, batch: list[Rec],
                                                  dedupe_result: DedupeResult) -> None:
    from ultralytics.data.utils import img2label_paths

    from selftrain.data import SegDataset
    from selftrain.evaluate import split_items

    ws = init_ws(tmp_path / "ws")
    res = dataset.add_records(ws, batch, dedupe_result, cycle=1, val_fraction=0.5, seed=7, log=quiet)
    v1 = ws.dataset_dir(dataset.create_version(ws, log=quiet))
    spec = yaml.safe_load((v1 / "dataset.yaml").read_text())
    images = sorted(str(p) for p in Path(spec["train"]).glob("*.jpg"))
    assert all(Path(label).is_file() for label in img2label_paths(images))   # Ultralytics' rule
    names = lambda root, split: [n for n, _, _ in split_items(root, split)]  # noqa: E731
    assert names(v1, "test") == names(ws.dataset_dir(0), "test")
    train = SegDataset(v1, "train", 64)
    added_train = [n for n in res.added.values() if n in {f.stem for f in train.files}]
    assert added_train
    x, y = train[[f.stem for f in train.files].index(added_train[0])]
    assert tuple(x.shape) == (3, 64, 64) and tuple(y.shape) == (64, 64)


@requires_data
def test_val_share_is_kept_across_cycles(tmp_path: Path, batch: list[Rec]) -> None:
    ws = init_ws(tmp_path / "ws")
    two_new = DedupeResult([[0], [1]], [], [], 0)
    first = dataset.add_records(ws, [batch[2], batch[4]], two_new, cycle=1, val_fraction=0.15,
                                seed=0, log=quiet)
    assert sorted(first.new_photographs.values()) == ["train", "train"]     # 0.15 * 2 -> 0
    second = dataset.add_records(ws, [batch[3], batch[5]], two_new, cycle=2, val_fraction=0.15,
                                 seed=0, log=quiet)
    assert sorted(second.new_photographs.values()) == ["train", "val"]      # 0.15 * 4 -> 1
    assert dataset.added_photographs(ws.read_manifest()) == (4, 1)
    assert dataset.create_version(ws, log=quiet) == 1
    assert dataset.create_version(ws, log=quiet) == 2                       # versions never reuse


@requires_data
def test_add_records_is_deterministic(tmp_path: Path, batch: list[Rec],
                                      dedupe_result: DedupeResult) -> None:
    rows = [dataset.add_records(init_ws(tmp_path / f"ws{k}"), batch, dedupe_result, cycle=1,
                                val_fraction=0.5, seed=11, log=quiet).rows for k in range(2)]
    assert rows[0] == rows[1]


@requires_data
def test_add_records_rejects_bad_input_without_writing(tmp_path: Path, batch: list[Rec],
                                                       dedupe_result: DedupeResult) -> None:
    ws = init_ws(tmp_path / "ws")
    n_files = len(list((dataset.pool_dir(ws) / "images").iterdir()))
    bad_calls = [dict(records=batch[:-1]), dict(cycle=0), dict(seed=-1), dict(val_fraction=1.0),
                 dict(records=[Rec(r.path, r.polygons, source="") for r in batch])]
    for bad in bad_calls:
        args = dict(records=batch, dedupe_result=dedupe_result, cycle=1, val_fraction=0.15, seed=0)
        with pytest.raises(ValueError):
            dataset.add_records(ws, **{**args, **bad}, log=quiet)
    assert len(ws.read_manifest()) == len(pd.read_csv(SMOKE / "manifest.csv"))
    assert len(list((dataset.pool_dir(ws) / "images").iterdir())) == n_files
    with pytest.raises(ValueError, match="initialise"):
        dataset.add_records(Workspace(tmp_path / "empty"), batch, dedupe_result, 1, 0.15, 0)


@requires_data
def test_create_version_rejects_a_changed_test_split(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws")
    pool = ws.read_manifest()
    single = pool[(pool["split"] == "train")
                  & ~pool["photograph"].duplicated(keep=False)].index[0]
    pool.loc[single, "split"] = "test"
    # written directly, bypassing Workspace.write_manifest's own test-set check,
    # as a manual edit or a faulty module would
    ws.manifest_path.write_text(pool.to_csv(index=False))
    with pytest.raises(FrozenSplitError, match="test_frozen.txt"):
        dataset.create_version(ws, log=quiet)
    assert not ws.dataset_dir(1).exists() and not list(ws.datasets.glob(".*"))


@requires_data
def test_create_version_rejects_changed_frozen_content(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws", copy=True)     # copies: editing them leaves data/seg_smoke intact
    name = (ws.dataset_dir(0) / "test_frozen.txt").read_text().split()[0]
    label = pool_file(ws, "labels", name)
    assert not label.is_symlink()
    label.write_text(label.read_text() + "0 0.1 0.1 0.2 0.1 0.2 0.2\n")
    with pytest.raises(FrozenSplitError, match="frozen.sha256"):
        dataset.create_version(ws, log=quiet)
    assert not ws.dataset_dir(1).exists() and not list(ws.datasets.glob(".*"))


@requires_data
def test_create_version_requires_every_pool_file(tmp_path: Path) -> None:
    ws = init_ws(tmp_path / "ws", copy=True)
    pool = ws.read_manifest()
    pool_file(ws, "masks", pool.loc[pool["split"] == "train", "name"].iloc[0]).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        dataset.create_version(ws, log=quiet)
    with pytest.raises(ValueError, match="empty"):
        dataset.create_version(Workspace(tmp_path / "empty"), log=quiet)
