"""Tests of selftrain.system.ingest (SPEC §2) and of the two steps that consume its
Records first: quality control (SPEC §3 step 2) and the drift test (step 4).

Batches are built in tmp_path from a few real photographs and YOLO-seg labels of
data/raw/roboflow_mobile/train (classes 0 NonPU, 1-4 Stage1-Stage4). Every label format
is derived from the same polygons, so the formats are checked against each other. Quality
control and the drift test are called as selftrain.system.steps calls them:
quality.check_all(records, cfg) and drift.drift_report(paths, drift.pool_reference(ws),
alpha)."""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import uuid
import zipfile
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from selftrain.config import ROOT
from selftrain.system import dataset, drift, ingest, quality
from selftrain.system.active import list_unlabeled
from selftrain.system.cycle import inbox_signature
from selftrain.system.dataset import pool_file
from selftrain.system.ingest import BACKGROUND, DROP, Batch, IngestError, Record, SourceSpec
from selftrain.system.steps import (CycleContext, step_dataset, step_dedupe, step_drift,
                                    step_ingest, step_quality)
from selftrain.system.workspace import UNSTATED_LICENSE, Workspace

MOBILE = ROOT / "data" / "raw" / "roboflow_mobile"
SMOKE = ROOT / "data" / "seg_smoke"
pytestmark = pytest.mark.skipif(not (MOBILE / "train" / "labels").is_dir(),
                                reason="data/raw/roboflow_mobile missing")

MOBILE_MAP = {0: "background", 1: 1, 2: 2, 3: 3, 4: 4}
SOURCE = {"name": "mobile_test", "license": "CC BY 4.0", "class_map": MOBILE_MAP}
TRIANGLE = "0.1 0.1 0.5 0.1 0.3 0.5"
Sample = tuple[Path, Path]          # (image, YOLO label) of roboflow_mobile/train


# ------------------------------------------------------------------ helpers

def read_yolo(path: Path) -> list[tuple[int, np.ndarray]]:
    """(class, (N, 2) float32) for every line of a YOLO label file."""
    return [(int(v[0]), np.asarray(v[1:], np.float32).reshape(-1, 2))
            for v in (line.split() for line in path.read_text().splitlines()) if v]


def expected(label: Path) -> list[tuple[int, np.ndarray]]:
    """Stage polygons of a mobile label under MOBILE_MAP (class c -> stage index c - 1)."""
    return [(c - 1, p) for c, p in read_yolo(label)]


def image_hw(image: Path) -> tuple[int, int]:
    """(height, width) of an image as OpenCV decodes it."""
    h, w = cv2.imread(str(image)).shape[:2]
    return h, w


def write_source(folder: Path, spec: dict | None = SOURCE) -> None:
    """Write `spec` as the source.yaml of a batch folder (none if `spec` is None)."""
    folder.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (folder / "source.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def yolo_unit(folder: Path, samples: list[Sample]) -> Path:
    """images/ + labels/ copies of the samples below `folder`."""
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    for image, label in samples:
        shutil.copy(image, folder / "images" / image.name)
        shutil.copy(label, folder / "labels" / label.name)
    return folder


def stage_mask(image: Path, label: Path) -> np.ndarray:
    """The pool mask of a sample (values stage + 1), via build_seg_dataset.rasterise."""
    return quality.bsd.rasterise(expected(label), *image_hw(image))


def coco_document(samples: list[Sample], categories: list[dict], category_of: dict[int, int]
                  ) -> dict:
    """COCO instance segmentation of the samples: polygons in pixels, one annotation
    per YOLO line (class c -> category category_of[c])."""
    images, annotations = [], []
    for k, (image, label) in enumerate(samples, 1):
        h, w = image_hw(image)
        images.append({"id": k, "file_name": image.name, "width": w, "height": h})
        for c, pts in read_yolo(label):
            px = pts * np.array([w, h], np.float32)
            x0, y0 = px.min(axis=0)
            x1, y1 = px.max(axis=0)
            annotations.append({"id": len(annotations) + 1, "image_id": k,
                                "category_id": category_of[c], "iscrowd": 0,
                                "segmentation": [px.ravel().astype(float).tolist()],
                                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]})
    return {"images": images, "annotations": annotations, "categories": categories}


ROBOFLOW_CATEGORIES = [{"id": 0, "name": "pressure-ulcers", "supercategory": "none"},
                       {"id": 1, "name": "NonPU"}, {"id": 2, "name": "Stage1"},
                       {"id": 3, "name": "Stage2"}, {"id": 4, "name": "Stage3"},
                       {"id": 5, "name": "Stage4"}]


def ingest_only_batch(ws: Workspace) -> tuple[Batch, list[Record]]:
    """scan_inbox (exactly one batch expected) and load_batch."""
    batches = ingest.scan_inbox(ws)
    assert len(batches) == 1, batches
    return batches[0], ingest.load_batch(batches[0])


def by_name(records: list[Record]) -> dict[str, Record]:
    """Records by image file name."""
    return {r.path.name: r for r in records}


def same_polygons(got: list[tuple[int, np.ndarray]], want: list[tuple[int, np.ndarray]],
                  atol: float = 0.0) -> None:
    """Same stages in the same order; float32 (N, 2) vertices equal within `atol`."""
    assert [s for s, _ in got] == [s for s, _ in want]
    for (_, g), (_, w) in zip(got, want):
        assert g.dtype == np.float32 and g.ndim == 2 and g.shape[1] == 2
        assert g.shape == w.shape
        assert np.allclose(g, w, rtol=0.0, atol=atol)


def report_of(folder: Path) -> dict:
    """The ingest_report.json of a processed or rejected batch folder."""
    return json.loads((folder / ingest.INGEST_REPORT).read_text())


def codes(rejected: list[tuple[Record, str]]) -> dict[str, str]:
    """{image file name: rejection code}."""
    return {rec.path.name: quality.reason_code(reason) for rec, reason in rejected}


@pytest.fixture(scope="module")
def samples() -> list[Sample]:
    """Six mobile photographs whose labels hold only stage outlines inside the image."""
    picked: list[Sample] = []
    for label in sorted((MOBILE / "train" / "labels").glob("*.txt")):
        image = MOBILE / "train" / "images" / f"{label.stem}.jpg"
        regions = read_yolo(label)
        if image.is_file() and regions and all(1 <= c <= 4 and len(p) >= 3 and (p >= 0).all()
                                               and (p <= 1).all() for c, p in regions):
            picked.append((image, label))
        if len(picked) == 6:
            break
    assert len(picked) == 6
    return picked


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    """A fresh workspace below tmp_path."""
    return Workspace(tmp_path / "ws").ensure()


@pytest.fixture
def cfg(ws: Workspace):
    """The default SystemConfig of the workspace."""
    return ws.load_config()


# ------------------------------------------------------------------ data types and ids

def test_record_has_the_spec_fields_and_identity_equality(tmp_path: Path) -> None:
    assert [f.name for f in fields(Record)] == ["path", "polygons", "source", "license",
                                                "batch", "issue"]
    a = Record(tmp_path / "x.jpg", [(0, np.zeros((3, 2), np.float32))], "s", "l", "b")
    b = Record(tmp_path / "x.jpg", [(0, np.zeros((3, 2), np.float32))], "s", "l", "b")
    assert a.issue is None and a == a and a != b and a in [a] and b not in [a]


@pytest.mark.parametrize("entry, kind, expected_id", [
    ("clinic A", "folder", "clinic_A"),
    ("Klinik München (2).zip", "zip", "Klinik_M_nchen_2"),
    ("Archive.ZIP", "zip", "Archive"),
    ("photo.zip.txt", "other", "photo.zip.txt"),
    ("...", "folder", "batch"),
])
def test_batch_id(entry: str, kind: str, expected_id: str) -> None:
    assert ingest.batch_id(entry, kind) == expected_id


def test_scan_inbox_lists_every_waiting_entry_with_a_unique_id(ws: Workspace) -> None:
    inbox = ws.inbox
    (inbox / "clinic A").mkdir()
    (inbox / "clinic A" / "x.jpg").write_bytes(b"x")
    (inbox / "clinic_A.zip").write_bytes(b"not read by the scan")
    (inbox / "notes.txt").write_text("a loose file")
    (inbox / "empty").mkdir()                                      # no file yet
    (inbox / "hidden_only").mkdir()
    (inbox / "hidden_only" / ".DS_Store").write_bytes(b"")
    (inbox / ".upload-1234").mkdir()                               # dashboard staging
    (inbox / ".upload-1234" / "a.jpg").write_bytes(b"x")
    (ws.unlabeled / "y.jpg").write_bytes(b"x")                     # reserved folder
    (ws.processed / "Notes.txt").mkdir()                           # id taken (any case)
    batches = ingest.scan_inbox(ws)
    assert [(b.name, b.kind, b.path.name) for b in batches] == [
        ("clinic_A", "folder", "clinic A"), ("clinic_A_2", "zip", "clinic_A.zip"),
        ("notes.txt_2", "other", "notes.txt")]
    # exactly the entries that make the cycle runner start a cycle are batches
    assert {rel.split("/")[0] for rel, _, _ in inbox_signature(ws)} == {b.path.name for b in batches}
    assert all(b.ws == ws for b in batches)
    assert inbox.joinpath("clinic A", "x.jpg").exists()           # scanning moves nothing


def test_scan_inbox_without_inbox(tmp_path: Path) -> None:
    assert ingest.scan_inbox(Workspace(tmp_path / "nothing")) == []


@pytest.mark.parametrize("name, stage", [
    ("Stage1", 1), ("stage-2", 2), ("STAGE_3", 3), ("Stage 4", 4), ("stage5", None),
    ("NonPU", None), ("DTI", None), ("State1", None), ("1", None)])
def test_stage_from_name(name: str, stage: int | None) -> None:
    assert ingest.stage_from_name(name) == stage


# ------------------------------------------------------------------ formats

def test_yolo_batch_is_read_filed_and_reported(ws: Workspace, samples: list[Sample]) -> None:
    folder = yolo_unit(ws.inbox / "mobile", samples[:3])
    write_source(folder)
    batch, records = ingest_only_batch(ws)
    target = ws.processed / "mobile"
    assert (batch.name, batch.kind) == ("mobile", "folder")
    assert not folder.exists() and target.is_dir()
    assert sorted(r.path for r in records) == sorted(target / "images" / i.name for i, _ in samples[:3])
    recs = by_name(records)
    for image, label in samples[:3]:
        rec = recs[image.name]
        assert rec.path.is_file() and rec.path.is_absolute()
        assert (rec.source, rec.license, rec.batch, rec.issue) == ("mobile_test", "CC BY 4.0",
                                                                   "mobile", None)
        same_polygons(rec.polygons, expected(label))
    report = report_of(target)
    assert report["status"] == "processed" and report["batch"] == "mobile"
    assert report["counts"]["labelled"] == 3 and report["counts"]["records"] == 3
    assert report["label_formats"] == {"yolo": 3}
    assert report["source"]["class_map"] == {"0": "background", "1": 1, "2": 2, "3": 3, "4": 4}
    assert {e["image"] for e in report["images"]} == {f"images/{i.name}" for i, _ in samples[:3]}
    assert all(e["label_file"].startswith("labels/") for e in report["images"])


def test_defaults_without_source_yaml(ws: Workspace, samples: list[Sample], cfg) -> None:
    folder = yolo_unit(ws.inbox / "plain", samples[:1])
    label = folder / "labels" / samples[0][1].name
    label.write_text(f"4 {TRIANGLE}\n2 0.6 0.6 0.9 0.6 0.8 0.9\n")
    _, records = ingest_only_batch(ws)
    (rec,) = records
    assert (rec.source, rec.license) == ("plain", UNSTATED_LICENSE)
    # default class map: YOLO class c -> stage c + 1 (index c); class 4 is not covered
    assert [s for s, _ in rec.polygons] == [quality.unknown_stage(4), 2]
    accepted, rejected = quality.check_all(records, cfg)
    assert accepted == [] and quality.reason_code(rejected[0][1]) == "unknown_class"
    assert "source class 4" in rejected[0][1]


def test_known_source_name_brings_its_licence() -> None:
    spec = ingest.parse_source({"name": "roboflow_mobile"}, "b1")
    assert (spec.name, spec.license, spec.negatives, spec.class_map) == (
        "roboflow_mobile", "CC BY 4.0", False, None)


def test_roboflow_export_layout(ws: Workspace, samples: list[Sample]) -> None:
    root = ws.inbox / "rf_export"
    for split, part in (("train", samples[:2]), ("valid", samples[2:3]), ("test", samples[3:4])):
        yolo_unit(root / split, part)
    shutil.copy(MOBILE / "data.yaml", root / "data.yaml")
    (root / "README.roboflow.txt").write_text("exported from roboflow")
    write_source(root)
    batch, records = ingest_only_batch(ws)
    target = ws.processed / batch.name
    splits = sorted(r.path.relative_to(target).parts[0] for r in records)
    assert splits == ["test", "train", "train", "valid"]
    recs = by_name(records)
    for image, label in samples[:4]:
        same_polygons(recs[image.name].polygons, expected(label))
    assert report_of(target)["units"] == [{"folder": "train", "images": 2},
                                          {"folder": "valid", "images": 1},
                                          {"folder": "test", "images": 1}]


def test_layout_folder_names_are_matched_ignoring_case(ws: Workspace, samples: list[Sample]) -> None:
    root = ws.inbox / "cases"
    for sub in ("Train/Images", "Train/Labels"):
        (root / sub).mkdir(parents=True)
    image, label = samples[0]
    shutil.copy(image, root / "Train" / "Images" / image.name)
    shutil.copy(label, root / "Train" / "Labels" / label.name)
    write_source(root)
    _, (rec,) = ingest_only_batch(ws)
    assert rec.path == ws.processed / "cases" / "Train" / "Images" / image.name
    same_polygons(rec.polygons, expected(label))


@pytest.mark.skipif(not (SMOKE / "manifest.csv").is_file(), reason="data/seg_smoke missing")
def test_pool_export_layout_with_nested_split_folders(ws: Workspace) -> None:
    """images|labels|masks/{train,val}/ as written by build_seg_dataset: labels are
    looked up by the path below images/, the YOLO labels (stage indices, the default
    map) win over the equivalent masks."""
    folder = ws.inbox / "seg_export"
    picked = []
    for split in ("train", "val"):
        for image in sorted((SMOKE / "images" / split).glob("*.jpg"))[:2]:
            for kind, suffix in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
                dst = folder / kind / split / f"{image.stem}{suffix}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(SMOKE / kind / split / f"{image.stem}{suffix}", dst)
            picked.append((split, image.stem))
    _, records = ingest_only_batch(ws)
    target = ws.processed / "seg_export"
    assert sorted(r.path for r in records) == sorted(target / "images" / sp / f"{st}.jpg"
                                                    for sp, st in picked)
    for rec in records:
        split = rec.path.parent.name
        want = read_yolo(SMOKE / "labels" / split / f"{rec.path.stem}.txt")
        same_polygons(rec.polygons, want, atol=1e-6)
    report = report_of(target)
    assert report["label_formats"] == {"yolo": 4}
    assert {i["reason"] for i in report["ignored"]} == {"superseded by the YOLO or COCO label of "
                                                        "the image"}


def test_data_yaml_contradicting_the_default_map_rejects_the_batch(ws: Workspace,
                                                                   samples: list[Sample]) -> None:
    root = ws.inbox / "rf_no_map"
    yolo_unit(root / "train", samples[:2])
    shutil.copy(MOBILE / "data.yaml", root / "data.yaml")         # names: NonPU, Stage1, ...
    batch = ingest.scan_inbox(ws)[0]
    with pytest.raises(IngestError, match="NonPU"):
        ingest.load_batch(batch)
    assert not root.exists() and not (ws.processed / "rf_no_map").exists()
    rejected = ws.rejected / "rf_no_map"
    assert (rejected / "train" / "images" / samples[0][0].name).is_file()   # moved unchanged
    report = report_of(rejected)
    assert report["status"] == "rejected" and "class_map" in report["reason"]


def test_mask_batch_round_trips_the_polygons(ws: Workspace, samples: list[Sample]) -> None:
    folder = ws.inbox / "masks"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    for image, label in samples[:4]:
        shutil.copy(image, folder / "images" / image.name)
        assert cv2.imwrite(str(folder / "masks" / f"{image.stem}.png"), stage_mask(image, label))
    write_source(folder, {"name": "mobile_masks"})                 # default: value v -> stage v
    _, records = ingest_only_batch(ws)
    recs = by_name(records)
    for image, label in samples[:4]:
        rec = recs[image.name]
        assert rec.issue is None and rec.polygons
        assert all(p.dtype == np.float32 and p.shape[1] == 2 for _, p in rec.polygons)
        original = stage_mask(image, label)
        again = quality.bsd.rasterise([(s, np.clip(p, 0, 1)) for s, p in rec.polygons],
                                      *original.shape)
        fg_a, fg_b = original > 0, again > 0
        assert (fg_a & fg_b).sum() / (fg_a | fg_b).sum() > 0.95          # foreground IoU
        both = fg_a & fg_b
        assert (original[both] == again[both]).mean() > 0.99            # same stages
        assert {s for s, _ in rec.polygons} == {int(v) - 1 for v in np.unique(original) if v}


def test_mask_specks_holes_and_nested_stages(ws: Workspace, samples: list[Sample]) -> None:
    folder = ws.inbox / "nested"
    (folder / "images").mkdir(parents=True)
    (folder / "masks").mkdir()
    image = samples[0][0]
    shutil.copy(image, folder / "images" / image.name)
    mask = np.zeros(image_hw(image), np.uint8)
    mask[100:300, 100:300] = 2                       # stage 2 around ...
    mask[150:250, 150:250] = 4                       # ... a nested stage-4 region
    mask[500:503, 500:503] = 1                       # a 3 x 3 speck
    cv2.imwrite(str(folder / "masks" / f"{image.stem}.png"), mask)
    _, (rec,) = ingest_only_batch(ws)
    assert sorted(s for s, _ in rec.polygons) == [1, 3]
    again = quality.bsd.rasterise(rec.polygons, *mask.shape)
    mask[500:503, 500:503] = 0
    assert np.array_equal(again, mask)
    assert report_of(ws.processed / "nested")["counts"]["mask_specks_removed"] == 1


@pytest.mark.parametrize("layout", ["images_and_annotations", "roboflow_coco_export"])
def test_coco_batch_with_default_category_names(ws: Workspace, samples: list[Sample],
                                                layout: str) -> None:
    doc = coco_document(samples[:3], ROBOFLOW_CATEGORIES, {c: c + 1 for c in range(5)})
    if layout == "images_and_annotations":
        unit = ws.inbox / "coco"
        image_dir, json_name = unit / "images", "annotations.json"
    else:                                            # Roboflow COCO: images beside the json
        unit = ws.inbox / "coco" / "train"
        image_dir, json_name = unit, "_annotations.coco.json"
    image_dir.mkdir(parents=True)
    for image, _ in samples[:3]:
        shutil.copy(image, image_dir / image.name)
    (unit / json_name).write_text(json.dumps(doc))
    _, records = ingest_only_batch(ws)
    recs = by_name(records)
    for image, label in samples[:3]:
        same_polygons(recs[image.name].polygons, expected(label), atol=2e-6)
    report = report_of(ws.processed / "coco")
    assert report["label_formats"] == {"coco": 3} and report["counts"]["labelled"] == 3


def test_coco_class_map_by_name_and_id(ws: Workspace, samples: list[Sample], cfg) -> None:
    doc = coco_document(samples[:2], ROBOFLOW_CATEGORIES, {c: c + 1 for c in range(5)})
    doc["annotations"].append({"id": 99, "image_id": 1, "category_id": 1, "iscrowd": 0,
                               "segmentation": [[10, 10, 60, 10, 30, 50]]})       # NonPU
    for name, source in (("by_default", None),
                         ("by_name", {"class_map": {"NonPU": "background", "Stage1": 1,
                                                    "Stage2": 2, "Stage3": 3, "Stage4": 4}}),
                         ("by_id", {"class_map": {1: "background", 2: 1, 3: 2, 4: 3, 5: 4}})):
        folder = ws.inbox / name
        (folder / "images").mkdir(parents=True)
        for image, _ in samples[:2]:
            shutil.copy(image, folder / "images" / image.name)
        (folder / "annotations.json").write_text(json.dumps(doc))
        write_source(folder, source)
    results = {b.name: by_name(ingest.load_batch(b)) for b in ingest.scan_inbox(ws)}
    first = samples[0][0].name
    for name in ("by_name", "by_id"):
        same_polygons(results[name][first].polygons, expected(samples[0][1]), atol=2e-6)
    # without a class map "NonPU" is no stage name: the image is rejected as unknown_class
    stages = [s for s, _ in results["by_default"][first].polygons]
    assert quality.unknown_stage(1) in stages
    accepted, rejected = quality.check_all(list(results["by_default"].values()), cfg)
    assert codes(rejected) == {first: "unknown_class"} and len(accepted) == 1


def test_coco_class_map_conflict_rejects_the_batch(ws: Workspace, samples: list[Sample]) -> None:
    folder = ws.inbox / "conflict"
    (folder / "images").mkdir(parents=True)
    shutil.copy(samples[0][0], folder / "images" / samples[0][0].name)
    doc = coco_document(samples[:1], ROBOFLOW_CATEGORIES, {c: c + 1 for c in range(5)})
    (folder / "annotations.json").write_text(json.dumps(doc))
    write_source(folder, {"class_map": {2: 1, "Stage1": 3}})        # category 2 is "Stage1"
    with pytest.raises(IngestError, match="by its name"):
        ingest_only_batch(ws)
    assert report_of(ws.rejected / "conflict")["status"] == "rejected"


# ------------------------------------------------------------------ labels, negatives, unlabeled

def test_unlabeled_photos_are_moved_for_active_learning(ws: Workspace, samples: list[Sample]) -> None:
    folder = yolo_unit(ws.inbox / "partial", samples[:2])
    for image, _ in samples[2:4]:
        shutil.copy(image, folder / "images" / image.name)        # no label
    (folder / "labels" / f"{samples[1][0].stem}.txt").write_text("")   # empty label
    write_source(folder)
    _, records = ingest_only_batch(ws)
    recs = by_name(records)
    assert set(recs) == {samples[0][0].name, samples[1][0].name}
    assert recs[samples[1][0].name].polygons == []                # labelled ulcer-free
    moved = sorted(ws.unlabeled.joinpath("partial", "images", i.name) for i, _ in samples[2:4])
    assert all(p.is_file() for p in moved) and list_unlabeled(ws) == moved
    assert not any((ws.processed / "partial" / "images" / i.name).exists() for i, _ in samples[2:4])
    counts = report_of(ws.processed / "partial")["counts"]
    assert (counts["labelled"], counts["ulcer_free"], counts["unlabeled"], counts["records"]) == (1, 1, 2, 2)


def test_negatives_flag_turns_unlabeled_photos_into_ulcer_free_records(
        ws: Workspace, samples: list[Sample]) -> None:
    folder = ws.inbox / "healthy_skin"
    (folder / "images").mkdir(parents=True)
    for image, _ in samples[:3]:
        shutil.copy(image, folder / "images" / image.name)
    write_source(folder, {"name": "healthy", "license": "CC0", "negatives": True})
    _, records = ingest_only_batch(ws)
    assert len(records) == 3 and all(r.polygons == [] and r.issue is None for r in records)
    assert list_unlabeled(ws) == []


def test_coco_unlisted_image_is_unlabeled_and_listed_without_annotations_is_ulcer_free(
        ws: Workspace, samples: list[Sample]) -> None:
    folder = ws.inbox / "coco_partial"
    (folder / "images").mkdir(parents=True)
    for image, _ in samples[:3]:
        shutil.copy(image, folder / "images" / image.name)
    doc = coco_document(samples[:2], ROBOFLOW_CATEGORIES, {c: c + 1 for c in range(5)})
    doc["annotations"] = [a for a in doc["annotations"] if a["image_id"] == 1]
    doc["images"].append({"id": 7, "file_name": "../outside.jpg", "width": 1, "height": 1})
    (folder / "annotations.json").write_text(json.dumps(doc))
    _, records = ingest_only_batch(ws)
    recs = by_name(records)
    assert set(recs) == {samples[0][0].name, samples[1][0].name}
    assert recs[samples[0][0].name].polygons and recs[samples[1][0].name].polygons == []
    assert list_unlabeled(ws) == [ws.unlabeled / "coco_partial" / "images" / samples[2][0].name]
    ignored = report_of(ws.processed / "coco_partial")["ignored"]
    assert any("../outside.jpg" in item["file"] and "unsafe" in item["reason"] for item in ignored)


def test_ingest_issues_are_rejected_by_quality_with_their_reason(
        ws: Workspace, samples: list[Sample], cfg) -> None:
    folder = ws.inbox / "issues"
    for sub in ("images", "labels", "masks"):
        (folder / sub).mkdir(parents=True)
    photo = samples[0][0]
    h, w = image_hw(photo)
    names = ("malformed odd box dropped background_only size wrong_bytes colour rle bbox "
             "no_width coco_size").split()
    for name in names:
        shutil.copy(photo, folder / "images" / f"{name}.jpg")
    labels = {"malformed": "1 0.1 0.1 abc 0.5 0.3 0.5", "odd": "1 0.1 0.1 0.5 0.1 0.3",
              "box": "2 0.5 0.5 0.2 0.2", "dropped": f"5 {TRIANGLE}\n1 0.6 0.6 0.9 0.6 0.8 0.9",
              "background_only": f"0 {TRIANGLE}"}
    for name, text in labels.items():
        (folder / "labels" / f"{name}.txt").write_text(text + "\n")
    cv2.imwrite(str(folder / "masks" / "size.png"), np.ones((100, 100), np.uint8))
    (folder / "masks" / "wrong_bytes.png").write_bytes(b"this is no png")
    colour = np.zeros((h, w, 3), np.uint8)
    colour[10:50, 10:50] = (0, 0, 255)
    cv2.imwrite(str(folder / "masks" / "colour.png"), colour)
    entries = {"rle": {"segmentation": {"size": [h, w], "counts": "abc"}, "iscrowd": 1},
               "bbox": {"segmentation": [], "bbox": [1, 1, 20, 20]},
               "no_width": {"segmentation": [[1, 1, 30, 1, 15, 25]]},
               "coco_size": {"segmentation": [[1, 1, 30, 1, 15, 25]]}}
    doc = {"categories": [{"id": 2, "name": "Stage2"}], "images": [], "annotations": []}
    for k, (name, ann) in enumerate(entries.items(), 1):
        size = {"no_width": {"height": h}, "coco_size": {"width": 100, "height": 100}}.get(
            name, {"width": w, "height": h})
        doc["images"].append({"id": k, "file_name": f"{name}.jpg", **size})
        doc["annotations"].append({"id": k, "image_id": k, "category_id": 2, **ann})
    (folder / "annotations.json").write_text(json.dumps(doc))
    write_source(folder, {"class_map": {**MOBILE_MAP, 5: "drop"}})
    _, records = ingest_only_batch(ws)
    issue_codes = {r.path.stem: quality.reason_code(r.issue) for r in records if r.issue}
    assert issue_codes == {"malformed": "malformed_label", "odd": "malformed_label",
                           "box": "box_label", "dropped": "dropped_class",
                           "size": "size_mismatch", "wrong_bytes": "unreadable_mask",
                           "colour": "unreadable_mask", "rle": "malformed_label",
                           "bbox": "box_label", "no_width": "malformed_label",
                           "coco_size": "size_mismatch"}
    assert set(issue_codes.values()) <= set(quality.INGEST_CODES)
    assert all(r.polygons == [] for r in records if r.issue)
    accepted, rejected = quality.check_all(records, cfg)
    assert [r.path.stem for r in accepted] == ["background_only"]
    assert {rec.path.stem: reason for rec, reason in rejected} == {
        r.path.stem: r.issue for r in records if r.issue}                    # reason unchanged
    report = report_of(ws.processed / "issues")
    assert report["counts"]["issue"] == 11 and report["counts"]["ulcer_free"] == 1
    assert "line 1" in next(e["issue"] for e in report["images"] if e["image"] == "images/malformed.jpg")


def test_quality_refuses_an_issue_without_a_known_code(samples: list[Sample], cfg) -> None:
    record = Record(samples[0][0], [], "s", "l", "b", issue="label looks odd")
    with pytest.raises(ValueError, match="not a rejection reason"):
        quality.check_all([record], cfg)
    record.issue = "strange_code: label looks odd"
    with pytest.raises(ValueError, match="no rejection code"):
        quality.check_all([record], cfg)


def test_quality_rejects_bad_geometry_and_unreadable_images(
        ws: Workspace, samples: list[Sample], cfg) -> None:
    folder = yolo_unit(ws.inbox / "quality", samples[:1])          # one good photograph
    (folder / "masks").mkdir()
    photo = samples[1][0]
    for name in ("whole_frame", "tiny", "outside"):
        shutil.copy(photo, folder / "images" / f"{name}.jpg")
    (folder / "images" / "broken.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    whole = np.full(image_hw(photo), 2, np.uint8)                  # every pixel stage 2
    cv2.imwrite(str(folder / "masks" / "whole_frame.png"), whole)
    (folder / "labels" / "tiny.txt").write_text("1 0.5 0.5 0.502 0.5 0.5 0.502\n")
    (folder / "labels" / "outside.txt").write_text("1 0.1 0.1 1.05 0.1 0.5 0.6\n")
    (folder / "labels" / "broken.txt").write_text(f"1 {TRIANGLE}\n")
    write_source(folder)
    _, records = ingest_only_batch(ws)
    assert all(r.issue is None for r in records)
    outside = by_name(records)["outside.jpg"].polygons[0][1]
    assert outside[:, 0].max() == pytest.approx(1.05)              # passed on unclipped
    accepted, rejected = quality.check_all(records, cfg)
    assert [r.path.name for r in accepted] == [samples[0][0].name]
    assert codes(rejected) == {"whole_frame.jpg": "whole_frame_label",
                               "tiny.jpg": "region_too_small",
                               "outside.jpg": "coordinates_out_of_range",
                               "broken.jpg": "unreadable_image"}


# ------------------------------------------------------------------ source.yaml

@pytest.mark.parametrize("text, message", [
    ("licence: CC BY 4.0\n", "unknown key"),
    ("class_map: {1: 5}\n", "allowed"),
    ("class_map: {1: stage1}\n", "allowed"),
    ("class_map: [1, 2]\n", "non-empty mapping"),
    ("class_map: {}\n", "non-empty mapping"),
    ("class_map: {1: 1, '1': 2}\n", "twice"),
    ("class_map: {-1: 1}\n", "integer >= 0"),
    ("class_map: {Stage1: 1}\n", "no COCO category"),
    ("negatives: 'yes'\n", "true or false"),
    ("name: 123\n", "must be text"),
    ("name: [unclosed\n", "cannot be read"),
    ("- just\n- a list\n", "must be a mapping"),
])
def test_invalid_source_yaml_rejects_the_batch(ws: Workspace, samples: list[Sample],
                                               text: str, message: str) -> None:
    folder = yolo_unit(ws.inbox / "bad_source", samples[:1])
    (folder / "source.yaml").write_text(text)
    with pytest.raises(IngestError, match=message):
        ingest_only_batch(ws)
    rejected = ws.rejected / "bad_source"
    assert (rejected / "source.yaml").read_text() == text
    assert message in report_of(rejected)["reason"]
    assert list(ws.processed.iterdir()) == []


def test_source_yaml_and_yml_together_are_ambiguous(ws: Workspace, samples: list[Sample]) -> None:
    folder = yolo_unit(ws.inbox / "two_sources", samples[:1])
    write_source(folder)
    (folder / "source.yml").write_text("name: other\n")
    with pytest.raises(IngestError, match="keep one"):
        ingest_only_batch(ws)


def test_parse_source_normalises_the_class_map() -> None:
    spec = ingest.parse_source({"name": " clinic  north ", "license": "CC BY 4.0",
                                "negatives": True,
                                "class_map": {0: "Background", "1": 1, 4: 4, 7: "DROP",
                                              "NonPU": "background"}}, "b1", "source.yaml")
    assert spec == SourceSpec("clinic north", "CC BY 4.0", True,
                              {0: BACKGROUND, 1: 0, 4: 3, 7: DROP, "NonPU": BACKGROUND},
                              "source.yaml")
    assert ingest.parse_source(None, "b1") == SourceSpec("b1", UNSTATED_LICENSE)


# ------------------------------------------------------------------ archives

def zip_folder(folder: Path, archive: Path, prefix: str = "", extra: dict[str, bytes] | None = None
               ) -> None:
    """Zip every file below `folder` (member names prefixed) plus `extra` members."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, prefix + path.relative_to(folder).as_posix())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)


def test_zip_batch_is_extracted_filed_and_removed(ws: Workspace, samples: list[Sample],
                                                  tmp_path: Path) -> None:
    src = yolo_unit(tmp_path / "src", samples[:2])
    write_source(src)
    archive = ws.inbox / "clinic B.zip"
    zip_folder(src, archive, prefix="clinicB/",
               extra={"__MACOSX/clinicB/._x.jpg": b"resource fork", "clinicB/.DS_Store": b""})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    batch, records = ingest_only_batch(ws)
    assert (batch.name, batch.kind) == ("clinic_B", "zip")
    target = ws.processed / "clinic_B"
    assert not archive.exists()
    assert sorted(r.path for r in records) == sorted(target / "clinicB" / "images" / i.name
                                                     for i, _ in samples[:2])
    recs = by_name(records)
    for image, label in samples[:2]:
        same_polygons(recs[image.name].polygons, expected(label))
    report = report_of(target)
    assert report["origin"]["sha256"] == digest and report["origin"]["kind"] == "zip"
    assert report["root"] == "clinicB"
    assert not (target / "__MACOSX").exists() and not (target / "clinicB" / ".DS_Store").exists()
    assert [p.name for p in ws.processed.iterdir()] == ["clinic_B"]          # no staging left


def _encrypted_flag(data: bytes) -> bytes:
    """Set the 'encrypted' bit of the first member in its local and central header."""
    out = bytearray(data)
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        out[out.find(signature) + offset] |= 0x1
    return bytes(out)


def _malicious(kind: str, image: bytes, marker: str) -> bytes:
    import io
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if kind == "encrypted":
            zf.writestr("images/secret.jpg", image)
        zf.writestr("images/ok.jpg", image)
        if kind == "dotdot":
            zf.writestr(f"../{marker}.txt", b"escaped")
        elif kind == "nested_dotdot":
            zf.writestr(f"images/../../{marker}.txt", b"escaped")
        elif kind == "backslash_dotdot":
            zf.writestr(f"images\\..\\..\\{marker}.txt", b"escaped")
        elif kind == "absolute":
            zf.writestr(f"/tmp/{marker}.txt", b"escaped")
        elif kind == "drive":
            zf.writestr(f"C:/{marker}.txt", b"escaped")
        elif kind == "symlink":
            info = zipfile.ZipInfo("images/link.jpg")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
        elif kind == "same_file":
            zf.writestr("images/OK.jpg", image)
    data = buffer.getvalue()
    if kind == "encrypted":
        data = _encrypted_flag(data)
    if kind == "not_a_zip":
        data = b"this is not a zip archive"
    return data


@pytest.mark.parametrize("kind, message", [
    ("dotdot", "'..'"), ("nested_dotdot", "'..'"), ("backslash_dotdot", "'..'"),
    ("absolute", "absolute path"), ("drive", "absolute path"), ("symlink", "symbolic link"),
    ("encrypted", "encrypted"), ("same_file", "same file"), ("not_a_zip", "cannot be extracted"),
])
def test_unsafe_or_broken_zip_is_rejected_whole(ws: Workspace, samples: list[Sample],
                                                tmp_path: Path, kind: str, message: str) -> None:
    marker = f"evil_{uuid.uuid4().hex}"
    data = _malicious(kind, samples[0][0].read_bytes(), marker)
    archive = ws.inbox / f"{kind}.zip"
    archive.write_bytes(data)
    with pytest.raises(IngestError, match=message):
        ingest_only_batch(ws)
    kept = ws.rejected / kind / f"{kind}.zip"
    assert kept.read_bytes() == data                                # moved unchanged
    report = report_of(ws.rejected / kind)
    assert report["status"] == "rejected" and message in report["reason"]
    assert list(ws.processed.iterdir()) == [] and not archive.exists()
    assert not list(tmp_path.rglob(f"{marker}*")) and not Path(f"/tmp/{marker}.txt").exists()
    assert not Path(f"C:/{marker}.txt").exists()


# ------------------------------------------------------------------ other entries, links, claims

def test_loose_file_and_batch_without_images_are_rejected(ws: Workspace,
                                                          samples: list[Sample]) -> None:
    shutil.copy(samples[0][0], ws.inbox / "photo.jpg")
    labels_only = ws.inbox / "labels_only" / "labels"
    labels_only.mkdir(parents=True)
    (labels_only / "x.txt").write_text(f"1 {TRIANGLE}\n")
    outcomes = {}
    for batch in ingest.scan_inbox(ws):
        with pytest.raises(IngestError) as info:
            ingest.load_batch(batch)
        outcomes[batch.name] = str(info.value)
    assert "folder or a .zip" in outcomes["photo.jpg"] and "no image" in outcomes["labels_only"]
    assert (ws.rejected / "photo.jpg" / "photo.jpg").is_file()
    assert (ws.rejected / "labels_only" / "labels" / "x.txt").is_file()
    assert inbox_signature(ws) == ()                               # nothing left to trigger a cycle


def test_symbolic_links_inside_a_batch_are_not_followed(ws: Workspace, samples: list[Sample],
                                                       tmp_path: Path) -> None:
    folder = yolo_unit(ws.inbox / "links", samples[:1])
    write_source(folder)
    outside = tmp_path / "outside"
    yolo_unit(outside, samples[1:2])
    (folder / "images" / "linked.jpg").symlink_to(samples[1][0])
    (folder / "labels" / "linked.txt").symlink_to(samples[1][1])
    (folder / "images" / "linked_dir").symlink_to(outside / "images", target_is_directory=True)
    _, records = ingest_only_batch(ws)
    assert [r.path.name for r in records] == [samples[0][0].name]
    ignored = {i["file"]: i["reason"] for i in report_of(ws.processed / "links")["ignored"]}
    assert {"images/linked.jpg", "labels/linked.txt", "images/linked_dir"} <= set(ignored)
    assert set(ignored.values()) == {"symbolic link: not followed"}


def test_orphan_labels_are_reported(ws: Workspace, samples: list[Sample]) -> None:
    folder = yolo_unit(ws.inbox / "orphans", samples[:1])
    (folder / "masks").mkdir()
    (folder / "labels" / "nobody.txt").write_text(f"1 {TRIANGLE}\n")
    cv2.imwrite(str(folder / "masks" / f"{samples[0][0].stem}.png"), np.zeros((8, 8), np.uint8))
    (folder / "images" / "notes.pdf").write_bytes(b"%PDF")
    write_source(folder)
    _, records = ingest_only_batch(ws)
    assert len(records) == 1 and records[0].polygons                # the YOLO label wins
    ignored = {i["file"]: i["reason"] for i in report_of(ws.processed / "orphans")["ignored"]}
    assert ignored == {"labels/nobody.txt": "label file without image",
                       f"masks/{samples[0][0].stem}.png": "superseded by the YOLO or COCO label "
                                                          "of the image",
                       "images/notes.pdf": "not an image file"}


def test_load_batch_leaves_the_entry_when_its_id_was_taken(ws: Workspace,
                                                           samples: list[Sample]) -> None:
    folder = yolo_unit(ws.inbox / "late", samples[:1])
    (batch,) = ingest.scan_inbox(ws)
    (ws.processed / "late").mkdir()
    with pytest.raises(IngestError, match="already used"):
        ingest.load_batch(batch)
    assert folder.is_dir() and not (ws.rejected / "late").exists()
    (ws.processed / "late").rmdir()
    shutil.rmtree(folder)
    with pytest.raises(IngestError, match="no longer in inbox"):
        ingest.load_batch(batch)


# ------------------------------------------------------------------ the cycle steps

def test_steps_ingest_and_quality(ws: Workspace, samples: list[Sample]) -> None:
    good = yolo_unit(ws.inbox / "good", samples[:2])
    (good / "labels" / samples[1][1].name).write_text("1 0.1 0.1 0.11 0.1 0.1 0.11\n")   # tiny
    write_source(good)
    (ws.inbox / "stray.txt").write_text("not a batch")
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1)
    result = step_ingest(ctx)
    assert result.ok and result.data["batches"] == {"good": 2}
    assert set(result.data["failed_batches"]) == {"stray.txt"}
    assert {r.batch for r in ctx.records} == {"good"}
    result = step_quality(ctx)
    assert result.ok and (result.data["accepted"], len(result.data["rejected"])) == (1, 1)
    processed = ws.processed / "good"
    assert (processed / ingest.INGEST_REPORT).is_file()
    quality_report = json.loads((processed / quality.QUALITY_REPORT).read_text())
    assert (quality_report["images"], quality_report["accepted"], quality_report["rejected"]) == (2, 1, 1)
    (rejection,) = quality_report["rejections"]
    assert rejection["code"] == "region_too_small"
    assert rejection["image"] == f"images/{samples[1][0].name}"


def textured_photo(seed: int, side: int = 320) -> np.ndarray:
    """A colour image of random filled rectangles: a photograph no pool image resembles."""
    rng = np.random.default_rng(seed)
    img = np.full((side, side, 3), 128, np.uint8)
    for _ in range(80):
        x0, y0 = (int(v) for v in rng.integers(0, side - 20, 2))
        w, h = (int(v) for v in rng.integers(8, 70, 2))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), colour, -1)
    return img


@pytest.mark.skipif(not (SMOKE / "manifest.csv").is_file(), reason="data/seg_smoke missing")
def test_records_flow_through_quality_dedupe_drift_and_dataset(ws: Workspace) -> None:
    """End to end on a pool initialised from data/seg_smoke: the Records of a batch
    are checked, matched, tested for drift and written to the pool and dataset v1."""
    assert dataset.init_from_seg(ws, SMOKE, copy=False, log=lambda _m: None) == 0
    folder = ws.inbox / "synthetic clinic"
    for sub in ("images", "labels"):
        (folder / sub).mkdir(parents=True)
    for k in range(7):
        cv2.imwrite(str(folder / "images" / f"photo{k}.jpg"), textured_photo(100 + k))
        x = 0.1 + 0.1 * (k % 4)
        line = f"{1 + k % 4} {x} 0.2 {x + 0.4} 0.2 {x + 0.4} 0.6 {x} 0.6"
        (folder / "labels" / f"photo{k}.txt").write_text(line + "\n")
    (folder / "labels" / "photo6.txt").write_text("2 0.1 0.2 0.5\n")      # odd coordinates
    write_source(folder, {"name": "synthetic_clinic", "license": "CC0", "class_map": MOBILE_MAP})
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1)
    for step in (step_ingest, step_quality, step_dedupe, step_drift, step_dataset):
        result = step(ctx)
        assert result.ok, (step.__name__, result.summary)
    assert len(ctx.records) == 7 and len(ctx.accepted) == 6
    assert [quality.reason_code(r) for _, r in ctx.rejected] == ["malformed_label"]
    assert len(ctx.groups.new_groups) == 6                         # six new photographs
    assert ctx.dataset_version == 1
    pool = ws.read_manifest()
    added = pool[pool["added_cycle"] == 1]
    assert len(added) == 6 and set(added["split"]) <= {"train", "val"}
    assert set(added["source"]) == {"synthetic_clinic"} and set(added["license"]) == {"CC0"}
    assert set(added["batch"]) == {"synthetic_clinic"}
    by_stem = {r.path.stem: r for r in ctx.accepted}
    for name in added["name"]:
        stem = name.split("__")[1]
        polygons = quality.bsd.read_polys(pool_file(ws, "labels", name), {c: c for c in range(4)})[0]
        same_polygons(polygons, by_stem[stem].polygons, atol=1e-6)
        assert cv2.imread(str(pool_file(ws, "masks", name)), cv2.IMREAD_UNCHANGED).max() >= 1
    assert (ws.processed / "synthetic_clinic" / quality.QUALITY_REPORT).is_file()


# ------------------------------------------------------------------ drift

def make_pool(ws: Workspace, images: list[Path]) -> None:
    """A pool of `images`, one photograph each, all in train."""
    rows = []
    for k, image in enumerate(images):
        name = f"roboflow_mobile__{k:03d}"
        shutil.copy(image, pool_file(ws, "images", name))
        rows.append({"name": name, "source": "roboflow_mobile", "photograph": k, "split": "train",
                     "batch": "", "license": "CC BY 4.0", "added_cycle": 0})
    ws.write_manifest(pd.DataFrame(rows))


def test_drift_detects_darkened_copies_but_not_identical_ones(ws: Workspace,
                                                             tmp_path: Path) -> None:
    images = sorted((MOBILE / "train" / "images").glob("*.jpg"))[:12]
    make_pool(ws, images)
    copies, darker = tmp_path / "copies", tmp_path / "darker"
    copies.mkdir()
    darker.mkdir()
    for image in images:
        shutil.copy(image, copies / image.name)
        img = cv2.imread(str(image))
        cv2.imwrite(str(darker / image.name), np.round(img * 0.35).astype(np.uint8))

    def test(folder: Path) -> dict:
        """The drift step's test of the photographs in `folder` against the pool."""
        records = [Record(p, [], "new", "CC BY 4.0", "b") for p in sorted(folder.iterdir())]
        return drift.drift_report([r.path for r in records], drift.pool_reference(ws), alpha=0.01)

    same = test(copies)
    assert same["status"] == "tested" and same["drift"] is False and same["shifted"] == []
    assert all(f["ks_statistic"] == 0.0 for f in same["features"].values())
    dark = test(darker)
    assert dark["drift"] is True and "lightness" in dark["shifted"]
    assert dark["features"]["lightness"]["direction"] == "lower"
    assert (dark["n_new"], dark["n_reference"]) == (12, 12)
    json.dumps(dark)                                               # JSON-ready for the events
