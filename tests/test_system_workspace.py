"""Tests of selftrain.system: package constants, Workspace and SystemConfig (SPEC section 1),
model registry (section 5) and model card (section 6).

No model is trained or evaluated: evaluation results are built with the experiment
library's own summarise/wilson_ci, so they have exactly the structure the evaluate step
writes. Every workspace lives under pytest's tmp_path.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import selftrain.system as system
from selftrain.config import NUM_STAGES, UNetConfig, YOLOConfig
from selftrain.evaluate import summarise
from selftrain.metrics import wilson_ci
from selftrain.system import modelcard, registry, workspace
from selftrain.system.modelcard import NA, fmt_ci, fmt_estimate, fmt_value, markdown_table
from selftrain.system.registry import VersionRecord
from selftrain.system.workspace import (KNOWN_SOURCES, MANIFEST_COLUMNS, SPLITS,
                                        UNSTATED_LICENSE, SystemConfig, Workspace,
                                        atomic_write_json, atomic_write_text, check_version,
                                        empty_manifest, json_safe, license_for_source,
                                        validate_manifest)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "seg_smoke"


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GPU belongs to the main experiments (nothing here needs a device anyway)."""
    monkeypatch.setenv("SELFTRAIN_DEVICE", "cpu")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()


def temp_files(folder: Path) -> list[Path]:
    """Leftovers of atomic writes (.<name>.<uuid>.tmp)."""
    return [p for p in folder.iterdir() if p.name.endswith(".tmp")]


# =========================================================================== package

def test_step_names_follow_spec() -> None:
    assert system.STEP_NAMES == ["ingest", "quality", "dedupe", "drift", "dataset",
                                 "train", "evaluate", "gate", "report", "notify"]
    assert system.STEP_STATUSES == ("started", "done", "failed", "skipped")
    assert system.Workspace is Workspace and system.SystemConfig is SystemConfig


def test_step_names_agree_with_the_cycle_steps() -> None:
    from selftrain.system import steps
    assert steps.STEP_NAMES == system.STEP_NAMES


# =========================================================================== SystemConfig

def test_config_defaults_follow_spec() -> None:
    cfg = SystemConfig()
    assert (cfg.architecture, cfg.demo_epochs, cfg.demo_max_train_images) == ("unet", 3, 120)
    assert (cfg.val_fraction_new, cfg.gate_margin, cfg.bootstrap) == (0.15, 0.02, 2000)
    assert (cfg.min_side_px, cfg.max_mask_frac, cfg.min_mask_frac) == (64, 0.95, 0.0005)
    assert (cfg.drift_alpha, cfg.telegram, cfg.seed) == (0.01, False, 0)
    assert (cfg.epochs, cfg.patience) == (UNetConfig().epochs, UNetConfig().patience)
    assert cfg.min_component_frac == 0.001


def test_config_yaml_round_trip_with_comments() -> None:
    cfg = SystemConfig(architecture="yoloseg", epochs=5, gate_margin=0.05, telegram=True,
                       min_mask_frac=1e-5)
    text = cfg.to_yaml()
    assert SystemConfig.from_yaml(text) == cfg
    assert set(yaml.safe_load(text)) == {f.name for f in dataclasses.fields(SystemConfig)}
    comments = [line for line in text.splitlines() if line.startswith("# ")]
    assert len(comments) >= len(dataclasses.fields(SystemConfig))


def test_config_missing_keys_take_defaults_and_scalars_are_coerced() -> None:
    assert SystemConfig.from_yaml("") == SystemConfig()
    cfg = SystemConfig.from_yaml("epochs: 10.0\nmin_mask_frac: 5e-4\ngate_margin: 0\n")
    assert cfg.epochs == 10 and isinstance(cfg.epochs, int)
    assert cfg.min_mask_frac == 0.0005           # YAML 1.1 reads 5e-4 as a string
    assert cfg.gate_margin == 0.0 and isinstance(cfg.gate_margin, float)
    assert cfg.patience == SystemConfig().patience


@pytest.mark.parametrize("text, fragment", [
    ("unknown_key: 1", "unknown configuration key"),
    ("epochs: true", "epochs"),
    ("epochs: 0", "epochs must be >= 1"),
    ("architecture: resnet", "architecture must be one of"),
    ("gate_margin: -0.1", "gate_margin"),
    ("telegram: 'yes'", "telegram"),
    ("val_fraction_new: .nan", "val_fraction_new"),
    ("drift_alpha: 1.5", "drift_alpha"),
    ("max_mask_frac: 0.0001\nmin_mask_frac: 0.001", "min_mask_frac"),
    ("- a\n- b", "YAML mapping"),
    ("epochs: [1", "not valid YAML"),
])
def test_config_rejects_invalid_values(text: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=re.escape(fragment)):
        SystemConfig.from_yaml(text)


def test_config_error_names_every_offending_key_and_replace_validates() -> None:
    with pytest.raises(ValueError) as info:
        SystemConfig(epochs=0, patience=0)
    assert "epochs" in str(info.value) and "patience" in str(info.value)
    with pytest.raises(ValueError, match="seed"):
        dataclasses.replace(SystemConfig(), seed=-1)


# =========================================================================== workspace

def test_workspace_root_is_absolute_and_nothing_is_created(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert Workspace(Path("rel")).root == tmp_path.resolve() / "rel"
    assert Workspace().root == tmp_path.resolve() / "system_ws"
    assert Workspace("rel") == Workspace(Path("rel"))
    assert not (tmp_path / "system_ws").exists() and not (tmp_path / "rel").exists()


def test_ensure_creates_the_layout_and_a_default_config(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    assert ws.ensure() is ws
    for rel in ("inbox", "inbox/unlabeled", "inbox/processed", "inbox/rejected", "pool/images",
                "pool/labels", "pool/masks", "datasets", "models", "reports", "reports/figures",
                "active"):
        assert (ws.root / rel).is_dir(), rel
    assert ws.config_path == ws.root / "config.yaml"
    assert ws.load_config() == SystemConfig()
    assert (ws.manifest_path, ws.registry_path) == (ws.root / "pool" / "manifest.csv",
                                                    ws.root / "registry.json")
    assert (ws.events_path, ws.status_path) == (ws.root / "events.jsonl", ws.root / "status.json")


def test_ensure_is_idempotent_and_keeps_an_edited_config(ws: Workspace) -> None:
    ws.save_config(SystemConfig(epochs=7, architecture="yoloseg"))
    ws.ensure()
    assert ws.load_config() == SystemConfig(epochs=7, architecture="yoloseg")
    assert temp_files(ws.root) == []


def test_load_config_defaults_without_file_and_names_a_broken_file(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")
    assert ws.load_config() == SystemConfig()
    ws.root.mkdir()
    ws.config_path.write_text("epochs: 0\n")
    with pytest.raises(ValueError, match=re.escape(str(ws.config_path))):
        ws.load_config()


def test_save_config_revalidates_a_mutated_object(ws: Workspace) -> None:
    before = ws.config_path.read_text()
    cfg = SystemConfig()
    cfg.epochs = 0                                # bypasses __post_init__
    with pytest.raises(ValueError, match="epochs"):
        ws.save_config(cfg)
    with pytest.raises(TypeError):
        ws.save_config({"epochs": 3})             # type: ignore[arg-type]
    assert ws.config_path.read_text() == before


def test_version_paths_and_version_numbers(ws: Workspace) -> None:
    assert ws.dataset_dir(3) == ws.root / "datasets" / "v3"
    assert ws.model_dir(np.int64(0)) == ws.root / "models" / "v0"
    with pytest.raises(ValueError):
        ws.dataset_dir(-1)
    for bad in (True, 1.5, "2"):
        with pytest.raises(TypeError):
            check_version(bad)


def test_next_dataset_version_counts_directories_and_registry(ws: Workspace) -> None:
    assert ws.next_dataset_version() == 0
    for name in ("v0", "v2", "v01", ".v3.partial", "v4_tmp"):
        (ws.datasets / name).mkdir()
    (ws.datasets / "v9").write_text("a file, not a version")
    assert ws.next_dataset_version() == 3
    registry.add_version(ws, make_record(ws, 0, dataset_version=6))
    assert ws.next_dataset_version() == 7


def test_next_model_version_counts_directories_and_registry(ws: Workspace) -> None:
    assert ws.next_model_version() == registry.next_version(ws) == 0
    (ws.models / "v1").mkdir()
    assert ws.next_model_version() == 2
    registry.add_version(ws, make_record(ws, 4))
    assert ws.next_model_version() == 5


# =========================================================================== manifest

def make_manifest(rows: list[tuple[str, str, Any, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["name", "source", "photograph", "split"])
    return df.assign(batch="b1", license="CC BY 4.0", added_cycle=0)


BASE_ROWS = [("a", "s1", 0, "train"), ("b", "s1", 1, "val"), ("c", "s2", 2, "test"),
             ("d", "s2", 2, "test")]                # c and d: one photograph, two images


def test_read_manifest_without_file_is_empty_with_columns(ws: Workspace) -> None:
    df = ws.read_manifest()
    assert list(df.columns) == list(MANIFEST_COLUMNS) and df.empty
    assert str(df["photograph"].dtype) == str(df["added_cycle"].dtype) == "int64"
    ws.write_manifest(empty_manifest())
    assert ws.read_manifest().empty


def test_manifest_round_trip_normalises(ws: Workspace) -> None:
    df = make_manifest(BASE_ROWS)
    df["photograph"] = df["photograph"].astype(str)              # "0", "1", ...
    df = df.assign(note=["x", "", "y", "z"])[["note", *reversed(MANIFEST_COLUMNS)]]
    ws.write_manifest(df, allow_test_change=True)
    back = ws.read_manifest()
    assert list(back.columns) == [*MANIFEST_COLUMNS, "note"]
    assert back["photograph"].tolist() == [0, 1, 2, 2]
    assert str(back["photograph"].dtype) == "int64"
    assert back["note"].tolist() == ["x", "", "y", "z"]
    assert ws.manifest_path.read_text().endswith("\n")
    assert temp_files(ws.pool) == []


INVALID_MANIFESTS = {
    "lacks required column": lambda df: df.drop(columns="split"),
    "names must be unique": lambda df: df.assign(name=["a", "a", "c", "d"]),
    "is empty": lambda df: df.assign(source=["", "s1", "s2", "s2"]),
    "unknown split": lambda df: df.assign(split=["train", "val", "test", "holdout"]),
    "more than one split": lambda df: df.assign(photograph=[0, 0, 2, 2]),
    "must hold integers": lambda df: df.assign(photograph=["0", "x", "2", "2"]),
    "integers; offending": lambda df: df.assign(photograph=[0, 1.5, 2, 2]),
    "added_cycle must be >= 0": lambda df: df.assign(added_cycle=[0, -1, 0, 0]),
    "duplicate column": lambda df: pd.concat([df, df[["name"]]], axis=1),
}


@pytest.mark.parametrize("fragment", sorted(INVALID_MANIFESTS))
def test_validate_manifest_rejects(fragment: str, ws: Workspace) -> None:
    bad = INVALID_MANIFESTS[fragment](make_manifest(BASE_ROWS))
    with pytest.raises(ValueError, match=re.escape(fragment)):
        validate_manifest(bad)
    with pytest.raises(ValueError):
        ws.write_manifest(bad, allow_test_change=True)
    assert not ws.manifest_path.exists()


def test_validate_manifest_requires_a_dataframe() -> None:
    with pytest.raises(TypeError):
        validate_manifest(BASE_ROWS)                                  # type: ignore[arg-type]


def test_frozen_test_split_cannot_change(ws: Workspace) -> None:
    ws.write_manifest(make_manifest(BASE_ROWS), allow_test_change=True)
    grown = make_manifest([*BASE_ROWS, ("e", "s3", 3, "train")])
    ws.write_manifest(grown)                                          # train may grow
    with pytest.raises(ValueError, match="frozen test split"):
        ws.write_manifest(make_manifest([*BASE_ROWS, ("e", "s3", 3, "train"),
                                         ("f", "s3", 4, "test")]))
    with pytest.raises(ValueError, match="frozen test split"):
        ws.write_manifest(make_manifest([r for r in BASE_ROWS if r[0] != "d"]))
    assert ws.read_manifest()["name"].tolist() == ["a", "b", "c", "d", "e"]
    ws.write_manifest(make_manifest(BASE_ROWS[:2]), allow_test_change=True)
    assert ws.read_manifest()["name"].tolist() == ["a", "b"]


# =========================================================================== file helpers

@dataclasses.dataclass
class _Point:
    x: float
    y: Any


def test_json_safe_produces_strict_json_values() -> None:
    obj = {"a": np.float32(1.5), "b": float("nan"), "c": (1, np.int64(2)), "d": Path("x/y"),
           3: np.array([1.0, np.inf]), "e": np.bool_(True), "f": _Point(0.5, float("-inf"))}
    assert json_safe(obj) == {"a": 1.5, "b": None, "c": [1, 2], "d": "x/y", "3": [1.0, None],
                              "e": True, "f": {"x": 0.5, "y": None}}
    for bad in ({1, 2}, object()):
        with pytest.raises(TypeError):
            json_safe(bad)


def test_atomic_write_replaces_content_without_leftovers(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "file.txt"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second\n")
    assert path.read_text() == "second\n"
    assert temp_files(path.parent) == []


def test_atomic_write_keeps_old_content_when_the_rename_fails(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file.txt"
    atomic_write_text(path, "old")

    def failing_replace(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(workspace.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(path, "new")
    monkeypatch.undo()
    assert path.read_text() == "old"
    assert temp_files(tmp_path) == []


def test_atomic_write_json_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    atomic_write_json(path, {"x": float("nan"), "y": np.float64(2.0)})
    assert json.loads(path.read_text(), parse_constant=_reject_constant) == {"x": None, "y": 2.0}


def _reject_constant(name: str) -> None:
    raise AssertionError(f"non-standard JSON constant {name}")


def test_licence_of_known_sources() -> None:
    roboflow = [s for s in KNOWN_SOURCES if s.startswith(("roboflow_", "rf_"))]
    assert len(roboflow) >= 5
    assert all(license_for_source(s) == "CC BY 4.0" for s in roboflow)
    assert license_for_source("neg_azh") == license_for_source("neg_fuseg") == UNSTATED_LICENSE
    assert license_for_source("a_new_batch_source") is None


# =========================================================================== registry

def make_record(ws: Workspace, version: int, *, create_weights: bool = True,
                **overrides: Any) -> VersionRecord:
    """A valid record (fields can be overridden); with `create_weights` its default weights
    file models/v{N}/best.pt is created."""
    path = ws.model_dir(version) / "best.pt"
    if create_weights:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"weights of v{version}".encode())
    fields: dict[str, Any] = dict(
        version=version, dataset_version=version, architecture="unet", train_images=10,
        train_photographs=9, metrics={"test_dice_mean": 0.5, "test_dice_ci95": [0.4, 0.6]},
        gate={"promoted": True, "reason": "test"}, weights=str(path))
    fields.update(overrides)
    return VersionRecord(**fields)


def test_empty_registry(ws: Workspace) -> None:
    assert registry.load(ws) == {"current": None, "versions": [], "history": []}
    assert registry.current(ws) is None
    assert registry.next_version(ws) == 0
    assert not ws.registry_path.exists()


def test_add_promote_and_current(ws: Workspace) -> None:
    registry.add_version(ws, make_record(ws, 0), make_current=True)
    assert registry.current(ws).version == 0
    registry.add_version(ws, make_record(ws, 1))
    assert registry.get_version(ws, 1).parent == 0               # baseline of its gate
    assert registry.current(ws).version == 0                       # registered, not promoted
    reg = registry.promote(ws, 1)
    assert reg["current"] == 1 and registry.current(ws).version == 1
    actions = [(h["action"], h["version"], h["previous"]) for h in reg["history"]]
    assert actions == [("promote", 0, None), ("promote", 1, 0)]
    registry.promote(ws, 1)                                        # no-op
    assert len(registry.load(ws)["history"]) == 2
    data = json.loads(ws.registry_path.read_text(), parse_constant=_reject_constant)
    assert [v["version"] for v in data["versions"]] == [0, 1]
    assert data["current"] == 1
    assert temp_files(ws.root) == []


def test_rollback_restores_any_registered_version(ws: Workspace) -> None:
    for v in range(3):
        registry.add_version(ws, make_record(ws, v), make_current=True)
    reg = registry.rollback(ws, 0)
    assert reg["current"] == 0
    assert reg["history"][-1]["action"] == "rollback" and reg["history"][-1]["previous"] == 2
    with pytest.raises(KeyError):
        registry.rollback(ws, 7)
    with pytest.raises(KeyError):
        registry.promote(ws, 7)


def test_a_version_without_weights_cannot_become_current(ws: Workspace) -> None:
    registry.add_version(ws, make_record(ws, 0), make_current=True)
    before = ws.registry_path.read_text()
    with pytest.raises(FileNotFoundError):
        registry.add_version(ws, make_record(ws, 1, create_weights=False), make_current=True)
    assert ws.registry_path.read_text() == before                 # nothing registered
    registry.add_version(ws, make_record(ws, 1, create_weights=False))
    with pytest.raises(FileNotFoundError):
        registry.promote(ws, 1)
    assert registry.current(ws).version == 0


def test_add_version_rejects_duplicates_and_other_types(ws: Workspace) -> None:
    registry.add_version(ws, make_record(ws, 0))
    with pytest.raises(ValueError, match="already registered"):
        registry.add_version(ws, make_record(ws, 0))
    with pytest.raises(TypeError):
        registry.add_version(ws, {"version": 1})                   # type: ignore[arg-type]


def test_relative_weights_are_resolved_against_the_root(ws: Workspace) -> None:
    rec = make_record(ws, 3, weights="models/v3/best.pt")
    assert rec.weights_path(ws) == ws.root / "models" / "v3" / "best.pt"
    registry.add_version(ws, rec, make_current=True)
    assert registry.current(ws).weights == "models/v3/best.pt"


@pytest.mark.parametrize("overrides", [
    {"architecture": "resnet"}, {"train_images": -1}, {"dataset_version": True},
    {"created": "yesterday"}, {"gate": {"promoted": "yes"}}, {"gate": {"reason": 3}},
    {"weights": ""}, {"parent": 5}, {"metrics": "high"},
])
def test_version_record_validation(ws: Workspace, overrides: dict[str, Any]) -> None:
    fields = {"version": 5, **overrides}
    with pytest.raises(ValueError):
        make_record(ws, **fields)


def test_version_record_round_trip_is_strict_json(ws: Workspace) -> None:
    rec = make_record(ws, 2, metrics={"test_dice_mean": np.float32(0.5),
                                      "test_fpr": float("nan"), "test_dice_ci95": (0.4, 0.6)},
                      gate={"promoted": False, "ci95": (-0.1, 0.05), "mean_diff": np.nan,
                            "reason": "worse"})
    assert rec.metrics == {"test_dice_mean": 0.5, "test_fpr": None, "test_dice_ci95": [0.4, 0.6]}
    assert list(rec.gate)[:4] == ["promoted", "mean_diff", "ci95", "reason"]
    assert rec.gate["mean_diff"] is None and rec.promoted is False
    assert VersionRecord.from_dict(json.loads(json.dumps(rec.to_dict()))) == rec
    with pytest.raises(ValueError, match="unknown key"):
        VersionRecord.from_dict({**rec.to_dict(), "colour": "red"})
    with pytest.raises(ValueError, match="missing key"):
        VersionRecord.from_dict({k: v for k, v in rec.to_dict().items() if k != "weights"})


def test_save_validates_the_whole_registry(ws: Workspace) -> None:
    rec0, rec1 = make_record(ws, 0).to_dict(), make_record(ws, 1, parent=0).to_dict()
    registry.save(ws, {"current": 0, "versions": [rec1, rec0], "history": []})
    assert [r.version for r in registry.load(ws)["versions"]] == [0, 1]
    invalid = [
        {"current": 4, "versions": [rec0]},                               # unregistered current
        {"current": None, "versions": [rec0, rec0]},                      # duplicate version
        {"current": None, "versions": [rec1]},                            # dangling parent
        {"current": None, "versions": [rec0], "extra": 1},                # unknown key
        {"current": None, "versions": [rec0], "history": [{"action": "delete"}]},
        {"current": None, "versions": "v0"},
    ]
    before = ws.registry_path.read_text()
    for reg in invalid:
        with pytest.raises(ValueError):
            registry.save(ws, reg)
    assert ws.registry_path.read_text() == before


def test_a_corrupt_registry_file_is_reported_with_its_path(ws: Workspace) -> None:
    ws.registry_path.write_text("{not json")
    with pytest.raises(ValueError, match=re.escape(str(ws.registry_path))):
        registry.load(ws)


def test_update_version(ws: Workspace) -> None:
    registry.add_version(ws, make_record(ws, 0))
    updated = registry.update_version(ws, 0, model_card="models/v0/model_card.md")
    assert updated.model_card == "models/v0/model_card.md"
    assert registry.get_version(ws, 0) == updated
    before = ws.registry_path.read_text()
    with pytest.raises(ValueError):
        registry.update_version(ws, 0, version=3)
    with pytest.raises(ValueError):
        registry.update_version(ws, 0, architecture="resnet")
    with pytest.raises(KeyError):
        registry.update_version(ws, 9, model_card="x.md")
    assert ws.registry_path.read_text() == before


def test_concurrent_registrations_are_not_lost(ws: Workspace) -> None:
    records = [make_record(ws, v) for v in range(12)]
    barrier = threading.Barrier(len(records))
    errors: list[BaseException] = []

    def register(rec: VersionRecord) -> None:
        try:
            barrier.wait()
            registry.add_version(ws, rec)
        except BaseException as exc:                     # noqa: BLE001 -- reported below
            errors.append(exc)

    threads = [threading.Thread(target=register, args=(r,)) for r in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert [r.version for r in registry.load(ws)["versions"]] == list(range(12))


# =========================================================================== evaluation fixture

PIXEL_CONFUSION = np.array([[1000, 10, 5, 0, 2],          # rows: labelled class 0-4
                            [20, 300, 10, 0, 0],          # stage 3 occurs nowhere -> NaN
                            [10, 5, 200, 0, 5],
                            [0, 0, 0, 0, 0],
                            [5, 0, 0, 0, 100]])


def per_image_rows(n_pos: int, n_neg: int, seed: int) -> list[dict[str, Any]]:
    """Rows in the format of selftrain.evaluate.evaluate_cache."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos):
        stage = 1 + i % NUM_STAGES
        rows.append({"name": f"pos{i}", "source": "src", "gt_positive": True,
                     "pred_positive": i != 1, "dice": float(rng.uniform(0.3, 0.9)),
                     "iou": float(rng.uniform(0.2, 0.8)), "gt_stage": stage,
                     "pred_stage": stage if i % 3 else 0, "gt_area": 0.1, "pred_area": 0.1})
    for i in range(n_neg):
        rows.append({"name": f"neg{i}", "source": "neg", "gt_positive": False,
                     "pred_positive": i == 0, "dice": 1.0, "iou": 1.0, "gt_stage": 0,
                     "pred_stage": 0, "gt_area": 0.0, "pred_area": 0.0})
    return rows


@pytest.fixture
def evaluation() -> dict[str, Any]:
    """A full_evaluation result: test with NaN stage-3 Dice, an external split without
    ulcer-free photographs (NaN false-positive rate) and extra negatives per wound type."""
    test = summarise(per_image_rows(9, 3, seed=0), PIXEL_CONFUSION, b=200)
    extra = {}
    for code, (k, n) in {"BG": (2, 20), "D": (5, 30), "V": (1, 25)}.items():
        rate, ci = wilson_ci(k, n)
        extra[code] = {"n": n, "false_positive_rate": rate, "ci95": ci}
    rate, ci = wilson_ci(8, 75)
    extra["all"] = {"n": 75, "false_positive_rate": rate, "ci95": ci}
    return {"model": "v1", "threshold": 0.45, "val_threshold_table": {"0.45": 0.6},
            "val": test, "test": test,
            "external": summarise(per_image_rows(5, 0, seed=1), PIXEL_CONFUSION, b=200),
            "extra_negatives": extra, "inference_ms_per_image": 12.34}


def test_metrics_from_evaluation(evaluation: dict[str, Any]) -> None:
    m = registry.metrics_from_evaluation(evaluation)
    t = evaluation["test"]
    assert m["test_dice_mean"] == pytest.approx(t["dice_mean"])
    assert m["test_dice_ci95"] == pytest.approx(list(t["dice_ci95"]))
    assert m["test_fpr"] == pytest.approx(1 / 3)
    assert (m["n_test_positive"], m["n_test_negative"], m["threshold"]) == (9, 3, 0.45)
    assert m["extra_negatives_fpr"] == pytest.approx(8 / 75)
    assert m["external_dice_mean"] == pytest.approx(evaluation["external"]["dice_mean"])
    assert json.loads(json.dumps(m, allow_nan=False)) == m        # strict JSON
    with pytest.raises(ValueError, match="test"):
        registry.metrics_from_evaluation({"threshold": 0.5})


def test_metrics_from_evaluation_matches_the_cycle_summary(evaluation: dict[str, Any]) -> None:
    from selftrain.system import steps
    assert registry.metrics_from_evaluation(evaluation) == json_safe(
        steps.summary_metrics(evaluation))


# =========================================================================== model card

def build_dataset_v0(ws: Workspace) -> pd.DataFrame:
    """Pool manifest and datasets/v0 (manifest snapshot, label symlinks) from the smoke
    dataset, in the layout of SPEC section 1."""
    seg = pd.read_csv(SMOKE / "manifest.csv", dtype=str, keep_default_na=False)
    pool = seg[["name", "source", "photograph", "split"]].assign(
        batch="init", license=[license_for_source(s) or "unknown" for s in seg["source"]],
        added_cycle=0)
    ws.write_manifest(pool, allow_test_change=True)
    vdir = ws.dataset_dir(0)
    for name, split in zip(pool["name"], pool["split"]):
        link = vdir / "labels" / split / f"{name}.txt"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(SMOKE / "labels" / split / f"{name}.txt")
    manifest = ws.read_manifest()
    manifest.to_csv(vdir / "manifest.csv", index=False)
    return manifest


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))           # NaN allowed, as the library writes


GATE_V1 = {"promoted": True, "mean_diff": 0.034, "ci95": [-0.004, 0.071], "p_value": 0.08,
           "n": 9, "margin": 0.02, "compared_to": 0, "reason": "better on average"}
GATE_V2 = {"promoted": False, "mean_diff": -0.05, "ci95": [-0.09, -0.01], "p_value": 0.01,
           "n": 9, "margin": 0.02, "compared_to": 1, "reason": "worse than v1"}


@pytest.fixture
def card_ws(ws: Workspace, evaluation: dict[str, Any]) -> Workspace:
    """v0: U-Net without evaluation or run files, current first. v1: U-Net with run files
    and evaluation.json, promoted. v2: YOLOv8-seg demo subset with registry metrics only,
    rejected by the gate. All on dataset v0."""
    build_dataset_v0(ws)
    registry.add_version(ws, make_record(ws, 0, dataset_version=0, metrics={}, train_images=57,
                                         gate={"promoted": True, "reason": "first model"}),
                         make_current=True)
    run = ws.model_dir(1) / "train"
    write_json(run / "config.json", {**UNetConfig(epochs=1).to_dict(), "train_images": 57})
    write_json(run / "summary.json", {"best_val_dice": 0.61, "best_epoch": 1, "epochs_run": 1})
    write_json(ws.model_dir(1) / "evaluation.json", evaluation)
    registry.add_version(ws, make_record(ws, 1, dataset_version=0, train_images=57,
                                         metrics=registry.metrics_from_evaluation(evaluation),
                                         gate=GATE_V1))
    registry.promote(ws, 1)
    run = ws.model_dir(2) / "train"
    write_json(run / "config.json", YOLOConfig(epochs=1).to_dict())
    write_json(run / "selection.json", {"criterion": "mean val binary Dice at conf 0.25",
                                        "selected": "epoch1.pt",
                                        "scores": {"epoch0.pt": 0.4, "epoch1.pt": 0.5}})
    registry.add_version(ws, make_record(ws, 2, dataset_version=0, architecture="yoloseg",
                                         train_images=20, train_photographs=18,
                                         metrics=registry.metrics_from_evaluation(evaluation),
                                         gate=GATE_V2))
    return ws


def section(card: str, heading: str) -> str:
    """Text of the '## heading' section of a card."""
    parts = re.split(r"^## ", card, flags=re.M)
    matches = [p for p in parts if p.startswith(heading + "\n")]
    assert len(matches) == 1, f"section {heading!r} not found once"
    return matches[0]


def table(text: str, first_header: str) -> list[list[str]]:
    """Body rows (cells) of the first Markdown table whose header starts with `first_header`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]])
    return rows


SECTIONS = ["Model details", "Intended use", "Data", "Training configuration",
            "Metrics on the frozen test split", "False-positive rates per wound type",
            "Gate decision", "Known limitations", "Version lineage"]


def test_card_is_written_atomically_and_registered(card_ws: Workspace) -> None:
    assert registry.get_version(card_ws, 1).model_card == ""
    path = modelcard.write(card_ws, 1)
    assert path == card_ws.model_dir(1) / "model_card.md" == modelcard.card_path(card_ws, 1)
    assert registry.get_version(card_ws, 1).model_card == str(path)
    card = path.read_text(encoding="utf-8")
    assert card.isascii() and card.endswith("\n")
    assert card.startswith("# Model card: pressure-ulcer segmentation model v1\n")
    assert re.findall(r"^## (.+)$", card, flags=re.M) == SECTIONS
    assert temp_files(path.parent) == []


def test_card_keeps_a_registered_path(card_ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    registry.update_version(card_ws, 1, model_card="models/v1/model_card.md")

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the registry already points to the card")

    monkeypatch.setattr(registry, "update_version", unexpected)
    assert modelcard.write(card_ws, 1).is_file()


def test_card_states_the_intended_use(card_ws: Workspace) -> None:
    text = section(modelcard.render(card_ws, 1), "Intended use")
    assert "**Research prototype.**" in text and "**NOT a medical device.**" in text
    assert "must not be used to diagnose" in text and "are not diagnoses" in text


def test_card_lists_sources_with_licences_and_counts(card_ws: Workspace) -> None:
    text = section(modelcard.render(card_ws, 1), "Data")
    manifest = card_ws.read_manifest()
    assert "`datasets/v0/manifest.csv`, the snapshot of the pool manifest" in text
    rows = {r[0].strip("`*"): r for r in table(text, "Source")}
    assert set(rows) == set(manifest["source"]) | {"total"}
    for source, part in manifest.groupby("source"):
        row = rows[source]
        expected = [int((part["split"] == s).sum()) for s in SPLITS]
        assert [int(c) for c in row[3:7]] == expected
        assert int(row[7]) == part["photograph"].nunique()
        if source in KNOWN_SOURCES:
            assert KNOWN_SOURCES[source].url in row[1]
            assert row[2] == KNOWN_SOURCES[source].license
    assert all(rows[s][2] == "CC BY 4.0" for s in rows if s.startswith(("roboflow_", "rf_")))
    total = rows["total"]
    assert [int(c) for c in total[3:7]] == [int((manifest["split"] == s).sum()) for s in SPLITS]
    assert int(total[7]) == manifest["photograph"].nunique()
    assert "https://creativecommons.org/licenses/by/4.0/" in text
    assert "trained on a subset" not in text


def count_smoke_labels() -> dict[str, list[int]]:
    """Per split: [images, ulcer-free, regions of stage 1..4], counted independently."""
    out = {}
    for split in SPLITS:
        files = sorted((SMOKE / "labels" / split).glob("*.txt"))
        classes = [line.split()[0] for f in files for line in f.read_text().splitlines()
                   if line.strip()]
        empty = sum(1 for f in files if not f.read_text().strip())
        out[split] = [len(files), empty, *(classes.count(str(c)) for c in range(NUM_STAGES))]
    return out


def test_card_counts_labelled_regions_per_stage(card_ws: Workspace) -> None:
    text = section(modelcard.render(card_ws, 1), "Data")
    counted = {r[0]: [int(c) for c in r[1:]] for r in table(text, "Split")}
    assert counted == count_smoke_labels()
    limitations = section(modelcard.render(card_ws, 1), "Known limitations")
    train = counted["train"]
    share = 100 * train[2] / sum(train[2:])
    assert (f"Stage 1 accounts for {train[2]} of the {sum(train[2:])} labelled regions "
            f"({share:.1f} %)") in limitations


def test_card_reports_test_metrics_with_cis(card_ws: Workspace,
                                            evaluation: dict[str, Any]) -> None:
    card = modelcard.render(card_ws, 1)
    text = section(card, "Metrics on the frozen test split")
    t = evaluation["test"]
    rows = {r[0]: r[1:] for r in table(text, "Metric")}
    assert rows["Dice, binary: mean per photograph"][:3] == [
        fmt_value(t["dice_mean"]), fmt_ci(t["dice_ci95"]), "9"]
    assert rows["IoU, binary: mean per photograph"][1] == fmt_ci(t["iou_ci95"])
    assert rows["False-positive rate: ulcer-free photographs with a predicted ulcer"][:3] == [
        "0.333", fmt_ci(t["false_positive_rate_ci95"]), "3"]
    staged = sum(1 for i in range(9) if i % 3)                    # pred_stage > 0
    assert rows["Dominant-stage accuracy: predicted stage = labelled stage"][2] == str(staged)
    stages = {r[0]: r[1] for r in table(text, "Stage")}
    assert stages["Stage 3"] == NA                                 # NaN in the evaluation
    assert stages["Stage 1"] == fmt_value(t["stage_dice"]["stage1"])
    assert stages["Macro mean"] == fmt_value(t["stage_dice_macro"])
    external = evaluation["external"]
    assert f"Dice {fmt_estimate(external['dice_mean'], external['dice_ci95'])}, n = 5" in text
    assert "12.3 ms per photograph" in text
    training = section(card, "Training configuration")
    assert "| Operating threshold | 0.45: the value of the grid 0.05, 0.10, ..., 0.95" in training


def test_card_reports_false_positive_rates_per_wound_type(card_ws: Workspace,
                                                          evaluation: dict[str, Any]) -> None:
    text = section(modelcard.render(card_ws, 1), "False-positive rates per wound type")
    rows = {r[0]: r[1:] for r in table(text, "Photographs")}
    extra = evaluation["extra_negatives"]
    for code, name in (("BG", "background"), ("D", "diabetic ulcer"), ("V", "venous ulcer")):
        assert rows[f"Extra: {name} ({code})"] == [
            str(extra[code]["n"]), fmt_value(extra[code]["false_positive_rate"]),
            fmt_ci(extra[code]["ci95"])]
    assert rows["All extra ulcer-free photographs"][0] == "75"
    assert rows["Ulcer-free photographs of the frozen test split"][:2] == ["3", "0.333"]


def test_card_training_configuration_of_a_unet_run(card_ws: Workspace) -> None:
    text = section(modelcard.render(card_ws, 1), "Training configuration")
    params = {r[0]: r[1] for r in table(text, "Parameter")}
    assert params["encoder"] == "resnet34" and params["epochs"] == "1"
    assert "| Run directory | `models/v1/train` |" in text
    assert ("Training outcome (`summary.json`): best validation binary Dice 0.610 at epoch 1; "
            "1 epoch(s) run.") in text


def test_card_of_a_rejected_yolo_demo_version(card_ws: Workspace) -> None:
    card = modelcard.render(card_ws, 2)
    assert "YOLOv8-seg (Ultralytics)" in section(card, "Model details")
    assert "not current (the current model is v1)" in card
    training = section(card, "Training configuration")
    assert "checkpoint `epoch1.pt` selected among 2 epoch checkpoint(s)" in training
    assert "trained on a subset: 20 of the 57 training images" in section(card, "Data")
    metrics = section(card, "Metrics on the frozen test split")
    assert "`models/v2/evaluation.json` was not found" in metrics
    assert "stored only in `evaluation.json`" in section(card, "False-positive rates per wound type")
    gate = {r[0]: r[1] for r in table(section(card, "Gate decision"), "Field")}
    assert gate["Decision"] == "not promoted" and gate["Compared with"] == "v1"
    assert gate["Mean Dice difference (candidate - baseline)"] == "-0.050"
    assert gate["95 % CI (paired bootstrap)"] == "[-0.090, -0.010]"
    assert gate["Reason"] == "worse than v1"


def test_card_of_an_unevaluated_version_without_run_files(card_ws: Workspace) -> None:
    card = modelcard.render(card_ws, 0)
    assert "has not been evaluated on the frozen test split" in card
    assert "No evaluation is available for this version." in card
    assert "not chosen yet" in card and "No training run directory" in card
    gate = {r[0]: r[1] for r in table(section(card, "Gate decision"), "Field")}
    assert gate["Compared with"] == "no model (none was current)"


def test_card_lineage_and_history(card_ws: Workspace) -> None:
    text = section(modelcard.render(card_ws, 2), "Version lineage")
    assert text.splitlines()[2].startswith("v2 <- v1 <- v0.")
    rows = table(text, "Version")
    assert [r[0] for r in rows] == ["**v2** (this card)", "v1", "v0"]
    assert rows[1][6] == "promoted (Dice +0.034 vs v0)"
    assert rows[0][6] == "not promoted (Dice -0.050 vs v1)"
    history = table(text, "Time (UTC)")
    assert [r[1:] for r in history] == [["promote", "v0", "-"], ["promote", "v1", "v0"]]
    registry.rollback(card_ws, 0)
    card = modelcard.render(card_ws, 1)
    assert table(section(card, "Version lineage"), "Time (UTC)")[-1][1:] == ["rollback", "v0", "v1"]
    assert "not current (the current model is v0)" in card


def test_card_identifies_the_weights(card_ws: Workspace) -> None:
    digest = hashlib.sha256(b"weights of v1").hexdigest()
    details = {r[0]: r[1] for r in table(modelcard.render(card_ws, 1), "Field")}
    assert details["Weights SHA-256"] == f"`{digest}`"
    assert details["Weights"] == "`models/v1/best.pt`"
    (card_ws.model_dir(1) / "best.pt").unlink()
    details = {r[0]: r[1] for r in table(modelcard.render(card_ws, 1), "Field")}
    assert details["Weights SHA-256"] == "weights file not found"


def test_card_falls_back_to_the_pool_manifest(card_ws: Workspace) -> None:
    (card_ws.dataset_dir(0) / "manifest.csv").unlink()
    for link in (card_ws.dataset_dir(0) / "labels").rglob("*.txt"):
        link.unlink()
    for folder in sorted((card_ws.dataset_dir(0) / "labels").iterdir()):
        folder.rmdir()
    text = section(modelcard.render(card_ws, 1), "Data")
    assert "`pool/manifest.csv`, the current pool (the snapshot of dataset v0 is missing)" in text
    assert "so labelled regions per stage cannot be counted" in text
    assert "Stage 1 is under-represented" in modelcard.render(card_ws, 1)


def test_card_for_an_unregistered_version(card_ws: Workspace) -> None:
    with pytest.raises(KeyError):
        modelcard.write(card_ws, 9)
    assert not modelcard.card_path(card_ws, 9).exists()


def test_card_with_the_dataset_module(tmp_path: Path, evaluation: dict[str, Any]) -> None:
    """Integration with the real producer of the pool and dataset v0 (selftrain.system.dataset)."""
    from selftrain.system import dataset
    ws = Workspace(tmp_path / "ws").ensure()
    assert dataset.init_from_seg(ws, SMOKE) == 0
    write_json(ws.model_dir(0) / "evaluation.json", evaluation)
    registry.add_version(ws, make_record(ws, 0, dataset_version=0, train_images=57,
                                         metrics=registry.metrics_from_evaluation(evaluation)),
                         make_current=True)
    card = modelcard.write(ws, 0).read_text()
    data = section(card, "Data")
    counted = {r[0]: [int(c) for c in r[1:]] for r in table(data, "Split")}
    assert counted == count_smoke_labels()
    total = {r[0]: r for r in table(data, "Source")}["**total**"]
    assert sum(int(c) for c in total[3:7]) == len(ws.read_manifest())


# =========================================================================== card helpers

def test_markdown_table_escapes_cells_and_aligns() -> None:
    assert markdown_table(("a", "b"), [("x|y", "line 1\nline 2"), (1, None)], align="lr") == (
        "| a | b |\n| :--- | ---: |\n| x\\|y | line 1 line 2 |\n| 1 | None |")
    with pytest.raises(ValueError):
        markdown_table(("a", "b"), [("only one cell",)])
    with pytest.raises(ValueError):
        markdown_table(("a", "b"), [], align="lx")


def test_number_formatting() -> None:
    assert [fmt_value(x) for x in (None, float("nan"), float("inf"), True, "0.5")] == [NA] * 5
    assert fmt_value(np.float32(0.5)) == "0.500" and fmt_value(2, digits=1) == "2.0"
    assert fmt_value(0.0123, signed=True) == "+0.012"
    assert fmt_ci([0.1, 0.2]) == "[0.100, 0.200]" and fmt_ci((-0.1, 0.2), signed=True) == (
        "[-0.100, +0.200]")
    assert [fmt_ci(c) for c in ([0.1, None], "ab", (0.1,), None, [float("nan")] * 2)] == [NA] * 5
    assert fmt_estimate(0.5, None) == "0.500"
    assert fmt_estimate(0.5, [0.4, 0.6]) == "0.500 [0.400, 0.600]"


def test_label_statistics_counts_invalid_lines(tmp_path: Path) -> None:
    folder = tmp_path / "labels" / "train"
    folder.mkdir(parents=True)
    poly = "0.1 0.1 0.2 0.1 0.2 0.2"
    (folder / "a.txt").write_text(f"0 {poly}\n2 {poly}\n\n")
    (folder / "b.txt").write_text("")                                   # ulcer-free
    (folder / "c.txt").write_text(f"7 {poly}\nx {poly}\n3 {poly}\n")    # two invalid lines
    (folder / "d.txt").symlink_to(tmp_path / "missing.txt")             # dangling symlink
    stats = modelcard.label_statistics(tmp_path)
    assert stats == {"train": {"images": 4, "ulcer_free": 1, "regions": [1, 0, 1, 1],
                               "invalid": 3}}
    assert modelcard.label_statistics(tmp_path / "nothing") == {}


def test_lineage_stops_at_cycles(ws: Workspace) -> None:
    one, two = make_record(ws, 1, parent=2), make_record(ws, 2, parent=1)
    reg: registry.Registry = {"current": None, "versions": [one, two], "history": []}
    assert modelcard.lineage(reg, 1) == [one, two]
    with pytest.raises(KeyError):
        modelcard.lineage(reg, 3)


def test_find_train_dir_for_external_weights(ws: Workspace) -> None:
    run = ws.root.parent / "runs" / "yoloseg"
    write_json(run / "config.json", YOLOConfig().to_dict())
    (run / "weights").mkdir()
    (run / "weights" / "selected.pt").write_bytes(b"w")
    rec = make_record(ws, 0, architecture="yoloseg", create_weights=False,
                      weights=str(run / "weights" / "selected.pt"))
    assert modelcard.find_train_dir(ws, rec) == run
    assert modelcard.find_train_dir(ws, make_record(ws, 1)) is None
