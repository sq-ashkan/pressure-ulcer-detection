"""End-to-end test of the self-retraining system (SPEC sections 1-8) through its
command-line interface: `init` creates a workspace from a dataset in the format of the
thesis dataset (data/seg) and registers a trained model as version 0; `run --demo` then
runs one complete cycle -- ingest, quality, dedupe, drift, dataset, train, evaluate,
gate, report, notify -- on one inbox batch built to exercise the protections:

* six labelled photographs of a Roboflow export (data/raw/roboflow_mobile/valid) with
  the export's own classes (NonPU, Stage1-4), mapped by the batch's source.yaml;
* a byte-identical copy of a frozen test photograph of the pool: frozen-test protection
  must keep it out of the pool and report it;
* a photograph whose label covers the whole frame: the quality gate must reject it
  (whole_frame_label, the defect of PPA Fig. 6.4);
* a photograph without a label: ingest files it for active learning, and
  `rank-unlabeled` ranks it with the current model.

The pool is a copy of data/seg_smoke whose list of extra ulcer-free photographs is cut
to two per wound class, so the three evaluations stay short on the CPU; the cycle trains
one demo epoch on at most eight images (config.yaml) and Telegram is replaced by a
recording fake. The expensive run happens once (module fixture); the tests check its
outcome from different sides.
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pandas as pd
import pytest

from selftrain.__main__ import main
from selftrain.system import STEP_NAMES, STEP_STATUSES, registry, steps
from selftrain.system.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "seg_smoke"
UNET_WEIGHTS = ROOT / "runs" / "smoke" / "unet" / "best.pt"
RAW = ROOT / "data" / "raw" / "roboflow_mobile"
BATCH = "mobile_valid_e2e"
NEW_PHOTOS = ("Stage1_13_jpg.rf.e7f39702b2ffb692e83fedd1f49d1af0",
              "Stage2_12_jpg.rf.1e9beec281b611b23b181867322f2570",
              "p18_JPG.rf.067538fb74ddc016ff3712101ddf1ff0",
              "pu22_JPG.rf.a47acd3467591f54dc0ce7e38b1a3af1",
              "pu38_JPG.rf.b3c075cfb1ff8488d52d43f67533762e",
              "s10_jpg.rf.521cef6dad12c36a4cadf903a54057ea")
WHOLE_FRAME = "p19_jpg.rf.b93f0c59631b50040b2253331793a540"
UNLABELED = "s18_jpg.rf.6210295f3b73b3bb2169b13151687558"
FROZEN_COPY = "roboflow_mobile__p20_JPG.rf.7ffe40c278abc673dd2917618c09c666__dce406"
NEGATIVES_PER_CLASS = 2
EVENT_KEYS = {"ts", "cycle", "step", "status", "summary", "data"}

pytestmark = pytest.mark.skipif(
    not (UNET_WEIGHTS.is_file() and (SMOKE / "manifest.csv").is_file()
         and (RAW / "valid" / "images").is_dir()),
    reason="runs/smoke/unet/best.pt, data/seg_smoke or data/raw/roboflow_mobile missing")


# ============================================================================ helpers

def image_file(folder: Path, stem: str) -> Path:
    """The image named `stem` in `folder` (any suffix)."""
    (path,) = [p for p in folder.iterdir() if p.stem == stem]
    return path


def trimmed_seg_root(target: Path) -> Path:
    """A copy of data/seg_smoke whose negatives_extra.csv keeps the first
    NEGATIVES_PER_CLASS photographs of every wound class (sorted by path)."""
    shutil.copytree(SMOKE, target, ignore=shutil.ignore_patterns("negatives_extra.csv"))
    extra = pd.read_csv(SMOKE / "negatives_extra.csv").sort_values("path")
    extra.groupby("azh_class").head(NEGATIVES_PER_CLASS).to_csv(target / "negatives_extra.csv",
                                                                 index=False)
    return target


def make_batch(inbox: Path) -> Path:
    """inbox/<BATCH>/ as a Roboflow export: valid/images|labels, data.yaml, source.yaml.

    The labels of the frozen-test copy are rewritten from the pool's classes (c = stage
    c + 1) into the export's (c + 1 = Stage c + 1), so one class map fits every label."""
    batch = inbox / BATCH
    images, labels = batch / "valid" / "images", batch / "valid" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for stem in NEW_PHOTOS:
        shutil.copy2(image_file(RAW / "valid" / "images", stem), images)
        shutil.copy2(RAW / "valid" / "labels" / f"{stem}.txt", labels)
    shutil.copy2(image_file(RAW / "valid" / "images", WHOLE_FRAME), images)
    (labels / f"{WHOLE_FRAME}.txt").write_text("1 0 0 1 0 1 1 0 1\n")
    shutil.copy2(image_file(RAW / "valid" / "images", UNLABELED), images)      # no label
    shutil.copy2(image_file(SMOKE / "images" / "test", FROZEN_COPY), images)
    pool_label = (SMOKE / "labels" / "test" / f"{FROZEN_COPY}.txt").read_text().splitlines()
    (labels / f"{FROZEN_COPY}.txt").write_text(
        "".join(f"{int(line.split()[0]) + 1} {line.split(maxsplit=1)[1]}\n" for line in pool_label))
    shutil.copy2(RAW / "data.yaml", batch / "data.yaml")
    (batch / "source.yaml").write_text("name: roboflow_mobile\nlicense: CC BY 4.0\n"
                                       "class_map: {0: background, 1: 1, 2: 2, 3: 3, 4: 4}\n")
    return batch


def cli(*argv: str) -> tuple[int, str]:
    """Run `python -m selftrain` in-process; returns (exit status, standard output)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        status = main(list(argv))
    return status, out.getvalue()


class ErrorRecords(logging.Handler):
    """Collects the ERROR records of the selftrain loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@dataclass
class EndToEnd:
    ws: Workspace
    init_out: str
    run_status: int
    run_out: str
    errors: list[str]
    telegram: list[tuple[str, dict[str, Any]]]

    def events(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.ws.events_path.read_text(encoding="utf-8").splitlines()]

    def final(self, step: str) -> dict[str, Any]:
        """The last event of `step` (its final status)."""
        return [e for e in self.events() if e["step"] == step][-1]


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> Iterator[EndToEnd]:
    tmp = tmp_path_factory.mktemp("e2e")
    seg = trimmed_seg_root(tmp / "seg")
    root = tmp / "ws"
    telegram: list[tuple[str, dict[str, Any]]] = []
    fake = SimpleNamespace(
        send_document=lambda path, caption="", silent=True: telegram.append(
            ("document", {"path": path, "caption": caption, "silent": silent})),
        send_message=lambda text, silent=True: telegram.append(
            ("message", {"text": text, "silent": silent})))
    handler = ErrorRecords()
    logger = logging.getLogger("selftrain")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SELFTRAIN_DEVICE", "cpu")              # the GPU belongs to the experiments
        mp.setattr(steps, "load_notifier", lambda: fake)  # never the real Telegram API
        status, init_out = cli("init", "--from", str(seg), "--architecture", "unet",
                               "--weights", str(UNET_WEIGHTS), "--workspace", str(root))
        assert status == 0, init_out
        ws = Workspace(root)
        ws.save_config(replace(ws.load_config(), demo_epochs=1, demo_max_train_images=8,
                               telegram=True))
        make_batch(ws.inbox)
        logger.addHandler(handler)
        try:
            run_status, run_out = cli("run", "--demo", "--workspace", str(root))
        finally:
            logger.removeHandler(handler)
        yield EndToEnd(ws, init_out, run_status, run_out,
                       [handler.format(r) for r in handler.records], telegram)


# ============================================================================== tests

def test_init_makes_the_thesis_split_version_0(e2e: EndToEnd) -> None:
    ws = e2e.ws
    reg = registry.load(ws)
    v0 = registry.get_version(ws, 0)
    assert v0.dataset_version == 0 and v0.gate["promoted"] is True
    assert (ws.model_dir(0) / "evaluation.json").is_file() and (ws.root / v0.model_card).is_file()
    assert reg["history"][0]["action"] == "promote" and reg["history"][0]["version"] == 0
    evaluation = json.loads((ws.model_dir(0) / "evaluation.json").read_text())
    assert evaluation["extra_negatives"]["all"]["n"] == 5 * NEGATIVES_PER_CLASS
    smoke = pd.read_csv(SMOKE / "manifest.csv")
    v0_manifest = pd.read_csv(ws.dataset_dir(0) / "manifest.csv")
    assert sorted(v0_manifest["name"]) == sorted(smoke["name"])
    assert "model v0 registered and current" in e2e.init_out


def test_the_cycle_runs_all_ten_steps_and_reports_every_event(e2e: EndToEnd) -> None:
    assert e2e.run_status == 0, e2e.run_out
    events = e2e.events()
    for event in events:
        assert set(event) == EVENT_KEYS and event["cycle"] == 1
        assert event["step"] in STEP_NAMES and event["status"] in STEP_STATUSES
        assert datetime.fromisoformat(event["ts"]).tzinfo is not None
    finals = [e for e in events if e["status"] != "started"]
    assert [(e["step"], e["status"]) for e in finals] == [(name, "done") for name in STEP_NAMES]
    status = json.loads(e2e.ws.status_path.read_text())
    assert (status["cycle"], status["running"], status["result"]) == (1, False, "done")
    assert all(state["status"] == "done" for state in status["steps"].values())
    # the CLI listener printed every final event; nothing was logged as an error
    # (regression: the drift step's data has its own "status" key)
    for name in STEP_NAMES:
        assert f"  {name:<8} done     " in e2e.run_out
    assert e2e.errors == []
    assert "cycle 1: done" in e2e.run_out


def test_quality_rejects_the_whole_frame_label(e2e: EndToEnd) -> None:
    ingest = e2e.final("ingest")["data"]
    assert ingest["batches"] == {BATCH: len(NEW_PHOTOS) + 2}         # + whole frame + copy
    quality = e2e.final("quality")["data"]
    assert (quality["checked"], quality["accepted"]) == (len(NEW_PHOTOS) + 2, len(NEW_PHOTOS) + 1)
    (rejected,) = quality["rejected"]
    assert Path(rejected["file"]).stem == WHOLE_FRAME
    assert rejected["reason"].startswith("whole_frame_label:")
    saved = json.loads((e2e.ws.processed / BATCH / "quality_report.json").read_text())
    assert saved["rejected_by_code"] == {"whole_frame_label": 1}
    assert (e2e.ws.processed / BATCH / "ingest_report.json").is_file()


def test_a_copy_of_a_frozen_test_photograph_never_enters_the_pool(e2e: EndToEnd) -> None:
    dedupe = e2e.final("dedupe")["data"]
    (dropped,) = dedupe["dropped_test_matches"]
    assert Path(dropped["file"]).stem == FROZEN_COPY
    assert dropped["split"] == "test" and dropped["pool_name"] == FROZEN_COPY
    pool = e2e.ws.read_manifest()
    added = pool[pool["added_cycle"].astype(int) == 1]
    assert not added["name"].str.contains(FROZEN_COPY.split("__")[1], regex=False).any()
    assert not added["name"].str.contains(WHOLE_FRAME, regex=False).any()
    assert set(added["split"]) <= {"train", "val"}
    smoke_test = set(pd.read_csv(SMOKE / "manifest.csv").query("split == 'test'")["name"])
    assert set(pool.query("split == 'test'")["name"]) == smoke_test
    # dataset v1 lists exactly the frozen test images of v0, byte for byte
    assert (e2e.ws.dataset_dir(1) / "test_frozen.txt").read_text() == \
        (e2e.ws.dataset_dir(0) / "test_frozen.txt").read_text()
    report_text = (e2e.ws.reports / "cycle_1.html").read_text(encoding="utf-8")
    assert "copy of a frozen test photograph" in report_text


def test_every_accepted_image_is_added_or_reported(e2e: EndToEnd) -> None:
    dedupe = e2e.final("dedupe")["data"]
    dataset = e2e.final("dataset")["data"]
    accepted = e2e.final("quality")["data"]["accepted"]
    assert len(dataset["rows"]) + len(dataset["skipped"]) == accepted
    assert len(dataset["new_photographs"]) == len(dedupe["new_groups"]) >= 1
    assert dataset["version"] == 1 and (e2e.ws.dataset_dir(1) / "dataset.yaml").is_file()
    v0, v1 = (pd.read_csv(e2e.ws.dataset_dir(v) / "manifest.csv") for v in (0, 1))
    assert len(v1) == len(v0) + len(dataset["rows"])


def test_the_candidate_is_registered_with_its_gate_decision(e2e: EndToEnd) -> None:
    v1 = registry.get_version(e2e.ws, 1)
    gate = v1.gate
    assert isinstance(gate["promoted"], bool) and gate["compared_to"] == 0
    assert gate["n"] == v1.metrics["n_test_positive"] and len(gate["ci95"]) == 2
    assert gate["reason"] and gate["demo"] is True
    assert (v1.dataset_version, v1.parent, v1.train_images) == (1, 0, 8)
    assert registry.load(e2e.ws)["current"] == (1 if gate["promoted"] else 0)
    assert e2e.final("train")["data"]["epochs_run"] == 1
    assert v1.metrics["extra_negatives_fpr"] is not None


def test_the_report_and_the_model_card_are_written(e2e: EndToEnd) -> None:
    ws = e2e.ws
    data = e2e.final("report")["data"]
    assert data["pdf"] == str(ws.reports / "cycle_1.pdf") and data["html"] == str(ws.reports / "cycle_1.html")
    pdf = (ws.reports / "cycle_1.pdf").read_bytes()
    assert pdf[:5] == b"%PDF-" and len(pdf) > 10_000
    if shutil.which("latexmk"):
        assert data["engine"] == "latex", data["pdf_error"]
    html = (ws.reports / "cycle_1.html").read_text(encoding="utf-8")
    assert "whole_frame_label" in html and "Gate decision" in html
    card = ws.root / registry.get_version(ws, 1).model_card
    assert card.is_file() and "not a medical device" in card.read_text().lower()
    summary = json.loads((ws.reports / "cycle_1.json").read_text())
    assert summary["model_version"] == 1 and summary["report_context"]["received"] == {
        "roboflow_mobile": len(NEW_PHOTOS) + 2}


def test_telegram_receives_the_pdf_silently(e2e: EndToEnd) -> None:
    (kind, call), = e2e.telegram
    assert kind == "document" and call["path"] == str(e2e.ws.reports / "cycle_1.pdf")
    assert call["silent"] is True
    assert call["caption"].startswith("Pressure-ulcer system: cycle 1 (demo)")
    assert e2e.final("notify")["summary"] == "report cycle_1.pdf sent to Telegram"


def test_the_unlabeled_photo_is_filed_and_ranked(e2e: EndToEnd) -> None:
    filed = sorted(p.relative_to(e2e.ws.unlabeled).as_posix()
                   for p in e2e.ws.unlabeled.rglob("*") if p.is_file())
    assert filed == [f"{BATCH}/valid/images/{image_file(RAW / 'valid' / 'images', UNLABELED).name}"]
    status, out = cli("rank-unlabeled", "--workspace", str(e2e.ws.root))
    assert status == 0 and "1 photos ranked" in out
    ranking = pd.read_csv(e2e.ws.active / "ranking.csv")
    assert len(ranking) == 1 and 0.0 <= ranking.loc[0, "uncertainty"] <= 1.0


def test_status_and_report_commands_read_the_finished_cycle(e2e: EndToEnd) -> None:
    status, out = cli("status", "--workspace", str(e2e.ws.root))
    assert status == 0 and "cycle 1 (demo): done" in out
    assert " v   1 v  1 unet" in out and " v   0 v  0 unet" in out
    (e2e.ws.reports / "cycle_1.html").unlink()
    status, out = cli("report", "--version", "1", "--workspace", str(e2e.ws.root))
    assert status == 0 and "report of cycle 1 (model v1)" in out
    assert (e2e.ws.reports / "cycle_1.html").is_file()
