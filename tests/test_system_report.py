"""Tests of the cycle report (SPEC section 3 step 9, feature F8): HTML + PDF from a cycle's
context and the workspace, LaTeX typesetting and its matplotlib (PdfPages) fallback.

The workspace is synthetic but complete: pool manifest (data/seg_smoke plus images added
in cycle 1), a registry with model v0 (baseline) and v1 (candidate), and evaluations built
with the experiment library's own `summarise` and `wilson_ci`, so they have exactly the
structure the evaluate step stores. The per-image results name the real frozen-test
images of data/seg_smoke, and the example-overlay figure runs the smoke U-Net on them
(CPU). The context strings contain every LaTeX special character, so a failure of the
escaping makes LaTeX fail and the test see the fallback engine.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from types import SimpleNamespace

import cv2
import markupsafe
import numpy as np
import pandas as pd
import pytest

from selftrain.evaluate import summarise, write_rows
from selftrain.metrics import wilson_ci
from selftrain.system import STEP_NAMES, drift, gate, registry, report, steps
from selftrain.system.registry import VersionRecord
from selftrain.system.steps import CycleContext, StepResult
from selftrain.system.workspace import (Workspace, atomic_write_json, license_for_source,
                                        utc_now_iso)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "seg_smoke"
UNET_WEIGHTS = ROOT / "runs" / "smoke" / "unet" / "best.pt"
HAS_LATEX = shutil.which("latexmk") is not None
BATCH = "20261106T050000Z_upload_ward#3"
PIXEL_CONFUSION = np.array([[90000, 900, 400, 100, 50],      # rows: labelled class 0-4
                            [700, 3000, 300, 0, 0],
                            [500, 200, 5000, 300, 0],
                            [100, 0, 200, 2500, 0],
                            [0, 0, 0, 0, 0]])                # stage 4 absent -> NaN
FIGURE_KEYS = ("version_history", "stage_dice", "fpr_wound_type", "data_added", "examples")
MIN_PDF_BYTES = 10_000

requires_smoke = pytest.mark.skipif(
    not (UNET_WEIGHTS.is_file() and (SMOKE / "manifest.csv").is_file()),
    reason="runs/smoke/unet/best.pt or data/seg_smoke missing")
requires_latex = pytest.mark.skipif(not HAS_LATEX, reason="latexmk is not installed")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Apple GPU belongs to the main experiments."""
    monkeypatch.setenv("SELFTRAIN_DEVICE", "cpu")


# ============================================================================ workspace

def frozen_test_images() -> tuple[list[str], list[str]]:
    """Names of the ulcer-containing and the ulcer-free test images of data/seg_smoke."""
    positives, negatives = [], []
    for path in sorted((SMOKE / "images" / "test").glob("*.jpg")):
        mask = cv2.imread(str(SMOKE / "masks" / "test" / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        (positives if (mask > 0).any() else negatives).append(path.stem)
    return positives, negatives


def evaluation_rows(dice: dict[str, float], negatives: list[str], false_positive: str | None
                    ) -> list[dict[str, Any]]:
    """Per-image rows in the format of selftrain.evaluate.evaluate_cache."""
    rows = [{"name": n, "source": n.split("__")[0], "gt_positive": True, "pred_positive": d > 0,
             "dice": d, "iou": d / (2 - d), "gt_stage": 1 + i % 3, "pred_stage": 1 + i % 3,
             "gt_area": 0.08, "pred_area": 0.07} for i, (n, d) in enumerate(dice.items())]
    rows += [{"name": n, "source": n.split("__")[0], "gt_positive": False,
              "pred_positive": n == false_positive, "dice": 0.0 if n == false_positive else 1.0,
              "iou": 0.0 if n == false_positive else 1.0, "gt_stage": 0,
              "pred_stage": 2 if n == false_positive else 0, "gt_area": 0.0,
              "pred_area": 0.03 if n == false_positive else 0.0} for n in negatives]
    return rows


def store_evaluation(ws: Workspace, version: int, rows: list[dict[str, Any]],
                     threshold: float) -> dict[str, Any]:
    """models/v{N}/evaluation.json and its per-image CSV, as steps.evaluate_version stores
    them (strict JSON, CSV path relative to models/v{N})."""
    rel = f"eval/v{version}_test_per_image.csv"
    write_rows(rows, ws.model_dir(version) / rel)
    extra = {}
    for code, (k, n) in {"BG": (1, 40), "D": (6, 50), "N": (0, 30), "V": (2, 25)}.items():
        rate, ci = wilson_ci(k, n)
        extra[code] = {"n": n, "false_positive_rate": rate, "ci95": ci}
    rate, ci = wilson_ci(9, 145)
    extra["all"] = {"n": 145, "false_positive_rate": rate, "ci95": ci}
    res = {"model": f"v{version}", "threshold": threshold, "val_threshold_table": {str(threshold): 0.4},
           "test": summarise(rows, PIXEL_CONFUSION, b=200), "extra_negatives": extra,
           "inference_ms_per_image": 30.0 + version, "version": version,
           "dataset_version": version, "architecture": "unet",
           "weights": str(ws.model_dir(version) / "best.pt"), "data_root": str(SMOKE),
           "per_image_csv": {"test": rel}, "evaluated": utc_now_iso()}
    atomic_write_json(ws.model_dir(version) / "evaluation.json", res)
    return json.loads((ws.model_dir(version) / "evaluation.json").read_text())


def write_pool_manifest(ws: Workspace) -> pd.DataFrame:
    """data/seg_smoke as pool (cycle 0) with five train and one val image marked as
    added in cycle 1."""
    seg = pd.read_csv(SMOKE / "manifest.csv", dtype=str, keep_default_na=False)
    pool = seg[["name", "source", "photograph", "split"]].assign(
        batch="init", license=[license_for_source(s) or "unknown" for s in seg["source"]],
        added_cycle=0)
    added = (pool.loc[pool["split"] == "train"].index[:5].tolist()
             + pool.loc[pool["split"] == "val"].index[:1].tolist())
    pool.loc[added, ["batch", "added_cycle"]] = [BATCH, 1]
    ws.write_manifest(pool, allow_test_change=True)
    return ws.read_manifest()


MODEL_CARD = """# Model card: v1 (unet)

Research prototype -- **not a medical device**; no diagnosis. Special characters:
50 % & 3 # $ _ ~ ^ \\ { } < > | "quotes", `test_dice_ci95`, Δ → ≤ 0.5.

## Metrics on the frozen test split

| Metric | Value | 95 % CI |
| --- | --- | --- |
| Dice, binary | 0.612 | [0.55, 0.67] |
| FPR & extra | 6.2 % | [2.9, 12.8] |

- stage 1 is scarce
- skin-tone coverage unknown

```
python -m selftrain report --version 1
```
"""


def build_workspace(root: Path, weights: Path | None) -> tuple[Workspace, dict[str, Any]]:
    """The synthetic workspace (module docstring) and the context of cycle 1, in which v1
    was trained on dataset v1 and compared with v0. `weights` is linked as both versions'
    checkpoint; None stores a placeholder file (the overlay figure then cannot be drawn)."""
    ws = Workspace(root).ensure()
    write_pool_manifest(ws)
    positives, negatives = frozen_test_images()
    base = dict(zip(positives, np.linspace(0.35, 0.75, len(positives))))
    cand = {n: d + 0.06 + (0.01 if i % 2 else -0.01) for i, (n, d) in enumerate(base.items())}
    evaluations = {0: store_evaluation(ws, 0, evaluation_rows(base, negatives, negatives[0]), 0.5),
                   1: store_evaluation(ws, 1, evaluation_rows(cand, negatives, None), 0.45)}
    for version in (0, 1):
        target = ws.model_dir(version) / "best.pt"
        if weights is None:
            target.write_bytes(b"placeholder: not a checkpoint")
        else:
            target.symlink_to(weights)
    _, cand_dice, base_dice = gate.pair_by_name(cand, base)
    decision = gate.decide(cand_dice, base_dice, current_version=0)
    gate_entry = {**decision.to_dict(), "version": 1, "demo": True}
    (ws.model_dir(1) / "model_card.md").write_text(MODEL_CARD, encoding="utf-8")
    registry.add_version(ws, VersionRecord(
        version=0, dataset_version=0, architecture="unet", train_images=57, train_photographs=51,
        metrics=steps.summary_metrics(evaluations[0]), weights="models/v0/best.pt",
        gate={"promoted": True, "reason": "initial model registered by init"}), make_current=True)
    registry.add_version(ws, VersionRecord(
        version=1, dataset_version=1, architecture="unet", train_images=62, train_photographs=55,
        metrics=steps.summary_metrics(evaluations[1]), gate=gate_entry,
        weights="models/v1/best.pt", model_card="models/v1/model_card.md"),
        make_current=decision.promoted)
    rng = np.random.default_rng(0)
    new, reference = rng.normal(size=(12, 6)), rng.normal(size=(60, 6))
    new[:, 0] += 3.0                                        # lightness shifted
    context = {
        "demo": True, "date": "2026-11-06T05:00:00+00:00", "batches": [BATCH],
        "dataset_version": 1, "model_version": 1, "gate": gate_entry,
        "metrics": steps.summary_metrics(evaluations[1]),
        "steps": {name: {"status": "done", "summary": f"{name}: 100 % done & ok_{{x}} \\ ~ ^ $"}
                  for name in STEP_NAMES[:8]},
        "received": {"roboflow_fid": 7, "rf_dekubitus": 2},
        "rejected": [
            {"name": "frame_01.jpg", "source": "roboflow_fid", "batch": BATCH,
             "reason": "whole_frame_label: the label covers 98.0 % of the image (> 95 %)"},
            {"name": "tiny_#2.jpg", "source": "rf_dekubitus", "batch": BATCH,
             "reason": "image_too_small: shorter side 40 px < 64 px"}],
        "duplicates": [{"image": "copy_of_train_1.jpg", "matched": positives[0], "split": "train",
                        "method": "phash", "action": "added to train (same photograph)"}],
        "drift": drift.compare_features(new, reference, alpha=0.01),
    }
    return ws, context


@pytest.fixture
def full_ws(tmp_path: Path) -> tuple[Workspace, dict[str, Any]]:
    if not (UNET_WEIGHTS.is_file() and (SMOKE / "manifest.csv").is_file()):
        pytest.skip("runs/smoke/unet/best.pt or data/seg_smoke missing")
    return build_workspace(tmp_path / "ws", UNET_WEIGHTS)


@pytest.fixture
def light_ws(tmp_path: Path) -> tuple[Workspace, dict[str, Any]]:
    """Without a usable checkpoint: no model inference, the overlay figure is left out."""
    if not (SMOKE / "manifest.csv").is_file():
        pytest.skip("data/seg_smoke missing")
    return build_workspace(tmp_path / "ws", None)


def spy_latex_logs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collect the LaTeX log of every report.compile_latex call."""
    logs: list[str] = []
    real = report.compile_latex

    def spy(*args: Any, **kwargs: Any) -> tuple[bool, str, str]:
        ok, reason, log_text = real(*args, **kwargs)
        logs.append(log_text)
        return ok, reason, log_text

    monkeypatch.setattr(report, "compile_latex", spy)
    return logs


def fake_latexmk(folder: Path, body: str) -> Path:
    """An executable that stands in for latexmk."""
    folder.mkdir(parents=True, exist_ok=True)
    exe = folder / "latexmk"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return exe


def assert_pdf(path: Path) -> None:
    data = path.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > MIN_PDF_BYTES


def assert_standalone_html(html: str) -> None:
    """No external resource: no scripts, stylesheets or URLs (figures as data URIs)."""
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html and "<link" not in html


# ============================================================================ build_report

@requires_smoke
@requires_latex
def test_build_report_typesets_pdf_and_html_with_latex(full_ws: tuple[Workspace, dict[str, Any]]) -> None:
    ws, context = full_ws
    events: list[tuple[str, str, str]] = []
    result = report.build_report(ws, 1, context, emit=lambda step, status, message, **_: events.append(
        (step, status, message)))
    assert result["engine"] == "latex", result["pdf_error"]
    assert result["pdf_error"] == "" and result["notes"] == []
    assert result["pdf"] == ws.reports / "cycle_1.pdf" and result["html"] == ws.reports / "cycle_1.html"
    assert_pdf(result["pdf"])
    assert result["pages"] == report.pdf_page_count(result["pdf"]) >= 2   # appendix on its own page
    assert not (ws.reports / "cycle_1_latex.log").exists()
    for key in FIGURE_KEYS:
        assert (ws.figures / f"cycle_1_{key}.pdf").is_file() and (ws.figures / f"cycle_1_{key}.png").is_file()
    assert len(result["figures"]) == 2 * len(FIGURE_KEYS)

    sentence = result["gate_sentence"]
    assert sentence.startswith("Version 1 was promoted and replaces version 0")
    assert f"n = {len(frozen_test_images()[0])} paired test images" in sentence
    html = result["html"].read_text(encoding="utf-8")
    assert sentence in html
    assert 'href="cycle_1.pdf"' in html and "matplotlib fallback" not in html
    assert html.count('src="data:image/png;base64,') == len(FIGURE_KEYS)
    assert_standalone_html(html)
    for text in ("whole_frame_label", "image_too_small", "copy_of_train_1.jpg", "lightness",
                 "Pool after cycle 1", "Model card of v1 (candidate)", "skin-tone coverage unknown"):
        assert text in html

    tex = result["tex"].read_text(encoding="utf-8")
    assert r"ward\#3" in tex and r"100 \% done \& ok\_" in tex
    assert r"\{x\} \textbackslash{} \textasciitilde{} \textasciicircum{} \$" in tex
    assert all(step == "report" and status == "started" for step, status, _ in events)
    assert any("LaTeX" in message for _, _, message in events)


@requires_smoke
def test_html_report_shows_metrics_data_and_drift(full_ws: tuple[Workspace, dict[str, Any]]) -> None:
    ws, context = full_ws
    data = report.collect(ws, 1, context)
    assert (data.candidate, data.baseline, data.in_service, data.subject) == (1, 0, 1, 1)
    assert data.outcome == "promoted" and data.demo is True
    assert data.drift is not None and data.drift.shifted == ["lightness"]
    assert data.data_total is not None and (data.data_total.added, data.data_total.train,
                                            data.data_total.val) == (6, 5, 1)
    assert data.data_total.received == 9 and data.n_rejected == 2
    labels = [row.label for row in data.metric_rows]
    assert labels[0].startswith("Dice") and any("threshold" in label for label in labels)
    dice_row = data.metric_rows[0]
    assert dice_row.candidate != report.DASH and dice_row.baseline != report.DASH
    assert [p.version for p in data.versions] == [0, 1]
    assert [p.current for p in data.versions] == [False, True]
    assert data.examples is not None and data.examples.threshold == 0.45


@requires_latex
def test_latex_missing_falls_back_to_pdfpages(light_ws: tuple[Workspace, dict[str, Any]]) -> None:
    ws, context = light_ws
    events: list[str] = []
    result = report.build_report(ws, 1, context, latexmk="selftrain-no-such-latexmk",
                                 emit=lambda step, status, message, **_: events.append(message))
    assert result["engine"] == "matplotlib"
    assert "not found on PATH" in result["pdf_error"]
    assert_pdf(result["pdf"])
    assert result["pages"] is not None and result["pages"] >= 2
    html = result["html"].read_text(encoding="utf-8")
    assert result["gate_sentence"] in html
    assert "matplotlib fallback because LaTeX failed" in html
    assert any("LaTeX failed" in message for message in events)
    assert len(result["notes"]) == 1 and result["notes"][0].startswith("Figure 'examples' is missing")
    assert str(markupsafe.escape(result["notes"][0])) in html
    assert_standalone_html(html)


def test_latex_error_falls_back_and_keeps_the_log(light_ws: tuple[Workspace, dict[str, Any]],
                                                  tmp_path: Path) -> None:
    ws, context = light_ws
    exe = fake_latexmk(tmp_path / "bin", "echo '! Undefined control sequence.' > report.log\n"
                                         "echo 'l.42 foo' >> report.log\nexit 12\n")
    result = report.build_report(ws, 1, context, latexmk=str(exe))
    assert result["engine"] == "matplotlib"
    assert "exit status 12" in result["pdf_error"] and "Undefined control sequence" in result["pdf_error"]
    assert "Undefined control sequence" in (ws.reports / "cycle_1_latex.log").read_text()
    assert_pdf(result["pdf"])
    assert result["gate_sentence"] in result["html"].read_text(encoding="utf-8")


def test_a_hanging_latex_is_stopped_after_the_timeout(light_ws: tuple[Workspace, dict[str, Any]],
                                                      tmp_path: Path) -> None:
    ws, context = light_ws
    exe = fake_latexmk(tmp_path / "bin", "sleep 60\n")
    result = report.build_report(ws, 1, context, latexmk=str(exe), timeout=1.0)
    assert result["engine"] == "matplotlib" and "timed out after 1 s" in result["pdf_error"]
    assert_pdf(result["pdf"])


def test_fallback_pdf_contains_every_section(light_ws: tuple[Workspace, dict[str, Any]]) -> None:
    ws, context = light_ws
    data = report.collect(ws, 1, context)
    lines = report.fallback_lines(data, "latexmk not found")
    for heading in ("GATE DECISION", "CYCLE STEPS", "DATA", "QUALITY CONTROL", "DUPLICATES",
                    "DRIFT", "METRICS ON THE FROZEN TEST SPLIT"):
        assert heading in lines
    assert any("latexmk not found" in line for line in lines)


def test_report_of_a_cycle_without_new_model(light_ws: tuple[Workspace, dict[str, Any]]) -> None:
    """Nothing trained (e.g. every image rejected): the version in service is shown."""
    ws, context = light_ws
    context = {"demo": False, "batches": [BATCH], "model_version": None, "gate": None,
               "steps": {"ingest": {"status": "done", "summary": "2 images"},
                         "train": {"status": "skipped", "summary": "no new dataset version: nothing to train"}},
               "rejected": context["rejected"]}
    result = report.build_report(ws, 2, context, latexmk="selftrain-no-such-latexmk")
    assert result["gate_sentence"] == ("No new model was trained in this cycle (no new dataset "
                                       "version: nothing to train); version 1 stays in service.")
    html = result["html"].read_text(encoding="utf-8")
    assert result["gate_sentence"] in html and "v1 (in service)" in html
    assert_pdf(result["pdf"])


def test_build_report_rejects_an_invalid_cycle_number(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws").ensure()
    for cycle in (-1, True, "x", 1.5):
        with pytest.raises(ValueError):
            report.build_report(ws, cycle, {})


# ============================================================================ gate sentence

GATE_PROMOTED = {"promoted": True, "mean_diff": 0.034, "ci95": [0.004, 0.071], "n": 9,
                 "margin": 0.02, "compared_to": 0, "reason": "promoted: ..."}


def test_gate_sentence_for_every_outcome() -> None:
    promoted = report.gate_sentence(1, GATE_PROMOTED, 1)
    assert promoted.startswith("Version 1 was promoted and replaces version 0: its mean test "
                               "Dice is 0.034 higher than that of version 0")
    assert "95 % CI +0.004 to +0.071, n = 9 paired test images" in promoted
    assert "lower CI bound above −0.02" in promoted
    worse = report.gate_sentence(2, {**GATE_PROMOTED, "promoted": False, "mean_diff": -0.01,
                                     "ci95": [-0.03, 0.01]}, 0)
    assert worse.startswith("Version 2 was not promoted and version 0 stays in service")
    assert "0.010 lower" in worse and "not better on average" in worse
    risky = report.gate_sentence(2, {**GATE_PROMOTED, "promoted": False, "mean_diff": 0.004,
                                     "ci95": [-0.04, 0.05]}, 0)
    assert "may be worse by more than the margin" in risky
    tie = report.gate_sentence(2, {**GATE_PROMOTED, "promoted": False, "mean_diff": 0.0,
                                   "ci95": [0.0, 0.0]}, 0)
    assert "equals that of version 0" in tie and "not better on average" in tie
    first = report.gate_sentence(0, {"promoted": True, "mean_diff": None, "ci95": None}, 0)
    assert first == ("Version 0 was promoted and is now in service because no model was in "
                     "service before it.")
    none = report.gate_sentence(None, None, None, "nothing ingested")
    assert none == "No new model was trained in this cycle (nothing ingested); no model is in service."
    for sentence in (promoted, worse, risky, tie, first, none):     # safe for HTML and LaTeX
        assert not set(sentence) & set("&<>\"'#$_{}\\^~")


# ============================================================================ write_report, step

def test_write_report_renders_a_stored_cycle_summary(light_ws: tuple[Workspace, dict[str, Any]]) -> None:
    ws, context = light_ws
    summary = json.loads(json.dumps({"cycle": 1, "model_version": 1, "report_context": context}))
    result = report.write_report(ws, summary, latexmk="selftrain-no-such-latexmk")
    assert result["html"] == ws.reports / "cycle_1.html"
    assert result["gate_sentence"] in result["html"].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="report_context"):
        report.write_report(ws, {"cycle": 1})
    with pytest.raises(ValueError):
        report.write_report(ws, ["not", "a", "summary"])        # type: ignore[arg-type]


@requires_smoke
@requires_latex
def test_report_step_writes_model_card_summary_and_report(full_ws: tuple[Workspace, dict[str, Any]],
                                                          capsys: pytest.CaptureFixture[str],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """Step 9 with the real model card (modelcard.write) typeset by LaTeX -- completely
    (every section) and without a line or table wider than the page -- then the CLI
    re-renders the report from the stored summary."""
    from selftrain.__main__ import main
    ws, context = full_ws
    events: list[tuple[str, str]] = []
    logs = spy_latex_logs(monkeypatch)
    record = SimpleNamespace(path=Path("a.jpg"), polygons=[], source="roboflow_fid",
                             license="CC BY 4.0", batch=BATCH)
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1, demo=True,
                       emit=lambda step, status, message, **_: events.append((step, status)),
                       batches=[BATCH], records=[record], dataset_version=1, model_version=1,
                       metrics=context["metrics"], artifacts={"gate": context["gate"]},
                       results={"ingest": StepResult(True, "1 labelled image from 1 batch(es)")})
    result = steps.step_report(ctx)
    assert result.ok and result.data["engine"] == "latex", result.data["pdf_error"]
    assert result.summary.startswith("report written: cycle_1.pdf (")
    card = ws.model_dir(1) / "model_card.md"
    assert card.is_file() and "not a medical device" in card.read_text().lower()
    assert registry.get_version(ws, 1).model_card == "models/v1/model_card.md"
    stored = json.loads((ws.reports / "cycle_1.json").read_text())
    assert stored["model_version"] == 1 and stored["report_context"]["received"] == {"roboflow_fid": 1}
    assert_pdf(ws.reports / "cycle_1.pdf")
    html = (ws.reports / "cycle_1.html").read_text(encoding="utf-8")
    sentence = report.gate_sentence(1, context["gate"], 1)
    assert sentence in html
    assert set(events) == {("report", "started")}
    assert ctx.artifacts["report_pdf"] == ws.reports / "cycle_1.pdf"
    assert len(logs) == 1 and "Overfull \\hbox" not in logs[0]
    tex = (ws.reports / "cycle_1.tex").read_text(encoding="utf-8")
    headings = re.findall(r"^#+ (.+)$", card.read_text(), flags=re.M)
    assert len(headings) >= 8 and all(report.latex_escape(h) in tex for h in headings)  # whole card
    assert "The model card continues" not in tex

    (ws.reports / "cycle_1.html").unlink()
    assert main(["report", "--version", "1", "--workspace", str(ws.root)]) == 0
    assert "report of cycle 1 (model v1)" in capsys.readouterr().out
    assert sentence in (ws.reports / "cycle_1.html").read_text(encoding="utf-8")


# ============================================================== model card in the PDF

SOURCES_TABLE = """\
| Source | Dataset | Licence | train | val | test | external | Photographs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `roboflow_woundcare` | [PI Staging TSYL, Roboflow Universe](https://universe.roboflow.com/pi-iqy1t/pi-staging-tsyl) | CC BY 4.0 | 10 | 1 | 1 | 0 | 12 |
| `neg_azh` | [AZH wound classification dataset (UWM BigData Lab), ulcer-free classes only](https://github.com/uwm-bigdata/wound-classification-using-images-and-locations) | not stated by the source | 8 | 3 | 1 | 0 | 12 |
| **total** |  |  | 18 | 4 | 2 | 0 | 24 |
"""
LINEAGE_TABLE = """\
| Version | Registered (UTC) | Dataset | Architecture | Train images | Test Dice [95 % CI] | Gate |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| **v1** (this card) | 2026-11-06T06:28:31+00:00 | v1 | unet | 62 | 0.416 [0.227, 0.614] | not promoted (Dice -0.053 vs v0) |
| v0 | 2026-11-06T06:23:58+00:00 | v0 | unet | 57 | 0.468 [0.309, 0.651] | promoted |
"""
DETAILS_TABLE = """\
| Field | Value |
| :--- | :--- |
| Task | pixel-wise segmentation of pressure ulcers in colour photographs into background and stages 1-4; the stage of a photograph is the stage covering most predicted ulcer pixels |
| Weights SHA-256 | `244f320d3557f827f4f4edefb64dbf93d8bda8dae04e2abc7a77109da7fd89cd` |
"""
COLUMN = re.compile(r">\{[^{}]*\}X|[lrc]")


def columns(layout: report.TableLayout) -> list[str]:
    """The column specifications of a layout, one per column."""
    found = COLUMN.findall(layout.spec)
    assert "".join(found) == layout.spec and len(found) == layout.columns
    return found


def factor(column: str) -> float:
    match = re.search(r"\\hsize=([0-9.]+)\\hsize", column)
    return float(match.group(1)) if match else 1.0


def test_markdown_tables_keep_the_alignment_of_their_separator_row() -> None:
    (block,) = report.parse_markdown(SOURCES_TABLE)
    assert block.kind == "table" and len(block.rows) == 4
    assert block.align == ("l", "l", "l", "r", "r", "r", "r", "r")
    (block,) = report.parse_markdown("| a | b | c | d |\n| --- | :---: | ---: | :--- |\n| 1 | 2 | 3 | 4 |")
    assert block.align == ("", "c", "r", "l")


def test_a_table_that_fits_the_line_keeps_its_natural_width() -> None:
    layout = report.table_layout([("Stage", "Dice"), ("Stage 1", "0.026"), ("Macro mean", "0.041")],
                                 ("l", "r"))
    assert layout == report.TableLayout("lr", 2, False)
    latex = report.markdown_to_latex("| Stage | Dice |\n| :--- | ---: |\n| Stage 1 | 0.026 |")
    assert r"\begin{tabular}{@{}lr@{}}" in latex and "tabularx" not in latex


def test_long_text_wraps_while_numbers_and_short_columns_keep_their_width() -> None:
    (block,) = report.parse_markdown(SOURCES_TABLE)
    layout = report.table_layout(block.rows, block.align)
    source, dataset, licence, *numbers = columns(layout)
    assert layout.wrap and source == "l" and numbers == ["r"] * 5
    assert dataset.endswith("X") and licence.endswith("X")
    assert factor(dataset) > factor(licence)                  # the description gets the room
    assert factor(dataset) + factor(licence) == pytest.approx(2.0, abs=1e-9)   # tabularx rule
    latex = report.markdown_to_latex(SOURCES_TABLE)
    assert latex.count(r"\begin{tabularx}{\linewidth}") == 1 and r"\end{tabularx}" in latex


def test_short_headers_are_never_squeezed_into_a_wrapping_column() -> None:
    """Regression: with equal X columns the headers "Dataset" and "Architecture" of the
    lineage table overlapped; columns whose cells are no longer than a wrapping
    column's minimum width are never wrapped."""
    (block,) = report.parse_markdown(LINEAGE_TABLE)
    layout = report.table_layout(block.rows, block.align)
    version, registered, dataset_, architecture, *_ = columns(layout)
    assert (dataset_, architecture) == ("l", "l")
    assert registered.endswith("X")                # 25 characters, breakable after "-"
    assert sum(factor(c) for c in columns(layout) if c.endswith("X")) == \
        pytest.approx(sum(c.endswith("X") for c in columns(layout)), abs=1e-9)


def test_a_wrapping_column_keeps_its_alignment() -> None:
    long_text = "word " * 40
    assert report.table_layout([("Field", "Value"), ("Reason", long_text)], ("l", "l")).spec == \
        r"l>{\raggedright\arraybackslash}X"
    assert report.table_layout([("Note", "Amount"), ("a", long_text)], ("l", "r")).spec == \
        r"l>{\raggedleft\arraybackslash}X"


def test_digests_and_export_hashes_get_break_points_but_words_do_not() -> None:
    digest = "244f320d3557f827f4f4edefb64dbf93d8bda8dae04e2abc7a77109da7fd89cd"
    assert str(report.latex_breakable(digest)).split(r"\allowbreak{}") == \
        [digest[i:i + 8] for i in range(0, len(digest), 8)]
    assert r"\allowbreak" not in report.latex_breakable("classification")
    assert r"\allowbreak" not in report.latex_breakable("photographs2020")      # 15 characters
    name = "roboflow_mobile__p20_JPG.rf.7ffe40c278abc673dd2917618c09c666__dce406"
    pieces = str(report.latex_breakable(name)).split(r"\allowbreak{}")
    assert "".join(pieces).replace(r"\_", "_") == name
    assert max(len(piece.replace(r"\_", "_")) for piece in pieces) <= 9


@requires_latex
def test_model_card_tables_typeset_within_the_page(tmp_path: Path) -> None:
    """The three table shapes of a model card -- many columns with a long description,
    short headers next to wrapping columns, a SHA-256 digest -- compiled by LaTeX in the
    report's page geometry and font size: no overfull line."""
    body = report.markdown_to_latex("\n\n".join([SOURCES_TABLE, LINEAGE_TABLE, DETAILS_TABLE]))
    tex = "\n".join([
        r"\documentclass[10pt,a4paper]{article}", r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}", r"\usepackage{lmodern}", r"\usepackage{textcomp}",
        r"\usepackage[a4paper,margin=2cm]{geometry}", r"\usepackage{booktabs}",
        r"\usepackage{tabularx}", r"\usepackage{enumitem}", r"\begin{document}",
        r"{\small", body, r"\par}", r"\end{document}"])
    ok, reason, log_text = report.compile_latex(tex, [], tmp_path / "tables.pdf")
    assert ok, reason
    assert "Overfull \\hbox" not in log_text
    assert (tmp_path / "tables.pdf").read_bytes()[:5] == b"%PDF-"


def test_a_long_model_card_is_cut_before_a_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    card = "\n".join(["# Card", "", "## A", "", "| x | y |", "| --- | --- |", "| 1 | 2 |",
                      "| 3 | 4 |", "", "## B", "", "```", "## not a heading", "```", "", "text"])
    assert report._card_for_pdf(card) == (card, False)               # within the limit
    monkeypatch.setattr(report, "MODEL_CARD_MAX_LINES", 9)
    cut, truncated = report._card_for_pdf(card)
    assert truncated and cut == "\n".join(card.splitlines()[:8])   # whole table, before "## B"
    monkeypatch.setattr(report, "MODEL_CARD_MAX_LINES", 13)        # "## not a heading" is code
    assert report._card_for_pdf(card) == (cut, True)
    monkeypatch.setattr(report, "MODEL_CARD_MAX_LINES", 4)
    paragraphs = "\n".join(["first", "", "second", "", "third", "", "fourth"])
    assert report._card_for_pdf(paragraphs) == ("first\n\nsecond", True)  # at a blank line
