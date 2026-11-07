"""Tests of the self-retraining cycle: promotion gate and rollback (SPEC section 3 step 8,
section 5), cycle runner with lock, events and status (sections 3 and 4), the real steps 2-6
on the smoke dataset, the notify step, active learning (section 7) and the CLI (section 8).

Everything runs on the CPU (SELFTRAIN_DEVICE=cpu) in workspaces under pytest's tmp_path.
The gate is tested on synthetic per-image Dice arrays whose differences are constructed so
that the decision follows from the rule alone; the cycle runner with injected step
functions; training uses one epoch on at most eight images; Telegram is replaced by a fake.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import pytest

from selftrain.__main__ import THESIS_PIPELINE, build_parser, main, print_cycle_result, print_event
from selftrain.system import (STEP_NAMES, STEP_STATUSES, active, cycle, dataset, gate,
                              registry, report, steps)
from selftrain.system.cycle import (CycleLock, CycleLockedError, inbox_signature, lock_owner,
                                    lock_path, read_status, run_cycle)
from selftrain.system.registry import VersionRecord
from selftrain.system.steps import CycleContext, StepResult
from selftrain.system.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "seg_smoke"
UNET_WEIGHTS = ROOT / "runs" / "smoke" / "unet" / "best.pt"
RAW = ROOT / "data" / "raw" / "roboflow_woundcare" / "train"
WOUNDCARE_STAGES = {2: 0, 3: 1, 4: 2, 5: 3}     # export classes stage-1..stage-4 -> stages 0..3
EVENT_KEYS = {"ts", "cycle", "step", "status", "summary", "data"}
SUBCOMMANDS = ["init", "run", "watch", "status", "report", "rank-unlabeled", "rollback",
               "dashboard", "reproduce"]
MARGIN = 0.02

requires_smoke_unet = pytest.mark.skipif(
    not (UNET_WEIGHTS.is_file() and (SMOKE / "manifest.csv").is_file()),
    reason="runs/smoke/unet/best.pt or data/seg_smoke missing")
requires_raw = pytest.mark.skipif(
    not ((SMOKE / "manifest.csv").is_file() and (RAW / "images").is_dir()),
    reason="data/seg_smoke or data/raw/roboflow_woundcare missing")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Apple GPU belongs to the main experiments."""
    monkeypatch.setenv("SELFTRAIN_DEVICE", "cpu")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()


# ============================================================================ helpers

def quiet(_message: str) -> None:
    """Log sink of the library calls."""


def register(ws: Workspace, version: int, *, current: bool = False, parent: int | None = None,
             weights: Path | None = None, gate_entry: dict[str, Any] | None = None,
             metrics: dict[str, Any] | None = None, architecture: str = "unet") -> VersionRecord:
    """Register a model version; without `weights` a placeholder file stands in for it
    (the registry only requires that the file exists)."""
    if weights is None:
        weights = ws.model_dir(version) / steps.WEIGHTS_FILE[architecture]
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"placeholder weights")
    rec = VersionRecord(version=version, dataset_version=version, architecture=architecture,
                        train_images=10 + version, train_photographs=8 + version,
                        metrics=metrics or {}, weights=str(weights), parent=parent,
                        gate=gate_entry or {"promoted": current, "reason": "registered by the test"})
    registry.add_version(ws, rec, make_current=current)
    return registry.get_version(ws, version)


def write_per_image(path: Path, dice: dict[str, float], negatives: int = 2) -> None:
    """A per-image CSV in the format of selftrain.evaluate (ulcer images, then ulcer-free)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"name": n, "source": "src", "gt_positive": True, "pred_positive": d > 0, "dice": d,
             "iou": d / (2 - d), "gt_stage": 2, "pred_stage": 2, "gt_area": 0.1, "pred_area": 0.1}
            for n, d in dice.items()]
    rows += [{"name": f"neg{i}", "source": "neg", "gt_positive": False, "pred_positive": False,
              "dice": 1.0, "iou": 1.0, "gt_stage": 0, "pred_stage": 0, "gt_area": 0.0,
              "pred_area": 0.0} for i in range(negatives)]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def store_evaluation(ws: Workspace, version: int, dice: dict[str, float]) -> str:
    """models/v{N}/evaluation.json with its per-image test CSV; returns the CSV path
    relative to models/v{N} (as steps.evaluate_version stores it)."""
    rel = f"eval/v{version}_test_per_image.csv"
    write_per_image(ws.model_dir(version) / rel, dice)
    (ws.model_dir(version) / "evaluation.json").write_text(json.dumps(
        {"threshold": 0.5, "per_image_csv": {"test": rel}, "test": {"dice_mean": float(np.mean(list(dice.values())))}}))
    return rel


def base_dice(n: int = 40, seed: int = 7) -> np.ndarray:
    """Per-image test Dice of the current model (uniform in [0.3, 0.8])."""
    return np.random.default_rng(seed).uniform(0.3, 0.8, n)


def alternating(n: int, amplitude: float) -> np.ndarray:
    """+a, -a, +a, ...: paired noise with mean exactly 0 (n even)."""
    return amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


def ok_step(summary: str) -> Callable[[CycleContext], StepResult]:
    def run(_ctx: CycleContext) -> StepResult:
        return StepResult(True, summary, {"answer": 42})
    return run


def make_steps(**overrides: Callable[[CycleContext], StepResult]) -> list[tuple[str, Any]]:
    """The ten steps as fakes that succeed, with some replaced."""
    return [(name, overrides.get(name, ok_step(f"{name} ok"))) for name in STEP_NAMES]


def drop_batch(ws: Workspace, name: str = "batch1") -> Path:
    """A batch folder waiting in inbox/ (the fake steps do not read it)."""
    folder = ws.inbox / name / "images"
    folder.mkdir(parents=True)
    (folder / "a.jpg").write_bytes(b"not read by the fake steps")
    return folder.parent


def read_events(ws: Workspace) -> list[dict[str, Any]]:
    return [json.loads(line) for line in ws.events_path.read_text(encoding="utf-8").splitlines()]


def assert_event_format(event: dict[str, Any]) -> None:
    """SPEC section 4: one JSON object per line with exactly these fields."""
    assert set(event) == EVENT_KEYS
    assert datetime.fromisoformat(event["ts"]).tzinfo is not None
    assert isinstance(event["cycle"], int) and event["cycle"] >= 1
    assert event["step"] in STEP_NAMES
    assert event["status"] in STEP_STATUSES
    assert isinstance(event["summary"], str) and "\n" not in event["summary"]
    assert isinstance(event["data"], dict)


# ============================================================================ gate rule

def test_gate_promotes_a_clearly_better_candidate() -> None:
    current = base_dice()
    candidate = current + 0.08 + alternating(current.size, 0.02)
    decision = gate.decide(candidate, current, margin=MARGIN, current_version=3)
    assert decision.promoted
    assert decision.mean_diff == pytest.approx(0.08)
    assert 0 < decision.ci95[0] < decision.mean_diff < decision.ci95[1]
    assert (decision.n, decision.compared_to, decision.margin) == (40, 3, MARGIN)
    assert decision.reason.startswith("promoted")


def test_gate_keeps_the_current_model_when_the_candidate_is_worse() -> None:
    current = base_dice()
    candidate = current - 0.08 + alternating(current.size, 0.02)
    decision = gate.decide(candidate, current, margin=MARGIN, current_version=3)
    assert not decision.promoted
    assert decision.mean_diff == pytest.approx(-0.08) and decision.ci95[1] < 0
    assert "no improvement on average" in decision.reason


def test_an_exact_tie_keeps_the_current_model() -> None:
    """mean_diff > 0 is required: equal Dice on every image is no reason to switch."""
    current = base_dice()
    decision = gate.decide(current.copy(), current, margin=MARGIN)
    assert decision.mean_diff == 0.0 and decision.ci95 == (0.0, 0.0)
    assert not decision.promoted


def test_a_small_gain_within_the_margin_is_promoted() -> None:
    """Better on average, not significantly (the CI covers 0), but the lower bound stays
    above -margin: the rule of SPEC section 3 step 8 promotes."""
    current = base_dice()
    candidate = current + 0.004 + alternating(current.size, 0.03)
    decision = gate.decide(candidate, current, margin=MARGIN)
    assert decision.mean_diff == pytest.approx(0.004)
    assert -MARGIN < decision.ci95[0] < 0 < decision.ci95[1]
    assert decision.promoted


def test_a_gain_that_may_hide_a_loss_beyond_the_margin_is_not_promoted() -> None:
    """Same mean gain, ten times noisier: the lower bound falls below -margin."""
    current = base_dice()
    candidate = current + 0.004 + alternating(current.size, 0.15)
    decision = gate.decide(candidate, current, margin=MARGIN)
    assert decision.mean_diff > 0 and decision.ci95[0] < -MARGIN
    assert not decision.promoted
    assert "worse by more than the margin" in decision.reason
    wide = gate.decide(candidate, current, margin=0.1)        # the margin decides
    assert wide.promoted and wide.ci95 == decision.ci95


def test_without_current_model_the_candidate_is_promoted() -> None:
    decision = gate.decide([0.5, 0.7, 0.2], None)
    assert decision.promoted and decision.n == 3
    assert decision.mean_diff is None and decision.ci95 is None and decision.compared_to is None
    assert "no current model" in decision.reason


def test_without_paired_images_the_candidate_is_not_promoted() -> None:
    decision = gate.decide([], [], current_version=0)
    assert not decision.promoted and decision.n == 0 and decision.compared_to == 0


def test_gate_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="margin"):
        gate.decide([0.5], [0.5], margin=-0.01)
    with pytest.raises(ValueError, match="shape"):
        gate.decide([0.5, 0.6], [0.5])
    with pytest.raises(ValueError, match="finite"):
        gate.decide([0.5, float("nan")], [0.5, 0.6])


def test_gate_decision_is_reproducible_and_json_ready() -> None:
    current = base_dice()
    candidate = current + 0.004 + alternating(current.size, 0.03)
    first = gate.decide(candidate, current, seed=5, current_version=1)
    assert gate.decide(candidate, current, seed=5, current_version=1) == first
    stored = json.loads(json.dumps(first.to_dict(), allow_nan=False))
    assert stored == {"promoted": True, "mean_diff": first.mean_diff, "ci95": list(first.ci95),
                      "p_value": first.p_value, "n": 40, "margin": MARGIN, "compared_to": 1,
                      "reason": first.reason}


def test_pairing_by_name_and_loading_per_image_dice(tmp_path: Path) -> None:
    csv_path = tmp_path / "v1_test_per_image.csv"
    write_per_image(csv_path, {"b": 0.5, "a": 0.25}, negatives=3)
    assert gate.load_test_dice(csv_path) == {"b": 0.5, "a": 0.25}            # ulcer images only
    assert len(gate.load_test_dice(csv_path, positives_only=False)) == 5
    names, cand, cur = gate.pair_by_name({"b": 0.5, "a": 0.25}, {"a": 0.2, "b": 0.6})
    assert names == ["a", "b"] and cand.tolist() == [0.25, 0.5] and cur.tolist() == [0.2, 0.6]
    with pytest.raises(ValueError, match="test images differ"):
        gate.pair_by_name({"a": 0.5}, {"b": 0.5})


# ============================================================================ gate step, rollback

def gate_context(ws: Workspace, candidate: dict[str, float]) -> CycleContext:
    """Context after the evaluate step of model v1 (its per-image test results stored)."""
    rel = store_evaluation(ws, 1, candidate)
    weights = ws.model_dir(1) / "best.pt"
    weights.write_bytes(b"placeholder weights")
    return CycleContext(ws=ws, cfg=ws.load_config(), cycle=1, dataset_version=1, model_version=1,
                        metrics={"test_dice_mean": float(np.mean(list(candidate.values())))},
                        artifacts={"evaluation": {"per_image_csv": {"test": rel}},
                                   "weights": weights, "train_images": 12, "train_photographs": 9})


@pytest.mark.parametrize("delta, promoted", [(0.1, True), (-0.1, False)])
def test_gate_step_registers_the_candidate_and_promotes_only_a_better_one(
        ws: Workspace, delta: float, promoted: bool) -> None:
    current = dict(zip([f"img{i:02d}" for i in range(30)], base_dice(30)))
    register(ws, 0, current=True)
    store_evaluation(ws, 0, current)
    ctx = gate_context(ws, {name: d + delta for name, d in current.items()})
    result = steps.step_gate(ctx)
    assert result.ok and result.data["promoted"] is promoted
    assert result.data["compared_to"] == 0 and result.data["n"] == 30
    assert result.data["mean_diff"] == pytest.approx(delta)
    rec = registry.get_version(ws, 1)                       # registered in any case
    assert rec.parent == 0 and rec.promoted is promoted and rec.weights == "models/v1/best.pt"
    assert registry.current(ws).version == (1 if promoted else 0)
    assert result.summary.startswith("v1 promoted" if promoted else "v0 kept")
    assert ctx.artifacts["gate"] == result.data


def test_gate_step_promotes_the_first_model(ws: Workspace) -> None:
    result = steps.step_gate(gate_context(ws, {"a": 0.4, "b": 0.6}))
    assert result.data["promoted"] is True and result.data["mean_diff"] is None
    assert registry.current(ws).version == 1


def test_gate_step_is_skipped_without_a_new_model(ws: Workspace) -> None:
    result = steps.step_gate(CycleContext(ws=ws, cfg=ws.load_config(), cycle=1))
    assert result.skipped and result.status == "skipped"


def test_rollback_restores_an_earlier_version(ws: Workspace) -> None:
    register(ws, 0, current=True)
    register(ws, 1, current=True, gate_entry={"promoted": True, "reason": "better"})
    rec = gate.rollback(ws, 0)
    assert rec.version == 0 and registry.current(ws).version == 0
    last = registry.load(ws)["history"][-1]
    assert (last["action"], last["version"], last["previous"]) == ("rollback", 0, 1)
    assert registry.get_version(ws, 1).gate["promoted"] is True     # decisions are kept
    assert json.loads(ws.status_path.read_text())["current_model"] == 0
    gate.rollback(ws, 1)                                             # any version, also forward
    assert registry.current(ws).version == 1
    with pytest.raises(KeyError):
        gate.rollback(ws, 7)


def test_rollback_is_refused_while_a_cycle_runs(ws: Workspace) -> None:
    register(ws, 0, current=True)
    register(ws, 1, current=True)
    with CycleLock(lock_path(ws)):
        with pytest.raises(CycleLockedError):
            gate.rollback(ws, 0)
    assert registry.current(ws).version == 1


# ============================================================================ cycle runner

def test_an_empty_inbox_starts_no_cycle(ws: Workspace) -> None:
    calls: list[int] = []

    def ingest(ctx: CycleContext) -> StepResult:
        calls.append(ctx.cycle)
        return StepResult(True, "read")

    result = run_cycle(ws, steps=make_steps(ingest=ingest))
    assert result["status"] == "idle" and result["cycle"] is None and calls == []
    assert not ws.events_path.exists()
    status = json.loads(ws.status_path.read_text())
    assert status["cycle"] == 0 and status["running"] is False


def test_a_successful_cycle_writes_events_and_status(ws: Workspace) -> None:
    drop_batch(ws)
    seen: list[tuple[str, str]] = []
    result = run_cycle(ws, demo=True, emit=lambda step, status, message, **data: seen.append((step, status)),
                       steps=make_steps())
    assert result["status"] == "done" and result["cycle"] == 1 and result["failed_step"] is None
    events = read_events(ws)
    for event in events:
        assert_event_format(event)
    assert [(e["step"], e["status"]) for e in events] == \
        [(name, status) for name in STEP_NAMES for status in ("started", "done")]
    assert seen == [(e["step"], e["status"]) for e in events]       # the listener saw every event
    assert events[1]["data"]["answer"] == 42 and "seconds" in events[1]["data"]
    status = json.loads(ws.status_path.read_text())
    assert status["cycle"] == 1 and status["running"] is False and status["result"] == "done"
    assert status["demo"] is True and list(status["steps"]) == STEP_NAMES
    assert all(s == {"status": "done", "summary": f"{n} ok"} for n, s in status["steps"].items())
    assert status["current_model"] is None and status["versions"] == []
    assert run_cycle(ws, steps=make_steps())["cycle"] == 2         # numbers count up


def test_event_data_may_use_the_names_of_the_event_fields(ws: Workspace,
                                                          capsys: pytest.CaptureFixture[str],
                                                          caplog: pytest.LogCaptureFixture) -> None:
    """Regression: the drift step's data carries the drift report's own "status"; the CLI
    listener print_event once failed on it ("got multiple values for argument 'status'")
    and the drift result line was missing from the output of `run`."""
    drop_batch(ws)

    def drift(ctx: CycleContext) -> StepResult:
        ctx.emit("drift", "started", "progress", status="tested", step="x", message="y")
        return StepResult(True, "no drift", {"status": "tested", "step": "x", "message": "y"})

    result = run_cycle(ws, emit=print_event, steps=make_steps(drift=drift))
    assert result["status"] == "done"
    assert "event listener failed" not in caplog.text
    lines = capsys.readouterr().out.splitlines()
    assert any(line.endswith("drift    done     no drift") for line in lines)
    assert sum(" drift " in line for line in lines) == 3            # started, progress, done
    final = [e for e in read_events(ws) if e["step"] == "drift"][-1]
    assert (final["status"], final["data"]["status"], final["data"]["step"]) == ("done", "tested", "x")


def test_status_is_written_when_a_step_fails(ws: Workspace) -> None:
    drop_batch(ws)
    result = run_cycle(ws, steps=make_steps(dedupe=lambda ctx: StepResult(False, "matching failed")))
    assert (result["status"], result["failed_step"]) == ("failed", "dedupe")
    status = json.loads(ws.status_path.read_text())
    assert status["running"] is False and status["result"] == "failed"
    assert status["failed_step"] == "dedupe" and status["finished"] is not None
    states = {name: s["status"] for name, s in status["steps"].items()}
    assert states == {**{n: "done" for n in ("ingest", "quality")}, "dedupe": "failed",
                      **{n: "skipped" for n in STEP_NAMES[3:]}}
    assert status["steps"]["dedupe"]["summary"] == "matching failed"
    events = read_events(ws)
    for event in events:
        assert_event_format(event)
    assert [e["step"] for e in events if e["status"] == "skipped"] == STEP_NAMES[3:]


def test_a_raising_step_fails_the_cycle_with_its_traceback(ws: Workspace) -> None:
    def broken(_ctx: CycleContext) -> StepResult:
        raise RuntimeError("disk full\nsecond line")

    drop_batch(ws)
    result = run_cycle(ws, steps=make_steps(train=broken))
    assert (result["status"], result["failed_step"]) == ("failed", "train")
    failed = [e for e in read_events(ws) if e["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["step"] == "train"
    assert failed[0]["summary"] == "RuntimeError: disk full"                # one line
    assert "Traceback" in failed[0]["data"]["traceback"]
    status = read_status(ws)
    assert status["result"] == "failed" and status["steps"]["evaluate"]["status"] == "skipped"


def test_an_interrupted_cycle_records_its_state_and_propagates(ws: Workspace) -> None:
    def interrupted(_ctx: CycleContext) -> StepResult:
        raise KeyboardInterrupt

    drop_batch(ws)
    with pytest.raises(KeyboardInterrupt):
        run_cycle(ws, steps=make_steps(drift=interrupted))
    status = json.loads(ws.status_path.read_text())
    assert status["running"] is False and status["result"] == "interrupted"
    assert status["steps"]["drift"] == {"status": "failed", "summary": "interrupted (KeyboardInterrupt)"}
    assert status["steps"]["notify"]["status"] == "skipped"
    with CycleLock(lock_path(ws)):                              # the lock was released
        pass


def test_steps_must_be_the_ten_spec_steps_in_order(ws: Workspace) -> None:
    shuffled = make_steps()
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    with pytest.raises(ValueError, match="in this order"):
        run_cycle(ws, steps=shuffled)
    with pytest.raises(ValueError):
        run_cycle(ws, steps=make_steps()[:-1])
    assert [name for name, _ in steps.STEPS] == STEP_NAMES


def test_the_lock_prevents_a_second_concurrent_cycle(ws: Workspace) -> None:
    drop_batch(ws)
    entered, release = threading.Event(), threading.Event()
    outcome: dict[str, Any] = {}

    def slow_ingest(_ctx: CycleContext) -> StepResult:
        entered.set()
        assert release.wait(30), "the test did not release the first cycle"
        return StepResult(True, "slow ingest done")

    def first_cycle() -> None:
        outcome["result"] = run_cycle(ws, steps=make_steps(ingest=slow_ingest))

    thread = threading.Thread(target=first_cycle)
    thread.start()
    try:
        assert entered.wait(30)
        owner = lock_owner(ws)
        assert owner is not None and owner["pid"] == os.getpid()
        assert read_status(ws)["running"] is True
        with pytest.raises(CycleLockedError) as info:
            run_cycle(ws, steps=make_steps())
        assert info.value.owner["pid"] == owner["pid"] and "another cycle" in str(info.value)
    finally:
        release.set()
        thread.join(30)
    assert outcome["result"]["status"] == "done" and outcome["result"]["cycle"] == 1
    assert {e["cycle"] for e in read_events(ws)} == {1}            # the refused cycle wrote nothing
    assert lock_owner(ws) is None
    assert run_cycle(ws, steps=make_steps())["cycle"] == 2


def test_the_lock_holds_across_processes_and_dies_with_its_owner(ws: Workspace) -> None:
    code = textwrap.dedent("""
        import sys, time
        from pathlib import Path
        from selftrain.system.cycle import CycleLock
        lock = CycleLock(Path(sys.argv[1]))     # referenced: closing the file would unlock
        lock.acquire()
        print("locked", flush=True)
        time.sleep(120)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code, str(lock_path(ws))], cwd=ROOT,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
        drop_batch(ws)
        with pytest.raises(CycleLockedError) as info:
            run_cycle(ws, steps=make_steps())
        assert info.value.owner["pid"] == proc.pid and lock_owner(ws)["pid"] == proc.pid
    finally:
        proc.kill()
        proc.wait(30)
    assert lock_owner(ws) is None               # the record of a dead process is ignored
    assert run_cycle(ws, steps=make_steps())["status"] == "done"


def test_a_crashed_cycle_is_reported_as_interrupted(ws: Workspace) -> None:
    ws.status_path.write_text(json.dumps({**cycle.idle_status(4), "running": True, "result": "running"}))
    status = read_status(ws)
    assert status["running"] is False and status["result"] == "interrupted" and status["cycle"] == 4


def test_watch_waits_until_the_inbox_is_unchanged(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple] = []

    def fake_run_cycle(ws_: Workspace, demo: bool = False, emit: Any = None) -> dict[str, Any]:
        started.append(inbox_signature(ws_))
        return {"cycle": len(started), "status": "done"}

    monkeypatch.setattr(cycle, "run_cycle", fake_run_cycle)
    batch = drop_batch(ws)
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 1:                    # a second file arrives while copying
            (batch / "images" / "b.jpg").write_bytes(b"later")

    results = cycle.watch(ws, interval=5, max_polls=4, sleep=fake_sleep)
    assert [r["cycle"] for r in results] == [1]                 # poll 3: unchanged since poll 2
    assert len(started[0]) == 2 and sleeps == [5, 5, 5]
    with pytest.raises(ValueError):
        cycle.watch(ws, interval=-1, max_polls=1)


# ============================================================================ real steps 2-6

@dataclass
class Rec:
    """Stand-in for selftrain.system.ingest.Record (SPEC section 2)."""
    path: Path
    polygons: list[tuple[int, np.ndarray]] = field(default_factory=list)
    source: str = "woundcare_new"
    license: str = "CC BY 4.0"
    batch: str = "batch1"


def read_label(path: Path) -> list[tuple[int, np.ndarray]]:
    polys = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 7 and int(parts[0]) in WOUNDCARE_STAGES:
            polys.append((WOUNDCARE_STAGES[int(parts[0])],
                          np.array(parts[1:], np.float32).reshape(-1, 2)))
    return polys


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory) -> CycleContext:
    """A workspace whose pool is data/seg_smoke, after steps 2-5 of cycle 1 on a batch of
    eight Roboflow woundcare photos plus two defective records."""
    if not ((SMOKE / "manifest.csv").is_file() and (RAW / "images").is_dir()):
        pytest.skip("data/seg_smoke or data/raw/roboflow_woundcare missing")
    tmp = tmp_path_factory.mktemp("cycle")
    ws = Workspace(tmp / "ws")
    dataset.init_from_seg(ws, SMOKE, log=quiet)
    ws.ensure()
    folder = tmp / "batch"
    folder.mkdir()
    records = []
    for stem in sorted(p.stem for p in (RAW / "images").glob("*.jpg"))[:8]:
        shutil.copyfile(RAW / "images" / f"{stem}.jpg", folder / f"{stem}.jpg")
        records.append(Rec(folder / f"{stem}.jpg", read_label(RAW / "labels" / f"{stem}.txt")))
    cv2.imwrite(str(folder / "tiny.jpg"), np.full((40, 30, 3), 128, np.uint8))
    records.append(Rec(folder / "tiny.jpg"))                       # ulcer-free, 40 x 30 px
    shutil.copyfile(records[0].path, folder / "whole_frame.jpg")
    whole = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    records.append(Rec(folder / "whole_frame.jpg", [(1, whole)]))  # label = whole photograph
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1, batches=["batch1"], records=records)
    for name in ("quality", "dedupe", "drift", "dataset"):
        ctx.results[name] = getattr(steps, f"step_{name}")(ctx)
    return ctx


@requires_raw
def test_quality_step_rejects_defective_records_and_writes_reports(processed: CycleContext) -> None:
    result = processed.results["quality"]
    assert result.ok and (result.data["checked"], result.data["accepted"]) == (10, 8)
    reasons = {Path(r["file"]).name: r["reason"] for r in result.data["rejected"]}
    assert reasons["tiny.jpg"].startswith("image_too_small:")
    assert reasons["whole_frame.jpg"].startswith("whole_frame_label:")
    report_file = processed.ws.processed / "batch1" / "quality_report.json"
    saved = json.loads(report_file.read_text())
    assert (saved["images"], saved["accepted"], saved["rejected"]) == (10, 8, 2)
    assert saved["rejected_by_code"] == {"image_too_small": 1, "whole_frame_label": 1}


@requires_raw
def test_dedupe_drift_and_dataset_steps_on_the_smoke_pool(processed: CycleContext) -> None:
    dedupe_result, drift_result, data_result = (processed.results[n] for n in ("dedupe", "drift", "dataset"))
    assert dedupe_result.ok and processed.groups is not None
    n_new = len(processed.groups.new_groups)
    assert 0 < n_new <= 8
    assert drift_result.status in ("done", "skipped")
    if drift_result.status == "done":
        assert drift_result.data["photographs_tested"] == n_new
        assert set(drift_result.data["features"]) == {"lightness", "red_green", "yellow_blue",
                                                      "colourfulness", "sharpness", "log_resolution"}
        assert processed.artifacts["drift"]["n_new"] == n_new
    assert data_result.ok and processed.dataset_version == 1 == data_result.data["version"]
    manifest = processed.ws.read_manifest()
    added = manifest[manifest["added_cycle"] == 1]
    assert len(added) == len(processed.artifacts["added"].rows) and set(added["split"]) <= {"train", "val"}
    assert (processed.ws.dataset_dir(1) / "dataset.yaml").is_file()
    smoke_test = pd.read_csv(SMOKE / "manifest.csv").query("split == 'test'")["name"]
    assert set(manifest.query("split == 'test'")["name"]) == set(smoke_test)     # frozen


@requires_raw
def test_cycle_summary_carries_the_report_context(processed: CycleContext) -> None:
    summary = steps.cycle_summary(processed)
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    context = summary["report_context"]
    assert context["received"] == {"woundcare_new": 10}
    assert sorted(r["name"] for r in context["rejected"]) == ["tiny.jpg", "whole_frame.jpg"]
    assert all(not d["image"].startswith("record ") for d in context["duplicates"])
    assert list(context["steps"]) == ["quality", "dedupe", "drift", "dataset"]
    assert context["dataset_version"] == 1 and context["model_version"] is None


@requires_raw
def test_train_step_one_epoch_on_eight_images(processed: CycleContext) -> None:
    cfg = dataclasses.replace(processed.cfg, demo_epochs=1, demo_max_train_images=8)
    ctx = dataclasses.replace(processed, cfg=cfg, demo=True, artifacts=dict(processed.artifacts),
                              results=dict(processed.results))
    result = steps.step_train(ctx)
    assert result.ok, result.summary
    assert (result.data["train_images"], result.data["epochs_run"], result.data["max_epochs"]) == (8, 1, 1)
    weights = ctx.ws.model_dir(1) / "best.pt"
    assert weights.is_file() and ctx.artifacts["weights"] == weights and ctx.model_version == 1
    assert 1 <= ctx.artifacts["train_photographs"] <= 8
    assert 0.0 <= result.data["val_dice"] <= 1.0
    assert (ctx.ws.model_dir(1) / "train" / "summary.json").is_file()


def test_training_steps_are_skipped_without_new_data(ws: Workspace) -> None:
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1)
    for step in (steps.step_quality, steps.step_dedupe, steps.step_drift, steps.step_dataset,
                 steps.step_train, steps.step_evaluate, steps.step_report):
        assert step(ctx).skipped


# ============================================================================ notify

class FakeTelegram:
    """Records what scripts/notify_telegram.py would send."""

    def __init__(self, configured: bool = True) -> None:
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.configured = configured

    def send_message(self, text: str, silent: bool = True) -> None:
        self._check()
        self.calls.append(("message", text, {"silent": silent}))

    def send_document(self, path: str, caption: str = "", silent: bool = True) -> None:
        self._check()
        self.calls.append(("document", path, {"caption": caption, "silent": silent}))

    def _check(self) -> None:
        if not self.configured:
            raise SystemExit("TELEGRAM_BOT_TOKEN missing in .env")


def notify_context(ws: Workspace, pdf: Path | None) -> CycleContext:
    ctx = CycleContext(ws=ws, cfg=dataclasses.replace(ws.load_config(), telegram=True), cycle=3)
    ctx.results["ingest"] = StepResult(True, "4 labelled images from 1 batch(es)")
    ctx.results["gate"] = StepResult(True, "v3 promoted")
    if pdf is not None:
        ctx.artifacts["report_pdf"] = pdf
    return ctx


def test_notify_sends_the_pdf_silently(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeTelegram()
    monkeypatch.setattr(steps, "load_notifier", lambda: fake)
    pdf = ws.reports / "cycle_3.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.5 test")
    result = steps.step_notify(notify_context(ws, pdf))
    assert result.ok and result.summary == "report cycle_3.pdf sent to Telegram"
    (kind, path, options), = fake.calls
    assert (kind, path, options["silent"]) == ("document", str(pdf), True)
    assert options["caption"].splitlines() == ["Pressure-ulcer system: cycle 3",
                                               "ingest: 4 labelled images from 1 batch(es)",
                                               "gate: v3 promoted"]


def test_notify_without_report_sends_the_summary(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeTelegram()
    monkeypatch.setattr(steps, "load_notifier", lambda: fake)
    assert steps.step_notify(notify_context(ws, None)).summary == "summary sent to Telegram"
    assert fake.calls[0][0] == "message" and fake.calls[0][2] == {"silent": True}


def test_notify_fails_when_telegram_is_not_configured(ws: Workspace,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steps, "load_notifier", lambda: FakeTelegram(configured=False))
    result = steps.step_notify(notify_context(ws, None))
    assert not result.ok and "not configured" in result.summary


def test_notify_is_skipped_when_disabled(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(steps, "load_notifier", lambda: pytest.fail("Telegram must not be loaded"))
    ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=1)
    assert ctx.cfg.telegram is False and steps.step_notify(ctx).skipped


def test_the_notifier_script_exposes_the_used_functions() -> None:
    notifier = steps.load_notifier()            # loading only: nothing is sent
    assert callable(notifier.send_message) and callable(notifier.send_document)


# ============================================================================ active learning

def test_uncertainty_score_follows_the_method() -> None:
    assert active.uncertainty_score(np.full((4, 4), 0.5), np.zeros((4, 4), bool)) == 1.0
    assert active.uncertainty_score(np.full((4, 4), 0.05), np.zeros((4, 4), bool)) == 0.0   # R empty
    p = np.zeros((2, 2))
    p[0, 0] = 0.25                                   # candidate pixel: u = 0.5
    predicted = np.zeros((2, 2), bool)
    predicted[1, 1] = True                           # predicted pixel with p = 0: u = 0
    assert active.uncertainty_score(p, predicted) == pytest.approx(0.25)


def test_ranking_without_photos_needs_no_model(ws: Workspace) -> None:
    ranking = active.rank_unlabeled(ws)
    assert ranking.empty and list(ranking.columns) == active.RANKING_COLUMNS
    assert (ws.active / "ranking.csv").is_file()


def test_every_image_type_ingest_files_as_unlabeled_is_ranked(ws: Workspace) -> None:
    """Regression: active learning listed only .jpg/.jpeg/.png/.bmp, so unlabeled .tif and
    .webp photos that ingest moved to inbox/unlabeled/ were never offered for labelling."""
    from selftrain.system import ingest
    folder = ws.unlabeled / "batch1" / "images"
    folder.mkdir(parents=True)
    names = sorted(f"photo{suffix}" for suffix in ingest.IMAGE_SUFFIXES)
    for name in names:
        cv2.imwrite(str(folder / name), np.full((64, 64, 3), 128, np.uint8))
    (folder / "notes.txt").write_text("not an image")
    (folder / ".hidden.jpg").write_bytes(b"")
    assert [p.name for p in active.list_unlabeled(ws)] == names
    assert all(cv2.imread(str(p)) is not None for p in active.list_unlabeled(ws))


def test_ranking_photos_without_a_current_model_fails(ws: Workspace) -> None:
    ws.unlabeled.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ws.unlabeled / "a.jpg"), np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(active.NoCurrentModelError):
        active.rank_unlabeled(ws)


@requires_smoke_unet
def test_rank_unlabeled_sorts_three_smoke_photos_by_uncertainty(ws: Workspace) -> None:
    register(ws, 0, current=True, weights=UNET_WEIGHTS)
    names = sorted((SMOKE / "images" / "test").glob("*.jpg"))[:3]
    ws.unlabeled.mkdir(parents=True, exist_ok=True)
    for path in names:
        shutil.copyfile(path, ws.unlabeled / path.name)
    ranking = active.rank_unlabeled(ws, top_k=2)
    assert len(ranking) == 3 and list(ranking.columns) == active.RANKING_COLUMNS
    u = ranking["uncertainty"].tolist()
    assert u == sorted(u, reverse=True) and all(0.0 <= x <= 1.0 for x in u)
    assert ranking["predicted_area"].between(0, 1).all() and ranking["predicted_stage"].isin(range(5)).all()
    assert {Path(p).name for p in ranking["path"]} == {p.name for p in names}
    assert len(list((ws.active / "thumbnails").glob("*.jpg"))) == 2
    assert [bool(t) for t in ranking["thumbnail"]] == [True, True, False]      # top_k = 2
    assert all((ws.active / t).is_file() for t in ranking["thumbnail"][:2])
    saved = pd.read_csv(ws.active / "ranking.csv", keep_default_na=False)
    assert saved["path"].tolist() == ranking["path"].tolist()
    assert np.allclose(saved["uncertainty"], ranking["uncertainty"])
    # the score of the top photo, recomputed with the documented method
    from selftrain.unet import UNetPredictor
    top = cv2.imread(ranking["path"].iloc[0])
    sem, p = active.predict(UNetPredictor(UNET_WEIGHTS), top, steps.DEFAULT_THRESHOLD["unet"],
                            ws.load_config().min_component_frac)
    assert active.uncertainty_score(p, sem > 0) == pytest.approx(u[0], abs=1e-6)


# ============================================================================ CLI

def test_the_cli_offers_the_spec_subcommands() -> None:
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert sorted(sub.choices) == sorted(SUBCOMMANDS)


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_every_subcommand_has_help_and_a_workspace_option(command: str,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main([command, "--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert f"python -m selftrain {command}" in out and "--workspace PATH" in out


def test_module_entry_point_prints_help() -> None:
    done = subprocess.run([sys.executable, "-m", "selftrain", "--help"], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert all(command in done.stdout for command in SUBCOMMANDS)


def test_cli_on_an_empty_workspace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = str(tmp_path / "ws")
    assert main(["status", "--workspace", root]) == 0
    out = capsys.readouterr().out
    assert "no cycle has run yet" in out and "current model: none" in out
    assert main(["status", "--json", "--workspace", root]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["running"] is False and status["current_model"] is None
    assert main(["run", "--workspace", root]) == 0
    assert "inbox is empty" in capsys.readouterr().out
    assert main(["rank-unlabeled", "--workspace", root]) == 0
    assert "no photos" in capsys.readouterr().out
    assert main(["rollback", "--to", "3", "--workspace", root]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["report", "--version", "3", "--workspace", root]) == 1


def test_cli_status_lists_versions_with_their_metrics(ws: Workspace,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    register(ws, 0, current=True, metrics={"test_dice_mean": 0.6123, "test_dice_ci95": [0.55, 0.67],
                                           "test_fpr": 0.25})
    assert main(["status", "--workspace", str(ws.root)]) == 0
    out = capsys.readouterr().out
    assert "current model: v0" in out
    row = next(line for line in out.splitlines() if "unet" in line)
    assert "0.612 [0.550, 0.670]" in row and "0.25" in row and row.lstrip().startswith("*")


def test_cli_rollback_and_run_respect_the_cycle_lock(ws: Workspace,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    register(ws, 0, current=True)
    register(ws, 1, current=True)
    assert main(["rollback", "--to", "0", "--workspace", str(ws.root)]) == 0
    assert "current model: v0" in capsys.readouterr().out
    drop_batch(ws)
    with CycleLock(lock_path(ws)):
        assert main(["run", "--workspace", str(ws.root)]) == 1
        assert "another cycle is running" in capsys.readouterr().err


def test_cli_reproduce_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["reproduce"]) == 0
    out = capsys.readouterr().out
    assert "thesis pipeline" in out and "dry run" in out
    assert all(step.script in out for step in THESIS_PIPELINE)


def test_cli_prints_cycle_results(capsys: pytest.CaptureFixture[str]) -> None:
    print_cycle_result({"cycle": 2, "status": "failed", "failed_step": "train", "error": None,
                        "gate": None})
    print_cycle_result({"cycle": 3, "status": "done", "failed_step": None, "error": None,
                        "gate": {"reason": "promoted: better"}, "report_pdf": "reports/cycle_3.pdf"})
    assert capsys.readouterr().out.splitlines() == [
        "cycle 2: failed at step train", "cycle 3: done", "gate: promoted: better",
        "report: reports/cycle_3.pdf"]


def test_refresh_status_mirrors_the_registry(ws: Workspace) -> None:
    register(ws, 0, current=True)
    status = cycle.refresh_status(ws)
    assert status["current_model"] == 0 and [v["version"] for v in status["versions"]] == [0]
    assert read_status(ws)["current_model"] == 0 and status["result"] == "idle"


def test_reproduce_lists_only_existing_scripts_in_order() -> None:
    from selftrain.__main__ import ROOT
    scripts = [step.script for step in THESIS_PIPELINE]
    assert [s for s in scripts if not (ROOT / s).is_file()] == []
    order = ["scripts/build_seg_dataset.py", "scripts/train_model.py", "scripts/compare_models.py",
             "scripts/external_piid.py", "scripts/annotator_agreement.py",
             "scripts/data_growth.sh", "scripts/learning_curve.py",
             "scripts/leakage_sensitivity.py", "scripts/make_figures.py"]
    firsts = [scripts.index(s) for s in order]
    assert firsts == sorted(firsts)
    seeds = {step.args[step.args.index("--seed") + 1] for step in THESIS_PIPELINE
             if "--seed" in step.args}
    assert seeds == {"0", "1", "2"}
    assert sum(s == "scripts/compare_models.py" for s in scripts) == 3
