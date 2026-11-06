"""Tests of the web dashboard (SPEC section 8, feature F9) with FastAPI's TestClient.

Every test builds the application with `create_app` on a workspace under tmp_path. The
cycle function is injected (a slow fake that blocks until released, or run_cycle with fake
steps), so no model is trained; /api/predict runs the smoke U-Net on the CPU.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import threading
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from selftrain.system import STEP_NAMES, dashboard, registry
from selftrain.system.cycle import CycleLock, inbox_signature, lock_path, refresh_status, run_cycle
from selftrain.system.registry import VersionRecord
from selftrain.system.steps import StepResult
from selftrain.system.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "seg_smoke"
UNET_WEIGHTS = ROOT / "runs" / "smoke" / "unet" / "best.pt"
YOLO_LABEL = b"1 0.30 0.30 0.70 0.30 0.70 0.70 0.30 0.70\n"

requires_smoke_unet = pytest.mark.skipif(
    not (UNET_WEIGHTS.is_file() and (SMOKE / "images" / "test").is_dir()),
    reason="runs/smoke/unet/best.pt or data/seg_smoke missing")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Apple GPU belongs to the main experiments."""
    monkeypatch.setenv("SELFTRAIN_DEVICE", "cpu")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "ws").ensure()


# ============================================================================ helpers

class SlowCycle:
    """Injected cycle function that runs until the test releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[bool] = []

    def __call__(self, ws: Workspace, demo: bool = False) -> dict[str, Any]:
        self.calls.append(demo)
        self.started.set()
        if not self.release.wait(30):
            raise TimeoutError("the test did not release the cycle")
        return {"cycle": len(self.calls), "status": "done"}


def client_for(ws: Workspace, **options: Any) -> TestClient:
    return TestClient(dashboard.create_app(ws, poll_s=0.01, heartbeat_s=0.05, **options))


def jpeg(side: int = 96, value: int = 128) -> bytes:
    ok, data = cv2.imencode(".jpg", np.full((side, side, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def register(ws: Workspace, version: int, *, current: bool, weights: Path | None = None,
             metrics: dict[str, Any] | None = None, gate: dict[str, Any] | None = None) -> None:
    if weights is None:
        weights = ws.model_dir(version) / "best.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"placeholder weights")
    registry.add_version(ws, VersionRecord(
        version=version, dataset_version=version, architecture="unet", train_images=50 + version,
        train_photographs=40 + version, metrics=metrics or {}, weights=str(weights),
        gate=gate or {"promoted": current, "reason": "registered by the test"}), make_current=current)


def sse_messages(text: str) -> list[dict[str, str]]:
    """Fields of every SSE message (blocks separated by a blank line)."""
    messages = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines()
                      if line and not line.startswith(":"))
        if fields:
            messages.append(fields)
    return messages


def write_events(ws: Workspace, events: list[dict[str, Any]], tail: str = "") -> None:
    ws.events_path.write_text("".join(json.dumps(e) + "\n" for e in events) + tail, encoding="utf-8")


def spec_event(n: int, step: str = "ingest", status: str = "done") -> dict[str, Any]:
    return {"ts": f"2026-11-06T05:00:{n:02d}+00:00", "cycle": 1, "step": step, "status": status,
            "summary": f"event {n}", "data": {"n": n}}


# ============================================================================ page

def test_index_page_is_a_single_offline_page(ws: Workspace) -> None:
    response = client_for(ws).get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"
    html = response.text
    assert html == dashboard.INDEX_HTML.read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html          # no CDN, works offline
    for value in re.findall(r"""(?:src|href)\s*=\s*["']([^"']*)""", html):
        assert value.startswith(("data:", "/", "#")), value
    assert "<svg" in html and "<style>" in html and "<script>" in html
    assert all(f'"{name}"' in html for name in STEP_NAMES)          # the pipeline of ten steps
    for endpoint in ("/api/status", "/api/events", "/api/versions", "/api/upload", "/api/run",
                     "/api/predict"):
        assert endpoint in html


def test_interactive_docs_that_need_a_cdn_are_disabled(ws: Workspace) -> None:
    client = client_for(ws)
    assert client.get("/docs").status_code == 404 and client.get("/redoc").status_code == 404
    assert client.get("/api/openapi.json").status_code == 200


# ============================================================================ status, versions

def test_status_combines_cycle_model_inbox_and_latest_report(ws: Workspace) -> None:
    register(ws, 0, current=True, metrics={"test_dice_mean": 0.61, "test_dice_ci95": [0.55, 0.67],
                                           "test_fpr": 0.2})
    refresh_status(ws)
    (ws.inbox / "batch1" / "images").mkdir(parents=True)
    (ws.inbox / "batch1" / "images" / "a.jpg").write_bytes(jpeg())
    ws.unlabeled.mkdir(parents=True, exist_ok=True)
    (ws.unlabeled / "photo.jpg").write_bytes(jpeg())
    ws.reports.mkdir(parents=True, exist_ok=True)
    (ws.reports / "cycle_1.html").write_text("<html>1</html>")
    (ws.reports / "cycle_2.html").write_text("<html>2</html>")
    status = client_for(ws).get("/api/status").json()
    assert status["cycle"] == 0 and status["running"] is False
    assert list(status["steps"]) == STEP_NAMES
    assert status["current_model"] == 0 and [v["version"] for v in status["versions"]] == [0]
    assert status["runner"]["busy"] is False
    model = status["model"]
    assert (model["version"], model["dice_mean"], model["dice_ci95"], model["fpr"]) == \
        (0, 0.61, [0.55, 0.67], 0.2)
    assert status["inbox"] == {"pending": ["batch1"], "unlabeled": 1}
    assert status["latest_report"] == {"cycle": 2, "html": "/api/report/2", "pdf": None}
    assert status["workspace"] == str(ws.root)


def test_versions_show_test_dice_with_ci_and_the_gate_decision(ws: Workspace) -> None:
    register(ws, 0, current=True, metrics={"test_dice_mean": 0.6, "test_dice_ci95": [0.5, 0.7]})
    register(ws, 1, current=False, metrics={"test_dice_mean": 0.58, "test_dice_ci95": [0.49, 0.66]},
             gate={"promoted": False, "mean_diff": -0.02, "ci95": [-0.05, 0.01],
                   "reason": "not promoted: no improvement on average"})
    body = client_for(ws).get("/api/versions").json()
    assert body["current"] == 0 and [h["action"] for h in body["history"]] == ["promote"]
    v0, v1 = body["versions"]
    assert (v0["version"], v0["current"], v0["promoted"], v0["dice_ci95"]) == (0, True, True, [0.5, 0.7])
    assert (v1["version"], v1["current"], v1["promoted"], v1["parent"]) == (1, False, False, 0)
    assert v1["gate"] == {"promoted": False, "mean_diff": -0.02, "ci95": [-0.05, 0.01],
                          "reason": "not promoted: no improvement on average"}


def test_a_corrupt_registry_is_reported_not_hidden(ws: Workspace) -> None:
    ws.registry_path.write_text("{broken")
    client = client_for(ws)
    assert client.get("/api/versions").status_code == 500
    status = client.get("/api/status").json()
    assert status["model"] is None and "invalid registry" in status["registry_error"]


# ============================================================================ upload

def test_upload_of_photos_and_labels_creates_one_inbox_batch(ws: Workspace) -> None:
    files = [("files", ("wound1.jpg", jpeg(), "image/jpeg")),
             ("files", ("wound1.txt", YOLO_LABEL, "text/plain")),
             ("files", ("wound2.jpg", jpeg(value=90), "image/jpeg")),
             ("files", ("notes.exe", b"MZ", "application/octet-stream"))]
    response = client_for(ws).post("/api/upload", files=files)
    assert response.status_code == 201, response.text
    body = response.json()
    (batch,) = body["batches"]
    assert batch["kind"] == "folder" and (batch["images"], batch["labels"], batch["files"]) == (2, 1, 3)
    assert re.fullmatch(r"\d{8}T\d{6}Z_upload", batch["name"]) and batch["path"] == f"inbox/{batch['name']}"
    assert body["ignored"] == [{"name": "notes.exe", "reason": "unsupported file type '.exe'"}]
    folder = ws.inbox / batch["name"]
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()) == \
        ["images/wound1.jpg", "images/wound2.jpg", "labels/wound1.txt"]
    assert (folder / "labels" / "wound1.txt").read_bytes() == YOLO_LABEL
    assert not [p for p in ws.inbox.iterdir() if p.name.startswith(".upload-")]   # no leftovers
    assert {rel.split("/")[0] for rel, _, _ in inbox_signature(ws)} == {batch["name"]}


def test_upload_of_a_zip_batch(ws: Workspace) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/a.jpg", jpeg())
        archive.writestr("labels/a.txt", YOLO_LABEL)
        archive.writestr("source.yaml", "name: ward3\nlicense: CC BY 4.0\n")
    response = client_for(ws).post("/api/upload", files=[("files", ("ward3.zip", buffer.getvalue(),
                                                                    "application/zip"))])
    assert response.status_code == 201, response.text
    (batch,) = response.json()["batches"]
    assert batch["kind"] == "zip" and batch["name"].endswith("_ward3.zip")
    with zipfile.ZipFile(ws.inbox / batch["name"]) as archive:
        assert sorted(archive.namelist()) == ["images/a.jpg", "labels/a.txt", "source.yaml"]


def test_unsafe_or_useless_uploads_are_rejected_without_writing(ws: Workspace) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.jpg", jpeg())
    client = client_for(ws)
    assert client.post("/api/upload", files=[("f", ("evil.zip", buffer.getvalue(), "application/zip"))]
                       ).status_code == 400
    assert client.post("/api/upload", files=[("f", ("tool.exe", b"MZ", "application/octet-stream"))]
                       ).status_code == 400
    assert client.post("/api/upload", files=[("f", ("labels.txt", YOLO_LABEL, "text/plain"))]
                       ).status_code == 400                                   # no image
    assert sorted(p.name for p in ws.inbox.iterdir()) == ["processed", "rejected", "unlabeled"]


# ============================================================================ run

def test_run_returns_409_while_a_cycle_is_running(ws: Workspace) -> None:
    slow = SlowCycle()
    client = client_for(ws, cycle_fn=slow)
    runner = client.app.state.dashboard.runner
    try:
        first = client.post("/api/run?demo=1")
        assert first.status_code == 202 and first.json()["started"] is True and first.json()["demo"] is True
        assert slow.started.wait(10)
        second = client.post("/api/run")
        assert second.status_code == 409 and "still running" in second.json()["detail"]
        assert client.get("/api/status").json()["runner"]["busy"] is True
    finally:
        slow.release.set()
    assert runner.wait(10)
    snapshot = client.get("/api/status").json()["runner"]
    assert snapshot["busy"] is False and snapshot["result"] == {"cycle": 1, "status": "done"}
    assert snapshot["error"] is None and snapshot["finished"] is not None
    assert client.post("/api/run").status_code == 202                       # free again
    assert runner.wait(10) and slow.calls == [True, False]


def test_run_returns_409_while_another_process_runs_a_cycle(ws: Workspace) -> None:
    slow = SlowCycle()
    slow.release.set()
    client = client_for(ws, cycle_fn=slow)
    with CycleLock(lock_path(ws)):
        response = client.post("/api/run")
    assert response.status_code == 409 and "another process" in response.json()["detail"]
    assert slow.calls == []


def test_a_failing_cycle_does_not_block_the_next(ws: Workspace) -> None:
    def broken(_ws: Workspace, demo: bool = False) -> dict[str, Any]:
        raise RuntimeError("training crashed")

    client = client_for(ws, cycle_fn=broken)
    assert client.post("/api/run").status_code == 202
    assert client.app.state.dashboard.runner.wait(10)
    runner = client.get("/api/status").json()["runner"]
    assert runner["busy"] is False and runner["error"] == "RuntimeError: training crashed"
    assert client.post("/api/run").status_code == 202


def test_a_cycle_started_from_the_dashboard_updates_status_and_events(ws: Workspace) -> None:
    def fake_cycle(ws_: Workspace, demo: bool = False) -> dict[str, Any]:
        def ok(name: str):
            return lambda _ctx: StepResult(True, f"{name} ok")
        return run_cycle(ws_, demo=demo, steps=[(name, ok(name)) for name in STEP_NAMES])

    (ws.inbox / "batch1").mkdir(parents=True)
    (ws.inbox / "batch1" / "a.jpg").write_bytes(jpeg())
    client = client_for(ws, cycle_fn=fake_cycle)
    assert client.post("/api/run?demo=1").status_code == 202
    assert client.app.state.dashboard.runner.wait(10)
    status = client.get("/api/status").json()
    assert (status["cycle"], status["result"], status["demo"]) == (1, "done", True)
    assert {s["status"] for s in status["steps"].values()} == {"done"}
    messages = sse_messages(client.get("/api/events?follow=0").text)
    assert len(messages) == 1 + 2 * len(STEP_NAMES)                     # retry + 20 events
    assert json.loads(messages[-1]["data"])["step"] == "notify"


# ============================================================================ events (SSE)

def test_events_sends_every_existing_complete_line(ws: Workspace) -> None:
    events = [spec_event(1), spec_event(2, "quality", "started"), spec_event(3, "quality")]
    write_events(ws, events[:2], tail="not json\n")
    with ws.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(events[2]) + "\n" + '{"half": "writ')     # line in progress
    response = client_for(ws).get("/api/events?follow=0")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    messages = sse_messages(response.text)
    assert messages[0] == {"retry": str(dashboard.SSE_RETRY_MS)}
    assert [m["id"] for m in messages[1:]] == ["1", "2", "4"]             # line 3 is not JSON
    assert [json.loads(m["data"]) for m in messages[1:]] == events


def test_events_resume_after_the_last_event_id(ws: Workspace) -> None:
    write_events(ws, [spec_event(n) for n in (1, 2, 3)])
    client = client_for(ws)
    messages = sse_messages(client.get("/api/events?follow=0", headers={"Last-Event-ID": "2"}).text)
    assert [m["id"] for m in messages[1:]] == ["3"]
    replaced = sse_messages(client.get("/api/events?follow=0", headers={"Last-Event-ID": "9"}).text)
    assert [m["id"] for m in replaced[1:]] == ["1", "2", "3"]               # file was replaced
    assert sse_messages(client.get("/api/events?follow=0").text)[0]["retry"]


def test_event_stream_follows_appended_lines_with_heartbeats(ws: Workspace) -> None:
    write_events(ws, [spec_event(1)])

    async def collect() -> list[str]:
        received: list[str] = []
        stream = dashboard.event_stream(ws.events_path, poll_s=0.01, heartbeat_s=0.05)
        try:
            async for chunk in stream:
                received.append(chunk)
                if chunk.startswith(": heartbeat"):
                    with ws.events_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(spec_event(2, "notify")) + "\n")
                if chunk.startswith("id: 2"):
                    break
        finally:
            await stream.aclose()
        return received

    received = asyncio.run(asyncio.wait_for(collect(), timeout=20))
    assert received[0].startswith("retry:") and received[1].startswith("id: 1\n")
    assert ": heartbeat\n\n" in received and received[-1].startswith("id: 2\n")
    assert json.loads(received[-1].split("data: ", 1)[1])["step"] == "notify"


# ============================================================================ predict, report

@requires_smoke_unet
def test_predict_with_the_smoke_unet(ws: Workspace) -> None:
    register(ws, 0, current=True, weights=UNET_WEIGHTS)
    photo = sorted((SMOKE / "images" / "test").glob("roboflow_*.jpg"))[0]
    img = cv2.imread(str(photo))
    client = client_for(ws)
    response = client.post("/api/predict", files={"file": (photo.name, photo.read_bytes(), "image/jpeg")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["version"], body["architecture"], body["threshold"]) == (0, "unet", 0.5)
    assert body["filename"] == photo.name and body["image_size"] == [img.shape[1], img.shape[0]]
    assert body["stage"] in range(5) and 0.0 <= body["area"] <= 1.0
    assert body["positive"] == (body["area_px"] > 0) == (body["stage"] > 0)
    assert set(body["stage_fractions"]) == {"1", "2", "3", "4"}
    if body["positive"]:
        assert 0.0 <= body["confidence"] <= 1.0 and body["regions"] >= 1
        assert sum(body["stage_fractions"].values()) == pytest.approx(1.0)
    else:
        assert body["confidence"] is None and body["regions"] == 0
    assert 0.0 <= body["max_confidence"] <= 1.0 and "not a medical device" in body["disclaimer"]
    overlay = cv2.imdecode(np.frombuffer(base64.b64decode(body["overlay_png"]), np.uint8), cv2.IMREAD_COLOR)
    assert base64.b64decode(body["overlay_png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert overlay is not None and max(overlay.shape[:2]) <= dashboard.OVERLAY_MAX_SIDE
    again = client.post("/api/predict", files={"file": (photo.name, photo.read_bytes(), "image/jpeg")})
    assert again.json()["area_px"] == body["area_px"]                       # cached model, same result


def test_predict_needs_a_model_and_an_image(ws: Workspace) -> None:
    client = client_for(ws)
    response = client.post("/api/predict", files={"file": ("a.jpg", jpeg(), "image/jpeg")})
    assert response.status_code == 503 and "no model version is current" in response.json()["detail"]
    register(ws, 0, current=True)
    loads: list[tuple[str, Path]] = []
    client = client_for(ws, model_loader=lambda arch, weights: loads.append((arch, weights)))
    assert client.post("/api/predict", files={"file": ("a.txt", b"no image", "text/plain")}).status_code == 400
    assert client.post("/api/predict", files={"file": ("a.jpg", b"", "image/jpeg")}).status_code == 400
    assert loads == []                                   # nothing is loaded for an invalid photo


def test_report_links_redirect_to_the_static_report(ws: Workspace) -> None:
    ws.reports.mkdir(parents=True, exist_ok=True)
    (ws.reports / "cycle_1.html").write_text("<html>report 1</html>")
    (ws.reports / "cycle_1.pdf").write_bytes(b"%PDF-1.5 report")
    client = client_for(ws)
    html = client.get("/api/report/1", follow_redirects=False)
    assert html.status_code == 307 and html.headers["location"].endswith("/reports/cycle_1.html")
    assert client.get("/api/report/1").text == "<html>report 1</html>"
    pdf = client.get("/api/report/1?format=pdf", follow_redirects=False)
    assert pdf.status_code == 307 and pdf.headers["location"].endswith("/reports/cycle_1.pdf")
    assert client.get("/api/report/2").status_code == 404
    assert client.get("/api/report/-1").status_code == 422
    assert client.get("/api/report/1?format=docx").status_code == 422


def test_the_default_cycle_function_is_run_cycle(ws: Workspace) -> None:
    """Without injection the dashboard runs selftrain.system.cycle.run_cycle: with an
    empty inbox it starts no cycle and reports "idle"."""
    client = client_for(ws)
    assert client.post("/api/run").status_code == 202
    assert client.app.state.dashboard.runner.wait(30)
    runner = client.get("/api/status").json()["runner"]
    assert runner["error"] is None and runner["result"]["status"] == "idle"
    assert not ws.events_path.exists()
