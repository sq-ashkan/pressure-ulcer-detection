"""Web dashboard of the self-retraining system (feature F9, SPEC section 8).

A FastAPI application over one workspace; it holds no state of its own beyond the
running cycle thread and the loaded model, so it can be restarted at any time::

    GET  /                  the single page static/index.html (inline CSS, JS and SVG;
                            no external resources, works offline on a projector)
    GET  /api/status        status.json (cycle.read_status) + current model, runner,
                            inbox and latest report
    GET  /api/events        Server-Sent Events: every line of events.jsonl, then new ones
    POST /api/upload        multipart photos / labelled folder / .zip -> new inbox batch(es)
    POST /api/run?demo=1    start a cycle in a background thread (409 while one runs)
    GET  /api/versions      registry versions with test Dice, 95 % CI and gate decision
    GET  /api/report/{n}    redirect to reports/cycle_{n}.html (or ?format=pdf)
    POST /api/predict       one photo -> stage, area, confidence + PNG overlay (base64)

Method
------
* Cycles. POST /api/run calls the injected cycle function (default
  ``selftrain.system.cycle.run_cycle``) in a daemon thread. A lock-protected busy flag is
  tested and set in one critical section and cleared in a ``finally`` clause, so at most
  one cycle runs per dashboard and a failing cycle never blocks the next. A cycle that
  another process (CLI, watch mode) is running is detected through the cycle lock
  (``cycle.lock_owner``); both cases answer HTTP 409.
* Events. /api/events tails events.jsonl: it sends every complete line (a line counts
  once its newline is written, so no half-written event is ever sent), then polls the
  file every 0.5 s for appended lines and sends a comment line as heartbeat after 15 s
  without data. The SSE id of an event is its line number, so a reconnecting browser
  (Last-Event-ID) receives only the lines it has not seen. A file that was replaced or
  truncated is read again from its first line.
* Uploads. Every .zip is one batch, checked before it is accepted (readable archive,
  no absolute or ``..`` member paths, no encryption, size limits, at least one image).
  All other files of a request form one batch folder; the relative paths a browser
  sends for a dropped folder are kept, common top folders are removed (the innermost
  one names the batch), and in a flat drop images go to images/ and a .txt with the
  stem of an uploaded image to labels/. Hidden files and unknown types are ignored and
  reported. Everything is written under hidden staging names (ignored by the inbox scan)
  and renamed into place only when the whole request succeeded, so a cycle or watch mode
  never sees a half-written batch.
* Prediction. The current registry version is loaded once with the experiment library
  (``selftrain.unet.UNetPredictor`` or ``selftrain.yoloseg.YOLOPredictor``, by the
  version's architecture) and reused until promotion or rollback changes the version or
  its weights file. The photograph is scored exactly as the evaluation scores test
  images: semantic map at the version's operating threshold (models/v{N}/evaluation.json,
  chosen on val), connected regions smaller than ``min_component_frac`` of the image
  removed. Reported: dominant stage, ulcer area (fraction of image pixels), number of
  regions, per-stage share of the ulcer pixels and the mean foreground confidence over
  the predicted ulcer pixels, plus an overlay PNG (stage colour at 45 % opacity, outlined).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from ..config import NUM_STAGES
from ..metrics import dominant_stage, remove_specks
from . import registry
from .cycle import (LOCK_NAME, CycleLock, CycleLockedError, inbox_signature, lock_owner,
                    read_status, run_cycle)
from .ingest import IMAGE_SUFFIXES
from .registry import VersionRecord
from .steps import load_predictor, operating_threshold, to_jsonable
from .workspace import Workspace, utc_now_iso

__all__ = [
    "CycleRunner", "Dashboard", "EventTail", "ModelCache", "NoModelError", "UploadError",
    "analyse_prediction", "check_zip", "create_app", "cycle_running_elsewhere", "decode_image",
    "event_stream",
    "inbox_view", "latest_report", "model_view", "parse_last_event_id", "render_overlay",
    "safe_relative_path", "serve", "stage_upload", "version_view",
]

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().with_name("static")
INDEX_HTML = STATIC_DIR / "index.html"

# events (Server-Sent Events)
HEARTBEAT_S = 15.0           # comment line after this many seconds without data
EVENT_POLL_S = 0.5           # how often events.jsonl is checked for appended lines
SSE_RETRY_MS = 3000          # browser reconnect delay after a lost connection
EVENT_READ_CHUNK = 1 << 20   # bytes read from events.jsonl per step
SSE_HEADERS: Mapping[str, str] = MappingProxyType(
    {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# uploads
MAX_UPLOAD_FILES = 20_000            # files per request (a dropped batch folder)
MAX_UPLOAD_BYTES = 4 << 30           # bytes per request
MAX_ZIP_MEMBERS = 200_000
MAX_ZIP_UNCOMPRESSED = 16 << 30      # zip-bomb guard: total uncompressed size
COPY_CHUNK = 1 << 20
LABEL_SUFFIX = ".txt"
META_SUFFIXES = frozenset({".json", ".yaml", ".yml"})
ACCEPTED_SUFFIXES = IMAGE_SUFFIXES | META_SUFFIXES | {LABEL_SUFFIX}
LAYOUT_DIRS = frozenset({"images", "labels", "masks", "train", "valid", "val", "test"})
_UNSAFE_CHARS = re.compile(r"[\x00-\x1f\x7f:]")
_SLUG_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# prediction
MAX_PREDICT_BYTES = 50 << 20
PREDICT_MAX_SIDE = 2048      # longer photographs are reduced before inference
OVERLAY_MAX_SIDE = 1280      # side limit of the returned overlay PNG
OVERLAY_ALPHA = 0.45
STAGE_COLOURS: Mapping[int, str] = MappingProxyType(
    {1: "#facc15", 2: "#fb923c", 3: "#f43f5e", 4: "#a78bfa"})
DISCLAIMER = "Research prototype: not a medical device, no diagnosis."

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_REPORT_FILE = re.compile(r"cycle_(0|[1-9][0-9]*)\.(html|pdf)")
_LAST_EVENT_ID = re.compile(r"[0-9]{1,18}")

CycleFn = Callable[..., Mapping[str, Any]]
"""Signature of the injected cycle function: ``fn(ws, demo=bool) -> result dict``."""
Loader = Callable[[str, Path], Any]
"""Signature of the predictor loader: ``loader(architecture, weights) -> predictor``."""


# --------------------------------------------------------------------------- views

def first_metric(metrics: Mapping[str, Any], *keys: str) -> Any:
    """Value of the first of `keys` present (not None) in `metrics`, else None.

    The cycle and the registry store headline metrics under the names of
    ``steps.summary_metrics`` = ``registry.metrics_from_evaluation`` (test_dice_mean,
    ...); a hand-made record may use the names of the evaluation's test block
    (dice_mean, ...). The dashboard reads both, the stored names first."""
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _interval(value: Any) -> list[float] | None:
    """A [low, high] pair of numbers, or None if `value` is not one."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return [float(value[0]), float(value[1])]
    return None


def version_view(rec: VersionRecord, current: int | None) -> dict[str, Any]:
    """One registry version as the dashboard shows it: frozen-test Dice with its 95 % CI,
    false-positive rate, training data, and the gate decision taken at registration."""
    m = rec.metrics
    gate = rec.gate
    return to_jsonable({
        "version": rec.version, "created": rec.created, "architecture": rec.architecture,
        "dataset_version": rec.dataset_version, "train_images": rec.train_images,
        "train_photographs": rec.train_photographs, "parent": rec.parent,
        "dice_mean": first_metric(m, "test_dice_mean", "dice_mean"),
        "dice_ci95": _interval(first_metric(m, "test_dice_ci95", "dice_ci95")),
        "fpr": first_metric(m, "test_fpr", "false_positive_rate"),
        "stage_dice_macro": m.get("stage_dice_macro"),
        "promoted": rec.promoted, "current": rec.version == current,
        "gate": {"promoted": gate.get("promoted"), "mean_diff": gate.get("mean_diff"),
                 "ci95": _interval(gate.get("ci95")), "reason": gate.get("reason", "")},
    })


def model_view(rec: VersionRecord | None) -> dict[str, Any] | None:
    """The current model for the page header (None when no version is current)."""
    return None if rec is None else version_view(rec, rec.version)


def report_files(ws: Workspace) -> dict[int, dict[str, Path]]:
    """Cycle reports on disk: {cycle: {"html": path, "pdf": path}} (formats present only)."""
    found: dict[int, dict[str, Path]] = {}
    if ws.reports.is_dir():
        for path in ws.reports.iterdir():
            match = _REPORT_FILE.fullmatch(path.name)
            if match and path.is_file():
                found.setdefault(int(match.group(1)), {})[match.group(2)] = path
    return found


def latest_report(ws: Workspace) -> dict[str, Any] | None:
    """Links to the report of the most recent cycle that has one, or None."""
    found = report_files(ws)
    if not found:
        return None
    cycle = max(found)
    return {"cycle": cycle,
            "html": f"/api/report/{cycle}" if "html" in found[cycle] else None,
            "pdf": f"/api/report/{cycle}?format=pdf" if "pdf" in found[cycle] else None}


def _count_images(directory: Path) -> int:
    """Image files below `directory`, hidden files and folders excluded."""
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.rglob("*")
               if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
               and not any(part.startswith(".") for part in p.relative_to(directory).parts))


def inbox_view(ws: Workspace) -> dict[str, Any]:
    """Batches waiting in inbox/ (top-level entries with at least one file, by the rule of
    ``cycle.inbox_signature``) and the number of photos queued in inbox/unlabeled/."""
    pending = sorted({PurePosixPath(rel).parts[0] for rel, _, _ in inbox_signature(ws)})
    return {"pending": pending, "unlabeled": _count_images(ws.unlabeled)}


# --------------------------------------------------------------------------- cycle runner

@dataclass(frozen=True)
class RunnerState:
    """What the dashboard knows about the cycle it started last."""
    busy: bool = False
    demo: bool | None = None
    started: str | None = None
    finished: str | None = None
    result: Any = None
    error: str | None = None


class CycleRunner:
    """Runs the injected cycle function in a background thread, at most one at a time.

    start() tests and sets the busy flag in one critical section, so two simultaneous
    requests cannot both start a cycle; the worker clears it in a ``finally`` clause.
    The thread is a daemon: stopping the dashboard does not wait for a training run;
    the cycle lock is then released by the operating system and ``cycle.read_status``
    reports the cycle as interrupted."""

    def __init__(self, ws: Workspace, cycle_fn: CycleFn) -> None:
        self._ws = ws
        self._fn = cycle_fn
        self._lock = threading.Lock()
        self._state = RunnerState()
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._state.busy

    def snapshot(self) -> dict[str, Any]:
        """The runner state as JSON-ready dict."""
        with self._lock:
            return to_jsonable(asdict(self._state))

    def start(self, demo: bool) -> bool:
        """Start a cycle (`demo` as given) unless one is running; True if started."""
        with self._lock:
            if self._state.busy:
                return False
            self._state = RunnerState(busy=True, demo=demo, started=utc_now_iso())
            thread = threading.Thread(target=self._work, args=(demo,),
                                      name="selftrain-cycle", daemon=True)
            try:
                thread.start()
            except BaseException:
                self._state = RunnerState()
                raise
            self._thread = thread
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until the running cycle (if any) has finished; True if idle afterwards."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return not self.busy

    def _work(self, demo: bool) -> None:
        result: Any = None
        error: str | None = None
        try:
            result = self._fn(self._ws, demo=demo)
        except Exception as exc:
            log.exception("cycle started from the dashboard failed")
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._state = replace(self._state, busy=False, finished=utc_now_iso(),
                                      result=to_jsonable(result), error=error)


# --------------------------------------------------------------------------- events (SSE)

class EventTail:
    """Incremental reader of the complete lines of a growing file (events.jsonl).

    read() returns the lines appended since the previous call as (line number, text).
    Bytes after the last newline are kept back until their newline arrives. If the file
    was replaced (other inode) or truncated (shorter than what was read), reading
    restarts at its first line and `generation` is incremented."""

    def __init__(self, path: Path, chunk: int = EVENT_READ_CHUNK) -> None:
        self.path = Path(path)
        self.chunk = chunk
        self.generation = 0
        self.restart()

    def restart(self) -> None:
        """Forget everything read so far; the next read() starts at line 1."""
        self.inode: int | None = None
        self.offset = 0
        self.lines = 0
        self.partial = b""

    def read(self) -> list[tuple[int, str]]:
        try:
            handle = open(self.path, "rb")
        except FileNotFoundError:
            return []
        with handle:
            st = os.fstat(handle.fileno())
            if self.inode is not None and (st.st_ino != self.inode or st.st_size < self.offset):
                self.restart()
                self.generation += 1
            self.inode = st.st_ino
            if st.st_size <= self.offset:
                return []
            handle.seek(self.offset)
            data = handle.read(min(self.chunk, st.st_size - self.offset))
        self.offset += len(data)
        pieces = (self.partial + data).split(b"\n")
        self.partial = pieces.pop()
        out = []
        for raw in pieces:
            self.lines += 1
            out.append((self.lines, raw.decode("utf-8", errors="replace").rstrip("\r")))
        return out


def parse_last_event_id(value: str | None) -> int:
    """Line number from an SSE Last-Event-ID header; 0 if absent or not a number."""
    if value is None or not _LAST_EVENT_ID.fullmatch(value.strip()):
        return 0
    return int(value.strip())


def parse_event_line(text: str) -> dict[str, Any] | None:
    """A line of events.jsonl as dict, or None if it is not a JSON object."""
    try:
        event = json.loads(text)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def format_event(number: int, event: Mapping[str, Any]) -> str:
    """One SSE message: id = line number, data = the event as single-line strict JSON."""
    data = json.dumps(to_jsonable(event), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
    return f"id: {number}\ndata: {data}\n\n"


async def event_stream(path: Path, *, follow: bool = True, last_id: int = 0,
                       poll_s: float = EVENT_POLL_S, heartbeat_s: float = HEARTBEAT_S,
                       is_disconnected: Callable[[], Awaitable[bool]] | None = None,
                       ) -> AsyncIterator[str]:
    """Server-Sent Events of events.jsonl (see module docstring, "Events").

    Sends ``retry:``, then every complete line after line `last_id` that is a JSON
    object (others are skipped), then -- with `follow` -- appended lines as they come,
    checking every `poll_s` seconds, with a ``: heartbeat`` comment after `heartbeat_s`
    seconds without data. Without `follow` it ends once the existing lines are sent. If
    the file holds fewer lines than `last_id`, it was replaced since the client read it
    and is sent from its first line."""
    tail = EventTail(path)
    generation = tail.generation
    caught_up = False
    yield f"retry: {SSE_RETRY_MS}\n\n"
    last_write = time.monotonic()
    while True:
        lines = tail.read()
        if tail.generation != generation:          # replaced while being followed
            generation, last_id = tail.generation, 0
        for number, text in lines:
            event = parse_event_line(text) if number > last_id else None
            if event is not None:
                yield format_event(number, event)
                last_write = time.monotonic()
        if lines:
            continue
        if not caught_up:
            caught_up = True
            if tail.lines < last_id:
                last_id = 0
                tail.restart()
                continue
        if not follow or (is_disconnected is not None and await is_disconnected()):
            return
        if time.monotonic() - last_write >= heartbeat_s:
            yield ": heartbeat\n\n"
            last_write = time.monotonic()
        await asyncio.sleep(poll_s)


# --------------------------------------------------------------------------- uploads

class UploadError(Exception):
    """An upload or prediction input that cannot be accepted (HTTP `status_code`)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def safe_relative_path(name: str) -> PurePosixPath | None:
    """The relative path of an uploaded file as sent by the browser, or None if unusable.

    Backslashes count as separators and empty or "." components are dropped. Refused:
    a ".." component, a hidden component (".DS_Store", ".git"), macOS "__MACOSX"
    metadata, control characters or ":" (drive letters), and paths without a name."""
    parts = [p for p in str(name).replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p.startswith(".") or p == "__MACOSX"
                        or _UNSAFE_CHARS.search(p) for p in parts):
        return None
    return PurePosixPath(*parts)


def slug(text: str, fallback: str = "upload", max_len: int = 60) -> str:
    """A file-name-safe form of `text` (letters, digits, ".", "_", "-")."""
    out = _SLUG_CHARS.sub("_", text).strip("._-")[:max_len].strip("._-")
    return out or fallback


def batch_layout(paths: Sequence[PurePosixPath]) -> tuple[str | None, list[PurePosixPath]]:
    """Map the relative paths of one upload onto a batch folder (SPEC section 2).

    A top folder shared by all paths is removed as long as it is not itself part of a
    batch layout (images, labels, masks, train, valid, val, test); the innermost removed
    folder is returned as the batch's name. If no file then lies in a sub-folder (a flat
    drop), images move to images/ and a .txt whose stem equals an image's stem moves to
    labels/; other files (source.yaml, annotations.json) stay at the batch root."""
    parts = [p.parts for p in paths]
    top: str | None = None
    while (parts and all(len(p) >= 2 for p in parts) and len({p[0] for p in parts}) == 1
           and parts[0][0].lower() not in LAYOUT_DIRS):
        top = parts[0][0]
        parts = [p[1:] for p in parts]
    rels = [PurePosixPath(*p) for p in parts]
    if any(len(p) > 1 for p in parts):
        return top, rels
    image_stems = {r.stem for r in rels if r.suffix.lower() in IMAGE_SUFFIXES}
    placed = []
    for rel in rels:
        suffix = rel.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            rel = PurePosixPath("images", rel.name)
        elif suffix == LABEL_SUFFIX and rel.stem in image_stems:
            rel = PurePosixPath("labels", rel.name)
        placed.append(rel)
    return top, placed


def size_text(n: int) -> str:
    """A byte count for messages: "4 GiB", "50 MiB", "512 bytes"."""
    for unit, shift in (("GiB", 30), ("MiB", 20), ("KiB", 10)):
        if n >= 1 << shift:
            return f"{n / (1 << shift):.3g} {unit}"
    return f"{n} bytes"


class _Budget:
    """Bytes still allowed in one upload request."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.left = limit

    def take(self, n: int) -> None:
        self.left -= n
        if self.left < 0:
            raise UploadError(413, f"upload too large: at most {size_text(self.limit)} "
                                   f"per request")


def _copy(stream: BinaryIO, target: Path, budget: _Budget) -> int:
    """Copy `stream` to the new file `target` in chunks, charging `budget`."""
    written = 0
    with open(target, "xb") as out:
        while chunk := stream.read(COPY_CHUNK):
            budget.take(len(chunk))
            out.write(chunk)
            written += len(chunk)
    return written


def _hidden(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    return any(p.startswith(".") or p == "__MACOSX" for p in parts)


def check_zip(path: Path, *, max_members: int = MAX_ZIP_MEMBERS,
              max_uncompressed: int = MAX_ZIP_UNCOMPRESSED) -> dict[str, int]:
    """Validate an uploaded .zip batch; returns its counts {files, images, labels}.

    Accepted only if the archive can be read, no member path is absolute, contains ".."
    or a drive letter (zip slip), no member is encrypted, it has at most `max_members`
    members whose total uncompressed size is at most `max_uncompressed` bytes (zip
    bomb), and at least one image outside hidden folders. Raises UploadError."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, ValueError) as exc:
        raise UploadError(400, f"not a readable zip archive ({exc})") from None
    if len(infos) > max_members:
        raise UploadError(413, f"zip archive has {len(infos)} members; at most "
                               f"{max_members} are accepted")
    counts = {"files": 0, "images": 0, "labels": 0}
    total = 0
    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            raise UploadError(400, f"zip member {info.filename!r} has an unsafe path")
        if info.flag_bits & 0x1:
            raise UploadError(400, f"zip member {info.filename!r} is encrypted")
        total += info.file_size
        if info.is_dir() or _hidden(name):
            continue
        counts["files"] += 1
        suffix = PurePosixPath(name).suffix.lower()
        counts["images"] += suffix in IMAGE_SUFFIXES
        counts["labels"] += suffix == LABEL_SUFFIX
    if total > max_uncompressed:
        raise UploadError(413, f"zip archive unpacks to {size_text(total)}; at most "
                               f"{size_text(max_uncompressed)} are accepted")
    if counts["images"] == 0:
        raise UploadError(400, "zip archive contains no image")
    return counts


def _publish(staged: Path, inbox: Path, base: str, suffix: str) -> str:
    """Rename a staged batch to the first free name base[_k]suffix in inbox/; its name.

    Files are hard-linked (os.link fails instead of overwriting) and the staging name is
    removed; folders are renamed after an existence check (os.rename fails on a
    non-empty target), so two simultaneous uploads never overwrite each other. On a
    file system without hard links a file is renamed after the existence check."""
    for k in range(1, 1000):
        name = f"{base}{'' if k == 1 else f'_{k}'}{suffix}"
        target = inbox / name
        if target.exists() or target.is_symlink():
            continue
        try:
            if staged.is_dir():
                os.rename(staged, target)
            else:
                _link_or_rename(staged, target)
        except FileExistsError:
            continue
        except OSError:
            if target.exists():
                continue
            raise
        return name
    raise UploadError(409, f"no free batch name for {base!r} in the inbox")


def _link_or_rename(staged: Path, target: Path) -> None:
    try:
        os.link(staged, target)
    except FileExistsError:
        raise
    except OSError:                       # no hard links on this file system
        if target.exists():
            raise FileExistsError(target) from None
        os.rename(staged, target)
        return
    staged.unlink()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def stage_upload(ws: Workspace, files: Sequence[tuple[str, BinaryIO]], *,
                 max_bytes: int = MAX_UPLOAD_BYTES,
                 now: datetime | None = None) -> dict[str, Any]:
    """Turn the files of one upload request into inbox batches (method: module docstring).

    `files` are (name as sent by the client, readable binary stream). Batch names are
    ``<UTC time>_<name>``: the zip's stem, or the dropped folder's name ("upload" for
    loose files). Every file of the request is staged and checked before the first batch
    is published, so an invalid zip or an oversized request adds nothing. Returns {"batches": [...], "ignored": [{"name", "reason"}]}; raises UploadError
    (400 nothing usable or invalid zip, 413 too large)."""
    ws.inbox.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    ignored: list[dict[str, str]] = []
    zips: list[tuple[PurePosixPath, BinaryIO]] = []
    loose: list[tuple[PurePosixPath, BinaryIO]] = []
    for name, stream in files:
        rel = safe_relative_path(name)
        suffix = rel.suffix.lower() if rel is not None else ""
        if rel is None:
            ignored.append({"name": name, "reason": "hidden or unsafe path"})
        elif suffix == ".zip":
            zips.append((rel, stream))
        elif suffix in ACCEPTED_SUFFIXES:
            loose.append((rel, stream))
        else:
            ignored.append({"name": name, "reason": f"unsupported file type '{suffix}'"})
    if not zips and not loose:
        raise UploadError(400, "no usable file: upload images (.jpg, .png, ...), label "
                               "files with their images, or a .zip batch")
    budget = _Budget(max_bytes)
    staged: list[tuple[Path, str, str, dict[str, Any]]] = []   # path, base, suffix, info
    try:
        for rel, stream in zips:
            tmp = ws.inbox / f".upload-{uuid.uuid4().hex}.zip"
            staged.append((tmp, f"{stamp}_{slug(rel.stem)}", ".zip", {"kind": "zip"}))
            _copy(stream, tmp, budget)
            staged[-1][3].update(check_zip(tmp))
        if loose:
            staged.append(_stage_folder(ws, loose, stamp, budget, ignored))
        batches = []
        for tmp, base, suffix, info in staged:
            name = _publish(tmp, ws.inbox, base, suffix)
            batches.append({"name": name, "path": f"inbox/{name}", **info})
    except BaseException:
        for tmp, *_ in staged:
            _remove(tmp)
        raise
    log.info("upload: %d batch(es) added to %s", len(batches), ws.inbox)
    return {"batches": batches, "ignored": ignored}


def _stage_folder(ws: Workspace, loose: Sequence[tuple[PurePosixPath, BinaryIO]], stamp: str,
                  budget: _Budget, ignored: list[dict[str, str]],
                  ) -> tuple[Path, str, str, dict[str, Any]]:
    """Write the loose files of a request into a hidden staging folder (batch_layout)."""
    top, rels = batch_layout([rel for rel, _ in loose])
    tmp = ws.inbox / f".upload-{uuid.uuid4().hex}"
    tmp.mkdir()
    info: dict[str, Any] = {"kind": "folder", "files": 0, "images": 0, "labels": 0}
    seen: set[PurePosixPath] = set()
    try:
        for (original, stream), rel in zip(loose, rels):
            if rel in seen:
                ignored.append({"name": str(original), "reason": "duplicate path"})
                continue
            seen.add(rel)
            target = tmp.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy(stream, target, budget)
            info["files"] += 1
            info["images"] += rel.suffix.lower() in IMAGE_SUFFIXES
            info["labels"] += rel.suffix.lower() == LABEL_SUFFIX
    except BaseException:
        _remove(tmp)
        raise
    if info["images"] == 0:
        _remove(tmp)
        raise UploadError(400, "the uploaded files contain no image")
    return tmp, f"{stamp}_{slug(top or 'upload')}", "", info


# --------------------------------------------------------------------------- prediction

class NoModelError(RuntimeError):
    """No model version is current, or its weights file is missing."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode an uploaded photograph to BGR uint8 (as cv2.imread reads dataset images,
    EXIF orientation applied); UploadError 400 if it is not a readable image."""
    if not data:
        raise UploadError(400, "the uploaded file is empty")
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None or img.size == 0:
        raise UploadError(400, "the uploaded file is not a readable image (JPEG, PNG, ...)")
    return img


def limit_side(img: np.ndarray, max_side: int,
               interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """`img` scaled (aspect kept) so that its longer side is at most `max_side`."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    r = max_side / max(h, w)
    return cv2.resize(img, (max(1, round(w * r)), max(1, round(h * r))),
                      interpolation=interpolation)


def analyse_prediction(predictor: Any, raw: Any, threshold: float,
                       min_frac: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Semantic map and summary of one prediction, scored as the evaluation scores it.

    sem = predictor.to_semantic(raw, threshold) with connected ulcer regions smaller than
    `min_frac` of the image removed (``selftrain.metrics.remove_specks``, 8-connected).
    Summary: positive (any ulcer pixel left), stage (``dominant_stage``: the stage with
    most ulcer pixels, 0 = none), area (ulcer pixels / image pixels), area_px, regions
    (8-connected ulcer regions), stage_fractions (share of the ulcer pixels per stage),
    confidence (mean foreground confidence ``predictor.fg_confidence`` over the ulcer
    pixels; None without ulcer) and max_confidence (its maximum over the image)."""
    sem = np.asarray(predictor.to_semantic(raw, threshold), dtype=np.uint8)
    sem = np.where(remove_specks(sem > 0, min_frac), sem, 0).astype(np.uint8)
    ulcer = sem > 0
    area_px = int(ulcer.sum())
    fg = np.asarray(predictor.fg_confidence(raw), dtype=np.float32)
    counts = np.bincount(sem.ravel(), minlength=NUM_STAGES + 1)
    regions = cv2.connectedComponents(ulcer.astype(np.uint8), connectivity=8)[0] - 1
    return sem, {
        "positive": area_px > 0,
        "stage": dominant_stage(sem),
        "area": area_px / sem.size,
        "area_px": area_px,
        "regions": int(regions),
        "stage_fractions": {str(s): (float(counts[s]) / area_px if area_px else 0.0)
                            for s in range(1, NUM_STAGES + 1)},
        "confidence": float(fg[ulcer].mean()) if area_px else None,
        "max_confidence": float(fg.max()) if fg.size else 0.0,
    }


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def render_overlay(img_bgr: np.ndarray, sem: np.ndarray, *, max_side: int = OVERLAY_MAX_SIDE,
                   alpha: float = OVERLAY_ALPHA) -> bytes:
    """PNG of the photograph with each predicted stage filled in its STAGE_COLOURS colour
    at opacity `alpha` and outlined; reduced to at most `max_side` pixels (mask resized
    nearest-neighbour, so stages are never blended)."""
    img = limit_side(img_bgr, max_side)
    if img.shape[:2] != sem.shape:
        sem = cv2.resize(sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = img.astype(np.float32)
    masks = {s: sem == s for s in STAGE_COLOURS}
    for stage, mask in masks.items():
        if mask.any():
            out[mask] = (1.0 - alpha) * out[mask] + alpha * np.float32(_bgr(STAGE_COLOURS[stage]))
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    thickness = max(2, round(max(out.shape[:2]) / 320))
    for stage, mask in masks.items():
        if mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, _bgr(STAGE_COLOURS[stage]), thickness,
                             lineType=cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encoding of the overlay failed")
    return buffer.tobytes()


@dataclass(frozen=True)
class ServingModel:
    """The loaded predictor of the current version and how it is operated."""
    version: int
    architecture: str
    weights: Path
    threshold: float
    predictor: Any


class ModelCache:
    """Predictor of the current registry version, loaded once and kept in memory.

    serving() asks the registry for the current version on every call and reuses the
    loaded predictor while (version, architecture, weights path, weights mtime) is
    unchanged; after a promotion or rollback the new version is loaded and the old one
    released. The threshold is re-read on every call (``steps.operating_threshold``:
    models/v{N}/evaluation.json, else the architecture's default). One lock serialises
    loading and inference, so one model is in memory and one forward pass runs at a time."""

    def __init__(self, ws: Workspace, loader: Loader | None = None) -> None:
        self._ws = ws
        self._loader = load_predictor if loader is None else loader
        self._lock = threading.Lock()
        self._key: tuple[Any, ...] | None = None
        self._predictor: Any = None

    def serving(self) -> ServingModel:
        """The current version with its loaded predictor (NoModelError if none)."""
        with self._lock:
            return self._serving()

    def _serving(self) -> ServingModel:
        rec = registry.current(self._ws)
        if rec is None:
            raise NoModelError("no model version is current: initialise the workspace or "
                               "run a cycle first")
        weights = rec.weights_path(self._ws)
        if not weights.is_file():
            raise NoModelError(f"weights of model v{rec.version} not found: {weights}")
        key = (rec.version, rec.architecture, str(weights), weights.stat().st_mtime_ns)
        if key != self._key:
            self._key, self._predictor = None, None       # release the old model first
            self._predictor = self._loader(rec.architecture, weights)
            self._key = key
            log.info("dashboard: loaded model v%d (%s) from %s",
                     rec.version, rec.architecture, weights)
        threshold = operating_threshold(self._ws, rec.version, rec.architecture)
        return ServingModel(rec.version, rec.architecture, weights, threshold, self._predictor)

    def predict(self, img_bgr: np.ndarray, min_frac: float) -> dict[str, Any]:
        """Run the current model on one BGR photograph (method: analyse_prediction) and
        return the summary with model provenance and the overlay PNG as base64."""
        with self._lock:
            model = self._serving()
            raw = model.predictor.infer(img_bgr)
            sem, summary = analyse_prediction(model.predictor, raw, model.threshold, min_frac)
        overlay = render_overlay(img_bgr, sem)
        h, w = img_bgr.shape[:2]
        return to_jsonable({
            "version": model.version, "architecture": model.architecture,
            "threshold": model.threshold, "min_component_frac": min_frac, **summary,
            "analysed_size": [w, h],
            "overlay_png": base64.b64encode(overlay).decode("ascii"),
            "stage_colours": {str(k): v for k, v in STAGE_COLOURS.items()},
            "disclaimer": DISCLAIMER,
        })


# --------------------------------------------------------------------------- application

@dataclass(frozen=True)
class Dashboard:
    """Everything the request handlers share; one instance per application."""
    ws: Workspace
    runner: CycleRunner
    models: ModelCache
    poll_s: float = EVENT_POLL_S
    heartbeat_s: float = HEARTBEAT_S


def _dashboard(request: Request) -> Dashboard:
    return request.app.state.dashboard


def index() -> HTMLResponse:
    """GET /: the single page (re-read on every request; never cached by the browser)."""
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-cache"})


def get_status(request: Request) -> dict[str, Any]:
    """GET /api/status: status.json via ``cycle.read_status`` (a cycle whose process died
    is reported as interrupted) plus runner, current model, inbox and latest report."""
    dash = _dashboard(request)
    status = read_status(dash.ws)
    try:
        model, registry_error = model_view(registry.current(dash.ws)), None
    except ValueError as exc:
        model, registry_error = None, str(exc)
    status.update(runner=dash.runner.snapshot(), model=model, registry_error=registry_error,
                  inbox=inbox_view(dash.ws), latest_report=latest_report(dash.ws),
                  workspace=str(dash.ws.root))
    return to_jsonable(status)


def get_versions(request: Request) -> dict[str, Any]:
    """GET /api/versions: every registered version (version_view), ascending."""
    ws = _dashboard(request).ws
    try:
        reg = registry.load(ws)
    except ValueError as exc:
        raise HTTPException(500, f"registry cannot be read: {exc}") from None
    return to_jsonable({"current": reg["current"],
                        "versions": [version_view(r, reg["current"]) for r in reg["versions"]],
                        "history": reg["history"]})


async def get_events(request: Request, follow: bool = True) -> StreamingResponse:
    """GET /api/events: Server-Sent Events of events.jsonl (event_stream). ``follow=0``
    sends the existing events and closes the stream."""
    dash = _dashboard(request)
    stream = event_stream(dash.ws.events_path, follow=follow,
                          last_id=parse_last_event_id(request.headers.get("last-event-id")),
                          poll_s=dash.poll_s, heartbeat_s=dash.heartbeat_s,
                          is_disconnected=request.is_disconnected)
    return StreamingResponse(stream, media_type="text/event-stream", headers=dict(SSE_HEADERS))


async def post_upload(request: Request) -> JSONResponse:
    """POST /api/upload (multipart, any field name): new inbox batch(es), see stage_upload."""
    ws = _dashboard(request).ws
    async with request.form(max_files=MAX_UPLOAD_FILES, max_fields=MAX_UPLOAD_FILES) as form:
        files = [(item.filename or "", item.file) for _, item in form.multi_items()
                 if isinstance(item, StarletteUploadFile)]
        try:
            result = await run_in_threadpool(stage_upload, ws, files)
        except UploadError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
    return JSONResponse(to_jsonable(result), status_code=201)


def cycle_running_elsewhere(ws: Workspace) -> dict[str, Any] | None:
    """Owner record of a cycle that another process (CLI, watch mode) is running, or None.

    The record in cycle.lock (``cycle.lock_owner``) is confirmed by trying the lock
    without blocking, so a record left behind by a crashed process whose pid now belongs
    to another program is not mistaken for a running cycle."""
    owner = lock_owner(ws)
    if owner is None:
        return None
    probe = CycleLock(ws.root / LOCK_NAME)
    try:
        probe.acquire()
    except CycleLockedError:
        return owner
    probe.release()
    return None


def post_run(request: Request, demo: bool = False) -> JSONResponse:
    """POST /api/run?demo=1: start a cycle in the background; 409 if one is running here
    or in another process of the same workspace."""
    dash = _dashboard(request)
    if dash.runner.busy:
        raise HTTPException(409, "a cycle started from this dashboard is still running")
    owner = cycle_running_elsewhere(dash.ws)
    if owner is not None:
        raise HTTPException(409, f"a cycle is running in another process (pid "
                                 f"{owner.get('pid', '?')}, since {owner.get('started', '?')})")
    if not dash.runner.start(demo):
        raise HTTPException(409, "a cycle started from this dashboard is still running")
    return JSONResponse({"started": True, "demo": demo, "runner": dash.runner.snapshot()},
                        status_code=202)


def get_report(request: Request, n: int = PathParam(..., ge=0),
               fmt: Literal["html", "pdf"] = Query("html", alias="format")) -> RedirectResponse:
    """GET /api/report/{n}: redirect to reports/cycle_{n}.html (or .pdf), which is served
    from /reports/ so that the report's relative links (figures/) resolve."""
    path = _dashboard(request).ws.reports / f"cycle_{n}.{fmt}"
    if not path.is_file():
        raise HTTPException(404, f"no {fmt.upper()} report for cycle {n}")
    return RedirectResponse(str(request.url_for("reports", path=path.name)), status_code=307)


def post_predict(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """POST /api/predict (multipart field "file"): the current model on one photograph.
    Photographs longer than PREDICT_MAX_SIDE are first reduced to it (INTER_AREA): the
    models see a 512 px letterbox anyway, so this bounds memory without changing what
    the network sees beyond interpolation. 503 without a usable current model."""
    dash = _dashboard(request)
    data = file.file.read(MAX_PREDICT_BYTES + 1)
    if len(data) > MAX_PREDICT_BYTES:
        raise HTTPException(413, f"photo too large (limit {MAX_PREDICT_BYTES >> 20} MiB)")
    try:
        img = decode_image(data)
    except UploadError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    original = [int(img.shape[1]), int(img.shape[0])]
    try:
        min_frac = dash.ws.load_config().min_component_frac
        result = dash.models.predict(limit_side(img, PREDICT_MAX_SIDE), min_frac)
    except NoModelError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception as exc:
        log.exception("prediction failed")
        raise HTTPException(500, f"prediction failed: {type(exc).__name__}: {exc}") from None
    return {**result, "image_size": original, "filename": file.filename}


def create_app(ws_root: Path | str | Workspace, *, cycle_fn: CycleFn | None = None,
               model_loader: Loader | None = None, poll_s: float = EVENT_POLL_S,
               heartbeat_s: float = HEARTBEAT_S) -> FastAPI:
    """The dashboard application for the workspace at `ws_root` (created if missing).

    `cycle_fn(ws, demo=...)` runs one cycle (default ``cycle.run_cycle``); `model_loader
    (architecture, weights)` builds a predictor (default ``steps.load_predictor``);
    `poll_s` and `heartbeat_s` time the event stream. Both are injectable for tests.
    FastAPI's interactive docs are disabled because they load scripts from a CDN; the
    OpenAPI schema is served at /api/openapi.json."""
    ws = ws_root if isinstance(ws_root, Workspace) else Workspace(Path(ws_root))
    ws.ensure()
    app = FastAPI(title="selftrain dashboard", docs_url=None, redoc_url=None,
                  openapi_url="/api/openapi.json")
    app.state.dashboard = Dashboard(
        ws=ws, runner=CycleRunner(ws, run_cycle if cycle_fn is None else cycle_fn),
        models=ModelCache(ws, model_loader), poll_s=poll_s, heartbeat_s=heartbeat_s)
    app.add_api_route("/", index, methods=["GET"], response_class=HTMLResponse,
                      include_in_schema=False)
    app.add_api_route("/api/status", get_status, methods=["GET"])
    app.add_api_route("/api/events", get_events, methods=["GET"])
    app.add_api_route("/api/upload", post_upload, methods=["POST"], status_code=201)
    app.add_api_route("/api/run", post_run, methods=["POST"], status_code=202)
    app.add_api_route("/api/versions", get_versions, methods=["GET"])
    app.add_api_route("/api/report/{n}", get_report, methods=["GET"])
    app.add_api_route("/api/predict", post_predict, methods=["POST"])
    app.mount("/reports", StaticFiles(directory=ws.reports, check_dir=False), name="reports")
    return app


def serve(ws_root: Path | str | Workspace, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve the dashboard with uvicorn until interrupted (blocking).

    The dashboard has no authentication (SPEC section 8); a host other than the loopback
    interface exposes upload, cycle start and prediction to the network and is logged as
    a warning. Access logging is off: the page polls /api/status every few seconds."""
    import uvicorn

    if host not in LOOPBACK_HOSTS:
        log.warning("dashboard has no authentication; binding to %s makes it reachable "
                    "from the network", host)
    uvicorn.run(create_app(ws_root), host=host, port=port, log_level="info", access_log=False)
