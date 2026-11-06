"""Cycle runner (SPEC §3, §4): one self-retraining cycle runs the ten steps in order.

Method
* Mutual exclusion. `<workspace>/cycle.lock` is locked with an exclusive, non-blocking
  advisory lock (POSIX `flock`) for the whole cycle and records the owner's pid, host
  and start time. A second `run_cycle` -- from the CLI, watch mode or the dashboard,
  in this or another process -- raises CycleLockedError while the lock is held. The
  operating system drops the lock when its process ends, so a crashed cycle can never
  block the next one; the file itself stays and only the lock state counts.
* Start. A cycle starts only if files are waiting in inbox/ (reserved sub-folders
  processed/, rejected/, unlabeled/ and hidden files excluded); an empty inbox starts
  no cycle and uses up no cycle number. Cycle numbers count up from 1.
* Events. Each step emits "started" (plus progress lines with the same status) and
  finally "done", "failed" or "skipped". Every event is appended as one JSON line to
  events.jsonl and mirrored into status.json, which is replaced atomically (temporary
  file, fsync, rename) so a reader never sees a half-written file. The model version
  fields of status.json are re-read from registry.json on every write.
* Failure policy. The first step that returns ok=False or raises ends the cycle; the
  remaining steps are recorded as skipped. status.json is written in a `finally`
  clause, so it is current after success, failure and interruption.
* Watch mode polls inbox/ and starts a cycle once the waiting files have stayed
  unchanged between two polls (a batch still being copied is not ingested half-way).
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import time
import traceback
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Sequence

from . import STEP_NAMES, STEP_STATUSES
from .steps import STARTED_MESSAGES, STEPS, CycleContext, StepResult, to_jsonable
from .workspace import INBOX_RESERVED, atomic_write_json, utc_now_iso

if TYPE_CHECKING:
    from .workspace import Workspace

log = logging.getLogger(__name__)

LOCK_NAME = "cycle.lock"
STATUS_NAME = "status.json"
EVENTS_NAME = "events.jsonl"
REGISTRY_NAME = "registry.json"
SUMMARY_MAX = 300
WAITING = "waiting"            # status.json state of a step that has not run yet

Listener = Callable[..., None]
"""listener(step, status, message, /, **data): called after every event. The three
event fields are passed positionally and should be positional-only parameters,
because event data may use any key (the drift step's data has its own "status")."""
StepFunction = Callable[[CycleContext], StepResult]
Signature = tuple[tuple[str, int, int], ...]


# ------------------------------------------------------------------------- lock

class CycleLockedError(RuntimeError):
    """Another cycle holds the workspace lock."""

    def __init__(self, path: Path, owner: dict[str, Any] | None) -> None:
        self.path, self.owner = Path(path), owner or {}
        super().__init__(f"another cycle is running in this workspace (lock {self.path}, "
                         f"pid {self.owner.get('pid', '?')}, since {self.owner.get('started', '?')})")


def lock_path(ws: Workspace) -> Path:
    """<workspace>/cycle.lock."""
    return Path(ws.root) / LOCK_NAME


def _parse_owner(text: str) -> dict[str, Any] | None:
    """Owner record stored in the lock file; None when empty or unreadable."""
    try:
        owner = json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        return None
    return owner if isinstance(owner, dict) else None


class CycleLock:
    """Exclusive cross-process lock on one workspace (context manager).

    acquire() never blocks: it raises CycleLockedError when the lock is held, also by
    another CycleLock object of the same process (each object opens the file anew, and
    flock locks belong to the open file description)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._file: IO[str] | None = None

    @property
    def held(self) -> bool:
        """True while this object holds the lock."""
        return self._file is not None

    def acquire(self) -> None:
        """Take the lock and record pid, host and start time in the file."""
        if self._file is not None:
            raise RuntimeError(f"{self.path} is already held by this CycleLock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644), "r+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:                     # EWOULDBLOCK: someone else holds it
            fh.seek(0)
            owner = _parse_owner(fh.read())
            fh.close()
            raise CycleLockedError(self.path, owner) from None
        fh.seek(0)
        fh.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started": utc_now_iso()}, fh)
        fh.flush()
        self._file = fh

    def release(self) -> None:
        """Clear the owner record and drop the lock (no-op when not held)."""
        if self._file is None:
            return
        try:
            self._file.seek(0)
            self._file.truncate()           # a free lock records no owner
            self._file.flush()
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> CycleLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def _pid_alive(pid: int) -> bool:
    """Whether a process with this pid exists on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                 # exists, belongs to another user
        return True
    return True


def lock_owner(ws: Workspace) -> dict[str, Any] | None:
    """Owner of a running cycle, read without touching the lock (display only): the
    record in cycle.lock, unless it is empty or its process no longer exists on this
    host (left behind by a crash)."""
    path = lock_path(ws)
    owner = _parse_owner(path.read_text()) if path.exists() else None
    if owner is None:
        return None
    pid = owner.get("pid")
    if owner.get("host") == socket.gethostname() and isinstance(pid, int) and not _pid_alive(pid):
        return None
    return owner


# --------------------------------------------------------------- status/events

def _one_line(text: str) -> str:
    """Dashboard summaries are single lines of bounded length."""
    lines = str(text).strip().splitlines()
    line = lines[0] if lines else ""
    return line if len(line) <= SUMMARY_MAX else line[:SUMMARY_MAX - 1] + "…"


def _read_json(path: Path) -> Any:
    """Parsed JSON file, or None when it is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None


def write_status_file(ws: Workspace, status: dict[str, Any]) -> None:
    """Replace status.json atomically with strict JSON."""
    atomic_write_json(Path(ws.root) / STATUS_NAME, to_jsonable(status))


def registry_view(ws: Workspace) -> dict[str, Any]:
    """`current_model` and `versions` for status.json, read from registry.json (the
    file format of SPEC §5; no validation, so a status write never fails on it)."""
    reg = _read_json(Path(ws.root) / REGISTRY_NAME)
    if not isinstance(reg, dict):
        return {"current_model": None, "versions": []}
    return {"current_model": reg.get("current"), "versions": reg.get("versions", [])}


def last_cycle_id(ws: Workspace) -> int:
    """Highest cycle number recorded in status.json or events.jsonl (0 if none)."""
    root = Path(ws.root)
    status = _read_json(root / STATUS_NAME)
    best = 0
    if isinstance(status, dict) and isinstance(status.get("cycle"), int):
        best = status["cycle"]
    events = root / EVENTS_NAME
    if events.exists():
        with open(events, encoding="utf-8") as f:
            for line in f:
                try:
                    best = max(best, int(json.loads(line).get("cycle") or 0))
                except (ValueError, AttributeError, TypeError):
                    continue
    return int(best)


def idle_status(cycle: int) -> dict[str, Any]:
    """status.json of a workspace in which no cycle is running."""
    return {"cycle": cycle, "running": False, "demo": False, "result": "idle",
            "started": None, "finished": None, "failed_step": None, "error": None,
            "steps": {name: {"status": WAITING, "summary": ""} for name in STEP_NAMES}}


def refresh_status(ws: Workspace) -> dict[str, Any]:
    """Rewrite status.json with the current registry state (after init, rollback or an
    idle run); creates it for a fresh workspace. Returns the written status."""
    status = _read_json(Path(ws.root) / STATUS_NAME)
    if not isinstance(status, dict):
        status = idle_status(last_cycle_id(ws))
    status.update(registry_view(ws), updated=utc_now_iso())
    write_status_file(ws, status)
    return to_jsonable(status)


def read_status(ws: Workspace) -> dict[str, Any]:
    """Current status for readers such as the CLI and the dashboard: status.json
    merged with the registry; a cycle that claims to be running while no live process
    holds the lock (crash) is reported as not running, result "interrupted"."""
    status = _read_json(Path(ws.root) / STATUS_NAME)
    if not isinstance(status, dict):
        status = idle_status(last_cycle_id(ws))
    if status.get("running") and lock_owner(ws) is None:
        status.update(running=False, result="interrupted")
    status.update(registry_view(ws))
    return to_jsonable(status)


class EventLog:
    """Persists the events of one cycle (events.jsonl) and mirrors them into
    status.json. `listener(step, status, message, /, **data)` -- e.g. the CLI
    printer -- is called after each event with the event fields as positional
    arguments and the data as keyword arguments; its exceptions are logged, never
    propagated."""

    def __init__(self, ws: Workspace, cycle: int, demo: bool,
                 listener: Listener | None = None) -> None:
        self.ws, self.root, self.cycle, self.listener = ws, Path(ws.root), cycle, listener
        self.status = {**idle_status(cycle), "running": True, "demo": demo,
                       "result": "running", "started": utc_now_iso()}
        self.write_status()

    def emit(self, step: str, status: str, message: str, /, **data: Any) -> None:
        """Record one event (the signature of CycleContext.emit). The event fields
        are positional-only, so `data` may contain any key, 'status' included."""
        self.record(step, status, message, data)

    def record(self, step: str, status: str, message: str, data: dict[str, Any]) -> None:
        """Append the event to events.jsonl, update status.json, notify the listener."""
        if step not in STEP_NAMES:
            raise ValueError(f"unknown step {step!r}")
        if status not in STEP_STATUSES:
            raise ValueError(f"unknown event status {status!r}")
        summary = _one_line(message)
        event = {"ts": utc_now_iso(), "cycle": self.cycle, "step": step, "status": status,
                 "summary": summary, "data": to_jsonable(data)}
        with open(self.root / EVENTS_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.status["steps"][step] = {"status": status, "summary": summary}
        self.write_status()
        if self.listener is not None:
            try:
                self.listener(step, status, summary, **data)
            except Exception:                       # a display problem must not end a cycle
                log.exception("event listener failed")

    def finish(self, result: str, failed_step: str | None, error: str | None) -> None:
        """Final status: result "done", "failed" or "interrupted"."""
        self.status.update(running=False, finished=utc_now_iso(), failed_step=failed_step,
                           error=error, result=result)
        self.write_status()

    def write_status(self) -> None:
        """status.json = this cycle's state + the registry's model versions."""
        write_status_file(self.ws, {**self.status, **registry_view(self.ws), "updated": utc_now_iso()})


# ------------------------------------------------------------------------ cycle

def inbox_signature(ws: Workspace) -> Signature:
    """(relative path, size, mtime in ns) of every file waiting in inbox/ -- reserved
    sub-folders (processed, rejected, unlabeled) and hidden files excluded. Empty when
    nothing is waiting. Reads the directory only (no side effects)."""
    inbox = Path(ws.root) / "inbox"
    if not inbox.is_dir():
        return ()
    items = []
    for top in sorted(inbox.iterdir()):
        if top.name in INBOX_RESERVED or top.name.startswith("."):
            continue
        for p in [top] if top.is_file() else sorted(top.rglob("*")):
            rel = p.relative_to(inbox)
            if p.is_file() and not any(part.startswith(".") for part in rel.parts):
                st = p.stat()
                items.append((rel.as_posix(), st.st_size, st.st_mtime_ns))
    return tuple(items)


def _run_step(ctx: CycleContext, events: EventLog, name: str, fn: StepFunction) -> StepResult:
    """Run one step: "started" event, call, final event with the step's duration.
    An exception becomes a failed StepResult (traceback in the event data);
    KeyboardInterrupt/SystemExit are recorded as failed and propagate."""
    events.record(name, "started", STARTED_MESSAGES.get(name, name), {})
    t0 = time.monotonic()
    try:
        result = fn(ctx)
        if not isinstance(result, StepResult):
            raise TypeError(f"step {name} returned {type(result).__name__}, not StepResult")
    except Exception as exc:
        log.exception("step %s failed", name)
        result = StepResult(False, f"{type(exc).__name__}: {exc}",
                            {"traceback": traceback.format_exc()})
    except BaseException as exc:
        events.record(name, "failed", f"interrupted ({type(exc).__name__})", {})
        raise
    events.record(name, result.status, result.summary,
                  {**result.data, "seconds": round(time.monotonic() - t0, 1)})
    return result


def _check_steps(steps: Sequence[tuple[str, StepFunction]]) -> None:
    """The steps of a cycle are exactly STEP_NAMES, in this order."""
    names = [name for name, _ in steps]
    if names != list(STEP_NAMES):
        raise ValueError(f"steps must be {list(STEP_NAMES)} in this order, got {names}")


def run_cycle(ws: Workspace, demo: bool = False, emit: Listener | None = None, *,
              steps: Sequence[tuple[str, StepFunction]] | None = None) -> dict[str, Any]:
    """Run one cycle over the batches waiting in inbox/ and return its summary.

    `emit(step, status, message, /, **data)` receives every event (optional). `steps`
    replaces the step functions (tests); the names must be STEP_NAMES in order.
    Raises CycleLockedError when another cycle is running. Returns {"status": "idle"}
    without starting a cycle when nothing is waiting in inbox/; otherwise "done" or
    "failed" with the failing step, the step states and the cycle's main results."""
    step_fns = STEPS if steps is None else steps
    _check_steps(step_fns)
    ws.ensure()
    with CycleLock(lock_path(ws)):
        if not inbox_signature(ws):
            refresh_status(ws)
            return {"cycle": None, "status": "idle", "summary": "inbox is empty: nothing to do"}
        cycle = last_cycle_id(ws) + 1
        events = EventLog(ws, cycle, demo, listener=emit)
        ctx: CycleContext | None = None
        current: str | None = None            # step being run
        failed: str | None = None
        error: str | None = None
        outcome = "failed"
        try:
            ctx = CycleContext(ws=ws, cfg=ws.load_config(), cycle=cycle, demo=demo,
                               emit=events.emit)
            for current, fn in step_fns:
                result = _run_step(ctx, events, current, fn)
                ctx.results[current] = result
                if not result.ok:
                    failed = current
                    break
            else:
                outcome = "done"
        except Exception as exc:             # outside a step's own code: config, event I/O
            log.exception("cycle %d failed", cycle)
            failed, error = current, f"{type(exc).__name__}: {exc}"
        except BaseException as exc:         # KeyboardInterrupt, SystemExit
            failed, error, outcome = current, f"interrupted ({type(exc).__name__})", "interrupted"
            raise
        finally:
            if outcome != "done":
                _close_unfinished(events, failed, error)
            events.finish(outcome, failed, error)
    return _cycle_result(cycle, demo, outcome, failed, error, events, ctx)


def _close_unfinished(events: EventLog, failed: str | None, error: str | None) -> None:
    """After a failure, give every unfinished step a final state: a step left
    "started" failed (with the error), a step still waiting was skipped."""
    if error:
        reason = f"not run: {_one_line(error)}"
    else:
        reason = f"not run: step '{failed}' failed" if failed else "not run: cycle failed"
    for name in STEP_NAMES:
        state = events.status["steps"][name]["status"]
        if state not in (WAITING, "started"):
            continue
        try:
            if state == "started":
                events.record(name, "failed", error or "failed", {})
            else:
                events.record(name, "skipped", reason, {})
        except Exception:                    # never mask the original failure
            log.exception("cannot record the final state of step %s", name)


def _cycle_result(cycle: int, demo: bool, outcome: str, failed: str | None, error: str | None,
                  events: EventLog, ctx: CycleContext | None) -> dict[str, Any]:
    """JSON-ready summary returned by run_cycle."""
    out: dict[str, Any] = {"cycle": cycle, "demo": demo, "status": outcome,
                           "failed_step": failed, "error": error,
                           "steps": events.status["steps"]}
    if ctx is not None:
        out.update(dataset_version=ctx.dataset_version, model_version=ctx.model_version,
                   metrics=ctx.metrics, gate=ctx.artifacts.get("gate"),
                   report_pdf=ctx.artifacts.get("report_pdf"),
                   report_html=ctx.artifacts.get("report_html"))
    return to_jsonable(out)


# ------------------------------------------------------------------------ watch

def _watch_finished(n_cycles: int, polls: int, max_cycles: int | None,
                    max_polls: int | None) -> bool:
    return (max_cycles is not None and n_cycles >= max_cycles) or \
        (max_polls is not None and polls >= max_polls)


def watch(ws: Workspace, interval: float = 60.0, demo: bool = False,
          emit: Listener | None = None, max_cycles: int | None = None,
          max_polls: int | None = None,
          sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    """Watch mode (F9): poll inbox/ every `interval` seconds and run a cycle when files
    are waiting and the inbox is unchanged since the previous poll, so a batch that is
    still being copied is not ingested half-way. The inbox content a cycle started
    from is not processed again until it changes -- after a failed cycle the batch
    stays in inbox/ and is retried only once the user has replaced or fixed it. A
    cycle started elsewhere (lock held) is waited out; an unexpected error is logged
    and watching continues. Stops after `max_cycles` cycles or `max_polls` polls
    (None: never). Returns the results of the cycles run."""
    if interval < 0:
        raise ValueError("interval must be >= 0")
    results: list[dict[str, Any]] = []
    previous: Signature | None = None    # inbox content at the previous poll
    handled: Signature | None = None     # inbox content the last cycle started from
    polls = 0
    while not _watch_finished(len(results), polls, max_cycles, max_polls):
        polls += 1
        current = inbox_signature(ws)
        if current and current == previous and current != handled:
            try:
                result = run_cycle(ws, demo=demo, emit=emit)
            except CycleLockedError as exc:
                log.info("%s; retrying at the next poll", exc)
            except Exception:
                log.exception("watch: the cycle could not be run")
                handled = current
            else:
                handled = current
                if result["status"] != "idle":
                    results.append(result)
        previous = current
        if not _watch_finished(len(results), polls, max_cycles, max_polls):
            sleep(interval)
    return results
