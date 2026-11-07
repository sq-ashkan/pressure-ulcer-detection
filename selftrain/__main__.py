"""Command-line interface of the self-retraining system (SPEC §8, feature F9).

    python -m selftrain <command> [--workspace PATH] [options]

Every command works on one workspace directory (default ./system_ws).

  init            create the workspace; --from DIR makes the thesis dataset (data/seg)
                  the pool and dataset version 0, --weights W registers the model W
                  (trained on that split) as model version 0 and makes it current
  run [--demo]    run one cycle over the batches waiting in inbox/
  watch           service mode: poll inbox/ and run a cycle when a batch has arrived
  status          state of the last cycle, current model and version history
  report          re-render the report of the cycle that produced model version N
  rank-unlabeled  rank the photos in inbox/unlabeled/ by model uncertainty (F14)
  rollback        make an earlier model version current again (F7)
  dashboard       web dashboard on http://127.0.0.1:8080 (FastAPI, no login)
  reproduce       print, and with --execute run, the thesis pipeline in order (F15)

Exit status: 0 success (an idle run included), 1 failure (failed cycle, a cycle
already running, invalid input), 2 command-line usage error.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import ROOT

if TYPE_CHECKING:
    from .system.registry import VersionRecord
    from .system.workspace import Workspace

DEFAULT_WORKSPACE = Path("system_ws")
DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"


# ------------------------------------------------------------------ helpers

def print_event(step: str, status: str, message: str, /, **data: Any) -> None:
    """CLI listener of run_cycle: one line per event. The three event fields are
    positional-only, so event data may use any key -- e.g. the drift step's data
    carries the drift report's own "status" ("tested" or "skipped")."""
    print(f"{time.strftime('%H:%M:%S')}  {step:<8} {status:<8} {message}", flush=True)


def open_workspace(args: argparse.Namespace) -> Workspace:
    """The workspace named by --workspace (its directories are created if missing)."""
    from .system.workspace import Workspace
    return Workspace(args.workspace).ensure()


def _fmt(x: Any, spec: str = ".3f") -> str:
    from .system.steps import fmt
    return fmt(x, spec)


def _ci(ci: Any) -> str:
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def format_versions(versions: Sequence[dict[str, Any]], current: int | None) -> list[str]:
    """Version history table: version, dataset, architecture, training images, test
    Dice with 95 % CI, false-positive rate, gate decision; * marks the current model.
    Metrics are read under the registry's names (registry.metrics_from_evaluation)."""
    if not versions:
        return ["  no model versions registered"]
    lines = [f"  {'':1} {'model':>5} {'data':>4} {'arch':<8} {'train':>6} "
             f"{'test Dice [95 % CI]':<24} {'FPR':>5}  gate"]
    for v in versions:
        m, g = v.get("metrics") or {}, v.get("gate") or {}
        dice = f"{_fmt(m.get('test_dice_mean'))} {_ci(m.get('test_dice_ci95'))}".strip()
        verdict = "promoted" if g.get("promoted") else "not promoted"
        mark = "*" if v.get("version") == current else ""
        lines.append(f"  {mark:1} v{v.get('version')!s:>4} v{v.get('dataset_version')!s:>3} "
                     f"{v.get('architecture', '?'):<8} {v.get('train_images', '?')!s:>6} "
                     f"{dice:<24} {_fmt(m.get('test_fpr'), '.2f'):>5}  {verdict}")
    return lines


def format_status(ws: Workspace, status: dict[str, Any]) -> str:
    """Human-readable form of read_status()."""
    from .system.cycle import lock_owner
    lines = [f"workspace {ws.root}"]
    cycle = status.get("cycle") or 0
    if cycle:
        state = "running" if status.get("running") else status.get("result", "?")
        demo = " (demo)" if status.get("demo") else ""
        lines.append(f"cycle {cycle}{demo}: {state}; started {status.get('started')}, "
                     f"finished {status.get('finished')}")
        if status.get("error"):
            lines.append(f"  error: {status['error']}")
        for name, step in (status.get("steps") or {}).items():
            lines.append(f"  {name:<8} {step.get('status', '?'):<8} {step.get('summary', '')}")
    else:
        lines.append("no cycle has run yet")
    owner = lock_owner(ws)
    if owner:
        lines.append(f"a cycle is running: pid {owner.get('pid')} since {owner.get('started')}")
    current = status.get("current_model")
    lines.append(f"current model: {'none' if current is None else f'v{current}'}")
    lines += format_versions(status.get("versions") or [], current)
    return "\n".join(lines)


# --------------------------------------------------------------------- init

def register_initial_model(ws: Workspace, weights: Path, evaluate: bool = True,
                           log: Any = print) -> VersionRecord:
    """Register `weights` -- a model trained on the thesis split, i.e. on dataset
    version 0 -- as model version 0 and make it current.

    Method: the file is copied into models/v0/ (best.pt for U-Net, selected.pt for
    YOLOv8-seg), so the workspace does not depend on the run directory it came from.
    With `evaluate` the model is evaluated on datasets/v0 with the pre-registered
    protocol (threshold chosen on val, frozen test once, extra negatives) and
    models/v0/evaluation.json is stored: its per-image test Dice is the reference
    against which the gate of the first cycle compares. Without evaluation the
    checkpoint is only loaded once to verify it, and the gate evaluates it on demand.
    The record's gate entry says that the version was registered by init (there is
    nothing to compare with); finally its model card is written."""
    from .system import modelcard, registry
    from .system.steps import (WEIGHTS_FILE, count_photographs, evaluate_version,
                               load_predictor, train_names, workspace_relative)
    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights file {weights} does not exist")
    if registry.load(ws)["versions"]:
        raise FileExistsError(f"{ws.root} already has model versions; "
                              f"the initial model can only be registered once")
    data_root = ws.dataset_dir(0)
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset version 0 is missing ({data_root}): "
                                f"initialise the pool with --from first")
    arch = ws.load_config().architecture
    target = ws.model_dir(0) / WEIGHTS_FILE[arch]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, target)
    try:
        if evaluate:
            log(f"evaluating {weights.name} on dataset v0 (frozen test split) ...")
            res = evaluate_version(ws, 0, arch, target, data_root, dataset_version=0, log_fn=log)
            metrics = registry.metrics_from_evaluation(res)
        else:
            load_predictor(arch, target)
            metrics = {}
    except Exception as exc:
        raise ValueError(f"cannot use {weights} as a {arch} model: {type(exc).__name__}: {exc}") from exc
    names = train_names(data_root)
    record = registry.VersionRecord(
        version=0, dataset_version=0, architecture=arch, train_images=len(names),
        train_photographs=count_photographs(data_root, names), metrics=metrics,
        gate={"promoted": True, "source_weights": str(weights),
              "reason": "initial model registered by init (thesis split; no gate: "
                        "there is no earlier model to compare with)"},
        weights=workspace_relative(ws, target))
    registry.add_version(ws, record, make_current=True)
    card = Path(modelcard.write(ws, 0))
    return registry.update_version(ws, 0, model_card=workspace_relative(ws, card))


def cmd_init(args: argparse.Namespace) -> int:
    """init: workspace layout, configuration, pool + dataset v0, model v0."""
    import dataclasses
    from .system.cycle import refresh_status
    ws = open_workspace(args)
    if args.architecture:
        ws.save_config(dataclasses.replace(ws.load_config(), architecture=args.architecture))
    if args.from_dir:
        from .system import dataset
        dataset.init_from_seg(ws, Path(args.from_dir), copy=args.copy, log=print)
    if args.weights:
        rec = register_initial_model(ws, Path(args.weights), evaluate=not args.skip_evaluation)
        dice = (rec.metrics or {}).get("test_dice_mean")
        print(f"model v0 registered and current ({rec.architecture}, {rec.train_images} "
              f"training images" + (f", test Dice {_fmt(dice)}" if dice is not None else "") + ")")
    refresh_status(ws)
    print(f"workspace ready: {ws.root} (architecture {ws.load_config().architecture})")
    return 0


# ------------------------------------------------------------ run and watch

def print_cycle_result(result: dict[str, Any]) -> None:
    """Final line(s) of `run`."""
    if result["status"] == "idle":
        print(result.get("summary", "inbox is empty: nothing to do"))
        return
    line = f"cycle {result['cycle']}: {result['status']}"
    if result.get("failed_step"):
        line += f" at step {result['failed_step']}"
    if result.get("error"):
        line += f" ({result['error']})"
    print(line)
    gate = result.get("gate")
    if gate:
        print(f"gate: {gate.get('reason')}")
    for key in ("report_html", "report_pdf"):
        if result.get(key):
            print(f"report: {result[key]}")


def cmd_run(args: argparse.Namespace) -> int:
    """run: one cycle; exit 1 if it failed."""
    from .system.cycle import run_cycle
    result = run_cycle(open_workspace(args), demo=args.demo, emit=print_event)
    print_cycle_result(result)
    return 0 if result["status"] in ("done", "idle") else 1


def cmd_watch(args: argparse.Namespace) -> int:
    """watch: poll inbox/ until interrupted (or --max-cycles cycles have run)."""
    from .system.cycle import watch
    ws = open_workspace(args)
    print(f"watching {ws.inbox} every {args.interval:g} s (Ctrl-C to stop)", flush=True)
    try:
        results = watch(ws, interval=args.interval, demo=args.demo, emit=print_event,
                        max_cycles=args.max_cycles)
    except KeyboardInterrupt:
        print("\nwatch stopped")
        return 0
    for result in results:
        print_cycle_result(result)
    return 0 if all(r["status"] == "done" for r in results) else 1


# ------------------------------------------------------------ status, report

def cmd_status(args: argparse.Namespace) -> int:
    """status: last cycle, current model, versions (or the raw JSON)."""
    from .system.cycle import read_status
    ws = open_workspace(args)
    status = read_status(ws)
    print(json.dumps(status, indent=2) if args.json else format_status(ws, status))
    return 0


def find_cycle_summary(ws: Workspace, version: int) -> Path | None:
    """reports/cycle_{N}.json of the cycle that produced model `version` (None if no
    cycle did, e.g. for the initial model of init)."""
    for path in sorted(ws.reports.glob("cycle_*.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(summary, dict) and summary.get("model_version") == version:
            return path
    return None


def cmd_report(args: argparse.Namespace) -> int:
    """report --version N: rewrite the model card of the version from the stored files
    (nothing is retrained), then re-render the cycle report from its stored summary, so
    the report embeds the new card; a version without a cycle (init's v0) has only the
    card."""
    from .system import modelcard, registry
    from .system.steps import workspace_relative
    ws = open_workspace(args)
    registry.get_version(ws, args.version)              # KeyError if not registered
    card = Path(modelcard.write(ws, args.version))
    registry.update_version(ws, args.version, model_card=workspace_relative(ws, card))
    summary_path = find_cycle_summary(ws, args.version)
    if summary_path is None:
        print(f"model v{args.version} was not produced by a cycle (registered by init): "
              f"no cycle report; model card: {card}")
        return 0
    from .system import report
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = report.write_report(ws, summary)
    print(f"report of cycle {summary.get('cycle')} (model v{args.version}):")
    print(f"  model card: {card}")
    for key in ("html", "pdf"):
        if paths.get(key):
            print(f"  {key}: {paths[key]}")
    return 0


# ------------------------------------------------------ active, rollback, web

def cmd_rank_unlabeled(args: argparse.Namespace) -> int:
    """rank-unlabeled: active-learning ranking of inbox/unlabeled/."""
    from .system.active import rank_unlabeled
    ws = open_workspace(args)
    ranking = rank_unlabeled(ws, top_k=args.top_k)
    if ranking.empty:
        print(f"no photos in {ws.unlabeled}")
        return 0
    print(f"{'rank':>4} {'uncertainty':>11} {'area':>6} {'stage':>5}  photo")
    for rank, row in enumerate(ranking.head(args.top_k).itertuples(index=False), start=1):
        print(f"{rank:>4} {row.uncertainty:>11.3f} {row.predicted_area:>6.3f} "
              f"{row.predicted_stage:>5}  {row.path}")
    print(f"{len(ranking)} photos ranked: {ws.active / 'ranking.csv'} "
          f"(thumbnails of the top {min(args.top_k, len(ranking))} in {ws.active / 'thumbnails'})")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """rollback --to N: make version N current (refused while a cycle runs)."""
    from .system import gate
    rec = gate.rollback(open_workspace(args), args.to)
    print(f"current model: v{rec.version} ({rec.architecture}, dataset v{rec.dataset_version}, "
          f"test Dice {_fmt((rec.metrics or {}).get('test_dice_mean'))})")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """dashboard: selftrain.system.dashboard.serve (uvicorn, blocking until Ctrl-C)."""
    from .system import dashboard
    ws = open_workspace(args)
    print(f"dashboard: http://{args.host}:{args.port}/  (workspace {ws.root}; Ctrl-C to stop)",
          flush=True)
    dashboard.serve(ws, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------- reproduce

@dataclass(frozen=True)
class PipelineStep:
    """One command of the thesis pipeline: a script of the repository and its
    arguments, run from the repository root with the current Python interpreter."""
    title: str
    script: str
    args: tuple[str, ...] = ()

    def command(self, python: str = sys.executable) -> list[str]:
        """argv of the step (a shell script, *.sh, is run with bash)."""
        runner = "bash" if self.script.endswith(".sh") else python
        return [runner, self.script, *self.args]


def _seed_dir(seed: int) -> str:
    """Run directory of a comparison seed: runs/compare (seed 0), runs/compare_seed{N}."""
    return "runs/compare" if seed == 0 else f"runs/compare_seed{seed}"


def _comparison_steps(seeds: tuple[int, ...] = (0, 1, 2)) -> tuple[PipelineStep, ...]:
    """Training of both models and the pre-registered comparison, per seed (seed 0 writes
    results/comparison, seed N results/comparison_seed{N})."""
    steps: list[PipelineStep] = []
    for seed in seeds:
        run = _seed_dir(seed)
        for model in ("unet", "yoloseg"):
            steps.append(PipelineStep(f"train {model}, seed {seed} (WP2)", "scripts/train_model.py",
                                      ("--model", model, "--seed", str(seed),
                                       "--out", f"{run}/{model}")))
    for seed in seeds:
        run = _seed_dir(seed)
        out = "results/comparison" if seed == 0 else f"results/comparison_seed{seed}"
        steps.append(PipelineStep(f"pre-registered model comparison, seed {seed} (WP2)",
                                  "scripts/compare_models.py",
                                  ("--unet", f"{run}/unet/best.pt",
                                   "--yolo", f"{run}/yoloseg/weights/selected.pt", "--out", out)))
    return tuple(steps)


THESIS_PIPELINE: tuple[PipelineStep, ...] = (
    PipelineStep("segmentation dataset data/seg (WP1)", "scripts/build_seg_dataset.py"),
    *_comparison_steps(),
    PipelineStep("external test on PIID, U-Net (F13)", "scripts/external_piid.py",
                 ("--model", "unet", "--weights", "runs/compare/unet/best.pt")),
    PipelineStep("external test on PIID, YOLOv8-seg (F13)", "scripts/external_piid.py",
                 ("--model", "yoloseg", "--weights", "runs/compare/yoloseg/weights/selected.pt")),
    PipelineStep("inter-annotator agreement (label-noise finding)",
                 "scripts/annotator_agreement.py"),
    PipelineStep("data growth: growth batches, selftrain init and three system cycles (WP4)",
                 "scripts/data_growth.sh"),
    PipelineStep("learning curve (25/50/75 % of train, seeds 0-2)", "scripts/learning_curve.py"),
    # review evidence (each script documents its command line and outputs)
    PipelineStep("source overlap of the photographs (data/seg/manifest.csv)",
                 "scripts/source_overlap.py"),
    PipelineStep("provenance check: OCR of copyright marks and watermarks",
                 "scripts/provenance_check.py"),
    PipelineStep("PIID: stage accuracy over all photographs, share marked (F13)",
                 "scripts/external_piid_extra.py"),
    PipelineStep("power of the promotion gate", "scripts/gate_power.py"),
    PipelineStep("learning-curve fit", "scripts/learning_curve_fit.py"),
    PipelineStep("robustness of the model comparison over the seeds",
                 "scripts/comparison_robustness.py"),
    PipelineStep("failure analysis of the test photographs", "scripts/failure_analysis.py"),
    PipelineStep("leakage sensitivity (cross-source near-duplicates)",
                 "scripts/leakage_sensitivity.py"),
    PipelineStep("engineering evidence: model size, training settings, code inventory",
                 "scripts/engineering_evidence.py"),
    PipelineStep("thesis figures and number macros", "scripts/make_figures.py"),
)
"""The thesis experiments in dependency order (docs/ROADMAP.md, WP1-WP4)."""


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def pipeline_numbers(root: Path = ROOT) -> list[str]:
    """Headline numbers of the result files the pipeline writes (those present)."""
    res = Path(root) / "results"
    lines = []
    manifest = Path(root) / "data" / "seg" / "manifest.csv"
    if manifest.is_file():                     # post-audit dataset, as evaluated
        with open(manifest, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        per_split = {sp: len({r["photograph"] for r in rows if r["split"] == sp})
                     for sp in ("train", "val", "test")}
        lines.append(f"dataset (data/seg/manifest.csv): {len({r['photograph'] for r in rows})} "
                     f"photographs, {len(rows)} images; photographs per split {per_split}")
    agree = _read_result(res / "annotator_agreement.json")
    if agree:
        lines.append(f"annotator agreement ({agree.get('matched_photographs')} photographs): "
                     f"binary Dice {_fmt(agree.get('binary_dice_mean'))} "
                     f"{_ci(agree.get('binary_dice_ci95'))}")
    comp = _read_result(res / "comparison" / "summary.json")
    if comp:
        paired = comp.get("paired_test") or {}
        lines.append(f"model comparison: winner {comp.get('winner')}; test Dice U-Net - YOLO "
                     f"{_fmt(paired.get('mean_diff'), '+.4f')} {_ci(paired.get('ci95'))}")
        for model in ("unet", "yoloseg"):
            test = (comp.get(model) or {}).get("test") or {}
            if test:
                lines.append(f"  {model}: test Dice {_fmt(test.get('dice_mean'))} "
                             f"{_ci(test.get('dice_ci95'))}, FPR "
                             f"{_fmt(test.get('false_positive_rate'), '.3f')}")
    extra = _read_result(res / "external" / "piid_extra.json") or {}
    for model in ("unet", "yoloseg"):
        ext = _read_result(res / "external" / f"external_piid_{model}.json")
        m = (extra.get("models") or {}).get(model) or {}
        if ext or m:
            ext = ext or {}
            n = extra.get("piid_n", ext.get("external_n"))
            marked = m.get("photographs_marked")
            share = marked / n if isinstance(marked, int) and n else None
            lines.append(f"PIID external test ({model}, n = {n}): share marked {_fmt(share)} "
                         f"({marked}/{n}), stage accuracy over all photographs "
                         f"{_fmt(m.get('stage_accuracy_all'))} {_ci(m.get('stage_accuracy_all_ci95'))}, "
                         f"quadratic kappa (marked) "
                         f"{_fmt(m.get('quadratic_kappa_detected', ext.get('quadratic_kappa')))}")
    return lines


def cmd_reproduce(args: argparse.Namespace) -> int:
    """reproduce: list the pipeline; with --execute run it in order, stopping at the
    first failing command, and print the resulting numbers."""
    print("thesis pipeline (run from the repository root):")
    for i, step in enumerate(THESIS_PIPELINE, start=1):
        missing = "" if (ROOT / step.script).is_file() else "   [script missing]"
        print(f"{i:>2}. {step.title}{missing}\n    {shlex.join(step.command())}")
    if args.execute:
        missing = [s.script for s in THESIS_PIPELINE if not (ROOT / s.script).is_file()]
        if missing:
            print(f"error: cannot reproduce, missing scripts: {', '.join(missing)}", file=sys.stderr)
            return 1
        for i, step in enumerate(THESIS_PIPELINE, start=1):
            print(f"\n[{i}/{len(THESIS_PIPELINE)}] {step.title}", flush=True)
            done = subprocess.run(step.command(), cwd=ROOT, check=False)
            if done.returncode != 0:
                print(f"error: step {i} ({step.script}) exited with {done.returncode}",
                      file=sys.stderr)
                return 1
    else:
        print("\ndry run: add --execute to run these commands (hours of GPU time)")
    numbers = pipeline_numbers(ROOT)
    print("\nresults:" if numbers else "\nno results yet")
    for line in numbers:
        print(f"  {line}")
    return 0


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    """argparse parser of all subcommands (each accepts --workspace)."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE, metavar="PATH",
                        help="workspace directory (default: ./system_ws)")
    common.add_argument("-v", "--verbose", action="store_true",
                        help="also print the log of the library (level INFO)")
    parser = argparse.ArgumentParser(
        prog="python -m selftrain",
        description="Self-retraining pressure-ulcer segmentation system (research "
                    "prototype, not a medical device).")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, func: Any, summary: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=[common], help=summary, description=summary)
        p.set_defaults(func=func)
        return p

    p = add("init", cmd_init, "create a workspace; optionally import the thesis dataset "
                              "as pool/dataset v0 and register a model as v0")
    p.add_argument("--from", dest="from_dir", metavar="DIR",
                   help="build_seg_dataset output (e.g. data/seg) that becomes the pool and "
                        "dataset version 0")
    p.add_argument("--architecture", choices=["unet", "yoloseg"],
                   help="architecture of every cycle (stored in config.yaml)")
    p.add_argument("--weights", metavar="W",
                   help="checkpoint trained on that split, registered as model v0")
    p.add_argument("--copy", action="store_true",
                   help="copy the dataset files into the pool instead of linking them")
    p.add_argument("--skip-evaluation", action="store_true",
                   help="register --weights without evaluating it now (the first gate "
                        "evaluates it on demand)")

    p = add("run", cmd_run, "run one cycle over the batches waiting in inbox/")
    p.add_argument("--demo", action="store_true",
                   help="demo mode: demo_epochs epochs on at most demo_max_train_images images")

    p = add("watch", cmd_watch, "poll inbox/ and run a cycle whenever a batch has arrived")
    p.add_argument("--interval", type=float, default=60.0, metavar="S",
                   help="seconds between two polls (default: 60)")
    p.add_argument("--demo", action="store_true", help="run every cycle in demo mode")
    p.add_argument("--max-cycles", type=int, default=None, metavar="N",
                   help="stop after N cycles (default: run until Ctrl-C)")

    p = add("status", cmd_status, "show the last cycle, the current model and all versions")
    p.add_argument("--json", action="store_true", help="print status.json as JSON")

    p = add("report", cmd_report, "rewrite the model card of a version and re-render its cycle report")
    p.add_argument("--version", type=int, required=True, metavar="N", help="model version")

    p = add("rank-unlabeled", cmd_rank_unlabeled,
            "rank the photos in inbox/unlabeled/ by uncertainty of the current model")
    p.add_argument("--top-k", type=int, default=20, metavar="K",
                   help="thumbnails and printed rows for the K most uncertain photos (default: 20)")

    p = add("rollback", cmd_rollback, "make an earlier model version current again")
    p.add_argument("--to", type=int, required=True, metavar="N", help="model version")

    p = add("dashboard", cmd_dashboard, "serve the web dashboard")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="port (default: 8080)")
    p.add_argument("--host", default=DEFAULT_HOST, help="interface (default: 127.0.0.1)")

    p = add("reproduce", cmd_reproduce,
            "print the thesis pipeline in order; run it with --execute")
    p.add_argument("--execute", action="store_true",
                   help="run the commands (stops at the first failure) and print the numbers")
    return parser


def _user_error(exc: Exception) -> str | None:
    """Message for an error caused by the input or the workspace state; None for a
    programming error (which keeps its traceback)."""
    from .system.cycle import CycleLockedError
    if isinstance(exc, KeyError):
        return str(exc.args[0]) if exc.args else "not found"
    if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, CycleLockedError)):
        return str(exc)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of `python -m selftrain`; returns the exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args))
    except Exception as exc:
        message = _user_error(exc)
        if message is None:
            raise
        print(f"error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
