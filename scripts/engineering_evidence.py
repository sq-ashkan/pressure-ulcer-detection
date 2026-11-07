"""Engineering evidence for the thesis: model size, training settings, protocol timing, code size.

Command line (CPU only, deterministic; GPU jobs may be running):
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/engineering_evidence.py

Outputs:
    results/model_complexity.json       parameters and GFLOPs at 512x512 of the comparison
                                        U-Net (runs/compare/unet/best.pt) and YOLOv8s-seg
                                        (runs/compare/yoloseg/weights/selected.pt)
    results/training_configs.json       hyper-parameters actually used and epochs actually trained
    results/preregistration_record.json commits that added/changed the evaluation protocol in
                                        docs/ROADMAP.md, SHA-256 of the current protocol section,
                                        time of the first test evaluation
    results/code_inventory.json         non-blank, non-comment Python lines per area
                                        (inherited practical project vs. written for the thesis)

GFLOPs are counted with thop (multiply-accumulates x 2), the same convention as
ultralytics' own model.info(); one 1x3x512x512 input, inference mode.
"""
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SELFTRAIN_DEVICE", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
UNET_DIR = ROOT / "runs/compare/unet"
YOLO_DIR = ROOT / "runs/compare/yoloseg"
IMG = 512


def dump(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    print(f"wrote results/{name}")


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------- (a) complexity
def model_complexity():
    import torch
    import thop
    from selftrain.unet import UNetPredictor
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    torch.set_grad_enabled(False)
    out = {"input": f"1x3x{IMG}x{IMG}", "flop_counter": f"thop {thop.__version__ if hasattr(thop, '__version__') else ''}".strip(),
           "flop_convention": "GFLOPs = 2 x multiply-accumulates (as ultralytics model.info)"}

    unet = UNetPredictor(UNET_DIR / "best.pt").model.cpu().eval()
    macs, _ = thop.profile(unet, inputs=(torch.zeros(1, 3, IMG, IMG),), verbose=False)
    out["unet"] = {"weights": rel(UNET_DIR / "best.pt"),
                   "architecture": "smp.Unet, ResNet-34 encoder, 5 output classes",
                   "parameters": sum(p.numel() for p in unet.parameters()),
                   "parameters_encoder": sum(p.numel() for p in unet.encoder.parameters()),
                   "gflops_512": round(2 * macs / 1e9, 2)}

    y = YOLO(str(YOLO_DIR / "weights/selected.pt")).model.cpu().float().eval()
    out["yoloseg"] = {"weights": rel(YOLO_DIR / "weights/selected.pt"),
                      "architecture": "YOLOv8s-seg (ultralytics), 4 classes",
                      "parameters": sum(p.numel() for p in y.parameters()),
                      "gflops_512": round(get_flops(y, IMG), 2)}
    out["ratio_parameters_unet_over_yolo"] = round(out["unet"]["parameters"] / out["yoloseg"]["parameters"], 2)
    out["ratio_gflops_unet_over_yolo"] = round(out["unet"]["gflops_512"] / out["yoloseg"]["gflops_512"], 2)
    dump("model_complexity.json", out)
    return out


# ---------------------------------------------------------------- (b) training configs
YOLO_KEYS = ["epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
             "warmup_epochs", "cos_lr", "mosaic", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "fliplr",
             "flipud", "degrees", "scale", "translate", "patience", "seed", "nbs", "deterministic", "pretrained"]


def training_configs():
    import yaml
    ucfg = json.loads((UNET_DIR / "config.json").read_text())
    hist = list(csv.DictReader(open(UNET_DIR / "history.csv")))
    best = max(hist, key=lambda r: float(r["val_dice"]))
    unet = {"source": ["runs/compare/unet/config.json", "selftrain/unet.py"], **ucfg,
            "optimizer": "AdamW", "lr_schedule": "CosineAnnealingLR(T_max=epochs), stepped per epoch",
            "loss": f"{ucfg['ce_weight']} x cross-entropy + {ucfg['dice_weight']} x multiclass Dice",
            "augmentation": "albumentations, online (selftrain/data.py train_augment)",
            "selection_rule": "best val binary Dice (early stopping on it)",
            "epochs_trained": len(hist),
            "best_epoch": int(best["epoch"]), "best_val_binary_dice": round(float(best["val_dice"]), 4),
            "mean_seconds_per_epoch": round(sum(float(r["seconds"]) for r in hist) / len(hist), 1)}

    args = yaml.safe_load((YOLO_DIR / "args.yaml").read_text())
    rows = list(csv.DictReader(open(YOLO_DIR / "results.csv")))
    sel = json.loads((YOLO_DIR / "selection.json").read_text())
    data = yaml.safe_load(open(args["data"]))
    n_train = len([p for p in Path(data["path"], data["train"]).iterdir() if not p.name.startswith(".")])
    # ultralytics 'optimizer: auto' (engine/trainer.py build_optimizer): iterations =
    # ceil(n_train / max(batch, nbs)) * epochs; > 10000 -> SGD-type lr 0.01, else AdamW with
    # lr0 = round(0.002 * 5 / (4 + nc), 6), momentum 0.9 (lr0 and momentum in args are ignored).
    iters = math.ceil(n_train / max(args["batch"], args["nbs"])) * args["epochs"]
    resolved = ({"optimizer": "SGD/MuSGD", "lr0": 0.01, "momentum": 0.9} if iters > 10000 else
                {"optimizer": "AdamW", "lr0": round(0.002 * 5 / (4 + data["nc"]), 6), "momentum": 0.9})
    yolo = {"source": ["runs/compare/yoloseg/args.yaml", "runs/compare/yoloseg/results.csv",
                       "runs/compare/yoloseg/selection.json"],
            **{k: args.get(k) for k in YOLO_KEYS},
            "optimizer_auto_resolved": {**resolved, "train_images": n_train, "nc": data["nc"],
                                        "auto_rule_iterations": iters,
                                        "note": "derived from the ultralytics auto rule; args lr0/momentum ignored"},
            "selection_rule": sel["criterion"],
            "epochs_trained": len(rows),
            "early_stopped": len(rows) < args["epochs"],
            "selected_checkpoint": sel["selected"],
            "selected_val_binary_dice": round(sel["scores"][sel["selected"]], 4),
            "mean_seconds_per_epoch": round(float(rows[-1]["time"]) / len(rows), 1)}
    out = {"unet": unet, "yoloseg": yolo}
    dump("training_configs.json", out)
    return out


# ---------------------------------------------------------------- (c) pre-registration
MARK = "**Pre-registered evaluation protocol**"


def protocol_section(text):
    """From the protocol heading line up to the next code fence / heading / end; trailing blanks dropped."""
    lines = text.splitlines()
    try:
        i = next(k for k, l in enumerate(lines) if l.startswith(MARK))
    except StopIteration:
        return None
    j = i + 1
    while j < len(lines) and not lines[j].startswith(("```", "## ")):
        j += 1
    return "\n".join(lines[i:j]).rstrip() + "\n"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest() if s is not None else None


def preregistration():
    log = git("log", "--follow", "--format=%H%x09%aI%x09%cI%x09%s", "--", "docs/ROADMAP.md")
    commits, prev = [], None
    for line in reversed(log.strip().splitlines()):              # oldest first
        h, at, ct, subj = line.split("\t", 3)
        try:
            sec = protocol_section(git("show", f"{h}:docs/ROADMAP.md"))
        except subprocess.CalledProcessError:
            sec = None
        if sha(sec) != prev:
            commits.append({"commit": h, "author_time": at, "commit_time": ct, "subject": subj,
                            "change": "added" if prev is None else ("removed" if sec is None else "changed"),
                            "protocol_sha256": sha(sec)})
        prev = sha(sec)
    current = protocol_section((ROOT / "docs/ROADMAP.md").read_text())
    committed_head = protocol_section(git("show", "HEAD:docs/ROADMAP.md"))

    evals = {rel(p): datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(timespec="seconds")
             for p in sorted(glob.glob(str(RES / "comparison/*_evaluation.json")))}
    logp = ROOT / "logs/compare_models.log"
    log_ts = []                                                     # the log carries no timestamps itself
    first = min(evals.values()) if evals else None
    out = {"protocol_file": "docs/ROADMAP.md",
           "section_rule": f"lines from '{MARK}' to the next code fence or '## ' heading",
           "commits_touching_protocol": commits,
           "current_protocol_sha256": sha(current),
           "current_equals_HEAD": current == committed_head,
           "current_equals_first_version": bool(commits) and sha(current) == commits[0]["protocol_sha256"],
           "test_evaluation_files_mtime_utc": evals,
           "compare_models_log": {"path": rel(logp),
                                  "mtime_utc": datetime.fromtimestamp(os.path.getmtime(logp), timezone.utc)
                                  .isoformat(timespec="seconds") if logp.exists() else None,
                                  "timestamps_in_log": log_ts,
                                  "note": "log lines carry no timestamps; file mtime given instead"},
           "first_test_evaluation_utc": first}
    if commits and first:
        added = datetime.fromisoformat(commits[0]["author_time"])
        out["protocol_added_before_first_test_evaluation"] = added < datetime.fromisoformat(first)
        out["hours_between_protocol_added_and_first_test_evaluation"] = round(
            (datetime.fromisoformat(first) - added).total_seconds() / 3600, 2)
    dump("preregistration_record.json", out)
    return out


# ---------------------------------------------------------------- (d) code inventory
INHERITED = ["scripts/download_*.py", "scripts/unify_dataset*.py", "scripts/augment*.py",
             "scripts/deduplicate.py", "scripts/eda.py", "scripts/generate_dataset.py",
             "scripts/convert_masks_to_bbox.py", "scripts/train_yolo.py", "app/**/*.py"]


def sloc(p):
    return sum(1 for l in open(p, encoding="utf-8", errors="replace")
               if l.strip() and not l.lstrip().startswith("#"))


def files(patterns):
    out = set()
    for pat in patterns:
        out |= {Path(p) for p in glob.glob(str(ROOT / pat), recursive=True) if "__pycache__" not in p}
    return sorted(out)


def code_inventory():
    inherited = files(INHERITED)
    areas = {
        "inherited_practical_project": inherited,
        "thesis_selftrain_library": [p for p in files(["selftrain/*.py"])],
        "thesis_selftrain_system": files(["selftrain/system/**/*.py"]),
        "thesis_other_scripts": [p for p in files(["scripts/*.py"]) if p not in inherited],
        "thesis_tests": files(["tests/**/*.py"]),
    }
    out = {"counting_rule": "non-blank lines not starting with '#' in .py files (docstrings count as code)",
           "areas": {}}
    for name, fs in areas.items():
        per = {rel(p): sloc(p) for p in fs}
        out["areas"][name] = {"files": len(per), "lines": sum(per.values()), "per_file": per}
    thesis = sum(v["lines"] for k, v in out["areas"].items() if k.startswith("thesis_"))
    inh = out["areas"]["inherited_practical_project"]["lines"]
    out["totals"] = {"inherited": inh, "thesis": thesis,
                     "thesis_share": round(thesis / (thesis + inh), 3)}
    web = [p for p in files(["selftrain/system/templates/**/*", "selftrain/system/static/**/*"]) if p.is_file()]
    out["non_python_dashboard_assets"] = {"files": len(web), "nonblank_lines": sum(
        sum(1 for l in open(p, encoding="utf-8", errors="replace") if l.strip()) for p in web)}
    dump("code_inventory.json", out)
    return out


if __name__ == "__main__":
    c = model_complexity(); t = training_configs(); p = preregistration(); k = code_inventory()
    print(json.dumps({"complexity": {m: c[m] for m in ("unet", "yoloseg")},
                      "epochs": {m: t[m]["epochs_trained"] for m in t},
                      "protocol_commits": [(x["commit"][:7], x["author_time"], x["change"]) for x in p["commits_touching_protocol"]],
                      "protocol_sha256": p["current_protocol_sha256"],
                      "first_test_eval": p["first_test_evaluation_utc"],
                      "code_totals": k["totals"]}, indent=1))
