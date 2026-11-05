"""Pre-registered comparison U-Net vs YOLOv8-seg (thesis chapter: model selection).

Decision rule (fixed before any test result was seen, docs/ROADMAP.md):
  primary   mean per-image binary Dice on ulcer-containing TEST images;
            paired bootstrap (2000 resamples) of the difference U-Net - YOLO.
  if the 95 % CI of the difference excludes 0 -> the higher model wins;
  otherwise the models are statistically tied on the primary metric and the
  tie is broken, in this order, by: macro per-stage Dice (test), false-positive
  rate on ulcer-free photographs (test + extra negatives, lower is better),
  inference time (lower is better).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selftrain import config  # noqa: E402
from selftrain.evaluate import full_evaluation  # noqa: E402
from selftrain.metrics import paired_bootstrap  # noqa: E402


def per_image(out_dir, label, split):
    p = Path(out_dir) / f"{label}_{split}_per_image.csv"
    with open(p) as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def paired(out_dir, split):
    a, b = per_image(out_dir, "unet", split), per_image(out_dir, "yoloseg", split)
    names = sorted(n for n in a if n in b and a[n]["gt_positive"] == "True")
    return paired_bootstrap([float(a[n]["dice"]) for n in names],
                            [float(b[n]["dice"]) for n in names], b=config.BOOTSTRAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default=str(config.RUNS / "compare" / "unet" / "best.pt"))
    ap.add_argument("--yolo", default=str(config.RUNS / "compare" / "yoloseg" / "weights" / "selected.pt"))
    ap.add_argument("--data", default=str(config.DATA_SEG))
    ap.add_argument("--out", default=str(config.RESULTS / "comparison"))
    a = ap.parse_args()
    out = Path(a.out)
    from selftrain.unet import UNetPredictor
    from selftrain.yoloseg import YOLOPredictor
    res = {
        "unet": full_evaluation(UNetPredictor(Path(a.unet)), Path(a.data), out, "unet"),
        "yoloseg": full_evaluation(YOLOPredictor(Path(a.yolo)), Path(a.data), out, "yoloseg"),
    }
    test_diff = paired(out, "test")
    ext_diff = paired(out, "external") if (out / "unet_external_per_image.csv").exists() else None
    u, y = res["unet"], res["yoloseg"]
    lo, hi = test_diff["ci95"]
    if lo > 0 or hi < 0:
        winner = "unet" if test_diff["mean_diff"] > 0 else "yoloseg"
        reason = (f"primary metric: test Dice difference U-Net - YOLO = {test_diff['mean_diff']:+.4f}, "
                  f"95 % CI [{lo:+.4f}, {hi:+.4f}] excludes 0")
    else:
        def fpr(r):
            return r["extra_negatives"]["all"]["false_positive_rate"] if "extra_negatives" in r \
                else r["test"]["false_positive_rate"]
        keys = [("macro per-stage Dice", lambda r: r["test"]["stage_dice_macro"], 1),
                ("false-positive rate", fpr, -1),
                ("inference time", lambda r: r["inference_ms_per_image"], -1)]
        winner, reason = None, f"primary metric tied (95 % CI [{lo:+.4f}, {hi:+.4f}] includes 0); "
        for name, fn, sign in keys:
            if fn(u) != fn(y):
                winner = "unet" if sign * (fn(u) - fn(y)) > 0 else "yoloseg"
                reason += f"tie broken by {name} ({fn(u):.4f} vs {fn(y):.4f})"
                break
        winner = winner or "yoloseg"
    summary = {"winner": winner, "reason": reason, "paired_test": test_diff, "paired_external": ext_diff,
               "unet": {k: u[k] for k in ("threshold", "inference_ms_per_image")} | {"test": u["test"]},
               "yoloseg": {k: y[k] for k in ("threshold", "inference_ms_per_image")} | {"test": y["test"]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"winner": winner, "reason": reason}, indent=2))


if __name__ == "__main__":
    main()
