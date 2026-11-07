"""Robustness of the seed-0 model comparison (U-Net vs YOLOv8-seg), reviewer request.

Command line:
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/comparison_robustness.py
Inputs:
    results/comparison/{unet,yoloseg}_test_per_image.csv   (136 test photographs, 91 with ulcer)
    results/comparison/{unet,yoloseg}_evaluation.json      (extra-negative false-positive rates)
Output:
    results/comparison_robustness.json

Reports, for seed 0 only:
  * ulcer photographs with Dice 0 and with no prediction at all (pred_positive False), per model;
  * the paired difference U-Net - YOLOv8-seg (per-image Dice, ulcer photographs) and the lower 95 %
    percentile-bootstrap bound (selftrain.metrics.paired_bootstrap, B = 2000) over 200 bootstrap
    seeds 0..199 (min, max, how many are > 0);
  * a two-sided Wilcoxon signed-rank p (scipy, zero differences dropped, 'wilcox');
  * dominant-stage accuracy over ALL 91 ulcer photographs (a miss, pred_stage 0, is an error) with
    Wilson 95 % interval, and over the photographs the model marked (pred_stage > 0), with n;
  * sensitivity (ulcer photographs) and false-positive rate (ulcer-free photographs) per source,
    with n and Wilson 95 % interval, both models;
  * the false-positive rate per extra-negative class (healthy skin N, diabetic D, venous V,
    surgical S, background BG) as stored in *_evaluation.json, both models.
Deterministic (fixed seeds).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.metrics import paired_bootstrap, wilson_ci  # noqa: E402

RUN = ROOT / "results" / "comparison"
OUT = ROOT / "results" / "comparison_robustness.json"
MODELS = ("unet", "yoloseg")
N_SEEDS, B = 200, 2000
EXTRA = {"N": "healthy skin", "D": "diabetic", "V": "venous", "S": "surgical", "BG": "background"}


def load(m: str) -> dict[str, dict]:
    with open(RUN / f"{m}_test_per_image.csv") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def prop(k: int, n: int) -> dict:
    p, ci = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": p, "wilson_ci95": list(ci)}


def main() -> None:
    R = {m: load(m) for m in MODELS}
    names = sorted(R["unet"])
    assert names == sorted(R["yoloseg"])
    pos = [n for n in names if R["unet"][n]["gt_positive"] == "True"]
    neg = [n for n in names if R["unet"][n]["gt_positive"] != "True"]
    dice = {m: np.array([float(R[m][n]["dice"]) for n in pos]) for m in MODELS}
    out: dict = {"run": "results/comparison (seed 0)", "n_ulcer": len(pos), "n_ulcer_free": len(neg), "models": {}}

    for m in MODELS:
        r = R[m]
        gt = np.array([int(r[n]["gt_stage"]) for n in pos]); pr = np.array([int(r[n]["pred_stage"]) for n in pos])
        marked = pr > 0
        ev = json.loads((RUN / f"{m}_evaluation.json").read_text())
        by_src = defaultdict(lambda: [0, 0, 0, 0])   # tp, n_pos, fp, n_neg
        for n in names:
            s = by_src[r[n]["source"]]
            if r[n]["gt_positive"] == "True":
                s[0] += r[n]["pred_positive"] == "True"; s[1] += 1
            else:
                s[2] += r[n]["pred_positive"] == "True"; s[3] += 1
        out["models"][m] = {
            "test_dice_ulcer": float(dice[m].mean()),
            "ulcer_photographs_dice0": int((dice[m] == 0).sum()),
            "ulcer_photographs_no_prediction": int(sum(r[n]["pred_positive"] != "True" for n in pos)),
            "dominant_stage_accuracy_all_ulcer": prop(int((pr == gt).sum()), len(pos)),
            "dominant_stage_accuracy_marked": prop(int((pr[marked] == gt[marked]).sum()), int(marked.sum())),
            "dominant_stage_accuracy_marked_matches_evaluation_json":
                bool(abs((pr[marked] == gt[marked]).mean() - ev["test"]["dominant_stage_accuracy"]) < 1e-9),
            "per_source": {s: {"sensitivity": prop(v[0], v[1]) if v[1] else None,
                               "false_positive_rate": prop(v[2], v[3]) if v[3] else None}
                           for s, v in sorted(by_src.items())},
            "extra_negatives_fpr": {f"{EXTRA[c]} ({c})": ev["extra_negatives"][c] for c in EXTRA}
                                   | {"all": ev["extra_negatives"]["all"]},
        }

    d = dice["unet"] - dice["yoloseg"]
    los = np.array([paired_bootstrap(dice["unet"], dice["yoloseg"], b=B, seed=s)["ci95"][0] for s in range(N_SEEDS)])
    w = wilcoxon(dice["unet"], dice["yoloseg"], zero_method="wilcox", alternative="two-sided")
    out["paired_difference_unet_minus_yoloseg"] = {
        "n": len(pos), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "n_nonzero": int((d != 0).sum()),
        "bootstrap_lower95_over_seeds": {"seeds": f"0..{N_SEEDS - 1}", "resamples": B, "min": float(los.min()),
                                         "max": float(los.max()), "n_above_0": int((los > 0).sum())},
        "wilcoxon_signed_rank": {"statistic": float(w.statistic), "p_value": float(w.pvalue),
                                 "zero_method": "wilcox (zero differences dropped)"}}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
