"""Extra evidence for the external PIID evaluation (reviewer request).

Command line:
    .venv/bin/python scripts/external_piid_extra.py

Inputs (written by scripts/external_piid.py):
    results/external/external_piid_{unet,yoloseg}_per_image.csv
    results/external/external_piid_{unet,yoloseg}.json
Output:
    results/external/piid_extra.json

Per model it reports
  * the number of photographs marked (a mask was predicted, detected=True);
  * stage accuracy over ALL PIID photographs, where a non-detection counts as an
    error, with a Wilson 95 % interval;
  * stage accuracy on the subset detected by BOTH models (same photographs for
    both, so the comparison is not biased by detection rate), with Wilson 95 %;
  * a percentile bootstrap 95 % interval (2000 resamples, seed 0) of the
    quadratic-weighted kappa on the detected photographs (same population as
    "quadratic_kappa" in external_piid_<model>.json).
On the photographs marked by BOTH models (n = 834) it further reports
  * the quadratic-weighted kappa of each model with a percentile bootstrap 95 %
    interval (2000 resamples, seed 0, the same resampled photographs for both
    models) and the interval of the paired kappa difference U-Net - YOLOv8-seg;
  * an exact McNemar test of stage correctness (two-sided binomial test of the
    discordant pairs, b = only U-Net correct, c = only YOLOv8-seg correct).
PIID contains only ulcer photographs (stages 1-4), so specificity cannot be
assessed on it; this is recorded in the output. Deterministic (fixed seed).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.metrics import wilson_ci  # noqa: E402

EXT = ROOT / "results" / "external"
MODELS = ("unet", "yoloseg")
N_BOOT, SEED = 2000, 0


def load(model):
    with open(EXT / f"external_piid_{model}_per_image.csv") as f:
        return {r["image"]: {"stage": int(r["piid_stage"]), "det": r["detected"] == "True",
                             "pred": int(r["pred_stage"])} for r in csv.DictReader(f)}


def kappa(y, p, labels):
    return float(cohen_kappa_score(y, p, labels=labels, weights="quadratic"))


def both_block(data, both):
    """Kappa (bootstrap CI) of both models and exact McNemar test on the photographs marked by both."""
    y = np.array([data[MODELS[0]][i]["stage"] for i in both])
    P = {m: np.array([data[m][i]["pred"] for i in both]) for m in MODELS}
    labels = sorted(set(y) | set(P[MODELS[0]]) | set(P[MODELS[1]]))
    rng = np.random.default_rng(SEED)
    boots = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        for m in MODELS:
            boots[m].append(kappa(y[idx], P[m][idx], labels))
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    k = {m: kappa(y, P[m], labels) for m in MODELS}
    cu, cy = P["unet"] == y, P["yoloseg"] == y
    b, c = int((cu & ~cy).sum()), int((~cu & cy).sum())
    return {"n": len(both),
            "quadratic_kappa": {m: {"kappa": k[m], "ci95": ci(boots[m])} for m in MODELS},
            "kappa_diff_unet_minus_yoloseg": {"diff": k["unet"] - k["yoloseg"],
                                              "ci95": ci(np.array(boots["unet"]) - np.array(boots["yoloseg"]))},
            "mcnemar_exact_stage_correct": {
                "both_correct": int((cu & cy).sum()), "only_unet_correct": b, "only_yoloseg_correct": c,
                "both_wrong": int((~cu & ~cy).sum()),
                "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
                "method": "exact two-sided binomial test of b vs c (p = 0.5)"}}


def main():
    data = {m: load(m) for m in MODELS}
    images = sorted(data[MODELS[0]])
    assert all(sorted(data[m]) == images for m in MODELS), "per-image CSVs cover different photographs"
    stages = sorted({data[MODELS[0]][i]["stage"] for i in images})
    both = [i for i in images if all(data[m][i]["det"] for m in MODELS)]
    out = {"piid_n": len(images), "piid_stages_present": stages,
           "ulcer_free_photographs": 0,
           "specificity": "not assessable: PIID contains only ulcer photographs (stages 1-4), no ulcer-free images",
           "detected_by_both_n": len(both), "bootstrap": {"resamples": N_BOOT, "seed": SEED, "method": "percentile"},
           "models": {}}
    for m in MODELS:
        d = data[m]
        summ = json.load(open(EXT / f"external_piid_{m}.json"))
        marked = [i for i in images if d[i]["det"]]
        correct_all = sum(d[i]["det"] and d[i]["pred"] == d[i]["stage"] for i in images)
        acc_all, ci_all = wilson_ci(correct_all, len(images))
        correct_both = sum(d[i]["pred"] == d[i]["stage"] for i in both)
        acc_both, ci_both = wilson_ci(correct_both, len(both))
        y = np.array([d[i]["stage"] for i in marked]); p = np.array([d[i]["pred"] for i in marked])
        labels = sorted(set(y) | set(p))
        k = kappa(y, p, labels)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            boots.append(kappa(y[idx], p[idx], labels))
        boots = np.array(boots)
        out["models"][m] = {
            "threshold": summ.get("threshold"),
            "photographs_marked": len(marked), "photographs_not_marked": len(images) - len(marked),
            "stage_accuracy_all": acc_all, "stage_accuracy_all_correct": correct_all,
            "stage_accuracy_all_ci95": list(ci_all),
            "stage_accuracy_detected_by_both": acc_both, "stage_accuracy_detected_by_both_correct": correct_both,
            "stage_accuracy_detected_by_both_n": len(both), "stage_accuracy_detected_by_both_ci95": list(ci_both),
            "quadratic_kappa_detected": k, "quadratic_kappa_detected_n": len(marked),
            "quadratic_kappa_matches_summary_json": bool(abs(k - summ["quadratic_kappa"]) < 1e-9),
            "quadratic_kappa_bootstrap_ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
        }
    out["detected_by_both"] = both_block(data, both)
    (EXT / "piid_extra.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
