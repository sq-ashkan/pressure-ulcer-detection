"""Residual-leakage check of the frozen evaluation sets and its effect on the model comparison.

  python scripts/leakage_sensitivity.py

During the data-growth cycles the system's frozen-set check (selftrain.system.dedupe) flagged 227
images of the growth batches -- which are drawn from the training split of data/seg -- as matches of
frozen evaluation photographs: 17 images against 4 test photographs and 210 images against 27
extra-negative photographs (system_ws/reports/cycle_*.json). All 4 test pairs and 12 extra-negative
pairs (all 5 pixel-correlation matches and 7 random dHash matches) were inspected visually
(2026-11-07): one test photograph (roboflow_fid Stage2_4, photograph 2934) is a mirrored, cropped and
recoloured copy of the training photograph roboflow_maskrcnn/woundcare "Chronic-ulcer-686" (ORB, 31
inliers); every other inspected pair shows different photographs (false matches of dHash <= 8 and of
the pixel correlation on low-texture skin images). This script records the matches and recomputes the
paired comparison without the copied test photograph and, as a bound, without all 4 flagged test
photographs. Output: results/leakage_sensitivity.json
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from selftrain.metrics import paired_bootstrap  # noqa: E402

TRUE_COPY = ["roboflow_fid__Stage2_4_PNG"]
FLAGGED = TRUE_COPY + ["roboflow_maskrcnn__Chronic-ulcer-690-", "neg_azh__neg_azh__b9600ef3__31",
                       "neg_fuseg__neg_fuseg__73c4a573__1331"]
INSPECTION = {"test_pairs_inspected": 4, "test_true_copies": 1,
              "extra_negative_pairs_inspected": 12, "extra_negative_true_copies": 0,
              "true_copy": "test roboflow_fid Stage2_4 (photograph 2934) = mirrored, cropped, recoloured "
                           "copy of train roboflow_maskrcnn/woundcare Chronic-ulcer-686 (8 training images)"}


def matches() -> dict:
    rows = []
    for n in (1, 2, 3):
        d = json.loads((REPO / f"system_ws/reports/cycle_{n}.json").read_text())
        rows += [dict(cycle=n, split=m["split"], method=m["method"], score=m["score"],
                      photograph=str(m["photograph"]).split("/")[-1]) for m in d["dedupe"]["dropped_test_matches"]]
    by = collections.Counter((r["split"], r["method"]) for r in rows)
    return {"images": len(rows),
            "by_split_and_method": {f"{s}/{m}": c for (s, m), c in sorted(by.items())},
            "distinct_photographs": {s: len({r["photograph"] for r in rows if r["split"] == s})
                                     for s in sorted({r["split"] for r in rows})}}


def comparison(run: str) -> dict:
    def load(model):
        return {r["name"]: r for r in csv.DictReader(open(REPO / f"results/{run}/{model}_test_per_image.csv"))}
    U, Y = load("unet"), load("yoloseg")
    out = {}
    for tag, excl in (("all", []), ("without_true_copy", TRUE_COPY), ("without_all_flagged", FLAGGED)):
        names = [n for n in U if not any(n.startswith(e) for e in excl)]
        pos = [n for n in names if U[n]["gt_positive"] == "True"]
        neg = [n for n in names if U[n]["gt_positive"] != "True"]
        du = np.array([float(U[n]["dice"]) for n in pos]); dy = np.array([float(Y[n]["dice"]) for n in pos])
        pb = paired_bootstrap(du, dy, b=2000, seed=0)
        out[tag] = {"n_pos": len(pos), "n_neg": len(neg), "unet_dice": round(float(du.mean()), 4),
                    "yolo_dice": round(float(dy.mean()), 4), "mean_diff": round(float(pb["mean_diff"]), 4),
                    "ci95": [round(float(x), 4) for x in pb["ci95"]], "p_value": float(pb["p_value"]),
                    "unet_fpr": round(sum(U[n]["pred_positive"] == "True" for n in neg) / len(neg), 4)}
    return out


def main() -> None:
    res = {"frozen_matches": matches(), "visual_inspection": INSPECTION,
           "comparison_seed0": comparison("comparison"), "comparison_seed1": comparison("comparison_seed1")}
    (REPO / "results/leakage_sensitivity.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
