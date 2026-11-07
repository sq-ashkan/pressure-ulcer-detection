"""Failure analysis of the selected U-Net (seed 0) on the frozen test split.

  python scripts/failure_analysis.py

Reads results/comparison/unet_test_per_image.csv (one row per evaluated test photograph) and reports,
for the ulcer-containing photographs: the missed ulcers (no predicted region) by source, the photographs
with Dice below 0.5, and the Dice by tercile of the reference region area (gt_area = fraction of the
image covered by the reference ulcer mask), plus the false positives on ulcer-free photographs by source.
Output: results/failure_analysis.json
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = list(csv.DictReader(open(REPO / "results/comparison/unet_test_per_image.csv")))
    pos = [r for r in rows if r["gt_positive"] == "True"]
    neg = [r for r in rows if r["gt_positive"] != "True"]
    missed = [r for r in pos if r["pred_positive"] != "True"]
    low = [r for r in pos if float(r["dice"]) < 0.5]
    area = np.array([float(r["gt_area"]) for r in pos]); dice = np.array([float(r["dice"]) for r in pos])
    q1, q2 = np.quantile(area, [1 / 3, 2 / 3])
    terc = {"small": area <= q1, "middle": (area > q1) & (area <= q2), "large": area > q2}
    low_small = sum(1 for r in low if float(r["gt_area"]) <= q1)
    res = {
        "n_positive": len(pos), "n_negative": len(neg),
        "missed": len(missed), "missed_by_source": dict(collections.Counter(r["source"] for r in missed)),
        "dice_below_0_5": len(low), "dice_below_0_5_in_smallest_tercile": low_small,
        "area_tercile_bounds": [round(float(q1), 4), round(float(q2), 4)],
        "dice_by_area_tercile": {k: {"n": int(m.sum()), "mean_dice": round(float(dice[m].mean()), 3)} for k, m in terc.items()},
        "dice_by_source": {src: {"n": len(v), "mean_dice": round(float(np.mean(v)), 3)} for src, v in
                           sorted(collections.defaultdict(list, {s: [float(r["dice"]) for r in pos if r["source"] == s]
                                                                   for s in {r["source"] for r in pos}}).items())},
        "false_positives": sum(r["pred_positive"] == "True" for r in neg),
        "false_positives_by_source": dict(collections.Counter(r["source"] for r in neg if r["pred_positive"] == "True")),
        "negatives_by_source": dict(collections.Counter(r["source"] for r in neg)),
    }
    (REPO / "results/failure_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
