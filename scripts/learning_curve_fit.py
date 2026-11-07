"""Log-linear fit of the U-Net learning curve with a joint seed/photograph bootstrap.

Command line:
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/learning_curve_fit.py
Inputs:
    results/learning_curve/subsets.json            (training photographs per run)
    results/learning_curve/f{25,50,75}_s{0,1,2}/unet_{evaluation.json,test_per_image.csv}
    results/comparison{,_seed1,_seed2}/unet_{evaluation.json,test_per_image.csv}  (650 photographs)
Output:
    results/learning_curve_fit.json

Model: test Dice = a + b * log2(training photographs), ordinary least squares over the
12 runs (4 sizes 162/325/488/650 x 3 seeds); b is the change in Dice per doubling of
the training photographs. Primary response: mean per-image Dice over the 91 ulcer test
photographs (one test image per photograph, data/seg/manifest.csv). The fit on the
evaluation JSON's test.dice_mean is reported as a cross-check (it is computed over the
same ulcer test images, so it is identical).

95 % interval: percentile bootstrap, 2000 resamples, numpy seed 0, resampling seeds and
test photographs jointly: per resample the 91 ulcer test photographs are drawn with
replacement (the same draw for all runs, so the pairing across runs is kept) and, within
each training size, the 3 seed runs are drawn with replacement; the 12 resampled run
means are refitted. The Dice at 1300 photographs (2x the full set) is an EXTRAPOLATION
of the log-linear fit beyond the observed range (162-650), not a measurement.
Deterministic.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "learning_curve_fit.json"
N_BOOT, SEED, X_EXTRA = 2000, 0, 1300


def per_image(run: Path) -> dict[str, float]:
    with open(run / "unet_test_per_image.csv") as f:
        return {r["name"]: float(r["dice"]) for r in csv.DictReader(f) if r["gt_positive"] == "True"}


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(np.log2(x), y, 1)
    return float(a), float(b)


def main() -> None:
    subsets = json.loads((ROOT / "results/learning_curve/subsets.json").read_text())
    runs = [(ROOT / "results/learning_curve" / t, s["photographs"], s["seed"]) for t, s in sorted(subsets.items())]
    runs += [(ROOT / f"results/{n}", 650, s) for s, n in enumerate(("comparison", "comparison_seed1",
                                                                     "comparison_seed2"))]
    names = sorted(per_image(runs[0][0]))
    D = np.array([[per_image(r)[n] for n in names] for r, _, _ in runs])     # 12 x 91
    x = np.array([p for _, p, _ in runs], float)
    y_all = np.array([json.loads((r / "unet_evaluation.json").read_text())["test"]["dice_mean"]
                      for r, _, _ in runs])
    y = D.mean(1)
    a, b = fit(x, y)
    a2, b2 = fit(x, y_all)
    sizes = sorted(set(x))
    groups = [np.flatnonzero(x == s) for s in sizes]

    rng = np.random.default_rng(SEED)
    bs, ext = np.empty(N_BOOT), np.empty(N_BOOT)
    for i in range(N_BOOT):
        img = rng.integers(0, D.shape[1], D.shape[1])
        rows = np.concatenate([g[rng.integers(0, g.size, g.size)] for g in groups])
        ai, bi = fit(x[rows], D[np.ix_(rows, img)].mean(1))
        bs[i], ext[i] = bi, ai + bi * np.log2(X_EXTRA)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731

    out = {
        "model": "test Dice = a + b * log2(training photographs), OLS over 12 runs",
        "response": "mean per-image Dice over the 91 ulcer test photographs",
        "runs": [{"run": str(r.relative_to(ROOT)), "photographs": int(p), "seed": int(s),
                  "dice_ulcer_photographs": float(v), "dice_mean_evaluation_json": float(w)}
                 for (r, p, s), v, w in zip(runs, y, y_all)],
        "mean_by_size": {str(int(s)): float(y[g].mean()) for s, g in zip(sizes, groups)},
        "intercept_a": a, "slope_b_per_doubling": b,
        "slope_b_ci95": q(bs),
        "r_squared": float(1 - ((y - (a + b * np.log2(x))) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "extrapolation_1300_photographs": {
            "label": "EXTRAPOLATION beyond the observed range 162-650 photographs (log-linear model "
                     "assumed to continue); not a measurement",
            "dice": a + b * float(np.log2(X_EXTRA)), "ci95": q(ext)},
        "cross_check_evaluation_json": {"response": "unet_evaluation.json test.dice_mean",
                                          "identical_to_primary": bool(np.allclose(y, y_all)),
                                          "intercept_a": a2, "slope_b_per_doubling": b2,
                                          "dice_at_1300_extrapolated": a2 + b2 * float(np.log2(X_EXTRA))},
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "method": "percentile; test photographs (shared "
                      "across runs) and seed runs within each size resampled jointly with replacement"},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("mean_by_size", "slope_b_per_doubling", "slope_b_ci95",
                                          "r_squared", "extrapolation_1300_photographs",
                                          "cross_check_evaluation_json")}, indent=1))


if __name__ == "__main__":
    main()
