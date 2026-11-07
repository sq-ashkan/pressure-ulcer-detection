"""Positive control of the drift check (selftrain/system/drift.py, drift_report).

Question: does the system's drift check flag photographs that come from another
distribution? Two batches of 100 photographs are compared, separately, with the
training reference:
  piid          100 photographs of the external PIID dataset (data/raw/piid/extracted,
                all images sorted by path, seeded uniform sample without replacement)
  azh_extra_neg 100 AZH ulcer-free extra-negative photographs (data/seg/negatives_extra.csv,
                same sampling)
Reference: the train split of data/seg (data/seg/manifest.csv), one image per photograph
(the first by name; independent observations, as drift.pool_reference does), i.e. the
photographs the models were trained on. Test and alpha as in the system: two-sample KS
per feature, Bonferroni over the six features, alpha = 0.01 (workspace default drift_alpha).

Command line (CPU, deterministic: numpy default_rng(0)):
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/drift_positive_control.py

Output:
    results/drift_positive_control.json  per batch: drift flag, shifted features, KS D,
                                         raw and Bonferroni-adjusted p-values, medians
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.system.drift import FEATURES, drift_report  # noqa: E402

N, SEED, ALPHA = 100, 0, 0.01
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def sample(paths, n=N, seed=SEED):
    paths = sorted(paths)
    pick = np.random.default_rng(seed).choice(len(paths), size=min(n, len(paths)), replace=False)
    return [paths[i] for i in sorted(pick)]


def reference():
    rows = sorted((r for r in csv.DictReader(open(ROOT / "data/seg/manifest.csv")) if r["split"] == "train"),
                  key=lambda r: r["name"])
    seen, out = set(), []
    for r in rows:
        if r["photograph"] not in seen:
            seen.add(r["photograph"])
            out.append(ROOT / "data/seg/images/train" / f"{r['name']}.jpg")
    return out


def main():
    ref = reference()
    piid = [p for p in (ROOT / "data/raw/piid/extracted").rglob("*") if p.suffix.lower() in IMG_EXT]
    azh = [Path(r["path"]) for r in csv.DictReader(open(ROOT / "data/seg/negatives_extra.csv"))]
    out = {"method": "selftrain.system.drift.drift_report (KS per feature, Bonferroni over 6 features)",
           "alpha": ALPHA, "sample_size": N, "seed": SEED,
           "reference": {"description": "data/seg train split, one image per photograph", "n": len(ref)},
           "batches": {}}
    for name, pool in (("piid", piid), ("azh_extra_negatives", azh)):
        r = drift_report(sample(pool), ref, ALPHA, log=lambda s: None)
        out["batches"][name] = {
            "pool_size": len(pool), "n_new": r["n_new"], "n_reference": r["n_reference"],
            "drift": r["drift"], "shifted": r["shifted"], "n_shifted": len(r["shifted"]),
            "summary": r["summary"], "alpha_per_test": r["alpha_per_test"],
            "features": {f: {k: r["features"][f][k] for k in
                             ("ks_statistic", "p_value", "p_adjusted", "shifted",
                              "new_median", "reference_median", "direction")}
                         for f in FEATURES}}
    (ROOT / "results/drift_positive_control.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, b in out["batches"].items():
        print(name, b["summary"])
        for f, v in b["features"].items():
            print(f"  {f:15s} D={v['ks_statistic']:.3f} p={v['p_value']:.3g} p_adj={v['p_adjusted']:.3g} "
                  f"{'SHIFT' if v['shifted'] else ''}")


if __name__ == "__main__":
    main()
