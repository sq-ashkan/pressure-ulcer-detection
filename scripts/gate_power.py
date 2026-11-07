"""Statistical power of the promotion gate (selftrain.system.gate.decide).

Command line:
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/gate_power.py
Output:
    results/gate_power.json

No model is run; only existing per-image test CSVs are read (ulcer-positive test
images, the gate's evidence, n = 91 per model in the comparison runs).

(1) Standard error of the paired mean Dice difference, SE = sd(d) / sqrt(n), for
    every available pair: U-Net vs YOLOv8-seg (results/comparison), U-Net seed 0 vs
    seed 1 (results/comparison vs results/comparison_seed1), the growth cycles
    v_k vs v_{k-1} (system_ws/models/v*/eval/v*_test_per_image.csv), and a shifted
    copy of the U-Net (U-Net + c vs U-Net), whose SE is 0 by construction.
(2) Power simulation: for a base set of observed paired differences d, centre them
    (d - mean(d)), and for each true gain Delta draw 200 simulated test sets
    (n images resampled with replacement from the centred d, plus Delta) and apply
    gate.decide(candidate=d_sim, current=0) -- B = 2000 bootstrap resamples, margin
    0.02, the gate's own fixed bootstrap seed 0. P(promote | Delta) is the share of
    promoted simulations; Delta = 0 gives the type-I rate. Fixed RNG seed 12345.
(3) Seed-to-seed difference of the U-Net test Dice from results/comparison*/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.system.gate import decide, load_test_dice, pair_by_name  # noqa: E402

DELTAS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
N_SIM, MARGIN, SEED = 200, 0.02, 12345
OUT = ROOT / "results" / "gate_power.json"


def diffs(cand_csv: Path, cur_csv: Path) -> np.ndarray:
    _, a, b = pair_by_name(load_test_dice(cand_csv), load_test_dice(cur_csv))
    return a - b


def se_entry(d: np.ndarray) -> dict:
    sd = float(d.std(ddof=1))
    return {"n": int(d.size), "mean_diff": float(d.mean()), "sd_diff": sd,
            "se_mean_diff": sd / np.sqrt(d.size),
            "se_at_n91": sd / np.sqrt(91)}


def power(d: np.ndarray, rng: np.random.Generator) -> dict:
    c = d - d.mean()
    probs = {}
    for delta in DELTAS:
        k = 0
        for _ in range(N_SIM):
            ds = c[rng.integers(0, c.size, c.size)] + delta
            k += decide(ds, np.zeros_like(ds), margin=MARGIN).promoted
        probs[f"{delta:.2f}"] = k / N_SIM

    def first(p):
        return next((dl for dl in DELTAS if probs[f"{dl:.2f}"] >= p), None)
    return {"p_promote": probs, "type1_rate_delta0": probs["0.00"],
            "min_delta_grid_p50": first(0.5), "min_delta_grid_p80": first(0.8)}


def main() -> None:
    cmp0, cmp1 = ROOT / "results/comparison", ROOT / "results/comparison_seed1"
    pairs: dict[str, np.ndarray] = {
        "unet_vs_yoloseg_seed0": diffs(cmp0 / "unet_test_per_image.csv",
                                       cmp0 / "yoloseg_test_per_image.csv")}
    if (cmp1 / "unet_test_per_image.csv").exists():
        pairs["unet_seed1_vs_seed0"] = diffs(cmp1 / "unet_test_per_image.csv",
                                             cmp0 / "unet_test_per_image.csv")
    u = np.array(list(load_test_dice(cmp0 / "unet_test_per_image.csv").values()))
    pairs["unet_shifted_copy_+0.02"] = (u + 0.02) - u
    vs = sorted((ROOT / "system_ws/models").glob("v*/eval/v*_test_per_image.csv"),
                key=lambda p: int(p.parent.parent.name[1:]))
    for prev, cur in zip(vs, vs[1:]):
        try:
            pairs[f"growth_{cur.parent.parent.name}_vs_{prev.parent.parent.name}"] = diffs(cur, prev)
        except ValueError as e:          # different test image sets
            print("skip", cur, prev, e)
    se = {k: se_entry(d) for k, d in pairs.items()}

    rng = np.random.default_rng(SEED)
    bases = [k for k in ("growth_v1_vs_v0", "unet_vs_yoloseg_seed0", "unet_seed1_vs_seed0")
             if k in pairs]
    pw = {k: {"base_sd_diff": se[k]["sd_diff"], "n": se[k]["n"], **power(pairs[k], rng)}
          for k in bases}

    seeds = {}
    for p in sorted(ROOT.glob("results/comparison*/summary.json")):
        s = json.loads(p.read_text())["unet"]
        name = p.parent.name
        pos = np.array(list(load_test_dice(p.parent / "unet_test_per_image.csv").values()))
        seeds["seed0" if name == "comparison" else name.replace("comparison_", "")] = {
            "summary": str(p.relative_to(ROOT)), "threshold": s["threshold"],
            "test_dice_mean_summary": s["test"]["dice_mean"],
            "test_dice_mean_positive_images": float(pos.mean()), "n_positive": int(pos.size)}
    vals = [v["test_dice_mean_positive_images"] for v in seeds.values()]
    vals_all = [v["test_dice_mean_summary"] for v in seeds.values()]
    seed_part = {"seeds": seeds,
                 "range_positive_images": float(max(vals) - min(vals)) if vals else None,
                 "range_summary_dice_mean": float(max(vals_all) - min(vals_all)) if vals_all else None,
                 "seed2_present": "seed2" in seeds}

    out = {
        "gate_rule": "promote iff mean(d) > 0 and lower 95 % percentile-bootstrap bound > -0.02 "
                     "(selftrain.system.gate.decide, B = 2000, bootstrap seed 0)",
        "bootstrap_p_value_formula": (
            "selftrain.metrics.paired_bootstrap: d = a - b; means_j = mean of d resampled "
            "with replacement (j = 1..B, B = 2000, seed 0); p = min(1, 2 * min(mean(means_j <= 0), "
            "mean(means_j >= 0))). This is the two-sided percentile-bootstrap p-value "
            "(share of the bootstrap distribution, not centred under H0, on the far side of 0); "
            "its resolution is 1/B = 0.0005 and it is not used by the gate decision."),
        "se_paired_mean_diff": se,
        "note_shifted_copy": "a shifted copy has d constant, so sd(d) = SE = 0: the SE is set by "
                             "the per-image disagreement of two models, not by the Dice level",
        "power_simulation": {"n_sim_per_delta": N_SIM, "deltas": DELTAS, "rng_seed": SEED,
                             "method": "centred observed paired differences resampled with "
                                       "replacement (n images) + Delta, gate.decide vs zeros",
                             "results": pw},
        "seed_to_seed_unet": seed_part,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"se": {k: round(v["se_mean_diff"], 4) for k, v in se.items()},
                      "power": pw, "seed": seed_part}, indent=1))


if __name__ == "__main__":
    main()
