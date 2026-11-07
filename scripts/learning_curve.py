"""Learning curve of the U-Net on fractions of the training photographs, with several seeds.

  python scripts/learning_curve.py --fractions 0.25 0.5 0.75 --seeds 0 1 2

For every (fraction, seed) the script draws round(fraction * N) of the N training photographs of
data/seg with a seeded random generator (every image of a drawn photograph is used), trains the U-Net
with the thesis configuration and that seed, and evaluates it with the pre-registered protocol of the
model comparison (threshold chosen on the validation split, metrics on the frozen test split and the
extra negatives; selftrain.evaluate.full_evaluation). Validation and test splits are never subsampled.
Runs that already have results are skipped, so the script can be restarted.
Outputs: runs/learning_curve/f<pct>_s<seed>/ (weights) and results/learning_curve/f<pct>_s<seed>/
(evaluation JSON and per-image CSVs) plus results/learning_curve/subsets.json (drawn photographs).
"""
from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from selftrain import config, unet  # noqa: E402
from selftrain.evaluate import full_evaluation  # noqa: E402


def train_photographs(manifest: Path) -> dict[str, list[str]]:
    by_photo: dict[str, list[str]] = collections.defaultdict(list)
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                by_photo[row["photograph"]].append(row["name"])
    return dict(sorted(by_photo.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", default=str(config.DATA_SEG))
    ap.add_argument("--runs", default=str(config.RUNS / "learning_curve"))
    ap.add_argument("--results", default=str(config.RESULTS / "learning_curve"))
    a = ap.parse_args()
    data = Path(a.data)
    photos = train_photographs(data / "manifest.csv")
    ids = list(photos)
    res_root = Path(a.results); res_root.mkdir(parents=True, exist_ok=True)
    subsets_file = res_root / "subsets.json"
    subsets = json.loads(subsets_file.read_text()) if subsets_file.exists() else {}
    for f in a.fractions:
        for s in a.seeds:
            tag = f"f{round(f * 100)}_s{s}"
            out_res = res_root / tag
            if (out_res / "unet_evaluation.json").exists():
                print(f"[{tag}] done already, skipped", flush=True)
                continue
            rng = random.Random(10_000 + 100 * s + round(f * 100))
            drawn = sorted(rng.sample(ids, round(f * len(ids))), key=lambda x: int(x) if x.isdigit() else x)
            names = [n for p in drawn for n in photos[p]]
            subsets[tag] = {"fraction": f, "seed": s, "photographs": len(drawn), "images": len(names),
                            "photograph_ids": drawn}
            subsets_file.write_text(json.dumps(subsets, indent=1))
            print(f"[{tag}] {len(drawn)} of {len(ids)} photographs, {len(names)} images", flush=True)
            cfg = dataclasses.replace(config.UNetConfig(), seed=s)
            run_dir = Path(a.runs) / tag
            best = run_dir / "best.pt"
            if not best.exists():
                unet.train(data, run_dir, cfg, train_names=names)
            res = full_evaluation(unet.UNetPredictor(best), data, out_res, "unet")
            t = res.get("test", {})
            print(f"[{tag}] test Dice {t.get('dice_mean')} threshold {res.get('threshold')}", flush=True)


if __name__ == "__main__":
    main()
