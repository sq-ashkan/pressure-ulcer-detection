"""Train one architecture on data/seg (used by the model comparison and the system).

  python scripts/train_model.py --model unet    --out runs/compare/unet
  python scripts/train_model.py --model yoloseg --out runs/compare/yoloseg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selftrain import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "yoloseg"], required=True)
    ap.add_argument("--data", default=str(config.DATA_SEG))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.model == "unet":
        from selftrain import unet
        cfg = config.UNetConfig()
        if a.epochs: cfg.epochs = a.epochs
        if a.seed is not None: cfg.seed = a.seed
        best = unet.train(Path(a.data), Path(a.out), cfg)
        print(f"BEST {best}")
    else:
        from selftrain import yoloseg
        cfg = config.YOLOConfig()
        if a.epochs: cfg.epochs = a.epochs
        if a.seed is not None: cfg.seed = a.seed
        yoloseg.train(Path(a.data) / "dataset.yaml", Path(a.out), cfg)
        sel, name, score = yoloseg.select_checkpoint(Path(a.out), Path(a.data))
        print(f"BEST {sel} ({name}, val Dice {score:.4f})")


if __name__ == "__main__":
    main()
