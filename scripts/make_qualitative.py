"""Qualitative comparison figure: frozen-test photographs with the ground-truth outline
(white) and the predicted ulcer regions of U-Net and YOLOv8-seg (colour = stage).
Cases chosen by U-Net per-image Dice: best, median, worst ulcer photograph, and the
ulcer-free photograph with the largest false-positive area.

Only photographs that may be reproduced are shown: test images of the five Roboflow
sources published under CC BY 4.0 (CC_BY_SOURCES; not neg_azh, neg_fuseg or PIID, which
state no licence), minus every image named in results/provenance_check.json (OCR found a
copyright mark or stock/clinic watermark) and stock-photo file names. The false-positive
case is therefore the CC BY ulcer-free photograph with the largest predicted area.

Command line (CPU, deterministic):
    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/make_qualitative.py
Outputs:
    docs/thesis/figures/qualitative_examples.pdf
    docs/thesis/generated/qualitative_cases.txt   case, image name, source, U-Net Dice
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.unet import UNetPredictor  # noqa: E402
from selftrain.yoloseg import YOLOPredictor  # noqa: E402

CC_BY_SOURCES = ("roboflow_maskrcnn", "roboflow_woundcare", "roboflow_fid", "roboflow_mobile", "rf_petchs_seg")
STAGE_RGB = {1: (86, 180, 233), 2: (0, 158, 115), 3: (230, 159, 0), 4: (213, 94, 0)}


def overlay(img, sem, gt):
    out = img.copy().astype(np.float32)
    for s, c in STAGE_RGB.items():
        m = sem == s
        out[m] = 0.55 * out[m] + 0.45 * np.array(c[::-1], np.float32)
    out = out.astype(np.uint8)
    cs, _ = cv2.findContours((gt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (255, 255, 255), max(2, img.shape[0] // 200))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def provenance_flagged():
    """Image names (without extension) mentioned anywhere in results/provenance_check.json."""
    names = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and Path(x).suffix.lower() in (".jpg", ".jpeg", ".png"):
            names.add(Path(x).stem)
    walk(json.load(open(ROOT / "results/provenance_check.json")))
    return names


def source(name):
    return name.split("__", 1)[0]


def main():
    stock = ("istock", "shutterstock", "gettyimages", "adobestock", "dreamstime", "123rf", "alamy", "depositphotos")
    flagged = provenance_flagged()
    rows = [r for r in csv.DictReader(open(ROOT / "results/comparison/unet_test_per_image.csv"))
            if not any(k in r["name"].lower() for k in stock)   # stock photos: no reuse licence
            and source(r["name"]) in CC_BY_SOURCES and r["name"] not in flagged]
    pos = sorted([r for r in rows if r["gt_positive"] == "True"], key=lambda r: float(r["dice"]))
    neg = sorted([r for r in rows if r["gt_positive"] == "False"], key=lambda r: -float(r["pred_area"]))
    cases = [("best", pos[-1]), ("median", pos[len(pos) // 2]), ("worst", pos[0]), ("false positive", neg[0])]
    thr = {m: json.load(open(ROOT / f"results/comparison/{m}_evaluation.json"))["threshold"] for m in ("unet", "yoloseg")}
    preds = {"unet": UNetPredictor(ROOT / "runs/compare/unet/best.pt"),
             "yoloseg": YOLOPredictor(ROOT / "runs/compare/yoloseg/weights/selected.pt")}
    fig, axes = plt.subplots(len(cases), 2, figsize=(5.9, 2.75 * len(cases)))
    for i, (title, r) in enumerate(cases):
        img = cv2.imread(str(ROOT / "data/seg/images/test" / f"{r['name']}.jpg"))
        gt = cv2.imread(str(ROOT / "data/seg/masks/test" / f"{r['name']}.png"), 0)
        for j, m in enumerate(("unet", "yoloseg")):
            sem = preds[m].to_semantic(preds[m].infer(img), thr[m])
            axes[i, j].imshow(overlay(img, sem, gt)); axes[i, j].axis("off")
            label = "U-Net" if m == "unet" else "YOLOv8-seg"
            axes[i, j].set_title(f"{label} — {title}", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in STAGE_RGB.values()]
    fig.legend(handles, [f"stage {s}" for s in STAGE_RGB], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "docs/thesis/figures/qualitative_examples.pdf", dpi=300)
    (ROOT / "docs/thesis/generated/qualitative_cases.txt").write_text(
        "\n".join(f"{t}: {r['name']} (source {source(r['name'])}, CC BY 4.0; U-Net Dice {float(r['dice']):.3f}, "
                  f"predicted area {float(r['pred_area']):.4f} of the image)" for t, r in cases) + "\n")
    print(f"ok: {len(rows)} eligible test images, {len(flagged)} provenance-flagged names excluded")


if __name__ == "__main__":
    main()
