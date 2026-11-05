"""External test on PIID (Ay et al., 2022): 1,091 photographs, one stage label each.

PIID has no outlines, so it tests what an outline-free label allows: does the model
find an ulcer at all (sensitivity), and which stage does it assign (dominant stage of the
predicted region vs. the PIID stage)? PIID photographs that are the same photograph as
any image of data/seg (any split) are removed first -- identity by 64-bit pHash under the
8 dihedral transforms (Hamming <= 8) confirmed by a 64x64 thumbnail correlation >= 0.90
-- so the remaining photographs were never seen in training, validation or test.

Usage: python scripts/external_piid.py --model unet --weights runs/compare/unet/best.pt
Output: results/external_piid_<model>.json (+ per-image CSV)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.metrics import dominant_stage, predicted_positive, remove_specks, wilson_ci  # noqa: E402

PIID = ROOT / "data" / "raw" / "piid" / "extracted"
SEG = ROOT / "data" / "seg"


def dihedral(a, k):
    b = np.rot90(a, k % 4)
    return np.fliplr(b) if k >= 4 else b


def thumb(gray, s=64):
    g = cv2.resize(gray, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n > 1e-6 else g


def overlap_filter(piid_items):
    seg_imgs = sorted(SEG.glob("images/*/*.jpg"))
    sh = np.array([imagehash.phash(Image.open(p)).hash.ravel() for p in seg_imgs], bool)
    st = [thumb(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in seg_imgs]
    keep, dropped = [], []
    for path, stage in piid_items:
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        t = thumb(g)
        hit = None
        for k in range(8):
            h = imagehash.phash(Image.fromarray(np.ascontiguousarray(dihedral(g, k)))).hash.ravel()
            for j in np.where((sh != h).sum(1) <= 8)[0]:
                if float((dihedral(t, k) * st[j]).sum()) >= 0.90:
                    hit = seg_imgs[j].name; break
            if hit:
                break
        (dropped if hit else keep).append((path, stage, hit))
    return keep, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["unet", "yoloseg"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, help="default: the val-chosen threshold in results/comparison")
    a = ap.parse_args()
    items = [(p, int(s)) for s in "1234" for p in sorted((PIID / s).glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    keep, dropped = overlap_filter(items)
    print(f"PIID: {len(items)} photographs, {len(dropped)} also in data/seg -> {len(keep)} external")
    thr = a.threshold
    if thr is None:
        ev = json.loads((ROOT / "results" / "comparison" / f"{a.model}_evaluation.json").read_text())
        thr = ev["threshold"]
    if a.model == "unet":
        from selftrain.unet import UNetPredictor as P
    else:
        from selftrain.yoloseg import YOLOPredictor as P
    pred = P(Path(a.weights))
    rows = []
    for path, stage, _ in keep:
        img = cv2.imread(str(path))
        sem = pred.to_semantic(pred.infer(img), thr)
        sem = np.where(remove_specks(sem > 0), sem, 0).astype(np.uint8)
        rows.append({"image": path.name, "piid_stage": stage, "detected": predicted_positive(sem),
                     "pred_stage": dominant_stage(sem), "pred_area": float((sem > 0).mean())})
    det = [r for r in rows if r["detected"]]
    sens, sens_ci = wilson_ci(len(det), len(rows))
    acc, acc_ci = wilson_ci(sum(r["pred_stage"] == r["piid_stage"] for r in det), len(det))
    kappa = float(cohen_kappa_score([r["piid_stage"] for r in det], [r["pred_stage"] for r in det],
                                    weights="quadratic")) if len(det) > 1 else float("nan")
    per_stage = {s: wilson_ci(sum(r["detected"] for r in rows if r["piid_stage"] == s),
                              sum(1 for r in rows if r["piid_stage"] == s)) for s in range(1, 5)}
    cm = np.zeros((4, 5), int)
    for r in rows:
        cm[r["piid_stage"] - 1, r["pred_stage"]] += 1
    res = {"model": a.model, "threshold": thr, "piid_total": len(items), "overlap_removed": len(dropped),
           "external_n": len(rows), "sensitivity": sens, "sensitivity_ci95": sens_ci,
           "sensitivity_per_stage": {f"stage{s}": {"rate": v[0], "ci95": v[1]} for s, v in per_stage.items()},
           "stage_accuracy_detected": acc, "stage_accuracy_ci95": acc_ci, "quadratic_kappa": kappa,
           "confusion_rows_piid_stage_cols_pred0to4": cm.tolist(),
           "overlap_examples": [(p.name, h) for p, _, h in dropped[:10]]}
    out = ROOT / "results" / "external"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"external_piid_{a.model}.json").write_text(json.dumps(res, indent=2))
    with open(out / f"external_piid_{a.model}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({k: res[k] for k in ("external_n", "sensitivity", "stage_accuracy_detected", "quadratic_kappa")}, indent=2))


if __name__ == "__main__":
    main()
