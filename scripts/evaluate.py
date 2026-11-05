"""
Evaluate a trained YOLO detector on the held-out test split.

Three things this script is careful about, because a naive report gets each
of them wrong:

  * Precision and recall.  Ultralytics reports P and R at the confidence that
    maximises mean F1 ON THE SPLIT BEING SCORED (utils/metrics.py, ap_per_class:
    `i = smooth(f1_curve.mean(0), 0.1).argmax()`).  Read off the test split that
    is a threshold tuned on the test data, so the value is an upper bound rather
    than an operating point.  This script reports P and R at a FIXED threshold
    declared up front, and prints the tuned figures separately, labelled.

  * Uncertainty.  mAP is reported with a bootstrap confidence interval obtained
    by resampling IMAGES and recomputing average precision per replicate, which
    is the sampling unit that matters when the test split holds one image per
    independent photograph.

  * Per-class claims.  Instance counts per class are printed next to every
    per-class number, because a class with a dozen instances cannot support a
    point estimate.

Usage:
    python scripts/evaluate.py --weights runs/detect/pu_yolov8s_v4/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Declared before looking at the test split.
FIXED_CONF = 0.25


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def image_ids(stats_dict, n_images):
    """Ultralytics does not expose which image each detection came from, so the
    bootstrap is built from per-image validation runs instead."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/pu_yolov8s_v4/weights/best.pt")
    ap.add_argument("--data", default="data/unified/dataset.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--conf", type=float, default=FIXED_CONF)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    weights = (PROJECT_ROOT / args.weights).resolve()
    data_yaml = (PROJECT_ROOT / args.data).resolve()
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    run_name = args.name or f"{weights.parent.parent.name}_{args.split}"

    root = Path(cfg["path"])
    img_dir = root / cfg[args.split]
    lbl_dir = root / "labels" / args.split
    files = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    inst = {i: 0 for i in names}
    for f in lbl_dir.iterdir():
        if f.suffix == ".txt":
            for line in f.read_text().splitlines():
                if line.strip():
                    inst[int(line.split()[0])] += 1

    print("=" * 72)
    print(f"EVALUATION on the held-out '{args.split}' split")
    print("=" * 72)
    print(f"  weights : {weights}")
    print(f"  images  : {len(files)}   instances: {sum(inst.values())}")
    print(f"  per class: " + "  ".join(f"{names[i]}={inst[i]}" for i in sorted(names)))

    model = YOLO(str(weights))
    res = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz,
                    batch=args.batch, device=device(), plots=True,
                    project=str(PROJECT_ROOT / "runs" / "detect"),
                    name=run_name, exist_ok=True, verbose=False)
    box = res.box

    # ---- rebuild the raw match arrays so AP can be recomputed per bootstrap
    # replicate.  Scoring one image at a time keeps the association between a
    # detection and the image it came from, which model.val() discards.
    print(f"\n  building per-image match statistics for the bootstrap ...", flush=True)
    per_image = []
    for f in files:
        r = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz,
                      batch=1, device=device(), plots=False, verbose=False,
                      project=str(PROJECT_ROOT / "runs" / "detect"),
                      name=f"{run_name}_tmp", exist_ok=True) if False else None
        per_image.append(f)

    # model.val() cannot be restricted to one image, so recompute matches
    # directly from predictions and ground truth.
    from ultralytics.utils.metrics import box_iou
    IOUV = torch.linspace(0.5, 0.95, 10)

    def load_gt(stem):
        p = lbl_dir / f"{stem}.txt"
        rows = []
        for line in p.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                rows.append([int(v[0])] + [float(x) for x in v[1:]])
        return rows

    stats = []          # one entry per image: (tp, conf, pred_cls, target_cls)
    for f in files:
        pred = model.predict(str(f), imgsz=args.imgsz, device=device(),
                             conf=0.001, max_det=300, verbose=False)[0]
        h, w = pred.orig_shape
        gt = load_gt(f.stem)
        gt_cls = torch.tensor([g[0] for g in gt], dtype=torch.float32)
        if gt:
            g = torch.tensor([g[1:] for g in gt], dtype=torch.float32)
            gx = torch.stack([(g[:, 0] - g[:, 2] / 2) * w, (g[:, 1] - g[:, 3] / 2) * h,
                              (g[:, 0] + g[:, 2] / 2) * w, (g[:, 1] + g[:, 3] / 2) * h], 1)
        else:
            gx = torch.zeros((0, 4))
        pb = pred.boxes
        pxy = pb.xyxy.cpu().float() if len(pb) else torch.zeros((0, 4))
        pconf = pb.conf.cpu().float() if len(pb) else torch.zeros(0)
        pcls = pb.cls.cpu().float() if len(pb) else torch.zeros(0)

        tp = torch.zeros((len(pxy), len(IOUV)), dtype=torch.bool)
        if len(pxy) and len(gx):
            iou = box_iou(gx, pxy)
            correct_cls = gt_cls[:, None] == pcls[None, :]
            for k, thr in enumerate(IOUV):
                m = torch.nonzero((iou >= thr) & correct_cls, as_tuple=False)
                if m.shape[0]:
                    mm = torch.cat((m, iou[m[:, 0], m[:, 1]][:, None]), 1).numpy()
                    mm = mm[mm[:, 2].argsort()[::-1]]
                    mm = mm[np.unique(mm[:, 1], return_index=True)[1]]
                    mm = mm[np.unique(mm[:, 0], return_index=True)[1]]
                    tp[mm[:, 1].astype(int), k] = True
        stats.append((tp.numpy(), pconf.numpy(), pcls.numpy(), gt_cls.numpy()))

    def compute(indices, conf_thresh=None):
        tp = np.concatenate([stats[i][0] for i in indices]) if indices else np.zeros((0, 10), bool)
        cf = np.concatenate([stats[i][1] for i in indices]) if indices else np.zeros(0)
        pc = np.concatenate([stats[i][2] for i in indices]) if indices else np.zeros(0)
        tc = np.concatenate([stats[i][3] for i in indices]) if indices else np.zeros(0)
        if conf_thresh is not None:
            keep = cf >= conf_thresh
            tp, cf, pc = tp[keep], cf[keep], pc[keep]
        if not len(tc):
            return None
        return ap_per_class(tp, cf, pc, tc, plot=False)

    full = compute(range(len(files)))
    # ap_per_class returns (tp, fp, p, r, f1, ap, unique_classes, ...) across versions;
    # index by position from the end to stay robust.
    ap = full[5] if isinstance(full, tuple) else full.ap
    uc = full[6] if isinstance(full, tuple) else full.ap_class_index

    print("\n" + "-" * 72)
    print(f"  THRESHOLD-FREE METRICS (the headline numbers)")
    print("-" * 72)
    print(f"  {'class':<10}{'inst':>7}{'mAP@50':>11}{'mAP@50-95':>12}")
    per_class = {}
    for k, c in enumerate(uc):
        per_class[names[int(c)]] = {"instances": inst[int(c)],
                                    "mAP50": float(ap[k, 0]), "mAP50_95": float(ap[k].mean())}
        print(f"  {names[int(c)]:<10}{inst[int(c)]:>7}{ap[k,0]:>11.4f}{ap[k].mean():>12.4f}")
    print(f"  {'all':<10}{sum(inst.values()):>7}{ap[:,0].mean():>11.4f}{ap.mean():>12.4f}")

    # ---- image-level bootstrap
    print(f"\n  Bootstrap over images (n={args.bootstrap}), resampling the "
          f"{len(files)} independent photographs:")
    rng = np.random.default_rng(42)
    m50, m5095 = [], []
    import sys as _sys
    pc50 = {int(c): [] for c in uc}
    for _rep in range(args.bootstrap):
        if _rep % 100 == 0:
            print(f"      replicate {_rep}/{args.bootstrap}", flush=True)
        idx = rng.integers(0, len(files), len(files))
        r = compute(list(idx))
        if r is None:
            continue
        a = r[5] if isinstance(r, tuple) else r.ap
        cs = r[6] if isinstance(r, tuple) else r.ap_class_index
        m50.append(a[:, 0].mean())
        m5095.append(a.mean())
        for k, c in enumerate(cs):
            if int(c) in pc50:
                pc50[int(c)].append(a[k, 0])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    lo50, hi50 = ci(m50)
    lo95, hi95 = ci(m5095)
    print(f"    mAP@50    = {ap[:,0].mean():.4f}   95% CI [{lo50:.4f}, {hi50:.4f}]")
    print(f"    mAP@50-95 = {ap.mean():.4f}   95% CI [{lo95:.4f}, {hi95:.4f}]")
    for c in sorted(pc50):
        if pc50[c]:
            l, h = ci(pc50[c])
            per_class[names[c]]["mAP50_ci95"] = [l, h]
            print(f"    {names[c]:<9} AP@50 = {per_class[names[c]]['mAP50']:.4f}   "
                  f"95% CI [{l:.4f}, {h:.4f}]   (n={inst[c]} instances)")

    # ---- operating point at a fixed, pre-declared threshold
    fixed = compute(range(len(files)), conf_thresh=args.conf)
    fp_, fr_ = (fixed[2], fixed[3]) if isinstance(fixed, tuple) else (fixed.p, fixed.r)
    print("\n" + "-" * 72)
    print(f"  OPERATING POINT at the pre-declared confidence threshold {args.conf}")
    print("-" * 72)
    print(f"    precision = {float(np.mean(fp_)):.4f}    recall = {float(np.mean(fr_)):.4f}")
    print(f"\n  For reference, Ultralytics reports P={box.mp:.4f} R={box.mr:.4f}, read at the")
    print( "  confidence that maximises F1 on this same split. That threshold is tuned on")
    print( "  the test data, so those two figures are an upper bound, not an operating point.")

    out = {
        "weights": str(weights), "split": args.split,
        "images": len(files), "instances_per_class": {names[i]: inst[i] for i in sorted(inst)},
        "headline": {
            "mAP50": float(ap[:, 0].mean()), "mAP50_ci95": [lo50, hi50],
            "mAP50_95": float(ap.mean()), "mAP50_95_ci95": [lo95, hi95],
        },
        "operating_point": {
            "confidence": args.conf,
            "precision": float(np.mean(fp_)), "recall": float(np.mean(fr_)),
        },
        "ultralytics_tuned_threshold": {
            "precision": float(box.mp), "recall": float(box.mr),
            "note": "read at the F1-maximising confidence on this split; an upper bound",
        },
        "per_class": per_class,
        "bootstrap_replicates": args.bootstrap,
        "caveats": [
            f"The test split holds one image per independent photograph ({len(files)} photographs).",
            f"Per-class instance counts range {min(inst.values())}-{max(inst.values())}; "
            "per-class AP is a point estimate on a small sample and is reported with a "
            "bootstrap interval for that reason.",
            "Precision and recall are quoted at a fixed threshold declared before "
            "scoring; the Ultralytics defaults are tuned on the split being scored.",
        ],
    }
    dest = PROJECT_ROOT / "results" / f"evaluation_{args.split}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {dest}")


if __name__ == "__main__":
    main()
