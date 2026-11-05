"""One evaluator for both architectures.

Protocol (pre-registered in docs/ROADMAP.md):
  1. the operating threshold is chosen on the VALIDATION split only: the value of
     THRESHOLD_GRID that maximises mean per-image binary Dice over all val images;
  2. the frozen TEST split is then evaluated once with that threshold;
  3. primary metric = mean per-image binary Dice over test images that contain an
     ulcer; secondary: IoU, per-stage pooled Dice, dominant-stage accuracy,
     sensitivity, false-positive rate on ulcer-free photographs, inference time.
"""
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .config import BOOTSTRAP, NUM_STAGES, THRESHOLD_GRID
from .metrics import (binary_scores, bootstrap_ci, dominant_stage, pixel_confusion,
                      predicted_positive, remove_specks, stage_dice, wilson_ci)


def split_items(data_root: Path, split: str):
    root = Path(data_root)
    out = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        out.append((img.stem, img, root / "masks" / split / f"{img.stem}.png"))
    return out


def load_cache(predictor, items):
    cache = []
    for name, img_p, mask_p in items:
        img = cv2.imread(str(img_p))
        gt = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        cache.append((name, predictor.infer(img), gt))
    return cache


def choose_threshold(predictor, val_cache):
    table = {}
    for t in THRESHOLD_GRID:
        d = [binary_scores(remove_specks(predictor.to_semantic(raw, t) > 0), gt > 0)[0]
             for _, raw, gt in val_cache]
        table[t] = float(np.mean(d))
    best = max(table, key=lambda k: (table[k], -abs(k - 0.5)))   # tie -> closest to 0.5
    return best, table


def evaluate_cache(predictor, cache, thr, source_of=lambda n: n.split("__")[0]):
    rows, conf = [], np.zeros((NUM_STAGES + 1,) * 2, np.int64)
    for name, raw, gt in cache:
        sem = predictor.to_semantic(raw, thr)
        sem = np.where(remove_specks(sem > 0), sem, 0).astype(np.uint8)
        dice, iou = binary_scores(sem > 0, gt > 0)
        conf += pixel_confusion(sem, gt)
        rows.append({
            "name": name, "source": source_of(name), "gt_positive": bool((gt > 0).any()),
            "pred_positive": predicted_positive(sem), "dice": dice, "iou": iou,
            "gt_stage": dominant_stage(gt), "pred_stage": dominant_stage(sem),
            "gt_area": float((gt > 0).mean()), "pred_area": float((sem > 0).mean()),
        })
    return rows, conf


def summarise(rows, conf, b=BOOTSTRAP):
    pos = [r for r in rows if r["gt_positive"]]
    neg = [r for r in rows if not r["gt_positive"]]
    dice_m, dice_ci = bootstrap_ci([r["dice"] for r in pos], b)
    iou_m, iou_ci = bootstrap_ci([r["iou"] for r in pos], b)
    sens, sens_ci = wilson_ci(sum(r["pred_positive"] for r in pos), len(pos))
    fpr, fpr_ci = wilson_ci(sum(r["pred_positive"] for r in neg), len(neg))
    sd = stage_dice(conf)
    staged = [r for r in pos if r["pred_stage"] > 0]
    stage_acc, stage_ci = wilson_ci(sum(r["pred_stage"] == r["gt_stage"] for r in staged), len(staged))
    cm = np.zeros((NUM_STAGES, NUM_STAGES + 1), int)      # rows gt stage 1-4, cols pred 0-4
    for r in pos:
        cm[r["gt_stage"] - 1, r["pred_stage"]] += 1
    return {
        "n_images": len(rows), "n_positive": len(pos), "n_negative": len(neg),
        "dice_mean": dice_m, "dice_ci95": dice_ci, "iou_mean": iou_m, "iou_ci95": iou_ci,
        "dice_median": float(np.median([r["dice"] for r in pos])) if pos else float("nan"),
        "sensitivity": sens, "sensitivity_ci95": sens_ci,
        "false_positive_rate": fpr, "false_positive_rate_ci95": fpr_ci,
        "stage_dice": dict(zip(["stage1", "stage2", "stage3", "stage4"], sd)),
        "stage_dice_macro": float(np.nanmean(sd)) if not np.all(np.isnan(sd)) else float("nan"),
        "dominant_stage_accuracy": stage_acc, "dominant_stage_accuracy_ci95": stage_ci,
        "stage_confusion_gt_rows_pred_cols0to4": cm.tolist(),
        "pixel_confusion": conf.tolist(),
    }


def evaluate_negatives(predictor, paths_with_class, thr):
    """False-positive rate on ulcer-free photographs, overall and per class."""
    by = {}
    for p, cls in paths_with_class:
        img = cv2.imread(str(p))
        if img is None:
            continue
        sem = predictor.to_semantic(predictor.infer(img), thr)
        by.setdefault(cls, []).append(predicted_positive(sem))
    out = {}
    for cls, v in sorted(by.items()):
        r, ci = wilson_ci(sum(v), len(v))
        out[cls] = {"n": len(v), "false_positive_rate": r, "ci95": ci}
    allv = [x for v in by.values() for x in v]
    r, ci = wilson_ci(sum(allv), len(allv))
    out["all"] = {"n": len(allv), "false_positive_rate": r, "ci95": ci}
    return out


def time_inference(predictor, items, n: int = 30, warmup: int = 5):
    imgs = [cv2.imread(str(p)) for _, p, _ in items[:n + warmup]]
    for im in imgs[:warmup]:
        predictor.infer(im)
    t0 = time.perf_counter()
    for im in imgs[warmup:]:
        predictor.infer(im)
    return 1000 * (time.perf_counter() - t0) / max(1, len(imgs) - warmup)


def write_rows(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def full_evaluation(predictor, data_root: Path, out_dir: Path, label: str, log=print):
    """Threshold on val, then test + external + extra negatives. Writes JSON/CSV."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)
    val_cache = load_cache(predictor, split_items(data_root, "val"))
    thr, table = choose_threshold(predictor, val_cache)
    log(f"[{label}] threshold chosen on val: {thr} (val Dice {table[thr]:.4f})")
    res = {"model": label, "threshold": thr, "val_threshold_table": table}
    for split in ("val", "test", "external"):
        items = split_items(data_root, split)
        if not items:
            continue
        cache = val_cache if split == "val" else load_cache(predictor, items)
        rows, conf = evaluate_cache(predictor, cache, thr)
        write_rows(rows, out_dir / f"{label}_{split}_per_image.csv")
        res[split] = summarise(rows, conf)
        log(f"[{label}] {split}: Dice {res[split]['dice_mean']:.4f} "
            f"CI {res[split]['dice_ci95'][0]:.3f}-{res[split]['dice_ci95'][1]:.3f} "
            f"FPR {res[split]['false_positive_rate']:.3f}")
    extra = data_root / "negatives_extra.csv"
    if extra.exists():
        with open(extra) as f:
            pairs = [(r["path"], r["azh_class"]) for r in csv.DictReader(f)]
        res["extra_negatives"] = evaluate_negatives(predictor, pairs, thr)
        log(f"[{label}] extra negatives FPR {res['extra_negatives']['all']['false_positive_rate']:.3f}")
    res["inference_ms_per_image"] = time_inference(predictor, split_items(data_root, "test"))
    (out_dir / f"{label}_evaluation.json").write_text(json.dumps(res, indent=2))
    return res
