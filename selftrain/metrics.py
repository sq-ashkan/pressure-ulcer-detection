"""Segmentation metrics and bootstrap statistics.

Conventions (stated in the thesis):
  * per-image binary Dice/IoU: ulcer (any stage) vs background; if both prediction
    and ground truth are empty the image scores 1 (correct rejection).
  * per-stage Dice: pooled over all pixels of a split (2TP / (2TP+FP+FN)).
  * an image counts as "predicted positive" if any predicted region larger than
    MIN_COMPONENT_FRAC of the image remains after removing tiny specks.
"""
import cv2
import numpy as np

from .config import MIN_COMPONENT_FRAC, NUM_STAGES


def binary_scores(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    ps, gs = pred.sum(), gt.sum()
    if ps + gs == 0:
        return 1.0, 1.0
    dice = 2.0 * inter / (ps + gs)
    iou = inter / (ps + gs - inter)
    return float(dice), float(iou)


def pixel_confusion(pred: np.ndarray, gt: np.ndarray, n: int = NUM_STAGES + 1):
    """(n x n) counts, rows = ground truth, columns = prediction."""
    idx = gt.astype(np.int64).ravel() * n + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def stage_dice(conf: np.ndarray):
    """Dice per stage (classes 1..n-1) from a pooled pixel confusion matrix."""
    out = []
    for c in range(1, conf.shape[0]):
        tp = conf[c, c]
        denom = conf[c, :].sum() + conf[:, c].sum()
        out.append(float(2 * tp / denom) if denom > 0 else float("nan"))
    return out


def remove_specks(fg: np.ndarray, min_frac: float = MIN_COMPONENT_FRAC):
    fg = fg.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_frac * fg.size
    return keep[lab]


def predicted_positive(sem: np.ndarray, min_frac: float = MIN_COMPONENT_FRAC) -> bool:
    return bool(remove_specks(sem > 0, min_frac).any())


def dominant_stage(sem: np.ndarray) -> int:
    """Stage (1-4) covering most ulcer pixels; 0 if there are none."""
    counts = np.bincount(sem.ravel(), minlength=NUM_STAGES + 1)[1:]
    return int(counts.argmax() + 1) if counts.sum() else 0


def bootstrap_ci(values, b: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (b, len(v)))].mean(1)
    return float(v.mean()), (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_bootstrap(a, b_, b: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean difference a-b over paired samples, percentile CI and two-sided p-value."""
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    assert a.shape == b_.shape
    d = a - b_
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (b, len(d)))].mean(1)
    lo, hi = np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_value": float(min(1.0, p)),
            "n": int(len(d))}


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (false-positive rates, sensitivity)."""
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), (float(max(0.0, c - h)), float(min(1.0, c + h)))
