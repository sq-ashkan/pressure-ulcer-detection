"""Distribution drift of new photographs (SPEC §3 step 4, feature F16).

Question: do the photographs of a new batch look like the photographs on which the
models were trained and evaluated? A batch from another camera, ward or lighting
situation can lower the accuracy of a model even when every label is correct; the drift
step makes such a shift visible in the cycle report. Drift is a warning, never a blocker:
the result is reported, it does not stop a cycle (SPEC §3).

Per-image features (FEATURES). Colour and sharpness are measured on the photograph as
both models see it: decoded by OpenCV (EXIF orientation applied) and scaled with the
letterbox rule of the models (selftrain.letterbox: longer side IMG_SIZE = 512 px,
bilinear), without the grey padding.
  lightness       mean L* of CIELAB (sRGB input, D65 white; 0 = black, 100 = white)
  red_green       mean a* of CIELAB (> 0 reddish, < 0 greenish)
  yellow_blue     mean b* of CIELAB (> 0 yellowish, < 0 bluish)
  colourfulness   Hasler & Suesstrunk (2003): with rg = R - G and yb = (R + G) / 2 - B on
                  0-255 RGB, M = sqrt(sd(rg)^2 + sd(yb)^2) + 0.3 sqrt(mean(rg)^2 + mean(yb)^2)
  sharpness       variance of the Laplacian (3 x 3 aperture) of the grey image
                  (Pech-Pacheco et al. 2000); measured at model resolution so that it does
                  not merely restate the resolution
  log_resolution  log10(width x height) of the ORIGINAL image

Test. For every feature a two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp,
two-sided; method "auto": exact p-value for samples of up to 10 000, asymptotic beyond)
compares the new photographs with the reference photographs. The family-wise error rate
over the k = 6 features is controlled with the Bonferroni correction: feature j has
shifted iff p_j <= alpha / k, i.e. iff the adjusted p-value min(1, k p_j) <= alpha
(alpha = drift_alpha, default 0.01; the rule of statsmodels' multipletests). Drift is
reported iff at least one feature shifted. The KS statistic D (the largest distance
between the two empirical distribution functions, 0-1) is the effect size; the medians
give the direction of a shift. The KS test is rank-based, so the log of the resolution
changes only the reported medians, not the test.

Power. With n new and m reference photographs the smallest p-value the test can return
is that of two completely separated samples (D = 1, p = 2 / C(n + m, n) for the exact
test). It is reported as min_attainable_p; when it exceeds alpha / k no shift can be
detected at this sample size and the report says so. With fewer than MIN_IMAGES = 5
readable photographs on either side the test is skipped with a note.

Independence. The KS test assumes independent observations. Pass ONE image per
photograph: pool_reference does so for the reference (the first image of every
photograph); augmented copies of one photograph would otherwise count as several
observations and make the test anti-conservative.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.stats import ks_2samp

from ..config import IMG_SIZE
from ..letterbox import letterbox

if TYPE_CHECKING:                      # type hints only
    from .workspace import Workspace

__all__ = [
    "FEATURES", "MIN_IMAGES", "REFERENCE_SPLITS", "compare_features", "drift_report",
    "feature_matrix", "features_of", "image_features", "min_attainable_p", "pool_reference",
]

_LOG = logging.getLogger(__name__)
Log = Callable[[str], None]

FEATURES: Mapping[str, str] = MappingProxyType({
    "lightness": "mean L* (CIELAB lightness, 0-100)",
    "red_green": "mean a* (CIELAB, + red / - green)",
    "yellow_blue": "mean b* (CIELAB, + yellow / - blue)",
    "colourfulness": "Hasler-Suesstrunk colourfulness M",
    "sharpness": "variance of the Laplacian of the grey image at model resolution",
    "log_resolution": "log10 of width x height of the original image",
})
"""Feature name -> description, in the column order of feature_matrix."""

MIN_IMAGES = 5
"""Fewer readable new or reference photographs than this: the test is skipped."""

REFERENCE_SPLITS: tuple[str, ...] = ("train", "val", "test")
"""Pool splits that form the reference distribution. The external split is excluded: it
is a held-out source (a "new clinic") that no model learns from."""


# ------------------------------------------------------------------ features

def features_of(img_bgr: np.ndarray) -> dict[str, float]:
    """The FEATURES of one decoded 8-bit BGR image (method: module docstring)."""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {img_bgr.shape}")
    height, width = img_bgr.shape[:2]
    canvas, (top, left, nh, nw) = letterbox(img_bgr, IMG_SIZE)
    view = canvas[top:top + nh, left:left + nw]
    lab = cv2.cvtColor(view.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    lightness, red_green, yellow_blue = (float(lab[..., i].mean()) for i in range(3))
    bgr = view.astype(np.float64)
    blue, green, red = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colourfulness = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    grey = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return {"lightness": lightness, "red_green": red_green, "yellow_blue": yellow_blue,
            "colourfulness": float(colourfulness), "sharpness": sharpness,
            "log_resolution": math.log10(height * width)}


def image_features(path: Path) -> dict[str, float] | None:
    """The FEATURES of an image file; None if OpenCV cannot decode it."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if img is None else features_of(img)


def feature_matrix(paths: Sequence[Path]) -> tuple[np.ndarray, list[Path], list[Path]]:
    """Features of many images: an (n, k) array (columns in FEATURES order) for the n
    readable images, the readable paths and the unreadable paths (both in input order)."""
    rows: list[list[float]] = []
    readable: list[Path] = []
    unreadable: list[Path] = []
    for path in paths:
        feats = image_features(Path(path))
        if feats is None:
            unreadable.append(Path(path))
            continue
        rows.append([feats[name] for name in FEATURES])
        readable.append(Path(path))
    table = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURES))
    return table, readable, unreadable


# ------------------------------------------------------------------ the test

def min_attainable_p(n_new: int, n_reference: int) -> float:
    """Smallest p-value ks_2samp can return for these sample sizes: the p-value of two
    completely separated samples (D = 1), computed with ks_2samp itself."""
    if n_new < 1 or n_reference < 1:
        raise ValueError("both samples need at least one observation")
    low, high = np.arange(n_new, dtype=float), np.arange(n_new, n_new + n_reference, dtype=float)
    return float(ks_2samp(low, high).pvalue)


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    return a


def compare_features(new: np.ndarray, reference: np.ndarray, alpha: float) -> dict[str, Any]:
    """KS test per feature with Bonferroni correction (module docstring).

    `new` (n, k) and `reference` (m, k) hold the features in FEATURES order. Returns the
    per-feature results (D, p, adjusted p, shifted, medians, means, direction), the list
    of shifted features, the overall decision and the power information."""
    alpha = _check_alpha(alpha)
    new, reference = np.asarray(new, np.float64), np.asarray(reference, np.float64)
    k = len(FEATURES)
    if new.ndim != 2 or reference.ndim != 2 or new.shape[1] != k or reference.shape[1] != k:
        raise ValueError(f"feature tables must have {k} columns, got {new.shape} and {reference.shape}")
    if len(new) < 1 or len(reference) < 1:
        raise ValueError("both feature tables need at least one row")
    alpha_per_test = alpha / k
    features: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(FEATURES):
        x, y = new[:, j], reference[:, j]
        res = ks_2samp(x, y, alternative="two-sided", method="auto")
        p = float(res.pvalue)
        med_x, med_y = float(np.median(x)), float(np.median(y))
        features[name] = {
            "description": FEATURES[name],
            "ks_statistic": float(res.statistic),
            "p_value": p,
            "p_adjusted": min(1.0, k * p),
            "shifted": p <= alpha_per_test,
            "new_median": med_x, "reference_median": med_y,
            "new_mean": float(x.mean()), "reference_mean": float(y.mean()),
            "direction": "higher" if med_x > med_y else "lower" if med_x < med_y else "equal",
        }
    shifted = [name for name in FEATURES if features[name]["shifted"]]
    floor = min_attainable_p(len(new), len(reference))
    return {"alpha": alpha, "correction": "bonferroni", "n_tests": k,
            "alpha_per_test": alpha_per_test, "n_new": int(len(new)),
            "n_reference": int(len(reference)), "drift": bool(shifted), "shifted": shifted,
            "min_attainable_p": floor, "detectable": floor <= alpha_per_test,
            "features": features}


def _summary(result: Mapping[str, Any]) -> str:
    """One line for the dashboard."""
    sizes = f"{result['n_new']} new vs {result['n_reference']} reference photographs"
    if result["drift"]:
        shifted = ", ".join(f"{name} {result['features'][name]['direction']}"
                            for name in result["shifted"])
        return (f"drift in {len(result['shifted'])} of {result['n_tests']} features: {shifted} "
                f"({sizes}; KS, Bonferroni alpha {result['alpha']:g})")
    if not result["detectable"]:
        return f"no drift detectable: too few photographs ({sizes})"
    return f"no drift in {result['n_tests']} features ({sizes})"


def drift_report(new_paths: Sequence[Path], reference_paths: Sequence[Path], alpha: float,
                 *, log: Log | None = None) -> dict[str, Any]:
    """Drift of the photographs `new_paths` against `reference_paths` (method: module
    docstring) as a JSON-ready dict.

    Keys: status ("tested" or "skipped"), drift (bool), shifted (feature names), summary
    (one line), note (explanation or None), alpha, correction, n_tests, alpha_per_test,
    n_new / n_reference (readable photographs), unreadable_new / unreadable_reference
    (paths), min_attainable_p, detectable, features ({name: {description, ks_statistic,
    p_value, p_adjusted, shifted, new_median, reference_median, new_mean, reference_mean,
    direction}})."""
    alpha = _check_alpha(alpha)
    log = log or _LOG.info
    new, new_ok, new_bad = feature_matrix(list(new_paths))
    ref, ref_ok, ref_bad = feature_matrix(list(reference_paths))
    unreadable = {"unreadable_new": [str(p) for p in new_bad],
                  "unreadable_reference": [str(p) for p in ref_bad]}
    if len(new_ok) < MIN_IMAGES or len(ref_ok) < MIN_IMAGES:
        short = [f"{n} readable {what} photograph{'s' if n != 1 else ''}"
                 for n, what in ((len(new_ok), "new"), (len(ref_ok), "reference"))
                 if n < MIN_IMAGES]
        note = (f"drift test skipped: {' and '.join(short)}; the KS tests need at least "
                f"{MIN_IMAGES} on each side")
        log(note)
        return {"status": "skipped", "drift": False, "shifted": [], "summary": note,
                "note": note, "alpha": alpha, "correction": "bonferroni",
                "n_tests": len(FEATURES), "alpha_per_test": alpha / len(FEATURES),
                "n_new": len(new_ok), "n_reference": len(ref_ok), **unreadable,
                "min_attainable_p": None, "detectable": False, "features": {}}
    result = compare_features(new, ref, alpha)
    note = None
    if not result["detectable"]:
        note = (f"with {result['n_new']} new and {result['n_reference']} reference photographs "
                f"the smallest attainable p-value ({result['min_attainable_p']:.2g}) exceeds "
                f"alpha / k = {result['alpha_per_test']:.2g}: a shift could not be detected")
    summary = _summary(result)
    log(summary)
    return {"status": "tested", "summary": summary, "note": note, **unreadable, **result}


# ------------------------------------------------------------------ reference sample

def pool_reference(ws: Workspace, splits: Sequence[str] = REFERENCE_SPLITS, *,
                   one_per_photograph: bool = True, max_images: int | None = None,
                   seed: int = 0) -> list[Path]:
    """Pool images that form the reference distribution of the drift test.

    Images of the pool manifest whose split is in `splits` (default train, val, test),
    sorted by name; with `one_per_photograph` only the first image of every photograph
    (independent observations, see the module docstring). With `max_images` a seeded
    uniform random subset of that size (numpy default_rng(seed), without replacement)
    bounds the cost; the result stays sorted by name."""
    from .dataset import pool_file     # imported here: dataset pulls in build_seg_dataset
    df = ws.read_manifest()
    df = df.loc[df["split"].isin(list(splits))].sort_values("name", kind="stable")
    if one_per_photograph:
        df = df.drop_duplicates("photograph", keep="first")
    names = df["name"].tolist()
    if max_images is not None:
        if max_images < 1:
            raise ValueError(f"max_images must be >= 1, got {max_images}")
        if len(names) > max_images:
            pick = np.random.default_rng(seed).choice(len(names), size=max_images, replace=False)
            names = [names[i] for i in sorted(pick)]
    return [pool_file(ws, "images", name) for name in names]
