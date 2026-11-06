"""Promotion gate and rollback (feature F7, SPEC §3 step 8).

A newly trained model (the *candidate*) replaces the deployed model (the *current*
version) only if the frozen test set shows that it is better. The rule was fixed
before any result was seen (docs/ROADMAP.md, "Gate (system)"):

1. Evidence: the per-image binary Dice (ulcer vs background) of both models on the
   ulcer-containing images of the frozen TEST split -- the primary metric of the
   thesis -- paired by image name, so both models are scored on identical images.
   Each model is applied at its own operating threshold (chosen on its validation
   split), i.e. as it would be deployed. Ulcer-free test images are not part of the
   comparison (their Dice is 1 or 0 and measures false positives, which the
   evaluation reports separately).
2. Statistic: d_i = Dice_candidate(i) - Dice_current(i). The paired bootstrap
   (`selftrain.metrics.paired_bootstrap`: B = 2000 resamples of the n paired images
   with replacement, fixed seed) gives the mean difference mean(d) and its
   percentile 95 % confidence interval [lo, hi].
3. Decision: promote iff there is no current model, or

       mean(d) > 0   and   lo > -margin          (margin = gate_margin = 0.02),

   i.e. the candidate is better on average and not plausibly worse by more than
   the non-inferiority margin. A tie (mean(d) = 0) keeps the current model; so does a
   candidate that cannot be compared because no paired image exists.

The registry keeps every version, promoted or not, so `rollback` can make any
earlier version current again.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..config import BOOTSTRAP
from ..metrics import paired_bootstrap

if TYPE_CHECKING:
    from .registry import VersionRecord
    from .workspace import Workspace

DEFAULT_MARGIN = 0.02          # non-inferiority margin of the pre-registered gate


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the gate for one candidate version.

    `mean_diff`, `ci95` and `p_value` describe candidate - current and are None when
    no comparison was made (no current model, or no paired test images). `n` is the
    number of paired test images; `compared_to` the current version."""
    promoted: bool
    reason: str
    mean_diff: float | None = None
    ci95: tuple[float, float] | None = None
    p_value: float | None = None
    n: int = 0
    margin: float = DEFAULT_MARGIN
    compared_to: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, stored as VersionRecord.gate in the registry."""
        return {"promoted": self.promoted, "mean_diff": self.mean_diff,
                "ci95": list(self.ci95) if self.ci95 is not None else None,
                "p_value": self.p_value, "n": self.n, "margin": self.margin,
                "compared_to": self.compared_to, "reason": self.reason}


def decide(candidate: Sequence[float], current: Sequence[float] | None,
           margin: float = DEFAULT_MARGIN, n_boot: int = BOOTSTRAP, seed: int = 0,
           current_version: int | None = None) -> GateDecision:
    """Apply the gate rule (module docstring) to paired per-image test Dice values.

    `candidate[i]` and `current[i]` must belong to the same test image. `current` is
    None when no model is deployed yet; the candidate is then promoted without a
    comparison. With zero paired images the rule cannot be evaluated and the
    candidate is not promoted. Raises ValueError for a negative margin, arrays of
    different shape and non-finite Dice values."""
    if margin < 0:
        raise ValueError(f"gate margin must be >= 0, got {margin}")
    if current is None:
        return GateDecision(True, "no current model: the first version is promoted",
                            n=len(candidate), margin=margin)
    a, b = np.asarray(candidate, dtype=float), np.asarray(current, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired Dice arrays differ in shape: {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("per-image Dice values must be finite")
    if a.size == 0:
        return GateDecision(False, "no paired ulcer-containing test images: the candidate "
                                   "cannot be compared and is not promoted",
                            margin=margin, compared_to=current_version)
    pb = paired_bootstrap(a, b, b=n_boot, seed=seed)
    mean_diff, (lo, hi) = pb["mean_diff"], pb["ci95"]
    better = mean_diff > 0
    non_inferior = lo > -margin
    promoted = better and non_inferior
    stats = (f"mean Dice difference {mean_diff:+.4f}, 95 % CI [{lo:+.4f}, {hi:+.4f}], "
             f"n = {pb['n']} paired test images")
    if promoted:
        reason = f"promoted: {stats}; better on average and lower bound > -{margin:.4f}"
    elif not better:
        reason = f"not promoted: {stats}; no improvement on average (mean difference <= 0)"
    else:
        reason = (f"not promoted: {stats}; lower bound <= -{margin:.4f}, the candidate may be "
                  f"worse by more than the margin")
    return GateDecision(promoted, reason, mean_diff=mean_diff, ci95=(lo, hi),
                        p_value=pb["p_value"], n=pb["n"], margin=margin,
                        compared_to=current_version)


def load_test_dice(per_image_csv: Path, positives_only: bool = True) -> dict[str, float]:
    """Per-image binary Dice from a `selftrain.evaluate` per-image CSV, keyed by image
    name. With `positives_only` (the gate's evidence) only images whose ground truth
    contains an ulcer (`gt_positive` = True) are returned."""
    out: dict[str, float] = {}
    with open(per_image_csv, newline="") as f:
        for row in csv.DictReader(f):
            if positives_only and row["gt_positive"] != "True":
                continue
            out[row["name"]] = float(row["dice"])
    return out


def pair_by_name(candidate: Mapping[str, float], current: Mapping[str, float]
                 ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align two name -> Dice maps (sorted by name). The gate compares models on
    identical images, so the two maps must cover exactly the same names; otherwise
    ValueError."""
    if set(candidate) != set(current):
        only_c = sorted(set(candidate) - set(current))
        only_r = sorted(set(current) - set(candidate))
        raise ValueError(f"test images differ: {len(only_c)} only in candidate "
                         f"{only_c[:3]}, {len(only_r)} only in current {only_r[:3]}")
    names = sorted(candidate)
    return (names, np.array([candidate[n] for n in names], dtype=float),
            np.array([current[n] for n in names], dtype=float))


def rollback(ws: Workspace, version: int) -> VersionRecord:
    """Manual override of the gate: make the registered `version` current again.

    Every version stays in the registry, so any earlier model -- promoted by the gate
    or not -- can be restored, e.g. when a promoted model turns out to be worse in
    practice. The registry records the change in its history (action "rollback");
    the gate decision stored with each version is not altered. The cycle lock is
    held while the registry changes, so a rollback is refused (CycleLockedError)
    while a cycle runs -- that cycle's gate compares its candidate with the current
    model. Afterwards status.json is refreshed, so the dashboard shows the new
    current model at once. Returns the record of the now current version."""
    from . import registry
    from .cycle import CycleLock, lock_path, refresh_status   # cycle imports this module
    with CycleLock(lock_path(ws)):
        registry.rollback(ws, version)
        refresh_status(ws)
    return registry.get_version(ws, version)
