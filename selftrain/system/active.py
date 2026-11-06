"""Active learning: rank unlabeled photographs by model uncertainty (feature F14).

Expert labelling time is the scarce resource, so photographs the current model is
least sure about are offered first. Method, for every image below inbox/unlabeled/:

1. The current model (registry) predicts the per-pixel foreground confidence
   p(x) in [0, 1] (`predictor.fg_confidence`: U-Net 1 - P(background); YOLOv8-seg the
   highest confidence of an instance mask covering x) and the semantic prediction at
   its operating threshold (chosen on validation data), with predicted regions
   smaller than `min_component_frac` of the image removed as in the evaluation.
2. Pixel uncertainty u(x) = 1 - |2 p(x) - 1|: 1 at p = 0.5, 0 at p = 0 or 1.
3. Image score = mean of u over the predicted-or-candidate region
   R = {x : predicted ulcer} ∪ {x : p(x) >= 0.1}. Averaging over R rather than the
   whole image keeps the large, confidently background area from diluting the
   score. An empty prediction with max p < 0.1 (R empty) scores 0: the model is
   confident the photograph shows no ulcer.
4. Photographs are ranked by decreasing score (ties by path). The ranking is written
   to <workspace>/active/ranking.csv (path, uncertainty, predicted_area,
   predicted_stage, thumbnail) and the top_k get overlay thumbnails in
   <workspace>/active/thumbnails/ (prediction coloured by stage, R outlined).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import pandas as pd

from ..config import MIN_COMPONENT_FRAC
from ..metrics import dominant_stage, remove_specks
from .ingest import IMAGE_SUFFIXES
from .steps import load_predictor, operating_threshold

if TYPE_CHECKING:
    from .workspace import Workspace

log = logging.getLogger(__name__)

CANDIDATE_P = 0.1                      # p >= 0.1: the model considers an ulcer possible
IMAGE_EXTENSIONS = IMAGE_SUFFIXES
"""Photographs ranked: every image type ingest moves to inbox/unlabeled/ (ingest.IMAGE_SUFFIXES)."""
RANKING_COLUMNS = ["path", "uncertainty", "predicted_area", "predicted_stage", "thumbnail"]
THUMB_SIDE = 320
OVERLAY_ALPHA = 0.45
STAGE_COLOURS_BGR = {1: (0, 215, 255), 2: (0, 140, 255), 3: (40, 40, 230), 4: (170, 60, 170)}
CANDIDATE_COLOUR_BGR = (255, 255, 0)


@dataclass(frozen=True)
class PhotoScore:
    """Ranking row for one unlabeled photograph."""
    path: Path
    uncertainty: float
    predicted_area: float
    predicted_stage: int


def uncertainty_score(p: np.ndarray, predicted: np.ndarray,
                      candidate_p: float = CANDIDATE_P) -> float:
    """Mean of 1 - |2p - 1| over R = predicted ∪ {p >= candidate_p}; 0 if R is empty."""
    p = np.asarray(p, dtype=np.float32)
    region = np.asarray(predicted, dtype=bool) | (p >= candidate_p)
    if not region.any():
        return 0.0
    return float(np.mean(1.0 - np.abs(2.0 * p[region] - 1.0)))


def list_unlabeled(ws: Workspace) -> list[Path]:
    """Image files below inbox/unlabeled/ (recursive, hidden files ignored), sorted."""
    root = ws.unlabeled
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def predict(predictor: Any, img_bgr: np.ndarray, threshold: float,
            min_component_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """(semantic prediction 0-4 after speck removal, foreground confidence p)."""
    raw = predictor.infer(img_bgr)
    sem = predictor.to_semantic(raw, threshold)
    sem = np.where(remove_specks(sem > 0, min_component_frac), sem, 0).astype(np.uint8)
    return sem, np.asarray(predictor.fg_confidence(raw), dtype=np.float32)


def render_thumbnail(img_bgr: np.ndarray, sem: np.ndarray, p: np.ndarray, caption: str,
                     side: int = THUMB_SIDE) -> np.ndarray:
    """Photograph with the prediction tinted by stage and R outlined, resized so the
    long side is `side` pixels, with `caption` in the top-left corner."""
    out = img_bgr.copy()
    for stage, colour in STAGE_COLOURS_BGR.items():
        m = sem == stage
        if m.any():
            out[m] = ((1 - OVERLAY_ALPHA) * out[m] + OVERLAY_ALPHA * np.array(colour)).astype(np.uint8)
    region = ((sem > 0) | (p >= CANDIDATE_P)).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = out.shape[:2]
    scale = side / max(h, w)
    cv2.drawContours(out, contours, -1, CANDIDATE_COLOUR_BGR, max(1, round(2 / scale)))
    out = cv2.resize(out, (max(1, round(w * scale)), max(1, round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    (tw, th), base = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (0, 0), (tw + 8, th + base + 8), (0, 0, 0), -1)
    cv2.putText(out, caption, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


class NoCurrentModelError(LookupError):
    """The workspace has no current model to rank with."""


def _current_model(ws: Workspace) -> tuple[Any, float]:
    """Predictor and operating threshold of the registry's current version."""
    from . import registry
    rec = registry.current(ws)
    if rec is None:
        raise NoCurrentModelError("no current model: run `python -m selftrain init` "
                                  "or a cycle first")
    return (load_predictor(rec.architecture, rec.weights_path(ws)),
            operating_threshold(ws, rec.version, rec.architecture))


def rank_unlabeled(ws: Workspace, top_k: int = 20, predictor: Any = None,
                   threshold: float | None = None,
                   min_component_frac: float | None = None) -> pd.DataFrame:
    """Score and rank every photograph in inbox/unlabeled/ (method: module docstring).

    `predictor`/`threshold` default to the registry's current version and its
    operating threshold; `min_component_frac` to the workspace configuration.
    Unreadable images are skipped with a warning. Writes active/ranking.csv and the
    top_k thumbnails (replacing those of an earlier ranking) and returns the ranking;
    without photos the ranking is empty and no model is needed. Raises
    NoCurrentModelError if photos wait but no model is current."""
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    photos = list_unlabeled(ws)
    if photos and predictor is None:
        predictor, default_threshold = _current_model(ws)
        threshold = default_threshold if threshold is None else threshold
    if photos and threshold is None:
        raise ValueError("a threshold is required together with an explicit predictor")
    if min_component_frac is None:
        min_component_frac = getattr(ws.load_config(), "min_component_frac", MIN_COMPONENT_FRAC)
    out_dir = ws.active
    thumbs = out_dir / "thumbnails"
    if thumbs.exists():
        shutil.rmtree(thumbs)
    thumbs.mkdir(parents=True)

    scores: list[PhotoScore] = []
    for path in photos:
        img = cv2.imread(str(path))
        if img is None:
            log.warning("skipping unreadable image %s", path)
            continue
        sem, p = predict(predictor, img, threshold, min_component_frac)
        scores.append(PhotoScore(path.resolve(), uncertainty_score(p, sem > 0),
                                 float((sem > 0).mean()), dominant_stage(sem)))
    scores.sort(key=lambda s: (-s.uncertainty, str(s.path)))

    rows = []
    for rank, score in enumerate(scores, start=1):
        thumb = ""
        if rank <= top_k:      # predicted again: full-resolution maps are not kept for all
            img = cv2.imread(str(score.path))
            sem, p = predict(predictor, img, threshold, min_component_frac)
            name = f"{rank:03d}_{score.path.stem}.jpg"
            caption = f"#{rank} u={score.uncertainty:.3f}"
            cv2.imwrite(str(thumbs / name), render_thumbnail(img, sem, p, caption),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            thumb = f"thumbnails/{name}"
        rows.append({"path": str(score.path), "uncertainty": score.uncertainty,
                     "predicted_area": score.predicted_area,
                     "predicted_stage": score.predicted_stage, "thumbnail": thumb})
    ranking = pd.DataFrame(rows, columns=RANKING_COLUMNS)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    return ranking
