"""
Inference wrapper around the trained pressure ulcer detector.

Kept separate from the user interface so that the same code path can be driven
from a script, a test, or an HTTP endpoint without importing Gradio.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "pu_yolov8s_v4" / "weights" / "best.pt"

# NPUAP/EPUAP stages, in the order the model was trained on.
STAGE_LABELS = {
    0: "Stage I",
    1: "Stage II",
    2: "Stage III",
    3: "Stage IV",
}

STAGE_DESCRIPTIONS = {
    0: "Non-blanchable erythema of intact skin.",
    1: "Partial-thickness skin loss with an exposed dermis.",
    2: "Full-thickness skin loss; subcutaneous fat may be visible.",
    3: "Full-thickness skin and tissue loss with exposed bone, tendon or muscle.",
}

# Colour per stage, deepening as severity rises. BGR, because OpenCV draws in BGR.
STAGE_COLOURS = {
    0: (120, 190, 255),
    1: (90, 150, 250),
    2: (70, 90, 230),
    3: (50, 40, 190),
}


@dataclass
class Detection:
    stage_id: int
    stage: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    area_fraction: float

    def as_dict(self) -> dict:
        return asdict(self)


class PressureUlcerDetector:
    """Loads the trained weights once and serves predictions."""

    def __init__(self, weights: Path | str = DEFAULT_WEIGHTS, device: str | None = None):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"No weights at {self.weights}. Train a model first with "
                f"scripts/train_yolo.py, or pass --weights."
            )
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(self.weights))
        self.imgsz = 640

    def predict(self, image: np.ndarray, conf: float = 0.25,
                iou: float = 0.45) -> tuple[list[Detection], float]:
        """Run detection on one RGB image. Returns detections and elapsed ms."""
        started = time.perf_counter()
        result = self.model.predict(image, imgsz=self.imgsz, conf=conf, iou=iou,
                                    device=self.device, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        h, w = image.shape[:2]
        frame_area = float(h * w)
        detections: list[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0])
            detections.append(Detection(
                stage_id=cls,
                stage=STAGE_LABELS.get(cls, f"class {cls}"),
                confidence=float(box.conf[0]),
                box_xyxy=(x1, y1, x2, y2),
                area_fraction=((x2 - x1) * (y2 - y1)) / frame_area if frame_area else 0.0,
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections, elapsed_ms

    @staticmethod
    def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Return a copy of the image with the detections drawn on it."""
        canvas = image.copy()
        h, w = canvas.shape[:2]
        thickness = max(2, int(round(min(h, w) / 300)))
        font_scale = max(0.5, min(h, w) / 900)

        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            colour = STAGE_COLOURS.get(det.stage_id, (200, 200, 200))
            # Gradio hands us RGB; the colour table is BGR, so flip it.
            colour = colour[::-1]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

            label = f"{det.stage}  {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Put the caption above the box, or inside it when there is no room.
            ty = y1 - baseline - 4
            if ty - th < 0:
                ty = y1 + th + baseline + 8
            cv2.rectangle(canvas, (x1, ty - th - baseline - 4),
                          (x1 + tw + 8, ty + baseline), colour, -1)
            cv2.putText(canvas, label, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def describe(self, detections: list[Detection], elapsed_ms: float) -> str:
        """A short clinical-style summary of what the model found."""
        if not detections:
            return (
                "**No pressure ulcer detected.**\n\n"
                "The model found nothing above the current confidence threshold. "
                "Lower the threshold to see weaker candidates. A negative result "
                "is not a clinical exclusion.\n\n"
                f"_Inference time: {elapsed_ms:.0f} ms_"
            )

        worst = max(detections, key=lambda d: d.stage_id)
        lines = [
            f"**{len(detections)} region{'s' if len(detections) > 1 else ''} detected.**",
            "",
            f"Most severe finding: **{worst.stage}** "
            f"({worst.confidence:.0%} confidence).",
            f"_{STAGE_DESCRIPTIONS.get(worst.stage_id, '')}_",
            "",
            "| # | Stage | Confidence | Share of frame |",
            "|---|-------|-----------|----------------|",
        ]
        for i, det in enumerate(detections, 1):
            lines.append(
                f"| {i} | {det.stage} | {det.confidence:.0%} | {det.area_fraction:.1%} |")
        lines += ["", f"_Inference time: {elapsed_ms:.0f} ms on {self.device.upper()}_"]
        return "\n".join(lines)
