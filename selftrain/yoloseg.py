"""YOLOv8-seg (Ultralytics): training and inference.

Ultralytics keeps its own validation fitness for early stopping. To compare with
U-Net under the SAME model-selection rule, every epoch is checkpointed and the
checkpoint with the best validation binary Dice is selected afterwards
(select_checkpoint). The test split is never used for any choice."""
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .config import YOLOConfig

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _device():
    """Apple GPU (MPS) when available; SELFTRAIN_DEVICE=cpu forces the CPU (tests)."""
    forced = os.environ.get("SELFTRAIN_DEVICE")
    if forced:
        return forced
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train(data_yaml: Path, out_dir: Path, cfg: YOLOConfig = YOLOConfig(), log=print):
    from ultralytics import YOLO
    # absolute: Ultralytics nests a relative `project` under its own runs_dir
    out_dir = Path(out_dir).resolve()
    model = YOLO(cfg.weights)
    model.train(data=str(data_yaml), imgsz=cfg.img_size, epochs=cfg.epochs, patience=cfg.patience,
                batch=cfg.batch, device=_device(), workers=cfg.workers, seed=cfg.seed,
                deterministic=True, flipud=cfg.flipud, project=str(out_dir.parent),
                name=out_dir.name, exist_ok=True, save_period=1, plots=True, verbose=False,
                **cfg.extra)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return out_dir / "weights"


class YOLOPredictor:
    """infer() -> instances {masks (N,H,W) bool at original resolution, conf, cls}."""
    kind = "yoloseg"
    MIN_CONF = 0.05          # lowest threshold of the evaluation grid

    def __init__(self, weights: Path, img_size: int = 640):
        from ultralytics import YOLO
        self.model, self.imgsz, self.dev = YOLO(str(weights)), img_size, _device()

    def infer(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.MIN_CONF, iou=0.7,
                               retina_masks=True, verbose=False, device=self.dev, max_det=50)[0]
        if r.masks is None or len(r.boxes) == 0:
            return {"masks": np.zeros((0, h, w), bool), "conf": np.zeros(0), "cls": np.zeros(0, int)}
        m = r.masks.data.cpu().numpy() > 0.5
        if m.shape[1:] != (h, w):            # safety: retina masks should already match
            import cv2
            m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                          for x in m])
        return {"masks": m, "conf": r.boxes.conf.cpu().numpy(), "cls": r.boxes.cls.cpu().numpy().astype(int)}

    @staticmethod
    def to_semantic(raw, thr: float):
        """Instances with confidence >= thr, painted in ascending confidence so the
        most confident instance decides overlapping pixels; value = stage (1-4)."""
        h, w = raw["masks"].shape[1:]
        out = np.zeros((h, w), np.uint8)
        for k in np.argsort(raw["conf"]):
            if raw["conf"][k] >= thr:
                out[raw["masks"][k]] = raw["cls"][k] + 1
        return out

    @staticmethod
    def fg_confidence(raw):
        h, w = raw["masks"].shape[1:]
        out = np.zeros((h, w), np.float32)
        for m, c in zip(raw["masks"], raw["conf"]):
            out[m] = np.maximum(out[m], c)
        return out


def select_checkpoint(run_dir: Path, data_root: Path, conf: float = 0.25, log=print):
    """Pick the epoch checkpoint with the best mean validation binary Dice
    (same criterion U-Net uses for early stopping). Writes weights/selected.pt."""
    import cv2
    from .metrics import binary_scores
    wdir = Path(run_dir) / "weights"
    ckpts = sorted(wdir.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]))
    val_imgs = sorted((Path(data_root) / "images" / "val").glob("*.jpg"))
    gts = [cv2.imread(str(Path(data_root) / "masks" / "val" / f"{p.stem}.png"), 0) > 0 for p in val_imgs]
    imgs = [cv2.imread(str(p)) for p in val_imgs]
    scores = {}
    for ck in ckpts:
        pr = YOLOPredictor(ck)
        d = [binary_scores(pr.to_semantic(pr.infer(im), conf) > 0, gt)[0] for im, gt in zip(imgs, gts)]
        scores[ck.name] = float(np.mean(d))
        log(f"[yolo-select] {ck.name}: val binary Dice {scores[ck.name]:.4f}")
    best = max(scores, key=scores.get)
    shutil.copy2(wdir / best, wdir / "selected.pt")
    (Path(run_dir) / "selection.json").write_text(json.dumps(
        {"criterion": f"mean val binary Dice at conf {conf}", "selected": best,
         "scores": scores}, indent=2))
    return wdir / "selected.pt", best, scores[best]
