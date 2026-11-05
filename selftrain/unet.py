"""U-Net (segmentation_models_pytorch): training with early stopping and inference.

Classes: 0 background + 4 stages. Loss: cross-entropy + multiclass Dice.
Model selection on the validation split only (binary Dice, ulcer vs background);
the test split is never touched here."""
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import NUM_STAGES, UNetConfig
from .data import SegDataset, to_tensor
from .letterbox import letterbox, unletterbox

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device():
    """Apple GPU (MPS) when available; SELFTRAIN_DEVICE=cpu forces the CPU (tests)."""
    forced = os.environ.get("SELFTRAIN_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _worker_init(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s + wid); random.seed(s + wid)


def build_model(cfg: UNetConfig, pretrained: bool = True):
    return smp.Unet(encoder_name=cfg.encoder,
                    encoder_weights=cfg.encoder_weights if pretrained else None,
                    in_channels=3, classes=NUM_STAGES + 1)


def binary_dice_batch(logits, target, thr=0.5):
    """Per-image binary Dice (ulcer vs background); empty-vs-empty counts as 1."""
    fg = 1.0 - logits.softmax(1)[:, 0]
    pred = fg > thr
    gt = target > 0
    inter = (pred & gt).flatten(1).sum(1).float()
    tot = pred.flatten(1).sum(1).float() + gt.flatten(1).sum(1).float()
    return torch.where(tot > 0, 2 * inter / tot.clamp(min=1), torch.ones_like(tot))


def train(data_root: Path, out_dir: Path, cfg: UNetConfig = UNetConfig(),
          train_names=None, log=print):
    """Train on data_root/train, select on data_root/val. Returns path to best.pt."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    dev = device()
    tr = SegDataset(data_root, "train", cfg.img_size, augment=True, names=train_names)
    va = SegDataset(data_root, "val", cfg.img_size, augment=False)
    g = torch.Generator(); g.manual_seed(cfg.seed)
    tl = DataLoader(tr, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
                    worker_init_fn=_worker_init, generator=g, drop_last=True,
                    persistent_workers=cfg.workers > 0)
    vl = DataLoader(va, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers,
                    persistent_workers=cfg.workers > 0)
    model = build_model(cfg).to(dev)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg.to_dict(), "train_images": len(tr), "val_images": len(va)}, indent=2))
    hist_f = open(out_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr", "seconds"])
    best, best_ep, wait = -1.0, -1, 0
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time(); model.train(); tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = cfg.ce_weight * F.cross_entropy(logits, y) + cfg.dice_weight * dice_loss(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(x); n += len(x)
        sched.step()
        model.eval(); vloss, vd, vn = 0.0, [], 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                vloss += (F.cross_entropy(logits, y) + dice_loss(logits, y)).item() * len(x)
                vd.append(binary_dice_batch(logits, y).cpu()); vn += len(x)
        vdice = float(torch.cat(vd).mean())
        hist.writerow([ep, tot / max(n, 1), vloss / max(vn, 1), vdice,
                       sched.get_last_lr()[0], round(time.time() - t0, 1)]); hist_f.flush()
        log(f"[unet] epoch {ep}/{cfg.epochs} loss {tot / max(n, 1):.4f} val_dice {vdice:.4f} "
            f"({time.time() - t0:.0f}s)")
        if vdice > best:
            best, best_ep, wait = vdice, ep, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                        "epoch": ep, "val_dice": vdice}, out_dir / "best.pt")
        else:
            wait += 1
            if wait >= cfg.patience:
                log(f"[unet] early stop at epoch {ep} (best {best:.4f} @ {best_ep})")
                break
    hist_f.close()
    (out_dir / "summary.json").write_text(json.dumps(
        {"best_val_dice": best, "best_epoch": best_ep, "epochs_run": ep}, indent=2))
    return out_dir / "best.pt"


class UNetPredictor:
    """infer() -> class probabilities (5,H,W) at the original resolution."""
    kind = "unet"

    def __init__(self, weights: Path):
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.cfg = UNetConfig(**{k: v for k, v in ck["config"].items() if k in UNetConfig.__dataclass_fields__})
        self.model = build_model(self.cfg, pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.dev = device(); self.model.to(self.dev).eval()

    @torch.no_grad()
    def infer(self, img_bgr: np.ndarray):
        rgb = img_bgr[..., ::-1]
        lb, box = letterbox(np.ascontiguousarray(rgb), self.cfg.img_size)
        x = to_tensor(lb)[None].to(self.dev)
        probs = self.model(x).softmax(1)[0].permute(1, 2, 0).float().cpu().numpy()
        probs = unletterbox(probs, box, img_bgr.shape[:2])
        return probs.transpose(2, 0, 1).astype(np.float16)

    @staticmethod
    def to_semantic(raw, thr: float):
        """Ulcer where P(not background) > thr; its stage is the most likely stage."""
        p = raw.astype(np.float32)
        fg = (1.0 - p[0]) > thr
        stage = p[1:].argmax(0).astype(np.uint8) + 1
        return np.where(fg, stage, 0).astype(np.uint8)

    @staticmethod
    def fg_confidence(raw):
        return 1.0 - raw[0].astype(np.float32)
