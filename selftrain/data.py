"""Dataset over data/seg: RGB image + uint8 mask (0 bg, 1-4 stage)."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .letterbox import letterbox

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet (encoder pre-training)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def train_augment():
    """Online augmentation for U-Net. Geometric transforms are free (wounds have no
    orientation); photometric ones are mild because colour carries stage information
    (e.g. non-blanchable redness in stage 1), so hue is not shifted."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.75, 1.25), translate_percent=(-0.1, 0.1), rotate=(-20, 20),
                 border_mode=cv2.BORDER_CONSTANT, fill=114, fill_mask=0, p=0.7),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
    ])


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SegDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, augment: bool = False, names=None):
        self.img_dir, self.mask_dir = Path(root) / "images" / split, Path(root) / "masks" / split
        files = sorted(self.img_dir.glob("*.jpg"))
        if names is not None:
            keep = set(names)
            files = [f for f in files if f.stem in keep]
        self.files, self.size = files, size
        self.aug = train_augment() if augment else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
        img, _ = letterbox(img, self.size)
        mask, _ = letterbox(mask, self.size, interp=cv2.INTER_NEAREST, pad_value=0)
        if self.aug is not None:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return to_tensor(img), torch.from_numpy(mask.astype(np.int64))
