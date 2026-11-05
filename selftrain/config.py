"""Central paths and defaults. Every experiment reads its settings from here, so the
thesis can state one configuration and the code cannot silently drift from it."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_SEG = ROOT / "data" / "seg"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"

NUM_STAGES = 4                     # stage 1..4 -> mask values 1..4, 0 = background
STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4"]
IMG_SIZE = 640                     # both architectures train and predict at 640 px
SEED = 0


@dataclass
class UNetConfig:
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    img_size: int = IMG_SIZE
    batch: int = 8
    epochs: int = 80
    patience: int = 15             # early stopping on val binary Dice
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 6
    seed: int = SEED
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class YOLOConfig:
    weights: str = str(MODELS / "yolov8" / "yolov8s-seg.pt")
    img_size: int = IMG_SIZE
    batch: int = 16
    epochs: int = 80
    patience: int = 15
    workers: int = 6
    seed: int = SEED
    flipud: float = 0.5            # wounds have no canonical orientation (U-Net uses V-flips too)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# Evaluation protocol (pre-registered in docs/ROADMAP.md)
THRESHOLD_GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95
MIN_COMPONENT_FRAC = 0.001        # predicted regions smaller than 0.1 % of the image are noise
BOOTSTRAP = 2000
