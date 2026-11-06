"""Split the training photographs of data/seg into four equal batches (seeded, by
photograph) for the data-growth demonstration of the self-retraining system.

  data/growth/batch{1..4}.txt     image names (all copies of a photograph stay together)
  data/growth/seed_v25/            dataset for the initial model: batch 1 as train,
                                   the full val split and the frozen test split (symlinks)
  data/growth/inbox_batch{2..4}/   batches 2-4 as inbox batches (images/ + labels/ + source.yaml)
                                   to be dropped into the system one after another
"""
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / "data" / "seg"
OUT = ROOT / "data" / "growth"
SEED = 0


def main():
    rows = list(csv.DictReader(open(SEG / "manifest.csv")))
    by_photo = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_photo[r["photograph"]].append(r["name"])
    photos = sorted(by_photo)
    random.Random(SEED).shuffle(photos)
    k = len(photos)
    cuts = [0, k // 4, k // 2, 3 * k // 4, k]
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    batches = []
    for b in range(4):
        names = [n for p in photos[cuts[b]:cuts[b + 1]] for n in by_photo[p]]
        batches.append(names)
        (OUT / f"batch{b + 1}.txt").write_text("".join(f"{n}\n" for n in names))
        print(f"batch {b + 1}: {cuts[b + 1] - cuts[b]} photographs, {len(names)} images")
    # initial dataset: batch 1 + full val + frozen test
    v = OUT / "seed_v25"
    for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
        for sp in ("train", "val", "test"):
            (v / sub / sp).mkdir(parents=True, exist_ok=True)
        for n in batches[0]:
            (v / sub / "train" / f"{n}{ext}").symlink_to(SEG / sub / "train" / f"{n}{ext}")
        for sp in ("val", "test"):
            for f in (SEG / sub / sp).glob(f"*{ext}"):
                (v / sub / sp / f.name).symlink_to(f)
    (v / "dataset.yaml").write_text(f"path: {v}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                    "nc: 4\nnames: [stage1, stage2, stage3, stage4]\n")
    for f in ("negatives_extra.csv", "test_frozen.txt"):
        shutil.copy2(SEG / f, v / f)
    keep = set(batches[0])
    with open(v / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        w.writerows(r for r in rows if r["split"] != "train" or r["name"] in keep)
    # inbox batches 2-4: real files (the system copies them into its pool)
    for b in (2, 3, 4):
        d = OUT / f"inbox_batch{b}"
        (d / "images").mkdir(parents=True); (d / "labels").mkdir()
        for n in batches[b - 1]:
            shutil.copy2(SEG / "images" / "train" / f"{n}.jpg", d / "images" / f"{n}.jpg")
            shutil.copy2(SEG / "labels" / "train" / f"{n}.txt", d / "labels" / f"{n}.txt")
        (d / "source.yaml").write_text(
            f"name: growth_batch{b}\nlicense: CC BY 4.0 (Roboflow Universe sources) / see data catalogue\n"
            "class_map: {0: 1, 1: 2, 2: 3, 3: 4}\nnegatives: true\n")
    print("done")


if __name__ == "__main__":
    main()
