"""
Build the photograph-disjoint SEGMENTATION dataset for the Bachelor thesis.

One set of photographs, two label formats, so U-Net and YOLOv8-seg are
compared on exactly the same images:

  data/seg/images/{train,val,test}/<name>.<ext>
  data/seg/labels/{train,val,test}/<name>.txt   YOLOv8-seg polygons (class 0-3)
  data/seg/masks/{train,val,test}/<name>.png    uint8: 0 = background, 1-4 = stage 1-4
  data/seg/dataset.yaml                          for Ultralytics
  data/seg/manifest.csv                          name, source, photograph id, split
  data/seg/test_frozen.txt                       the frozen test list (never retrained on)
  data/seg/*/external/                           held-out source (default rf_dekubitus):
                                                 a "new clinic" never seen in training
  data/seg/negatives_extra.csv                   every other AZH non-PU photo (surgical,
                                                 diabetic, venous, healthy skin, background)
                                                 for false-positive rates per wound type

Label policy (documented in the thesis):
  * Target classes are the four NPUAP/EPUAP stages. Stage polygons are kept.
  * NonPU lesions (fid, mobile) become background -- they are negative regions.
  * Images containing a DTI or Unstageable instance are dropped: those wounds are
    outside the four-stage scope and would otherwise be taught as background.
  * rf_petchs_seg also outlines tissue types (eschar/granulation/slough) inside
    wounds; only its stage outlines are used.
  * rf_thai_sores carries boxes only and PISeg is the same Roboflow project as
    roboflow_woundcare -- both are excluded.
  * rf_dekubitus is excluded: every one of its 126 photographs is a photograph of
    roboflow_maskrcnn annotated a second time, and the two annotations agree poorly
    (scripts/annotator_agreement.py). Keeping both would put two conflicting labels
    of the same photograph into training.
  * Negative photographs (other wound types, healthy skin, background) come from
    AZH (classes N, S, D, V, BG -- not P) and FUSeg (diabetic foot ulcers).

Photograph identity and the split reuse unify_dataset_v3 (stem + pHash + dHash +
dihedral correlation + ORB/RANSAC); the SIFT audit checks for residual leakage.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unify_dataset_v3 as u  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "seg"
RESULTS = ROOT / "results"
STAGES = {0: "stage1", 1: "stage2", 2: "stage3", 3: "stage4"}
SPLITS = ("train", "val", "test", "external")

# source class id -> unified stage (0-3), "bg" (negative region) or "drop" (drop image)
POLY_SOURCES = {
    "roboflow_maskrcnn":  {0: "drop", 1: "drop", 2: 0, 3: 1, 4: 2, 5: 3},
    "roboflow_woundcare": {0: "drop", 1: "drop", 2: 0, 3: 1, 4: 2, 5: 3},
    "roboflow_fid":       {0: "bg", 1: 2, 2: 3},
    "roboflow_mobile":    {0: "bg", 1: 0, 2: 1, 3: 2, 4: 3},
    # eschar, granulation, slough (tissue inside wounds) -> ignored; stage1-4; unstageable
    "rf_petchs_seg":      {0: "skip", 1: "skip", 2: "skip", 3: 0, 4: 1, 5: 2, 6: 3, 7: "drop"},
}
# rf_dekubitus PNG masks: 0 bg, 1 DTI, 2 Unstageable, 3-6 stage1-4
DEKUBITUS_MAP = {1: "drop", 2: "drop", 3: 0, 4: 1, 5: 2, 6: 3}
AZH_NEG_CLASSES = ("N", "S", "D", "V", "BG")

# Filename identity is trusted only inside the family of Roboflow exports that share
# the same original photographs (as in unify_dataset_v3). Every other source gets a
# namespaced stem, so a coincidental name such as "100.jpg" in two unrelated
# collections cannot merge two different photographs; true cross-source duplicates
# are still caught by pHash/dHash/dihedral/ORB.
STEM_FAMILY = {"roboflow_maskrcnn", "roboflow_woundcare", "roboflow_fid", "roboflow_mobile"}


def _md5(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode()).hexdigest()


def stage_for_identity(records, stage_dir: Path):
    """Return records whose path names encode the identity policy (symlinks).
    stage_dir is private to one build (derived from --out), so concurrent builds
    cannot delete each other's links."""
    STAGE_DIR = stage_dir
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    staged = []
    for img, polys, src in records:
        if src in STEM_FAMILY:
            staged.append((img, polys, src))
            continue
        if src.startswith("neg_"):
            name = f"{src}__{_md5(img)[:8]}__{img.name}"
        else:  # keep the Roboflow stem so augmented copies of one original still merge
            name = f"{src}__{img.name}"
        link = STAGE_DIR / name
        if link.exists() or link.is_symlink():
            link = STAGE_DIR / f"{src}__{_md5(img)[:8]}__{img.name}"
        link.symlink_to(img.resolve())
        staged.append((link, polys, src))
    return staged


def norm_poly(vals):
    pts = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
    return np.clip(pts, 0.0, 1.0)


def read_polys(lbl: Path, table):
    """Return (stage polygons [(cls, Nx2 normalised)], drop_image)."""
    polys, drop = [], False
    if not lbl.exists():
        return polys, drop
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls, vals = int(p[0]), [float(v) for v in p[1:]]
        target = table.get(cls, "skip")
        if target == "drop":
            drop = True
            continue
        if target in ("bg", "skip"):
            continue
        if len(vals) == 4:          # a stray box in a polygon export: use its outline
            cx, cy, w, h = vals
            vals = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2,
                    cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
        if len(vals) >= 6 and len(vals) % 2 == 0:
            polys.append((target, norm_poly(vals)))
    return polys, drop


def collect_poly_source(name):
    recs, dropped = [], 0
    for img, lbl in u.collect_roboflow(name):
        polys, drop = read_polys(lbl, POLY_SOURCES[name])
        if drop:
            dropped += 1
            continue
        recs.append((img, polys, name))     # empty polys = negative photo (NonPU only)
    return recs, dropped


def collect_dekubitus():
    recs, dropped = [], 0
    for split in ("train", "valid", "test"):
        d = RAW / "rf_dekubitus" / split
        for img in sorted(d.glob("*.jpg")):
            m = d / f"{img.stem}_mask.png"
            mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            present = set(np.unique(mask).tolist()) - {0}
            if any(DEKUBITUS_MAP.get(v) == "drop" for v in present):
                dropped += 1
                continue
            h, w = mask.shape
            polys = []
            for v, stage in DEKUBITUS_MAP.items():
                if stage == "drop" or v not in present:
                    continue
                cnts, _ = cv2.findContours((mask == v).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) < 25:
                        continue
                    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32)
                    if len(c) >= 3:
                        polys.append((stage, np.clip(c / [w, h], 0, 1)))
            recs.append((img, polys, "rf_dekubitus"))
    return recs, dropped


def collect_negatives(n_azh, n_fuseg, seed):
    rng = random.Random(seed)
    azh = []
    for cls in AZH_NEG_CLASSES:
        for part in ("Train/Train", "Test/Test"):
            azh += sorted((RAW / "azh_classification" / "dataset" / part / cls).glob("*"))
    azh = [p for p in azh if p.suffix.lower() in u.IMG_EXT]
    fus_dir = RAW / "fuseg" / "data" / "Foot Ulcer Segmentation Challenge"
    fus = sorted(p for p in fus_dir.glob("*/images/*") if p.suffix.lower() in u.IMG_EXT)
    rng.shuffle(azh)
    rng.shuffle(fus)
    return ([(p, [], "neg_azh") for p in azh[:n_azh]] +
            [(p, [], "neg_fuseg") for p in fus[:n_fuseg]])


def rasterise(polys, h, w):
    mask = np.zeros((h, w), np.uint8)
    # large outlines first so a smaller, nested outline keeps its own stage
    order = sorted(polys, key=lambda t: -cv2.contourArea((t[1] * [w, h]).astype(np.float32)))
    for stage, pts in order:
        cv2.fillPoly(mask, [np.round(pts * [w, h]).astype(np.int32)], int(stage) + 1)
    return mask


def audit(out: Path):
    """SIFT + RANSAC twin search for every evaluation image against training
    (same method as unify_dataset_v3.audit_leakage, extended to the external split)."""
    sift = cv2.SIFT_create()

    def load(path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (u.FEATURE_SIDE, u.FEATURE_SIDE))
        kp, des = sift.detectAndCompute(img, None)
        small = cv2.resize(img, (u.GRID_SIDE, u.GRID_SIDE)).astype(np.float32).ravel()
        small -= small.mean()
        nrm = float(np.linalg.norm(small))
        return kp, des, (small / nrm if nrm > 1e-6 else small)

    tr = [(f.name, *load(f)) for f in tqdm(sorted((out / "images" / "train").iterdir()),
                                            desc="  SIFT train", leave=False)]
    tr_vecs = np.stack([t[3] for t in tr])
    cross = []
    for sp in ("val", "test", "external"):
        for f in tqdm(sorted((out / "images" / sp).iterdir()), desc=f"  audit {sp}", leave=False):
            kp, des, vec = load(f)
            best_n, best_name = 0, None
            for j in np.argsort(-u._dihedral_scores(vec, tr_vecs))[:u.SIFT_SHORTLIST]:
                n_in = u._feature_inliers(kp, des, tr[j][1], tr[j][2], cv2.NORM_L2)
                if n_in > best_n:
                    best_n, best_name = n_in, tr[j][0]
            if best_n >= u.SIFT_MIN_INLIERS:
                cross.append((sp, f.name, best_name, best_n))
    (out / "leakage_audit.json").write_text(json.dumps(cross, indent=1))
    return cross


def drop_flagged(out: Path, cross):
    """Remove every evaluation image the SIFT audit matched to a training image.

    Grouping already merged most copies; whatever slips through (e.g. strong crops of
    the same photograph exported by two different projects) must not stay in val, test
    or external. Files and manifest rows are removed and the removal is recorded."""
    flagged = {(sp, name) for sp, name, *_ in cross}
    removed = []
    for sp, name in sorted(flagged):
        stem = Path(name).stem
        for sub, ext in (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")):
            f = out / sub / sp / f"{stem}{ext}"
            if f.exists():
                f.unlink()
        removed.append({"split": sp, "name": stem})
    man = out / "manifest.csv"
    with open(man) as f:
        rows = list(csv.DictReader(f))
    gone = {(r["split"], r["name"]) for r in removed}
    keep = [r for r in rows if (r["split"], r["name"]) not in gone]
    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(keep)
    (out / "test_frozen.txt").write_text("".join(f"{r['name']}\n" for r in keep if r["split"] == "test"))
    (out / "removed_by_audit.json").write_text(json.dumps(removed, indent=1))
    return removed


def extra_negatives(records, out: Path):
    """All AZH non-PU photos not already in the dataset, minus near-duplicates (pHash)."""
    used = {Path(r[0]).resolve() for r in records}
    ds_hashes = []
    for r in tqdm(records, desc="  pHash dataset", leave=False):
        try:
            ds_hashes.append(imagehash.phash(Image.open(r[0]), hash_size=u.PHASH_SIZE))
        except Exception:
            pass
    ds = np.array([h.hash.ravel() for h in ds_hashes], dtype=bool)
    rows = []
    for cls in AZH_NEG_CLASSES:
        for part in ("Train/Train", "Test/Test"):
            for p in sorted((RAW / "azh_classification" / "dataset" / part / cls).glob("*")):
                if p.suffix.lower() not in u.IMG_EXT or p.resolve() in used or not u.image_ok(p):
                    continue
                h = imagehash.phash(Image.open(p), hash_size=u.PHASH_SIZE).hash.ravel()
                if (ds != h).sum(axis=1).min() <= u.PHASH_THRESHOLD:
                    continue  # near-duplicate of a dataset image -> not independent
                rows.append((str(p.resolve()), cls))
    with open(out / "negatives_extra.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["path", "azh_class"]); wr.writerows(rows)
    print(f"  {len(rows)} extra negatives: " + json.dumps(Counter(c for _, c in rows)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--neg-azh", type=int, default=150)
    ap.add_argument("--neg-fuseg", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--external", default="",
                    help="comma-separated sources held out completely as an external test")
    ap.add_argument("--with-dekubitus", action="store_true",
                    help="include rf_dekubitus (excluded by default, see module docstring)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: keep only the first N images per source")
    args = ap.parse_args()
    out = Path(args.out)

    print("== Step 1: read sources")
    records, dropped = [], Counter()
    for name in POLY_SOURCES:
        recs, d = collect_poly_source(name)
        records += recs; dropped[name] = d
        print(f"  {name:20s} {len(recs):5d} images kept, {d} dropped (DTI/Unstageable)")
    if args.with_dekubitus:
        recs, d = collect_dekubitus()
        records += recs; dropped["rf_dekubitus"] = d
        print(f"  {'rf_dekubitus':20s} {len(recs):5d} images kept, {d} dropped (DTI/Unstageable)")
    negs = collect_negatives(args.neg_azh, args.neg_fuseg, args.seed)
    records += negs
    print(f"  negatives: {len(negs)} (AZH {args.neg_azh}, FUSeg {args.neg_fuseg})")
    if args.limit:
        by_src = defaultdict(list)
        for r in records:
            by_src[r[2]].append(r)
        records = [r for rs in by_src.values() for r in rs[:args.limit]]
    records = [r for r in tqdm(records, desc="  image check", leave=False) if u.image_ok(r[0])]
    print(f"  readable images: {len(records)}")

    print("== Step 2: photograph identity (stem + pHash + dHash + dihedral + ORB)")
    records = stage_for_identity(records, Path(str(out) + "_stage"))
    groups, _, merged = u.build_groups(records)
    print(f"  {len(records)} images -> {len(groups)} distinct photographs ({merged} perceptual merges)")

    print("== Step 3: split by photograph")
    assign = u.split_groups(groups, records, args.val_frac, args.test_frac, args.seed)
    if out.exists():
        shutil.rmtree(out)
    external = set(filter(None, args.external.split(",")))
    n_ext = 0
    for gid, idxs in groups.items():
        if any(records[i][2] in external for i in idxs):
            assign[gid] = "external"
            n_ext += 1
    print(f"  external hold-out ({', '.join(sorted(external)) or 'none'}): {n_ext} photographs")
    for sub in ("images", "labels", "masks"):
        for sp in SPLITS:
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    rows, per_split, per_src, per_cls, photos = [], Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(set)
    neg_imgs = Counter()
    for gid, idxs in tqdm(groups.items(), desc="  writing", leave=False):
        sp = assign[gid]
        keep = idxs if sp == "train" else idxs[:1]     # eval: one image per photograph
        for i in keep:
            img, polys, src = records[i]
            name = f"{src}__{img.stem[:80]}__{_md5(img)[:6]}"
            im = cv2.imread(str(img))
            if im is None:
                raise RuntimeError(f"cannot read {img} (source {src})")
            h, w = im.shape[:2]
            cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / sp / f"{name}.txt").write_text(
                "".join(f"{s} " + " ".join(f"{v:.6f}" for v in pts.ravel()) + "\n" for s, pts in polys))
            cv2.imwrite(str(out / "masks" / sp / f"{name}.png"), rasterise(polys, h, w))
            per_split[sp] += 1; per_src[src][sp] += 1; photos[sp].add(gid)
            for s, _ in polys:
                per_cls[sp][STAGES[s]] += 1
            if not polys:
                neg_imgs[sp] += 1
            rows.append((name, src, gid, sp, len(polys)))

    with open(out / "manifest.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["name", "source", "photograph", "split", "instances"]); wr.writerows(rows)
    (out / "test_frozen.txt").write_text("".join(f"{r[0]}\n" for r in rows if r[3] == "test"))
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: 4\nnames: [stage1, stage2, stage3, stage4]\n")

    print("== Step 4: SIFT leakage audit (val, test, external vs train)")
    cross = audit(out)
    removed = drop_flagged(out, cross)
    print(f"  {len(cross)} evaluation images matched a training image -> removed from evaluation")

    print("== Step 5: extra negatives for false-positive rates")
    extra = extra_negatives(records, out)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stats = {
        "generated": ts, "photographs": len(groups), "perceptual_merges": merged,
        "photographs_per_split": {k: len(v) for k, v in photos.items()},
        "images_per_split": dict(per_split), "negative_images_per_split": dict(neg_imgs),
        "per_source": {k: dict(v) for k, v in per_src.items()},
        "instances_per_class": {k: dict(v) for k, v in per_cls.items()},
        "dropped_dti_unstageable": dict(dropped), "eval_with_train_twin": len(cross),
        "extra_negatives": dict(Counter(c for _, c in extra)),
    }
    res = RESULTS if not args.limit else out
    res.mkdir(exist_ok=True)
    (res / "seg_dataset_stats.json").write_text(json.dumps(stats, indent=2))
    lines = [f"SEGMENTATION DATASET  ({ts})", "",
             f"distinct photographs: {len(groups)}  (perceptual merges: {merged})",
             f"eval images with a train twin (SIFT audit): {len(cross)}", ""]
    for sp in SPLITS:
        lines.append(f"{sp:5s} images={per_split[sp]:5d} photographs={len(photos[sp]):4d} "
                     f"negatives={neg_imgs[sp]:4d}  " +
                     " ".join(f"{c}={per_cls[sp][c]}" for c in STAGES.values()))
    lines += ["", "per source (train/val/test/external):"]
    lines += [f"  {k:20s} " + " ".join(f"{s}={v.get(s, 0)}" for s in SPLITS)
              for k, v in sorted(per_src.items())]
    lines += ["", "dropped images containing DTI/Unstageable: " + json.dumps(dict(dropped)),
              "extra negatives (AZH, for FP rates): " + json.dumps(dict(Counter(c for _, c in extra)))]
    (res / "seg_dataset_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
