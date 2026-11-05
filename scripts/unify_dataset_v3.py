"""
Unified dataset pipeline (v3) for four-class pressure ulcer DETECTION.

v3 fixes the two defects that survived v2:

  (1) Perceptual-hash threshold was mis-scaled.  v2 called
      imagehash.phash(hash_size=16), which returns a 256-bit hash, but kept the
      threshold of 5 that the library's documentation calibrates for the default
      64-bit hash (hash_size=8).  Five differing bits out of 256 only catches
      re-encoding noise, so near-duplicates survived.  Measured on the v2 output:
      hash_size=16/threshold=5 finds 0 near-duplicate pairs, while the standard
      hash_size=8/threshold=5 finds 684 pairs, 72 of them spanning splits.

  (2) Identity was defined by filename alone.  The same photograph is exported by
      different Roboflow projects under unrelated filenames (for example
      roboflow_fid_Stage1_21 and roboflow_mobile_Stage1_21), so no filename regex
      can group them.  v2's own leakage check reused the grouping regex and could
      therefore only ever report zero.

  v3 builds photograph groups by union-find over BOTH filename stem and
  perceptual identity, assigns whole groups to folds, and then verifies the
  result with dHash -- an independent hash family -- so the check cannot be
  tautological.  Validation and test keep one representative image per group so
  that the reported set size equals the number of independent photographs.

Usage:
    python scripts/unify_dataset_v3.py
    python scripts/unify_dataset_v3.py --include-fullframe --val-frac 0.15
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    sys.exit("pip install imagehash")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "unified"
RESULTS_DIR = PROJECT_ROOT / "results"

UNIFIED_CLASSES = {0: "stage1", 1: "stage2", 2: "stage3", 3: "stage4"}

CLASS_REMAPS = {
    # [DTI, Unstageable, stage-1, stage-2, stage-3, stage-4]
    "roboflow_maskrcnn":  {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},
    "roboflow_woundcare": {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},
    # [NonPU, Stage3, Stage4] -- this export cannot express stage 1 or 2; some of
    # its photographs are filed under Stage1/Stage2 names yet carry Stage3/Stage4
    # class ids.  data.yaml is the only machine-readable authority, so it wins,
    # and the ambiguity is recorded in the report.
    "roboflow_fid":       {0: None, 1: 2, 2: 3},
    # [NonPU, Stage1, Stage2, Stage3, Stage4]
    "roboflow_mobile":    {0: None, 1: 0, 2: 1, 3: 2, 4: 3},
    # Whole-frame labels -- see FULL_FRAME_SOURCES
    "roboflow_stage2":    {0: 0, 1: 1, 2: 2, 3: 3},
    "roboflow_project1":  {0: 0, 1: 1, 2: 2, 3: 3},
}

FULL_FRAME_SOURCES = {"roboflow_stage2", "roboflow_project1", "piid"}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
RF_SUFFIX = re.compile(r"(_(?:jpg|jpeg|png|bmp|JPG|JPEG|PNG))?\.rf\.[0-9a-f]+$")

# imagehash's documented threshold of 5 is calibrated for the 64-bit default.
PHASH_SIZE, PHASH_THRESHOLD = 8, 5
# A second, independent hash family. Two families disagree on which pairs they
# catch, so grouping on the union of both is stricter than either alone.
DHASH_SIZE, DHASH_THRESHOLD = 8, 8
# Grouping also compares downsampled pixels, under all eight dihedral
# transforms, because Roboflow augments by flipping and rotating and no
# orientation-sensitive signal can see through that.
GRID_SIDE, GRID_CORR = 32, 0.90
# Global appearance cannot see that two different crops of one photograph are
# the same scene, so grouping also matches LOCAL features: ORB keypoints
# filtered by a RANSAC homography. Candidates are shortlisted by the dihedral
# correlation, since running ORB over every pair would be intractable.
ORB_FEATURES, ORB_MIN_INLIERS, ORB_SHORTLIST = 1200, 25, 40
# The audit then re-checks the result with SIFT: a different detector and a
# different descriptor family (float gradients rather than binary tests), so a
# blind spot in ORB does not go unnoticed.
SIFT_MIN_INLIERS, SIFT_SHORTLIST = 25, 40
FEATURE_SIDE = 480


# ---------------------------------------------------------------- annotations

def polygon_to_box(coords):
    xs, ys = coords[0::2], coords[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def read_annotations(path: Path):
    """Read a YOLO label file holding boxes, polygons, or both."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(vals) == 4:
            cx, cy, w, h = vals
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            cx, cy, w, h = polygon_to_box(vals)
        else:
            continue
        out.append((cls, cx, cy, w, h))
    return out


def remap(source, entries):
    table, kept = CLASS_REMAPS[source], []
    for cls, cx, cy, w, h in entries:
        new = table.get(cls)
        if new is None:
            continue
        # Clip the corners, then rebuild the box, so the result always lies
        # inside the frame even if a source vertex did not.
        x0, x1 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
        y0, y1 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
        nw, nh = x1 - x0, y1 - y0
        if nw < 1e-3 or nh < 1e-3:
            continue
        kept.append((new, (x0 + x1) / 2, (y0 + y1) / 2, nw, nh))
    return kept


def write_labels(path: Path, entries):
    path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            for c, cx, cy, w, h in entries))


# -------------------------------------------------------------------- sources

def source_stem(filename: str) -> str:
    return RF_SUFFIX.sub("", Path(filename).stem)


def collect_roboflow(name: str):
    root, pairs = RAW_DIR / name, []
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


def collect_piid():
    base, out = RAW_DIR / "piid" / "extracted", []
    for folder, unified in (("1", 0), ("2", 1), ("3", 2), ("4", 3)):
        d = base / folder
        if not d.exists():
            continue
        for img in d.iterdir():
            if img.suffix.lower() in IMG_EXT:
                out.append((img, [(unified, 0.5, 0.5, 0.95, 0.95)]))
    return out


# ------------------------------------------------------------------ grouping

def _feature_inliers(kp1, des1, kp2, des2, norm):
    """RANSAC-verified match count between two keypoint sets."""
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0
    matches = cv2.BFMatcher(norm, crossCheck=True).match(des1, des2)
    if len(matches) < 8:
        return 0
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def _describe(path, detector):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return detector.detectAndCompute(cv2.resize(img, (FEATURE_SIDE, FEATURE_SIDE)), None)


def _dihedral_scores(vec, mat):
    """Best correlation of `vec` against every row of `mat`, over 8 orientations."""
    g = vec.reshape(GRID_SIDE, GRID_SIDE)
    best = np.zeros(mat.shape[0], dtype=np.float32)
    for k in range(4):
        r = np.rot90(g, k)
        for cand in (r, r[:, ::-1]):
            c = cand.ravel() - cand.mean()
            nrm = float(np.linalg.norm(c))
            if nrm > 1e-6:
                best = np.maximum(best, mat @ (c / nrm))
    return best


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(records):
    """Union-find over filename stem AND perceptual identity."""
    n = len(records)
    uf = UnionFind(n)

    # Key on the bare stem WITHOUT the source prefix: the same photograph is
    # exported by several Roboflow projects under the same base name, and
    # keeping the prefix would stop those copies from merging.
    by_stem = defaultdict(list)
    for i, (path, _, src) in enumerate(records):
        by_stem[source_stem(path.name)].append(i)
    for idxs in by_stem.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    phashes, dhashes = [], []
    for path, _, _ in tqdm(records, desc="  hashing", leave=False):
        try:
            im = Image.open(path)
            phashes.append(imagehash.phash(im, hash_size=PHASH_SIZE))
            dhashes.append(imagehash.dhash(im, hash_size=DHASH_SIZE))
        except Exception:
            phashes.append(None)
            dhashes.append(None)

    merged = 0
    for hashes, thr in ((phashes, PHASH_THRESHOLD), (dhashes, DHASH_THRESHOLD)):
        # Bucket on each half of the hash so the pairwise scan stays tractable;
        # any pair within the threshold must agree on at least one half.
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            if h is None:
                continue
            bits = h.hash.flatten()
            for half in (bits[: len(bits) // 2], bits[len(bits) // 2:]):
                buckets[half.tobytes()].append(i)
        for idxs in buckets.values():
            if len(idxs) < 2 or len(idxs) > 600:
                continue
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if uf.find(i) != uf.find(j) and hashes[i] - hashes[j] <= thr:
                        uf.union(i, j)
                        merged += 1

    # Third signal: raw pixel correlation. The two hash families share a
    # frequency-domain view and miss the same pairs; comparing downsampled
    # pixels directly is a different modality, so it closes their blind spot.
    def norm_vec(g):
        v = g.astype(np.float32).ravel()
        v -= v.mean()
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 1e-6 else v

    grids = np.zeros((n, GRID_SIDE, GRID_SIDE), dtype=np.float32)
    for i, (path, _, _) in enumerate(tqdm(records, desc="  pixels", leave=False)):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            grids[i] = cv2.resize(img, (GRID_SIDE, GRID_SIDE)).astype(np.float32)

    base = np.stack([norm_vec(g) for g in grids])
    # The eight dihedral transforms of the square: four rotations, each with and
    # without a mirror. Comparing every transform of one image against the
    # untransformed others makes the match orientation-invariant.
    variants = []
    for k in range(4):
        rot = np.rot90(grids, k, axes=(1, 2))
        variants.append(np.stack([norm_vec(g) for g in rot]))
        variants.append(np.stack([norm_vec(g) for g in rot[:, :, ::-1]]))

    pixel_merged = 0
    block = 512
    for var in variants:
        for start in range(0, n, block):
            corr = var[start:start + block] @ base.T
            for r, i in enumerate(range(start, min(start + block, n))):
                for j in np.nonzero(corr[r] >= GRID_CORR)[0]:
                    j = int(j)
                    if j != i and uf.find(i) != uf.find(j):
                        uf.union(i, j)
                        pixel_merged += 1
    merged += pixel_merged

    # Local-feature pass: catches two different crops of one photograph, which
    # every global signal above scores as only weakly similar.
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    feats = [_describe(path, orb) for path, _, _ in
             tqdm(records, desc="  ORB", leave=False)]
    orb_merged = 0
    for i in tqdm(range(n), desc="  ORB match", leave=False):
        scores = _dihedral_scores(base[i], base)
        for j in np.argsort(-scores)[:ORB_SHORTLIST + 1]:
            j = int(j)
            if j == i or uf.find(i) == uf.find(j):
                continue
            if _feature_inliers(*feats[i], *feats[j], cv2.NORM_HAMMING) >= ORB_MIN_INLIERS:
                uf.union(i, j)
                orb_merged += 1
    merged += orb_merged

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return groups, phashes, merged


def split_groups(groups, records, val_frac, test_frac, seed):
    """Assign whole photograph groups to folds.

    The fractions count PHOTOGRAPHS, not images: validation and test keep one
    image per photograph, so balancing image counts would starve them.
    """
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_val, n_test = round(val_frac * n), round(test_frac * n)
    out = {}
    for i, k in enumerate(keys):
        out[k] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return out


# --------------------------------------------------------------------- checks

def image_ok(path: Path) -> bool:
    try:
        img = cv2.imread(str(path))
        if img is None or img.ndim < 3:
            return False
        h, w = img.shape[:2]
        return h >= 32 and w >= 32
    except Exception:
        return False


def audit_leakage(out_dir: Path):
    """Check every evaluation image for a same-photograph twin in training.

    Grouping merged on filename, two perceptual hash families, dihedral pixel
    correlation and ORB features. This audit uses SIFT -- a different detector
    with a different descriptor family -- so it can surface pairs ORB missed
    rather than merely restating the grouping decision.
    """
    sift = cv2.SIFT_create()

    def load(path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, None
        img = cv2.resize(img, (FEATURE_SIDE, FEATURE_SIDE))
        kp, des = sift.detectAndCompute(img, None)
        small = cv2.resize(img, (GRID_SIDE, GRID_SIDE)).astype(np.float32).ravel()
        small -= small.mean()
        nrm = float(np.linalg.norm(small))
        return kp, des, (small / nrm if nrm > 1e-6 else small)

    cache = {}
    for sp in ("train", "val", "test"):
        cache[sp] = [(f.name, *load(f)) for f in
                     tqdm(sorted((out_dir / "images" / sp).iterdir()),
                          desc=f"  SIFT {sp}", leave=False)]

    tr = cache["train"]
    tr_vecs = np.stack([t[3] for t in tr]) if tr else np.zeros((0, GRID_SIDE ** 2))

    cross = []
    for sp in ("val", "test"):
        for name, kp, des, vec in tqdm(cache[sp], desc=f"  audit {sp}", leave=False):
            if not len(tr):
                break
            order = np.argsort(-_dihedral_scores(vec, tr_vecs))[:SIFT_SHORTLIST]
            best_n, best_name = 0, None
            for j in order:
                n_in = _feature_inliers(kp, des, tr[j][1], tr[j][2], cv2.NORM_L2)
                if n_in > best_n:
                    best_n, best_name = n_in, tr[j][0]
            if best_n >= SIFT_MIN_INLIERS:
                cross.append((sp, name, "train", best_name, best_n))
    return cross


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-fullframe", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--keep-duplicates-in-eval", action="store_true",
                    help="Keep every copy in val/test instead of one per photograph")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    print("=" * 70)
    print("DATASET UNIFICATION v3")
    print("=" * 70)

    active = [s for s in CLASS_REMAPS
              if args.include_fullframe or s not in FULL_FRAME_SOURCES]
    print(f"  sources: {', '.join(sorted(active))}")
    print(f"  full-frame sources: {'included' if args.include_fullframe else 'EXCLUDED'}")
    print(f"  phash: size={PHASH_SIZE} threshold={PHASH_THRESHOLD} "
          f"({PHASH_SIZE**2}-bit hash)")

    # Step 1 -- read and remap
    print("\n-- Step 1: read and remap")
    records, fmt = [], defaultdict(Counter)
    for src in active:
        if not (RAW_DIR / src).exists():
            print(f"  [skip] {src}: not downloaded")
            continue
        kept = 0
        for img, lbl in collect_roboflow(src):
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    n = len(line.split())
                    fmt[src]["box" if n == 5 else "polygon" if n > 5 else "empty"] += 1
            entries = remap(src, read_annotations(lbl))
            if entries:
                records.append((img, entries, src))
                kept += 1
        print(f"  {src:<22} -> {kept:>5} images with usable labels")

    if args.include_fullframe and (RAW_DIR / "piid" / "extracted").exists():
        piid = collect_piid()
        records.extend((img, ent, "piid") for img, ent in piid)
        print(f"  {'piid':<22} -> {len(piid):>5} images (synthesised whole-frame)")

    print(f"  total: {len(records)}")

    # Step 2 -- image quality
    print("\n-- Step 2: image quality")
    records = [r for r in tqdm(records, desc="  checking", leave=False) if image_ok(r[0])]
    print(f"  readable: {len(records)}")

    # Step 3 -- photograph groups
    print("\n-- Step 3: photograph identity")
    print("   (bare stem + pHash + dHash + dihedral pixel correlation + ORB/RANSAC)")
    groups, hashes, merged = build_groups(records)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"  {len(records)} images -> {len(groups)} distinct photographs")
    print(f"  cross-filename merges made by perceptual identity: {merged}")
    print(f"  group sizes: max={sizes[0]} median={sizes[len(sizes)//2]} singletons={sizes.count(1)}")

    # Step 4 -- split by group
    print("\n-- Step 4: group-level split")
    assign = split_groups(groups, records, args.val_frac, args.test_frac, args.seed)

    chosen = []          # (record_index, split)
    dropped_eval = 0
    for gid, idxs in groups.items():
        split = assign[gid]
        if split == "train" or args.keep_duplicates_in_eval:
            chosen.extend((i, split) for i in idxs)
        else:
            # one representative per photograph, so |val| and |test| equal the
            # number of independent photographs they contain
            chosen.append((idxs[0], split))
            dropped_eval += len(idxs) - 1
    print(f"  train keeps every copy; val/test keep 1 image per photograph "
          f"({dropped_eval} redundant eval copies dropped)")

    # Step 5 -- write
    print("\n-- Step 5: writing dataset")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    per_split, per_source, per_class = Counter(), defaultdict(Counter), defaultdict(Counter)
    groups_per_split = defaultdict(set)
    idx_to_group = {i: g for g, idxs in groups.items() for i in idxs}
    for i, split in tqdm(chosen, desc="  copying", leave=False):
        img, entries, src = records[i]
        name = f"{src}_{img.stem}"
        shutil.copy2(img, out_dir / "images" / split / f"{name}{img.suffix}")
        write_labels(out_dir / "labels" / split / f"{name}.txt", entries)
        per_split[split] += 1
        per_source[src][split] += 1
        groups_per_split[split].add(idx_to_group[i])
        for cls, *_ in entries:
            per_class[split][cls] += 1

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(UNIFIED_CLASSES)}\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in UNIFIED_CLASSES.items())
    )

    # Step 6 -- independent audit
    print(f"\n-- Step 6: leakage audit (SIFT + RANSAC homography, >= {SIFT_MIN_INLIERS} inliers;")
    print("   a different detector and descriptor family from the ORB pass used for grouping)")
    cross = audit_leakage(out_dir)
    print(f"  eval images with a train twin: {len(cross)} "
          f"({'PASS' if not cross else 'FAIL'})")
    for a in cross[:6]:
        print(f"    {a[4]} inliers  {a[0]:5} {a[1][:42]}  <->  train {a[3][:42]}")

    total = sum(per_split.values())
    lines = ["DATASET UNIFICATION REPORT (v3)", "=" * 70,
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"Full-frame sources: {'included' if args.include_fullframe else 'excluded'}",
             f"Perceptual hash: phash size={PHASH_SIZE} ({PHASH_SIZE**2}-bit) threshold={PHASH_THRESHOLD}",
             "", "Annotation formats read from source:"]
    for src, c in sorted(fmt.items()):
        lines.append(f"  {src:<22} box={c['box']:<6} polygon={c['polygon']:<6} empty={c['empty']}")
    lines += ["", "Images per source (final contribution; a source may reach zero "
                  "if its photographs duplicate another source):"]
    for src in sorted(CLASS_REMAPS):
        if src not in active:
            continue
        c = per_source.get(src, Counter())
        lines.append(f"  {src:<22} train={c['train']:<6} val={c['val']:<6} "
                     f"test={c['test']:<6} total={sum(c.values())}")
    lines += ["", "Split sizes (images / independent photographs):"]
    for s in ("train", "val", "test"):
        pct = 100 * per_split[s] / total if total else 0
        lines.append(f"  {s:<6} images={per_split[s]:<6} ({pct:4.1f}%)  "
                     f"photographs={len(groups_per_split[s])}")
    lines += ["", "Class distribution (annotation instances):"]
    for s in ("train", "val", "test"):
        lines.append(f"  {s:<6} " + "  ".join(
            f"{UNIFIED_CLASSES[c]}={per_class[s][c]}" for c in sorted(UNIFIED_CLASSES)))
    lines += ["", "Identity and leakage:",
              f"  distinct photographs (stem + perceptual): {len(groups)}",
              f"  merges made by perceptual identity alone: {merged}",
              f"  eval images with a train twin (SIFT+RANSAC audit, >={SIFT_MIN_INLIERS} inliers): {len(cross)}",
              "", f"Total images: {total}", f"Output: {out_dir}",
              "", "Known limitations:",
              "  * roboflow_fid declares only [NonPU, Stage3, Stage4]; some of its",
              "    photographs are filed under Stage1/Stage2 names but carry Stage3/",
              "    Stage4 class ids. data.yaml is treated as authoritative.",
              "  * Validation and test hold one image per photograph, so their sizes",
              "    equal the number of independent observations.",
              "  * Sources whose annotations span the whole frame are excluded by",
              "    default; they carry no localisation signal."]

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"unification_report_v3_{stamp}.txt").write_text(report)
    (RESULTS_DIR / "unification_report_v3_latest.txt").write_text(report)
    (RESULTS_DIR / f"unification_stats_v3_{stamp}.json").write_text(json.dumps({
        "formats": {k: dict(v) for k, v in fmt.items()},
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "per_split": dict(per_split),
        "photographs_per_split": {k: len(v) for k, v in groups_per_split.items()},
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "groups": len(groups), "perceptual_merges": merged,
        "cross_split_pairs": len(cross), "total": total,
    }, indent=2))
    print(f"\nReport: {RESULTS_DIR / f'unification_report_v3_{stamp}.txt'}")


if __name__ == "__main__":
    main()
