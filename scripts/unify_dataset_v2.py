"""
Unified dataset pipeline (v2) for four-class pressure ulcer DETECTION.

Fixes over v1 (scripts/unify_dataset.py):
  1. Polygon labels.  Four of the source datasets ship YOLO *segmentation*
     polygons, not boxes.  v1 read fields 1..4 of every line as (cx,cy,w,h),
     which for a polygon takes the first two vertices and yields a box whose
     area is wrong by a median factor of 3.6x (p90 23x).  v2 branches on the
     field count and converts polygons via min/max over the vertex arrays.
  2. Split leakage.  The Roboflow exports contain several augmented copies of
     each source photograph.  v1 split by individual image, so 33% of source
     photographs appeared in more than one split.  v2 groups by source stem so
     every copy of a photograph lands in the same fold.
  3. Localisation quality.  Some sources label the whole frame (cx,cy,w,h =
     0.5,0.5,1,1) and carry no localisation signal.  v2 classifies each source
     and, by default, keeps only detection-grade sources.
  4. Honest reporting.  v1 tallied per-dataset statistics before deduplication,
     so its report credited datasets that contributed zero surviving images.
     v2 re-tallies from the final image list.

Usage:
    python scripts/unify_dataset_v2.py                    # detection-grade only
    python scripts/unify_dataset_v2.py --include-fullframe # add classification-grade
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

# Source class id -> unified id.  None means "drop this annotation".
CLASS_REMAPS = {
    # [DTI, Unstageable, stage-1, stage-2, stage-3, stage-4]
    "roboflow_maskrcnn":  {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},
    "roboflow_woundcare": {0: None, 1: None, 2: 0, 3: 1, 4: 2, 5: 3},
    # [NonPU, Stage3, Stage4]
    "roboflow_fid":       {0: None, 1: 2, 2: 3},
    # [NonPU, Stage1, Stage2, Stage3, Stage4]
    "roboflow_mobile":    {0: None, 1: 0, 2: 1, 3: 2, 4: 3},
    # [stage1..stage4] -- labels are whole-frame, see FULL_FRAME_SOURCES
    "roboflow_stage2":    {0: 0, 1: 1, 2: 2, 3: 3},
    "roboflow_project1":  {0: 0, 1: 1, 2: 2, 3: 3},
}

# Sources whose annotations carry no localisation information: every box spans
# the frame.  Measured, not assumed -- see verify_localisation() below.
FULL_FRAME_SOURCES = {"roboflow_stage2", "roboflow_project1", "piid"}

# Body positions rather than ulcer stages.
EXCLUDED_SOURCES = {"roboflow_sciproj", "azh_localization", "azh_classification",
                    "fuseg", "medetec"}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# Roboflow appends "_<ext>.rf.<md5>" to every augmented copy of a photograph.
RF_SUFFIX = re.compile(r"(_(?:jpg|jpeg|png|bmp|JPG|JPEG|PNG))?\.rf\.[0-9a-f]+$")


# ---------------------------------------------------------------- annotations

def polygon_to_box(coords: list[float]) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box of a normalised polygon, as (cx, cy, w, h)."""
    xs, ys = coords[0::2], coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def read_annotations(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file that may hold boxes, polygons, or both."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            vals = [float(v) for v in parts[1:]]
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


def remap(source: str, entries):
    table = CLASS_REMAPS[source]
    kept = []
    for cls, cx, cy, w, h in entries:
        new = table.get(cls)
        if new is None:
            continue
        # Clamp to the unit square and require a non-degenerate box.
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w < 1e-3 or h < 1e-3:
            continue
        kept.append((new, cx, cy, w, h))
    return kept


def write_labels(path: Path, entries):
    path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            for c, cx, cy, w, h in entries))


# -------------------------------------------------------------------- sources

def source_stem(filename: str) -> str:
    """Strip the Roboflow augmentation suffix to recover the source photograph."""
    return RF_SUFFIX.sub("", Path(filename).stem)


def collect_roboflow(name: str):
    root = RAW_DIR / name
    pairs = []
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() in IMG_EXT:
                pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


def collect_piid():
    """PIID is a classification set: whole-frame boxes are synthesised."""
    base = RAW_DIR / "piid" / "extracted"
    out = []
    for folder, unified in (("1", 0), ("2", 1), ("3", 2), ("4", 3)):
        d = base / folder
        if not d.exists():
            continue
        for img in d.iterdir():
            if img.suffix.lower() in IMG_EXT:
                out.append((img, [(unified, 0.5, 0.5, 0.95, 0.95)]))
    return out


def verify_localisation(records) -> dict:
    """Measure, per source, what fraction of boxes span (almost) the whole frame."""
    stats = defaultdict(lambda: {"n": 0, "full": 0, "areas": []})
    for _, entries, src in records:
        for _, _, _, w, h in entries:
            a = w * h
            stats[src]["n"] += 1
            stats[src]["areas"].append(a)
            if a >= 0.95:
                stats[src]["full"] += 1
    out = {}
    for src, s in stats.items():
        areas = sorted(s["areas"])
        out[src] = {
            "annotations": s["n"],
            "full_frame_pct": round(100 * s["full"] / s["n"], 1) if s["n"] else 0.0,
            "median_area": round(areas[len(areas) // 2], 4) if areas else 0.0,
        }
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


def deduplicate(paths, hash_size=16, threshold=5):
    """Perceptual-hash near-duplicate removal.  Returns the set of duplicates."""
    seen, dups = {}, set()
    for p in tqdm(paths, desc="  hashing", leave=False):
        try:
            h = imagehash.phash(Image.open(p), hash_size=hash_size)
        except Exception:
            dups.add(str(p))
            continue
        for prev in seen:
            if abs(h - prev) <= threshold:
                dups.add(str(p))
                break
        else:
            seen[h] = str(p)
    return dups


def group_split(records, ratios=(0.7, 0.2, 0.1), seed=42):
    """Assign splits so that every copy of a source photograph shares a fold."""
    groups = defaultdict(list)
    for i, (path, _, src) in enumerate(records):
        groups[f"{src}/{source_stem(path.name)}"].append(i)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    total = len(records)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    counts = [0, 0, 0]
    assignment = {}
    for k in keys:
        # Send each group to whichever split is furthest below its target.
        deficits = [targets[i] - counts[i] for i in range(3)]
        j = int(np.argmax(deficits))
        assignment[k] = ("train", "val", "test")[j]
        counts[j] += len(groups[k])
    return {i: assignment[k] for k, idxs in groups.items() for i in idxs}, len(keys)


# ----------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-fullframe", action="store_true",
                    help="Also include sources whose boxes span the whole frame")
    ap.add_argument("--skip-dedup", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    print("=" * 70)
    print("DATASET UNIFICATION v2")
    print("=" * 70)

    active = [s for s in CLASS_REMAPS
              if args.include_fullframe or s not in FULL_FRAME_SOURCES]
    print(f"  sources           : {', '.join(sorted(active))}")
    print(f"  full-frame sources: {'included' if args.include_fullframe else 'EXCLUDED'} "
          f"({', '.join(sorted(FULL_FRAME_SOURCES))})")

    # Step 1 -- read and remap
    print("\n-- Step 1: read and remap annotations")
    records, fmt = [], defaultdict(Counter)
    for src in active:
        if not (RAW_DIR / src).exists():
            print(f"  [skip] {src}: not downloaded")
            continue
        pairs = collect_roboflow(src)
        kept = 0
        for img, lbl in pairs:
            raw_lines = lbl.read_text().splitlines() if lbl.exists() else []
            for line in raw_lines:
                n = len(line.split())
                fmt[src]["box" if n == 5 else "polygon" if n > 5 else "empty"] += 1
            entries = remap(src, read_annotations(lbl))
            if entries:
                records.append((img, entries, src))
                kept += 1
        print(f"  {src:<22} {len(pairs):>5} images -> {kept:>5} with usable labels")

    if args.include_fullframe and (RAW_DIR / "piid" / "extracted").exists():
        piid = collect_piid()
        records.extend((img, ent, "piid") for img, ent in piid)
        print(f"  {'piid':<22} {len(piid):>5} images -> {len(piid):>5} (synthesised whole-frame)")

    print(f"\n  annotation formats read:")
    for src, c in sorted(fmt.items()):
        print(f"    {src:<22} box={c['box']:<6} polygon={c['polygon']:<6} empty={c['empty']}")
    print(f"  total images with labels: {len(records)}")

    # Step 2 -- localisation audit
    print("\n-- Step 2: localisation audit (measured on the remapped boxes)")
    loc = verify_localisation(records)
    for src, s in sorted(loc.items()):
        flag = "  <-- WHOLE-FRAME" if s["full_frame_pct"] > 50 else ""
        print(f"  {src:<22} n={s['annotations']:<6} median_area={s['median_area']:<8} "
              f"full_frame={s['full_frame_pct']}%{flag}")

    # Step 3 -- image quality
    print("\n-- Step 3: image quality")
    records = [r for r in tqdm(records, desc="  checking", leave=False) if image_ok(r[0])]
    print(f"  readable images: {len(records)}")

    # Step 4 -- deduplication
    if args.skip_dedup:
        print("\n-- Step 4: deduplication SKIPPED")
    else:
        print("\n-- Step 4: perceptual-hash deduplication")
        dups = deduplicate([r[0] for r in records])
        before = len(records)
        records = [r for r in records if str(r[0]) not in dups]
        print(f"  removed {before - len(records)} duplicates -> {len(records)} unique")

    if not records:
        sys.exit("no images survived; nothing to write")

    # Step 5 -- group-aware split
    print("\n-- Step 5: group-aware split (grouped by source photograph)")
    splits, n_groups = group_split(records, seed=args.seed)
    print(f"  {len(records)} images from {n_groups} distinct source photographs")

    # Step 6 -- write
    print("\n-- Step 6: writing dataset")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    per_split = Counter()
    per_source = defaultdict(Counter)
    per_class = defaultdict(Counter)
    for i, (img, entries, src) in enumerate(tqdm(records, desc="  copying", leave=False)):
        split = splits[i]
        name = f"{src}_{img.stem}"
        shutil.copy2(img, out_dir / "images" / split / f"{name}{img.suffix}")
        write_labels(out_dir / "labels" / split / f"{name}.txt", entries)
        per_split[split] += 1
        per_source[src][split] += 1
        for cls, *_ in entries:
            per_class[split][cls] += 1

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(UNIFIED_CLASSES)}\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in UNIFIED_CLASSES.items())
    )

    # Step 7 -- verify no leakage, then report
    print("\n-- Step 7: verification")
    stem_splits = defaultdict(set)
    for s in ("train", "val", "test"):
        for f in (out_dir / "images" / s).iterdir():
            stem_splits[source_stem(f.name)].add(s)
    leaked = [k for k, v in stem_splits.items() if len(v) > 1]
    print(f"  source photographs spanning >1 split: {len(leaked)} "
          f"({'PASS' if not leaked else 'FAIL'})")

    total = sum(per_split.values())
    lines = [
        "DATASET UNIFICATION REPORT (v2)",
        "=" * 70,
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Full-frame sources: {'included' if args.include_fullframe else 'excluded'}",
        "",
        "Annotation formats read from source:",
    ]
    for src, c in sorted(fmt.items()):
        lines.append(f"  {src:<22} box={c['box']:<6} polygon={c['polygon']:<6} empty={c['empty']}")
    lines += ["", "Localisation audit (before dedup):"]
    for src, s in sorted(loc.items()):
        lines.append(f"  {src:<22} annotations={s['annotations']:<6} "
                     f"median_area={s['median_area']:<8} full_frame={s['full_frame_pct']}%")
    lines += ["", "Images per source AFTER deduplication (final contribution):"]
    for src in sorted(per_source):
        c = per_source[src]
        lines.append(f"  {src:<22} train={c['train']:<6} val={c['val']:<6} "
                     f"test={c['test']:<6} total={sum(c.values())}")
    lines += ["", "Split sizes:"]
    for s in ("train", "val", "test"):
        pct = 100 * per_split[s] / total if total else 0
        lines.append(f"  {s:<6} {per_split[s]:<6} ({pct:.1f}%)")
    lines += ["", "Class distribution:"]
    for s in ("train", "val", "test"):
        lines.append(f"  {s:<6} " + "  ".join(
            f"{UNIFIED_CLASSES[c]}={per_class[s][c]}" for c in sorted(UNIFIED_CLASSES)))
    lines += ["",
              f"Distinct source photographs: {n_groups}",
              f"Source photographs spanning >1 split: {len(leaked)}",
              f"Total images: {total}",
              f"Output: {out_dir}"]

    report = "\n".join(lines)
    print("\n" + report)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"unification_report_v2_{stamp}.txt").write_text(report)
    (RESULTS_DIR / "unification_report_v2_latest.txt").write_text(report)
    (RESULTS_DIR / f"unification_stats_v2_{stamp}.json").write_text(json.dumps({
        "formats": {k: dict(v) for k, v in fmt.items()},
        "localisation": loc,
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "per_split": dict(per_split),
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "source_groups": n_groups,
        "leaked_groups": len(leaked),
        "total": total,
    }, indent=2))
    print(f"\nReport: {RESULTS_DIR / f'unification_report_v2_{stamp}.txt'}")


if __name__ == "__main__":
    main()
