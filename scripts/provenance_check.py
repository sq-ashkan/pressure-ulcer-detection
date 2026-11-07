"""Provenance check: OCR the image borders for watermarks / copyright notices.

File names alone do not prove provenance; this script looks at the pixels.
For every image in data/seg/images/{train,val,test} the four outer border
strips (12 % of height for top/bottom, 12 % of width for left/right) are cut
out; the side strips are rotated both ways so vertical text is readable too.
The strips are stacked into one white-padded grayscale canvas, upscaled 2x and
read with the tesseract CLI (``--psm 11``, sparse text, single-threaded,
deterministic).  An image is flagged when the OCR text contains a watermark
keyword (copyright sign, "copyright", "mayo", "foundation", stock agency
names, "www"/".com", ...).  Flags are aggregated per split and source and per
photograph id (manifest.csv column ``photograph``), and cross-tabulated with
stock-photo-named files (istock, shutterstock, alamy, getty, ...).

Command line:
    .venv/bin/python scripts/provenance_check.py [--workers 6]

Outputs:
    results/provenance_check.json   method, counts, flagged images/photographs
"""
import argparse, csv, json, os, re, subprocess, tempfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data/seg/images"
SPLITS = ("train", "val", "test")
FRAC = 0.12
STOCK_RE = re.compile(r"istock|shutterstock|alamy|getty|dreamstime|adobe|depositphotos|123rf|medetec|mayo", re.I)
SERIES_RE = re.compile(r"chronic-ulcer", re.I)
# keyword hits in OCR text (lower-cased).  A bare "©" is NOT a hit: tesseract
# reads it out of skin texture; it counts only when followed by a year or a word
# ("©2017 Medline", "© Mayo").  Calibrated by inspecting contact sheets.
WM_RE = re.compile(r"©\s?(?:\d{4}|[A-Za-z]{3,})|\(c\)\s?\d{4}|copyright|mayo|foundation|istock|shutterstock|alamy|getty|dreamstime|"
                   r"adobe|depositphoto|123rf|medetec|www\.|\.com|\.org|rights reserved|photo by|credit", re.I)

# rights holder attribution from the OCR text (first match wins)
ISSUERS = [("Medetec", r"med[ea]?[lt]?[ea]c|medet"), ("NPIAP / Gordian Medical", r"npia|aplap|gordian"),
           ("Mayo Foundation", r"mayo|foundation for medical"), ("Shutterstock", r"shutterstock"),
           ("Medline", r"medline")]


def issuer(text):
    for name, rx in ISSUERS:
        if re.search(rx, text, re.I):
            return name
    return "other/unreadable"


def composite(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    bh, bw = max(1, int(round(h * FRAC))), max(1, int(round(w * FRAC)))
    a = np.asarray(im)
    strips = [a[:bh], a[h - bh:]]
    for s in (a[:, :bw], a[:, w - bw:]):
        strips += [np.rot90(s, 1), np.rot90(s, -1)]
    W = max(s.shape[1] for s in strips)
    pad = 20
    rows = []
    for s in strips:
        canvas = np.full((s.shape[0] + 2 * pad, W + 2 * pad), 255, np.uint8)
        canvas[pad:pad + s.shape[0], pad:pad + s.shape[1]] = s
        rows.append(canvas)
    c = Image.fromarray(np.vstack(rows))
    return c.resize((c.width * 2, c.height * 2), Image.BICUBIC)


def ocr(path):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        composite(path).save(f.name)
        out = subprocess.run(["tesseract", f.name, "stdout", "--psm", "11", "-l", "eng"],
                             capture_output=True, text=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    text = " ".join(out.stdout.split())
    hits = sorted({m.group(0).lower()[:12] for m in WM_RE.finditer(text)})
    return path.name, text, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    man = {}
    with open(ROOT / "data/seg/manifest.csv") as f:
        for r in csv.DictReader(f):
            man[r["name"]] = r
    files = [(sp, p) for sp in SPLITS for p in sorted((IMG / sp).glob("*.jpg"))]
    with Pool(args.workers) as pool:
        res = pool.map(ocr, [p for _, p in files], chunksize=8)
    per_split, per_source, total = Counter(), Counter(), Counter()
    flagged, photos = [], defaultdict(lambda: {"images": 0, "splits": set(), "stock_named": False, "series_chronic_ulcer": False})
    for (sp, p), (name, text, hits) in zip(files, res):
        m = man.get(p.stem, {})
        src = m.get("source") or name.split("__")[0]
        total[sp] += 1
        stock, series = bool(STOCK_RE.search(name)), bool(SERIES_RE.search(name))
        if not hits:
            continue
        per_split[sp] += 1
        per_source[src] += 1
        pid = f"{src}:{m.get('photograph', '?')}"
        ph = photos[pid]
        ph["images"] += 1; ph["splits"].add(sp)
        ph["stock_named"] |= stock; ph["series_chronic_ulcer"] |= series
        flagged.append({"file": name, "split": sp, "source": src, "photograph": pid,
                        "hits": hits, "issuer": issuer(text), "stock_named": stock, "ocr_text": text[:200]})
    photos = {k: {**v, "splits": sorted(v["splits"])} for k, v in sorted(photos.items())}
    out = {
        "method": ("tesseract %s CLI, --psm 11, eng; four outer border strips (%.0f %% of side), side strips "
                   "rotated +-90 deg, stacked, upscaled 2x; flag = OCR text matches %s"
                   % (subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split()[1],
                      FRAC * 100, WM_RE.pattern)),
        "images_checked": dict(total),
        "flagged_images_per_split": dict(per_split),
        "flagged_images_per_source": dict(per_source),
        "flagged_images_total": len(flagged),
        "flagged_photographs_total": len(photos),
        "flagged_photographs_stock_named": sum(v["stock_named"] for v in photos.values()),
        "flagged_photographs_chronic_ulcer_series": sum(v["series_chronic_ulcer"] for v in photos.values()),
        "flagged_images_per_issuer": dict(Counter(f["issuer"] for f in flagged)),
        "flagged_photographs_per_issuer": dict(Counter(issuer(" ".join(f["ocr_text"] for f in flagged if f["photograph"] == k)) for k in photos)),
        "flagged_test_images": [f["file"] for f in flagged if f["split"] == "test"],
        "visual_check": ("manual: contact sheets (48 thumbnails each) of one image per flagged photograph and of all "
                         "27 istock-named images were inspected; every flagged sheet image shows a printed notice; "
                         "istock-named images show no visible watermark (re-cropped previews); a bare OCR '(c)' sign "
                         "was a false positive on skin texture and is therefore not counted"),
        "stock_named_images_total": sum(bool(STOCK_RE.search(p.name)) for _, p in files),
        "flagged_photographs": photos,
        "flagged_images": flagged,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/provenance_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("flagged_photographs", "flagged_images", "method", "visual_check")}, indent=1))


if __name__ == "__main__":
    main()
