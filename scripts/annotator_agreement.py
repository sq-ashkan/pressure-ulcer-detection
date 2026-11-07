"""Inter-annotator agreement: rf_dekubitus vs. the maskrcnn/woundcare annotations.

The building of data/seg showed that almost every rf_dekubitus photograph is also in
roboflow_maskrcnn / roboflow_woundcare (same photo, independently annotated by a
different Roboflow user). Two independent outlines of the same wound give the level of
agreement between human annotators -- the practical ceiling for any model trained and
judged on such labels.

Method
  1. Every rf_dekubitus image (PNG semantic mask) is matched to the images of the two
     other projects by content: 64-bit pHash shortlist (Hamming <= 12 under any of the
     8 dihedral transforms), then the transform with the highest Pearson correlation of
     the 64x64 grey thumbnails; a pair is accepted when r >= 0.95 (same geometry: no
     crop or zoom between the two exports).
  2. The partner's polygons are rasterised at the partner's resolution, brought into
     the dekubitus frame with the same dihedral transform and resized (nearest).
  3. Agreement: binary Dice/IoU of "wound" (any class incl. DTI/Unstageable) between the
     two annotators; for pairs where both mark a stage 1-4 wound, agreement of the
     dominant stage and Cohen's kappa; bootstrap 95 % CIs.
  4. Extras (reviewer request), all over distinct photographs:
     - cohen_kappa_ci95: percentile bootstrap of Cohen's kappa (2000 resamples, seed 0;
       resamples with an undefined kappa -- a single category in both raters -- are
       dropped, their count is reported as cohen_kappa_bootstrap_undefined);
     - pabak = (k*p_o - 1)/(k - 1) with k = 4 stage categories (prevalence- and
       bias-adjusted kappa, Byrt et al. 1993);
     - gwet_ac1 = (p_o - p_e)/(1 - p_e), p_e = sum_k pi_k (1 - pi_k)/(k - 1), pi_k the mean
       of both raters' marginal proportions of stage k (Gwet 2008); both with bootstrap CIs;
     - binary_dice_by_stage: Dice mean, bootstrap CI and n per stratum, where the stratum is
       "stage_<s>" when both annotators give the same dominant stage s, "discordant" when
       they differ, "no_stage" when either has no stage 1-4 pixels; plus the same split by
       the rf_dekubitus annotator's stage alone (binary_dice_by_dekubitus_stage).
Run:    SELFTRAIN_DEVICE=cpu .venv/bin/python scripts/annotator_agreement.py
Output: results/annotator_agreement.json, results/annotator_agreement_pairs.csv
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selftrain.metrics import binary_scores, bootstrap_ci  # noqa: E402

RAW = ROOT / "data" / "raw"
PARTNERS = {"roboflow_maskrcnn": {2: 1, 3: 2, 4: 3, 5: 4, 0: 9, 1: 9},      # -> 1-4 stage, 9 = DTI/Unst.
            "roboflow_woundcare": {2: 1, 3: 2, 4: 3, 5: 4, 0: 9, 1: 9}}
DEK = {1: 9, 2: 9, 3: 1, 4: 2, 5: 3, 6: 4}


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    b = np.rot90(a, k % 4)
    return np.fliplr(b) if k >= 4 else b


def thumb(path: Path, s: int = 64) -> np.ndarray:
    g = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (s, s), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n > 1e-6 else g


def raster_partner(lbl: Path, table, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if not lbl.exists():
        return m
    polys = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        v = table.get(int(p[0]))
        if v is None:
            continue
        pts = (np.array(p[1:], np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        polys.append((v, pts))
    for v, pts in sorted(polys, key=lambda t: -cv2.contourArea(t[1].astype(np.float32))):
        cv2.fillPoly(m, [pts], v)
    return m


def dominant(m: np.ndarray) -> int:
    c = np.bincount(m.ravel(), minlength=10)[1:5]
    return int(c.argmax() + 1) if c.sum() else 0


K_STAGES = 4


def _po_pe_ac1(a, b, k: int = K_STAGES):
    a, b = np.asarray(a), np.asarray(b)
    po = float(np.mean(a == b))
    pi = np.array([(np.mean(a == c) + np.mean(b == c)) / 2 for c in range(1, k + 1)])
    pe = float((pi * (1 - pi)).sum() / (k - 1))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def stage_extras(st, b: int = 2000, seed: int = 0):
    """Bootstrap CI of Cohen's kappa, PABAK and Gwet's AC1 for the dominant-stage pairs."""
    a = np.array([x["stage_dek"] for x in st]); c = np.array([x["stage_partner"] for x in st])
    if len(st) < 2:
        return {}
    po, ac1 = _po_pe_ac1(a, c)
    pabak = (K_STAGES * po - 1) / (K_STAGES - 1)
    rng = np.random.default_rng(seed)
    ks, pb, ac = [], [], []
    for _ in range(b):
        i = rng.integers(0, len(a), len(a))
        with np.errstate(all="ignore"):
            k = cohen_kappa_score(a[i], c[i]) if len(set(a[i]) | set(c[i])) > 1 else float("nan")
        ks.append(k)
        p, g = _po_pe_ac1(a[i], c[i])
        pb.append((K_STAGES * p - 1) / (K_STAGES - 1)); ac.append(g)
    ks = np.array(ks, float)
    q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
    return {"cohen_kappa_ci95": q(ks), "cohen_kappa_bootstrap_undefined": int(np.isnan(ks).sum()),
            "pabak": float(pabak), "pabak_ci95": q(np.array(pb)), "pabak_categories": K_STAGES,
            "gwet_ac1": float(ac1), "gwet_ac1_ci95": q(np.array(ac, float))}


def dice_strata(uniq, key):
    groups = {}
    for x in uniq:
        groups.setdefault(key(x), []).append(x["dice"])
    out = {}
    for g in sorted(groups):
        m, ci = bootstrap_ci(groups[g])
        out[g] = {"n": len(groups[g]), "dice_mean": m, "dice_ci95": list(ci),
                  "dice_median": float(np.median(groups[g]))}
    return out


def pair_stratum(x):
    if not x["stage_dek"] or not x["stage_partner"]:
        return "no_stage"
    return f"stage_{x['stage_dek']}" if x["stage_dek"] == x["stage_partner"] else "discordant"


def main():
    partner_imgs = []
    for src in PARTNERS:
        for split in ("train", "valid", "test"):
            for img in sorted((RAW / src / split / "images").glob("*.jpg")):
                partner_imgs.append((src, img, RAW / src / split / "labels" / f"{img.stem}.txt"))
    p_hash = np.array([imagehash.phash(Image.open(p)).hash.ravel() for _, p, _ in partner_imgs], bool)
    p_thumb = [thumb(p) for _, p, _ in partner_imgs]

    rows = []
    for split in ("train", "valid", "test"):
        d = RAW / "rf_dekubitus" / split
        for img in sorted(d.glob("*.jpg")):
            mask = cv2.imread(str(d / f"{img.stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            g = thumb(img)
            base = np.asarray(Image.open(img).convert("L").resize((32, 32), Image.LANCZOS), np.float32)
            best = (-1.0, None, None)
            for k in range(8):
                hk = imagehash.phash(Image.fromarray(dihedral(np.asarray(Image.open(img).convert("L")), k).copy()))
                cand = np.where((p_hash != hk.hash.ravel()).sum(1) <= 12)[0]
                for j in cand:
                    r = float((dihedral(g, k) * p_thumb[j]).sum())
                    if r > best[0]:
                        best = (r, j, k)
            if best[1] is None or best[0] < 0.95:
                rows.append({"dekubitus": img.name, "matched": "", "r": round(best[0], 4)})
                continue
            r, j, k = best
            src, pimg, plbl = partner_imgs[j]
            ph, pw = cv2.imread(str(pimg)).shape[:2]
            pm = raster_partner(plbl, PARTNERS[src], ph, pw)
            # partner frame -> dekubitus frame: invert the dihedral transform k
            inv = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}[k]
            pm = np.ascontiguousarray(dihedral(pm, inv))
            pm = cv2.resize(pm, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            dm = np.vectorize(lambda v: DEK.get(int(v), 0))(mask).astype(np.uint8)
            dice, iou = binary_scores(dm > 0, pm > 0)
            rows.append({"dekubitus": img.name, "matched": f"{src}/{pimg.name}", "r": round(r, 4),
                         "dice": dice, "iou": iou, "stage_dek": dominant(dm), "stage_partner": dominant(pm)})

    pairs = [x for x in rows if x.get("matched")]
    # one pair per distinct dekubitus original (Roboflow augmentation copies share the stem)
    seen, uniq = set(), []
    for x in pairs:
        key = x["dekubitus"].split("_jpg.rf.")[0]
        if key not in seen:
            seen.add(key); uniq.append(x)
    dm, dci = bootstrap_ci([x["dice"] for x in uniq])
    im, ici = bootstrap_ci([x["iou"] for x in uniq])
    st = [x for x in uniq if x["stage_dek"] and x["stage_partner"]]
    agree = float(np.mean([x["stage_dek"] == x["stage_partner"] for x in st])) if st else float("nan")
    kappa = float(cohen_kappa_score([x["stage_dek"] for x in st], [x["stage_partner"] for x in st],
                                    weights=None)) if len(st) > 1 else float("nan")
    res = {"dekubitus_images": len(rows), "matched_images": len(pairs), "matched_photographs": len(uniq),
           "binary_dice_mean": dm, "binary_dice_ci95": dci, "binary_iou_mean": im, "binary_iou_ci95": ici,
           "binary_dice_median": float(np.median([x["dice"] for x in uniq])) if uniq else float("nan"),
           "stage_pairs": len(st), "dominant_stage_agreement": agree, "cohen_kappa": kappa,
           "stage_confusion": {f"{a}->{b}": sum(1 for x in st if x["stage_dek"] == a and x["stage_partner"] == b)
                               for a in range(1, 5) for b in range(1, 5)}}
    res.update(stage_extras(st))
    res["binary_dice_by_stage"] = dice_strata(uniq, pair_stratum)
    res["binary_dice_by_dekubitus_stage"] = dice_strata(uniq, lambda x: f"stage_{x['stage_dek']}")
    out = ROOT / "results"
    (out / "annotator_agreement.json").write_text(json.dumps(res, indent=2))
    with open(out / "annotator_agreement_pairs.csv", "w", newline="") as f:
        keys = ["dekubitus", "matched", "r", "dice", "iou", "stage_dek", "stage_partner"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k, "") for k in keys})
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
