"""
Download the extra sources that the Bachelor thesis needs on top of
download_all.py / download_github.py, and unpack the archived ones.

  * three further Roboflow Universe projects with polygon/mask labels
    (rf_petchs_seg, rf_thai_sores, rf_dekubitus), latest version, YOLOv8 format
    (rf_dekubitus is a semantic-segmentation project: PNG masks instead)
  * PISeg (github.com/meheditusar/PISeg), pressure-injury segmentation
  * PIID (1,091 images, four stages) -- the GitHub repo only holds a README;
    the images are a RAR on Google Drive. Unpacked to data/raw/piid/extracted/
    with one folder per stage (1..4), the layout unify_dataset_v3.py expects.
  * AZH wound datasets -- their images ship inside zip files; each zip is
    unpacked into its own sibling folder.

Usage:  python scripts/download_extra.py [--skip-roboflow] [--skip-piid]
Needs:  ROBOFLOW_API_KEY in .env, `unar` (brew install unar) for the RAR.
"""

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ROBOFLOW_EXTRA = [
    ("rf_petchs_seg", "petchs-workspace-v8pvc", "pressure-ulcer-segmentation"),
    ("rf_thai_sores", "thanyaratthadchpong", "pressure-sores-1rbco"),
    ("rf_dekubitus", "mweber1-rheinahrcampus-de", "dekubitus"),
]
PISEG_URL = "https://github.com/meheditusar/PISeg.git"
PIID_GDRIVE_ID = "1tSKSAmrmQ_kQz220GsMfaBMgTVga2IYB"  # PIID.rar in folder 12JouktrzXIo6ywpSe2OYWRYNNIxlEKvK
AZH_ZIPS = [
    "azh_classification/dataset/Train.zip",
    "azh_classification/dataset/Test.zip",
    "azh_classification/BodyMap.zip",
    "azh_localization/dataset/Localization_ROI_Dataset.zip",
]


def count_images(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMG_EXT) if d.exists() else 0


def roboflow_extra():
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY not found in .env")
    rf = Roboflow(api_key=key)
    for name, ws, proj in ROBOFLOW_EXTRA:
        out = RAW_DIR / name
        if count_images(out):
            print(f"  {name}: already present ({count_images(out)} images)")
            continue
        if out.exists():
            shutil.rmtree(out)
        try:
            p = rf.workspace(ws).project(proj)
            versions = sorted({int(str(v.version).rstrip("/").split("/")[-1]) for v in p.versions()})
            v = versions[-1]
            print(f"  {name}: versions {versions}, using v{v}")
            # Semantic-segmentation projects cannot export YOLO; take PNG masks instead
            fmt = "png-mask-semantic" if getattr(p, "type", "") == "semantic-segmentation" else "yolov8"
            p.version(v).download(fmt, location=str(out), overwrite=True)
            print(f"    -> {count_images(out)} images ({fmt})")
        except Exception as exc:  # keep going: one dead project must not stop the rest
            print(f"  {name}: FAILED {exc}")


def piseg():
    out = RAW_DIR / "piseg"
    if count_images(out):
        print(f"  piseg: already present ({count_images(out)} images)")
        return
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(["git", "clone", "--depth", "1", "-q", PISEG_URL, str(out)], check=True)
    print(f"  piseg: {count_images(out)} images")


def piid():
    import gdown

    base = RAW_DIR / "piid"
    extracted = base / "extracted"
    if count_images(extracted):
        print(f"  piid: already extracted ({count_images(extracted)} images)")
        return
    base.mkdir(parents=True, exist_ok=True)
    rar = base / "PIID.rar"
    if not rar.exists() or rar.stat().st_size < 1_000_000:
        gdown.download(id=PIID_GDRIVE_ID, output=str(rar), quiet=False)
    unar = shutil.which("unar")
    if not unar:
        raise SystemExit("`unar` not found -- install it with: brew install unar")
    tmp = base / "_unpack"
    if tmp.exists():
        shutil.rmtree(tmp)
    subprocess.run([unar, "-quiet", "-force-overwrite", "-output-directory", str(tmp), str(rar)], check=True)
    # Normalise to extracted/1..4 whatever the archive's top-level folder names are
    # (e.g. "Stage-1", "stage 1", "1").
    extracted.mkdir(parents=True, exist_ok=True)
    moved = 0
    for d in sorted(p for p in tmp.rglob("*") if p.is_dir()):
        m = re.search(r"(?:stage|category|kategorie)?[\s_-]*([1-4])$", d.name, re.I)
        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT]
        if m and imgs:
            dest = extracted / m.group(1)
            dest.mkdir(exist_ok=True)
            for img in imgs:
                shutil.move(str(img), dest / img.name)
                moved += 1
    shutil.rmtree(tmp)
    per_stage = {s: count_images(extracted / s) for s in "1234"}
    print(f"  piid: {moved} images -> {per_stage}")


def azh():
    for rel in AZH_ZIPS:
        z = RAW_DIR / rel
        if not z.exists():
            print(f"  {rel}: missing (run download_github.py first)")
            continue
        dest = z.with_suffix("")  # Train.zip -> Train/
        if count_images(dest):
            print(f"  {rel}: already unpacked ({count_images(dest)} images)")
            continue
        with zipfile.ZipFile(z) as zf:
            for member in zf.infolist():
                # refuse absolute paths and parent-directory escapes
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise SystemExit(f"unsafe path in {z}: {member.filename}")
            zf.extractall(dest)
        for junk in dest.rglob("__MACOSX"):
            shutil.rmtree(junk, ignore_errors=True)
        print(f"  {rel}: {count_images(dest)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-roboflow", action="store_true")
    ap.add_argument("--skip-piid", action="store_true")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_roboflow:
        print("== Roboflow extra projects"); roboflow_extra()
    print("== PISeg"); piseg()
    if not args.skip_piid:
        print("== PIID (Google Drive)"); piid()
    print("== AZH archives"); azh()
    print("\ndone.")


if __name__ == "__main__":
    main()
