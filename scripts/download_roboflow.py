"""
Download all Roboflow datasets for the Decubitus Detection project.

Usage: python scripts/download_roboflow.py
Requires: ROBOFLOW_API_KEY in .env

Notes:
  * The target directory must NOT be pre-created: roboflow's Version.download()
    silently skips when the location already exists and overwrite=False.
  * Version numbers are resolved dynamically, because pinned versions can be
    removed by the dataset owner.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    sys.exit("ROBOFLOW_API_KEY not found in .env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = [
    {"name": "roboflow_stage2",    "workspace": "stage2-n7xya",         "project": "pressure-ulcer-sxitf"},
    {"name": "roboflow_project1",  "workspace": "project-1-hozvp",      "project": "pressure-ulcer-fr7kn"},
    {"name": "roboflow_maskrcnn",  "workspace": "pressure-injury",      "project": "maskrcnn-swd7w"},
    {"name": "roboflow_fid",       "workspace": "fid-fvcc6",            "project": "pressureulcer-ctn4w"},
    {"name": "roboflow_woundcare", "workspace": "pi-iqy1t",             "project": "advances-in-wound-care-pi-dataset"},
    {"name": "roboflow_sciproj",   "workspace": "pressure-ulcer-pgayk", "project": "pressureulcer-sciproj-2024-fmokt"},
    {"name": "roboflow_mobile",    "workspace": "class-vx0ys",          "project": "mobile-application-for-diagnostic-of-pressure-ulcer"},
]

FMT = "yolov8"


def count_images(root: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in exts)


def resolve_version(project, preferred=1):
    """Return a usable version number, preferring `preferred`, else the highest."""
    try:
        nums = sorted({int(str(v.version).rstrip("/").split("/")[-1]) for v in project.versions()})
    except Exception as exc:
        print(f"    could not list versions ({exc}); trying {preferred}")
        return preferred
    if not nums:
        return preferred
    return preferred if preferred in nums else nums[-1]


def main():
    rf = Roboflow(api_key=API_KEY)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for ds in DATASETS:
        out = RAW_DIR / ds["name"]
        print("=" * 64)
        print(f"Downloading: {ds['name']}  ({ds['workspace']}/{ds['project']})")

        if out.exists() and count_images(out) > 0:
            n = count_images(out)
            print(f"  already present with {n} images -- skipping")
            summary.append((ds["name"], n, "cached"))
            continue

        # roboflow skips when the directory exists; make sure it does not.
        if out.exists():
            shutil.rmtree(out)

        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            ver = resolve_version(project, preferred=1)
            print(f"  using version {ver}")
            project.version(ver).download(FMT, location=str(out), overwrite=True)
            n = count_images(out)
            status = "ok" if n > 0 else "EMPTY"
            print(f"  -> {status}: {n} images")
            summary.append((ds["name"], n, status))
        except Exception as exc:
            print(f"  -> FAILED: {exc}")
            summary.append((ds["name"], 0, f"failed: {exc}"))

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    total = 0
    for name, n, status in summary:
        print(f"  {name:<22} {n:>6} images   {status}")
        total += n
    print(f"  {'TOTAL':<22} {total:>6} images")
    bad = [s for s in summary if s[1] == 0]
    if bad:
        print(f"\n  WARNING: {len(bad)} dataset(s) returned no images:")
        for name, _, status in bad:
            print(f"    - {name}: {status}")


if __name__ == "__main__":
    main()
