"""
Download all Roboflow datasets for Decubitus Detection project.
Usage: python scripts/download_all.py
Requires: ROBOFLOW_API_KEY in .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

RAW_DIR = Path("data/raw")

DATASETS = [
    {
        "name": "roboflow_stage2",
        "workspace": "stage2-n7xya",
        "project": "pressure-ulcer-sxitf",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "roboflow_project1",
        "workspace": "project-1-hozvp",
        "project": "pressure-ulcer-fr7kn",
        "version": 4,
        "format": "yolov8",
    },
    {
        "name": "roboflow_maskrcnn",
        "workspace": "pressure-injury",
        "project": "maskrcnn-swd7w",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "roboflow_fid",
        "workspace": "fid-fvcc6",
        "project": "pressureulcer-ctn4w",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "roboflow_woundcare",
        "workspace": "pi-iqy1t",
        "project": "advances-in-wound-care-pi-dataset",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "roboflow_sciproj",
        "workspace": "pressure-ulcer-pgayk",
        "project": "pressureulcer-sciproj-2024-fmokt",
        "version": 1,
        "format": "yolov8",
    },
    {
        "name": "roboflow_mobile",
        "workspace": "class-vx0ys",
        "project": "mobile-application-for-diagnostic-of-pressure-ulcer",
        "version": 1,
        "format": "yolov8",
    },
]


def download_all():
    rf = Roboflow(api_key=API_KEY)

    for ds in DATASETS:
        output_dir = RAW_DIR / ds["name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Downloading: {ds['name']}")
        print(f"  Workspace: {ds['workspace']}")
        print(f"  Project:   {ds['project']}")
        print(f"  Version:   {ds['version']}")
        print(f"  Format:    {ds['format']}")
        print(f"  Output:    {output_dir}")
        print(f"{'='*60}")

        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = project.version(ds["version"])
            version.download(
                ds["format"],
                location=str(output_dir.resolve()),
                overwrite=True,
            )
            # Count downloaded files
            img_count = sum(
                1
                for f in output_dir.rglob("*")
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            print(f"  -> Done! {img_count} images downloaded.")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            continue

    print(f"\n\nAll downloads complete! Check {RAW_DIR}/")


if __name__ == "__main__":
    download_all()
