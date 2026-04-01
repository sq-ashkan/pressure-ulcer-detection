"""
Download GitHub-hosted datasets for Decubitus Detection project.
Usage: python scripts/download_github.py
"""

import subprocess
import os
from pathlib import Path

RAW_DIR = Path("data/raw")

GITHUB_REPOS = [
    {
        "name": "azh_localization",
        "url": "https://github.com/uwm-bigdata/wound_localization.git",
    },
    {
        "name": "piid",
        "url": "https://github.com/FU-MedicalAI/PIID.git",
    },
    {
        "name": "medetec",
        "url": "https://github.com/mlaradji/deep-learning-for-wound-care.git",
    },
    {
        "name": "azh_classification",
        "url": "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations.git",
    },
    {
        "name": "fuseg",
        "url": "https://github.com/uwm-bigdata/wound-segmentation.git",
    },
]


def clone_all():
    for repo in GITHUB_REPOS:
        output_dir = RAW_DIR / repo["name"]

        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Skipping {repo['name']} — already exists at {output_dir}")
            continue

        print(f"\nCloning: {repo['name']}")
        print(f"  URL:    {repo['url']}")
        print(f"  Output: {output_dir}")

        try:
            subprocess.run(
                ["git", "clone", repo["url"], str(output_dir)],
                check=True,
            )
            print(f"  -> Done!")
        except subprocess.CalledProcessError as e:
            print(f"  -> FAILED: {e}")
            continue

    print(f"\n\nAll clones complete! Check {RAW_DIR}/")


if __name__ == "__main__":
    clone_all()
