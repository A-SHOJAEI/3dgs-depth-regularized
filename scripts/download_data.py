#!/usr/bin/env python3
"""Download Mip-NeRF 360 dataset (garden scene)."""

import os
import subprocess
import sys
from pathlib import Path


def download_mipnerf360(data_dir="data", scenes=None):
    """Download Mip-NeRF 360 dataset.

    Downloads the full 360_v2.zip and optionally keeps only specified scenes.
    """
    data_path = Path(data_dir)
    target_dir = data_path / "360_v2"

    if scenes is None:
        scenes = ["garden"]

    # Check if already downloaded
    all_exist = all((target_dir / scene).exists() for scene in scenes)
    if all_exist:
        print(f"Scenes already exist: {scenes}")
        return

    data_path.mkdir(parents=True, exist_ok=True)
    zip_path = data_path / "360_v2.zip"

    url = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

    if not zip_path.exists():
        print(f"Downloading Mip-NeRF 360 dataset...")
        print(f"  URL: {url}")
        print(f"  Target: {zip_path}")
        subprocess.run(
            ["curl", "-L", "-o", str(zip_path), url],
            check=True,
        )
        print("Download complete.")
    else:
        print(f"Zip already exists: {zip_path}")

    # Extract only requested scenes
    print(f"Extracting scenes: {scenes}")
    for scene in scenes:
        subprocess.run(
            ["unzip", "-o", "-q", str(zip_path), f"{scene}/*", "-d", str(target_dir)],
            check=True,
        )
        print(f"  Extracted: {scene}")

    # Remove zip to save space
    zip_path.unlink()
    print(f"Removed zip file to save space.")

    # Verify
    for scene in scenes:
        scene_dir = target_dir / scene
        if scene_dir.exists():
            n_images = len(list((scene_dir / "images").glob("*")))
            print(f"  {scene}: {n_images} images")
        else:
            print(f"  WARNING: {scene} not found!")


if __name__ == "__main__":
    scenes = sys.argv[1:] if len(sys.argv) > 1 else ["garden"]
    download_mipnerf360(scenes=scenes)
