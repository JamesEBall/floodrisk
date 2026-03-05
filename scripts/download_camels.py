#!/usr/bin/env python
"""Download the CAMELS-US dataset from NCAR.

Downloads basin_mean_forcing_daymet, usgs_streamflow, and camels_attributes
archives and extracts them into the specified output directory.

Usage::

    python scripts/download_camels.py --output_dir data/camels
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

CAMELS_URLS = {
    "basin_mean_forcing_daymet": (
        "https://gdex.ucar.edu/dataset/camels/file/"
        "basin_mean_forcing_daymet.tar.gz"
    ),
    "usgs_streamflow": (
        "https://gdex.ucar.edu/dataset/camels/file/"
        "usgs_streamflow.tar.gz"
    ),
    "camels_attributes": (
        "https://gdex.ucar.edu/dataset/camels/file/"
        "camels_attributes_v2.0.tar.gz"
    ),
}


def download_and_extract(url: str, tar_path: Path, output_dir: Path) -> None:
    """Download a tar.gz file and extract it."""
    if tar_path.exists():
        print(f"Already downloaded: {tar_path}")
    else:
        print(f"Downloading {url} ...")
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(tar_path))
        print(f"  Saved to {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")

    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir, filter="data")
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CAMELS-US dataset from NCAR."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/camels",
        help="Output directory for CAMELS data (default: data/camels)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, url in CAMELS_URLS.items():
        tar_path = output_dir / f"{name}.tar.gz"
        download_and_extract(url, tar_path, output_dir)

    print(f"\nCAMELS-US dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
