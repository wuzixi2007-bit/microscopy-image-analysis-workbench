"""Convert NIST_AM_Pores' 32x32x32 binary volumes into 2D max-intensity
projections (the pore's silhouette as it would appear looking down one axis),
paired with each pore's known equivalent spherical diameter from the
metadata CSV. Used to validate measurement.py's pixel-based sizing against
a real, independently-measured ground truth.

Note: the source readme only gives a dataset-wide voxel resolution range
(~2.43-2.77 um/voxel), not an exact per-pore value, so any px->um conversion
here carries that same +/-6% uncertainty baked into the source data.

Note: the metadata CSV's `equivalent_spherical_diameter` is in VOXEL units,
not micrometers (verified: sphere-volume-from-voxel-count math reproduces it
exactly, e.g. pore 0's 1605 voxels -> d=14.526 voxels, matching the CSV's
14.53). It must be multiplied by VOXEL_SIZE_UM before comparing against any
pixel/projection measurement in real units.

Usage:
    python scripts/prepare_nist_pores.py [--limit N]
"""

import argparse
import csv
import os

import numpy as np
import tifffile
from PIL import Image

DATA_DIR = "data/datasets/NIST_AM_Pores"
VOXEL_SIZE_UM = 2.6  # midpoint of the dataset's stated 2.43-2.77 um/voxel range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="how many pores to prepare (dataset has 3503)")
    args = parser.parse_args()

    out_dir = f"{DATA_DIR}/prepared"
    os.makedirs(f"{out_dir}/projections", exist_ok=True)

    rows = []
    with open(f"{DATA_DIR}/_download/synthetic_pore_metadata.csv") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= args.limit:
                break
            rows.append(row)

    records = []
    for row in rows:
        tiff_path = f"{DATA_DIR}/tiff_masks/{row['tiff_filename']}"
        volume = tifffile.imread(tiff_path) > 0  # (32, 32, 32) bool

        projection = volume.max(axis=0).astype(np.uint8) * 255  # silhouette looking down z
        out_name = row["tiff_filename"].replace(".tiff", ".png")
        Image.fromarray(projection).save(f"{out_dir}/projections/{out_name}")

        records.append(
            {
                "file": out_name,
                "known_equivalent_spherical_diameter_voxels": row["equivalent_spherical_diameter"],
                "known_equivalent_spherical_diameter_um": float(row["equivalent_spherical_diameter"]) * VOXEL_SIZE_UM,
                "projection_area_px": int((projection > 0).sum()),
                "projection_area_um2": int((projection > 0).sum()) * VOXEL_SIZE_UM**2,
            }
        )

    fieldnames = [
        "file",
        "known_equivalent_spherical_diameter_voxels",
        "known_equivalent_spherical_diameter_um",
        "projection_area_px",
        "projection_area_um2",
    ]
    with open(f"{out_dir}/projection_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"prepared {len(records)} projections -> {out_dir}/projections/")
    print(f"index with known sizes -> {out_dir}/projection_index.csv")
    print(f"voxel size used for px->um conversion: {VOXEL_SIZE_UM} um/voxel (dataset states 2.43-2.77 range)")


if __name__ == "__main__":
    main()
