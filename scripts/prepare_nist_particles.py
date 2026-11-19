"""Prepare NIST_CalibrationParticles SEM images for use with the pipeline:
crop off the instrument's burned-in metadata banner (bottom ~60px, not
micrograph content) and OCR the banner's "Pixel Size = X.XX nm" line for a
per-image px->nm calibration factor.

Important limitation: the .dat particle measurement files (LWT/proj) are
keyed by STL filenames from a *separate* X-ray CT acquisition and have no
shared ID with these SEM image filenames -- there is no way to look up "the
true 3D size of the specific particle in this specific image." The per-image
OCR'd pixel size is the only real calibration signal available here; treat
population-level size-distribution comparison against the CT data as a
sanity check, not a per-particle ground truth match.

Usage:
    python scripts/prepare_nist_particles.py [--limit N]
"""

import argparse
import csv
import os
import re

import pytesseract
from PIL import Image

DATA_DIR = "data/datasets/NIST_CalibrationParticles/SEM_images/SEM-images"
BANNER_TOP = 708  # row where the image content ends and the black banner begins (verified per-instrument)
PIXEL_SIZE_RE = re.compile(r"Pixel Size\s*=\s*([\d.]+)\s*(nm|um|µm)", re.IGNORECASE)


def ocr_pixel_size(banner_crop: Image.Image) -> tuple[float, str] | None:
    upscaled = banner_crop.resize((banner_crop.width * 3, banner_crop.height * 3))
    text = pytesseract.image_to_string(upscaled)
    m = PIXEL_SIZE_RE.search(text)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


def process_batch(batch_dir, out_images_dir, records, limit_remaining):
    count = 0
    for fname in sorted(os.listdir(batch_dir)):
        if not fname.lower().endswith((".tif", ".tiff")) or count >= limit_remaining:
            break
        path = f"{batch_dir}/{fname}"
        im = Image.open(path).convert("L")
        if im.height <= BANNER_TOP:
            continue  # unexpected size, skip rather than guess a wrong crop

        content = im.crop((0, 0, im.width, BANNER_TOP))
        banner = im.crop((0, BANNER_TOP, im.width, im.height))
        calib = ocr_pixel_size(banner)

        out_name = fname.replace(".tif", ".png").replace(".tiff", ".png")
        content.save(f"{out_images_dir}/{out_name}")
        records.append(
            {
                "file": out_name,
                "source": path,
                "pixel_size_value": calib[0] if calib else "",
                "pixel_size_unit": calib[1] if calib else "",
            }
        )
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="images to prepare per particle-batch dir")
    args = parser.parse_args()

    records = []
    for batch_name in sorted(os.listdir(DATA_DIR)):
        batch_root = f"{DATA_DIR}/{batch_name}"
        if not os.path.isdir(batch_root):
            continue
        for sub in sorted(os.listdir(batch_root)):
            sub_path = f"{batch_root}/{sub}"
            if not os.path.isdir(sub_path):
                continue
            out_images_dir = f"data/datasets/NIST_CalibrationParticles/prepared/{batch_name}/images"
            os.makedirs(out_images_dir, exist_ok=True)
            n = process_batch(sub_path, out_images_dir, records, args.limit)
            print(f"{batch_name}/{sub}: prepared {n} images")

    out_dir = "data/datasets/NIST_CalibrationParticles/prepared"
    with open(f"{out_dir}/calibration_index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "pixel_size_value", "pixel_size_unit"])
        writer.writeheader()
        writer.writerows(records)

    ocr_hits = sum(1 for r in records if r["pixel_size_value"])
    print(f"\nprepared {len(records)} images total, OCR'd pixel size for {ocr_hits}/{len(records)}")
    print(f"index -> {out_dir}/calibration_index.csv")


if __name__ == "__main__":
    main()
