"""BAM_TiO2 has a shape mismatch between images and masks for its TSEM half
(odd-numbered files): images are 768x1024, masks 712x1024. Both subsets'
raw images carry an identical 56px instrument-info banner at the bottom
("BAM FB 6.1 / File Name / EHT / Store resolution / Width = 7.520 um /
Mag..."); the SEM (even-numbered) masks keep that strip as all-zero
background (768 rows), but the TSEM masks were cropped to exclude it
entirely (712 rows) -- an inconsistency between the two original annotation
batches, not corruption. Elementwise image/mask ops (e.g. `image[mask]=...`)
crash on the mismatch, so this crops every image to match its mask.

Usage:
    python scripts/prepare_bam_tio2.py
"""

import os

import numpy as np
from PIL import Image

DATA_DIR = "data/datasets/BAM_TiO2"


def main():
    img_dir = f"{DATA_DIR}/images"
    msk_dir = f"{DATA_DIR}/masks"
    out_dir = f"{DATA_DIR}/prepared/images"
    os.makedirs(out_dir, exist_ok=True)

    cropped = 0
    unchanged = 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        image = np.array(Image.open(f"{img_dir}/{fname}").convert("RGB"))
        mask = np.array(Image.open(f"{msk_dir}/{fname}"))

        if image.shape[:2] != mask.shape[:2]:
            image = image[: mask.shape[0], : mask.shape[1]]
            cropped += 1
        else:
            unchanged += 1

        # keep the original filename (extension included) so prepared/images/<f>
        # still pairs directly with the untouched masks/<f>
        Image.fromarray(image).save(f"{out_dir}/{fname}")

    print(f"cropped {cropped} images to match their mask, {unchanged} already matched")
    print(f"prepared images -> {out_dir} (same filenames as masks/, pair directly)")


if __name__ == "__main__":
    main()
