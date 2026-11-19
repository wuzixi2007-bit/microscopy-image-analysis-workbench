"""Parse MoNuSeg's Aperio-style XML polygon annotations into rasterized
binary masks aligned with each .tif image, so the dataset can be fed into
FewShotDetector/validate_few_shot.py like Aachen-Heerlen's CSV polygons were.

Usage:
    python scripts/prepare_monuseg.py
"""

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image

DATA_DIR = "data/datasets/MoNuSeg"


def parse_polygons(xml_path):
    tree = ET.parse(xml_path)
    polygons = []
    for region in tree.getroot().iter("Region"):
        vertices = region.find("Vertices")
        if vertices is None:
            continue
        pts = [(float(v.get("X")), float(v.get("Y"))) for v in vertices.iter("Vertex")]
        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))
    return polygons


def rasterize(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 255)
    return mask


def process_split(split):
    images_dir = f"{DATA_DIR}/{split}/images"
    ann_dir = f"{DATA_DIR}/{split}/annotations"
    out_dir = f"{DATA_DIR}/prepared/{split}"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(fname)[0]
        xml_path = f"{ann_dir}/{stem}.xml"
        if not os.path.exists(xml_path):
            print(f"  skip {fname}: no matching annotation")
            continue

        image = np.array(Image.open(f"{images_dir}/{fname}").convert("RGB"))
        polygons = parse_polygons(xml_path)
        mask = rasterize(image.shape, polygons)

        Image.fromarray(image).save(f"{out_dir}/images/{stem}.png")
        Image.fromarray(mask).save(f"{out_dir}/masks/{stem}.png")
        count += 1
        print(f"  {fname}: {len(polygons)} nuclei -> mask coverage {mask.mean() / 255:.3f}")

    print(f"[{split}] prepared {count} image/mask pairs -> {out_dir}")


def main():
    for split in ("training", "testing"):
        process_split(split)


if __name__ == "__main__":
    main()
