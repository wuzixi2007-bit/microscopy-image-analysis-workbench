"""M1 validation: register 1-3 (reference image, reference mask) pairs, then
run PerSAM-style matching on a new image, outside the Streamlit app.

Usage:
    python scripts/validate_few_shot.py \
        --ref-image path/to/ref1.png --ref-mask path/to/ref1_mask.png \
        --image path/to/target.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import config
from core.detectors.few_shot import FewShotDetector
from core.reliability import check_few_shot_result
from core.segmentation.sam_wrapper import SamSegmenter


def overlay(image: np.ndarray, masks: np.ndarray, points, point_labels) -> np.ndarray:
    import cv2

    out = image.copy()
    for mask in masks:
        color = np.array([255, 80, 80])
        out[mask] = (0.5 * out[mask] + 0.5 * color).astype(np.uint8)
    for (x, y), label in zip(points, point_labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.drawMarker(out, (int(x), int(y)), color, markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-image", action="append", required=True, dest="ref_images")
    parser.add_argument("--ref-mask", action="append", required=True, dest="ref_masks")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sam = SamSegmenter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, config.DEVICE)
    detector = FewShotDetector(sam)

    for ref_image_path, ref_mask_path in zip(args.ref_images, args.ref_masks):
        ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
        ref_mask = np.array(Image.open(ref_mask_path).convert("L")) > 127
        detector.set_reference(ref_image, ref_mask)

    image = np.array(Image.open(args.image).convert("RGB"))
    prompt = detector.detect(image)
    masks = sam.segment(image, prompt)

    print(f"{len(args.ref_images)} reference(s) -> {masks.shape[0]} mask(s), sim_score={prompt.scores}")
    if masks.shape[0] > 0:
        reliability = check_few_shot_result(masks[0], detector.reference_mask_area)
        status = "OK" if reliability.is_reliable else "UNRELIABLE"
        print(f"reliability: {status} (area_ratio={reliability.area_ratio:.2f}) {'; '.join(reliability.reasons)}")

    out_path = args.out or str(config.OUTPUTS_DIR / f"few_shot_{Path(args.image).stem}.png")
    result = overlay(image, masks, prompt.points, prompt.point_labels)
    Image.fromarray(result).save(out_path)
    print(f"saved annotated result to {out_path}")


if __name__ == "__main__":
    main()
