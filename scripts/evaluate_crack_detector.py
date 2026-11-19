"""Strict one-to-one-matching evaluation (IoU>=0.5, Hungarian assignment) for
the crack instance segmenter trained by scripts/train_crack_segmenter.py --
same methodology as scripts/evaluate_detector.py (reuses its iou/match_strict/
run_eval directly), applied to CrackSeg9k's and DenseCrack's official test
splits instead of EMPS/BAM_TiO2.

GT instances come from connected components on the (properly thresholded --
see train_crack_segmenter.py's module docstring for why CrackSeg9k needs
threshold=127 while DenseCrack needs 0) binary mask, matching exactly how
CrackInstanceDataset derived training targets, so the evaluation measures
the model against the same definition of "instance" it was trained on.

**Merge-radius, settled at 10px (2026-09-27) -- read before changing
DEFAULT_MERGE_RADIUS**: the crack_segmenter.pth checkpoint (DenseCrack-only
training) has a real, diagnosed failure mode -- a single visually-continuous
crack often gets predicted as several separate local detections along its
length (confirmed directly: 67% of GT instances were substantially
overlapped by >=2 predictions, while those predictions had low IoU with
EACH OTHER, only 5.5% of prediction pairs overlapping -- consecutive
segments of one crack, not duplicate boxes stacked on one spot). Strict
one-to-one matching then only credits ONE segment per crack, so the rest
count as false positives, tanking precision (baseline 42.3%) without the
model actually hallucinating (only 4.6% of "false positives" had near-zero
pixel overlap with any real crack).

Fixing only one side made it WORSE, not better -- both are real findings,
not guesses:
  - GT-closing alone (bridging small gaps in the mask before connected-
    component labeling, to stop over-fragmenting GT itself): recall rose
    (fewer, bigger GT instances are individually easier to hit) but
    precision FELL (42.3%->30.8% at radius 5) -- predictions stayed
    fragmented, so merged GT instances became harder for any single
    fragment-sized prediction to reach 0.5 IoU against.
  - Merging predictions alone (union-find over dilated masks, radius 3):
    precision rose sharply (42.3%->64.6%) but recall COLLAPSED
    (75.4%->45.9%) -- a merged big prediction can now only be Hungarian-
    matched to ONE of GT's several small fragments, losing the other
    matches that were legitimate before merging.
  - Only merging BOTH sides at the SAME radius fixed it cleanly -- a sweep
    (0/3/5/8/10/15/20px) found radius=10 as the sweet spot: recall 74.8%
    (vs 75.4% baseline, essentially free), precision 73.0% (vs 42.3%
    baseline, +30.7pp), mean matched IoU 0.741 (vs 0.749, barely moved --
    confirms this is fixing instance-COUNTING, not segmentation quality).
    Beyond radius 10, IoU degrades (0.734 at 15, 0.721 at 20) as
    genuinely-separate nearby cracks start getting wrongly merged too.

DEFAULT_MERGE_RADIUS applies to BOTH --gt-closing-radius (evaluation-only,
no effect at real inference time since there's no GT then) and
--merge-radius (a real inference-time post-processing step, still only
implemented here in the eval script -- not yet ported into a production
crack-inference wrapper the way InstanceSegmenter wraps the particle
model). Pass 0 to either flag to reproduce the original unmerged numbers.

Usage:
    python scripts/evaluate_crack_detector.py --dataset crackseg9k
    python scripts/evaluate_crack_detector.py --dataset densecrack
    python scripts/evaluate_crack_detector.py --dataset crackseg9k --n-images 300
    python scripts/evaluate_crack_detector.py --dataset densecrack --gt-closing-radius 0 --merge-radius 0  # original unmerged numbers
"""

import argparse

import numpy as np
import torch
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk

import config
from core.device import resolve_device
from scripts.evaluate_detector import run_eval
from scripts.train_crack_segmenter import (
    CRACKSEG9K_DIR,
    CRACKSEG9K_MASK_THRESHOLD,
    DENSECRACK_DIR,
    MIN_INSTANCE_AREA_PX,
    build_model,
    crackseg9k_subset_tag,
)

DEFAULT_MERGE_RADIUS = 10  # px; see module docstring's sweep results


def crack_gt_instances(filename: str, threshold: int, image_shape: tuple[int, int], closing_radius: int = 0) -> list[np.ndarray]:
    """`filename` includes its dataset's directory prefix already baked
    into the path passed in by the caller -- kept as a plain path here
    rather than reconstructing dataset-specific paths, since the two
    datasets' mask filenames aren't guaranteed unique across each other.

    41/510 DenseCrack images (~8%, found while building this script) have a
    mask whose pixel size doesn't match its own image's (mostly a 512x512
    image against a 256x256 mask, one 512x512 against a 2000x2000 mask) --
    a real data quality issue in the source dataset, not something
    introduced here. This silently affected training too (CrackInstanceDataset
    never checked for it), likely degrading supervision quality on that
    ~3% slice of the training pool rather than crashing outright -- flagged
    honestly, not fixed retroactively (would need a retrain to fully
    correct). Here, evaluation needs GT and predictions in the same
    coordinate space to compute IoU at all, so mismatched masks are resized
    (nearest-neighbor, to keep them binary) to the image's own shape."""
    mask = np.array(Image.open(filename))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != image_shape:
        mask = np.array(Image.fromarray(mask).resize((image_shape[1], image_shape[0]), Image.NEAREST))
    binary = mask > threshold
    if closing_radius > 0:
        # A visually-continuous crack in a dense network can have a few-
        # pixel gap (faint/thin section falling below `threshold`) that
        # connected-component labeling then treats as two separate GT
        # "instances" -- real diagnosis from a low-precision investigation:
        # 95.4% of "false positive" predictions actually overlapped real
        # crack pixels (median best-IoU-to-any-single-GT-instance 0.222),
        # meaning the model was finding whole cracks that GT had fragmented,
        # not hallucinating. Closing (dilate then erode) bridges small gaps
        # before labeling, without changing the mask's overall footprint.
        binary = binary_closing(binary, disk(closing_radius))
    lbl = label(binary)
    return [lbl == r.label for r in regionprops(lbl) if r.area >= MIN_INSTANCE_AREA_PX]


def load_crack_segmenter(device: str, detections_per_img: int | None = None):
    model = build_model().to(device)
    state = torch.load(config.MODELS_DIR / "crack_segmenter.pth", map_location=device)
    model.load_state_dict(state)
    if detections_per_img is not None:
        model.roi_heads.detections_per_img = detections_per_img
    model.eval()
    return model


def crack_segmenter_predict(model, image: np.ndarray, device: str, score_threshold=0.5, mask_threshold=0.5):
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = model([image_t.to(device)])[0]
    masks = []
    for score, mask in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
        if score >= score_threshold:
            masks.append(mask[0] >= mask_threshold)
    return masks


def merge_nearby_predictions(pred_masks: list[np.ndarray], dilation_radius: int) -> list[np.ndarray]:
    """Union-find merge of predictions whose DILATED masks touch/overlap --
    targets the diagnosed real failure mode (not a guess): 67% of DenseCrack
    GT instances get substantially overlapped by >=2 separate predictions,
    but those predictions have low IoU with EACH OTHER (only 5.5% of pairs
    overlap directly) -- a single long crack gets split into several
    consecutive, non-overlapping local detections along its length, which
    ordinary pairwise-IoU NMS can't catch since split segments don't overlap
    each other directly. Dilating first lets nearby-but-not-touching
    segments register as connected before unioning their ORIGINAL
    (undilated) masks -- the merged mask's footprint is exactly the union of
    what was actually detected, not inflated by the dilation itself."""
    if not pred_masks:
        return []
    from scipy.ndimage import binary_dilation
    from skimage.morphology import disk

    n = len(pred_masks)
    dilated = [binary_dilation(m, disk(dilation_radius)) for m in pred_masks]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if (dilated[i] & dilated[j]).any():
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        u = np.zeros_like(pred_masks[0])
        for i in idxs:
            u |= pred_masks[i]
        merged.append(u)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["crackseg9k", "densecrack"], required=True)
    parser.add_argument("--n-images", type=int, default=None, help="evaluate a stratified subsample of this size instead of the full official test split (CrackSeg9k test.txt has 1827 images)")
    parser.add_argument("--detections-per-img", type=int, default=None)
    parser.add_argument("--gt-closing-radius", type=int, default=DEFAULT_MERGE_RADIUS, help=f"morphological closing radius (px) applied to GT before connected-component labeling, to bridge small gaps that over-fragment a visually-continuous crack into multiple GT instances (default {DEFAULT_MERGE_RADIUS}, validated by sweep -- see module docstring; pass 0 for the original unmerged behavior)")
    parser.add_argument("--merge-radius", type=int, default=DEFAULT_MERGE_RADIUS, help=f"dilation radius (px) for merging nearby/co-linear predictions that likely belong to the same crack split across several local detections (default {DEFAULT_MERGE_RADIUS}, validated by sweep -- see module docstring; pass 0 for the original unmerged behavior)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "crackseg9k":
        with open(f"{CRACKSEG9K_DIR}/test.txt") as f:
            stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
        mask_dir, image_dir, threshold = f"{CRACKSEG9K_DIR}/masks", f"{CRACKSEG9K_DIR}/images", CRACKSEG9K_MASK_THRESHOLD
        if args.n_images and args.n_images < len(stems):
            rng = np.random.default_rng(args.seed)
            by_subset: dict[str, list[str]] = {}
            for s in stems:
                by_subset.setdefault(crackseg9k_subset_tag(s), []).append(s)
            quota = max(1, args.n_images // len(by_subset))
            stems = []
            for subset_stems in by_subset.values():
                k = min(quota, len(subset_stems))
                idx = rng.choice(len(subset_stems), size=k, replace=False)
                stems.extend(subset_stems[i] for i in idx)
    else:
        with open(f"{DENSECRACK_DIR}/test.txt") as f:
            stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
        mask_dir, image_dir, threshold = f"{DENSECRACK_DIR}/masks", f"{DENSECRACK_DIR}/images", 0
        if args.n_images and args.n_images < len(stems):
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(stems), size=args.n_images, replace=False)
            stems = [stems[i] for i in idx]

    device = resolve_device("cuda")
    print(f"device: {device}")
    print(f"evaluating crack_segmenter on {args.dataset} ({len(stems)} images)\n")

    model = load_crack_segmenter(device, detections_per_img=args.detections_per_img)

    def gt_fn(stem):
        with Image.open(f"{image_dir}/{stem}.png") as img:
            image_shape = (img.height, img.width)
        return crack_gt_instances(f"{mask_dir}/{stem}.png", threshold, image_shape, args.gt_closing_radius)

    def predict_fn(stem, gt_instances):
        image = np.array(Image.open(f"{image_dir}/{stem}.png").convert("RGB"))
        masks = crack_segmenter_predict(model, image, device)
        if args.merge_radius > 0:
            masks = merge_nearby_predictions(masks, args.merge_radius)
        return masks

    run_eval(stems, gt_fn, predict_fn, f"crack_segmenter / {args.dataset}")

    print(
        "\n(GT here comes from connected components on a binary mask, not true per-instance "
        "annotation -- see train_crack_segmenter.py's module docstring on why that's a defensible "
        "approximation for cracks specifically, unlike the particle case.)"
    )


if __name__ == "__main__":
    main()
