"""Build a grid-cell-level (particle vs. background) training set from
per-instance masks, at the same 64x64 SAM feature-grid resolution
FewShotDetector's similarity heatmap operates on. Each sample is one feature
vector plus a binary label -- meant to train a lightweight classifier head
that replaces a fixed cosine-similarity threshold when scanning for multiple
candidate instances (see README.md "穷举检测" section for why the raw
threshold approach failed).

v2 additions (attacking recall, see README.md's "继续攻召回率" follow-up):
  - Feature vector is now 257-d: the raw 256-d SAM embedding at that grid
    cell, PLUS its cosine similarity to a self-reference embedding (one
    representative particle picked from the same image). At inference this
    second value is exactly FewShotDetector's own similarity heatmap value
    at that cell -- already computed, free to add. Previously similarity and
    classifier probability were combined as a hard AND-gate outside the
    model; letting the classifier see both jointly should recover recall
    the AND-gate was throwing away.
  - BAM_TiO2 TSEM images (odd-numbered) are now included in training (35 of
    40, minus a 5-image held-out TSEM eval set incl. 1908267.tif) -- v1 only
    trained on SEM, and TSEM was the worst-performing domain in the 9-image
    eval (see README.md).

Datasets combined for material/imaging-mode diversity:
  - BAM_TiO2: 32 train.txt SEM images + 35 TSEM images, binary masks -> skimage.label()
  - EMPS: random subset of 366 train.txt images, instance-ID masks (each
    particle already has its own pixel value, no labeling needed)

Usage:
    python scripts/build_particle_classifier_dataset.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import cv2

import config
from core.preprocessing import preprocess
from core.segmentation.sam_wrapper import SamSegmenter

FEATURE_SIZE = 64  # matches core/detectors/few_shot.py
MIN_AREA_PX = 25  # matches core/measurement.py's noise floor
NEG_PER_IMAGE_MULTIPLIER = 4  # negatives sampled per positive, per image
EMPS_SUBSET_SIZE = 60  # full 366 would take ~1hr on this CPU; subsample for diversity within time budget

# TSEM (odd-numbered) images reserved for held-out evaluation, never trained on
TSEM_HELDOUT = {"1908267", "1908277", "1908287", "1908297", "1908307"}

OUT_PATH = "data/datasets/particle_classifier_train.npz"


def grid_cells_for_mask(full_mask: np.ndarray) -> np.ndarray:
    """Downsample a full-res boolean mask to the 64x64 feature grid the same
    way FewShotDetector.set_reference() does, so training samples match
    inference-time candidate extraction exactly."""
    return cv2.resize(
        full_mask.astype(np.uint8), (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def bam_tio2_real_mask(gt: np.ndarray):
    """BAM_TiO2 masks are binary; connected components >= MIN_AREA_PX are
    real particles (see README.md's noise-filter section -- same floor as
    measurement.py)."""
    if gt.ndim == 3:
        gt = gt[..., 0]
    lbl = label(gt > 0)
    regions = [r for r in regionprops(lbl) if r.area >= MIN_AREA_PX]
    if not regions:
        return None, 0, None
    return np.isin(lbl, [r.label for r in regions]), len(regions), (lbl, regions)


def emps_real_mask(gt: np.ndarray):
    """EMPS masks are instance-ID (0=background, each particle its own ID)
    -- no labeling needed, just threshold each ID's area."""
    if gt.ndim == 3:
        gt = gt[..., 0]
    ids = np.unique(gt)
    ids = ids[ids != 0]
    keep_ids = [i for i in ids if (gt == i).sum() >= MIN_AREA_PX]
    if not keep_ids:
        return None, 0, None
    return np.isin(gt, keep_ids), len(keep_ids), None


def pick_self_reference_mask(real_mask_full, extra) -> np.ndarray:
    """One representative particle from this image, used to compute the
    similarity feature -- same "pick the median-area region" heuristic used
    throughout today's eval scripts."""
    if extra is not None:
        lbl, regions = extra
        med = sorted(r.area for r in regions)[len(regions) // 2]
        ref_region = min(regions, key=lambda r: abs(r.area - med))
        return lbl == ref_region.label
    # EMPS: no labeled regions handed back, re-derive from real_mask_full
    lbl = label(real_mask_full)
    regions = regionprops(lbl)
    med = sorted(r.area for r in regions)[len(regions) // 2]
    ref_region = min(regions, key=lambda r: abs(r.area - med))
    return lbl == ref_region.label


def collect_image_list():
    jobs = []
    with open("data/datasets/BAM_TiO2/train.txt") as f:
        for line in f:
            stem = line.strip().rsplit(".", 1)[0]
            if stem:
                jobs.append(("BAM_TiO2-SEM", stem,
                              f"data/datasets/BAM_TiO2/prepared/images/{stem}.tif",
                              f"data/datasets/BAM_TiO2/masks/{stem}.tif",
                              bam_tio2_real_mask))

    all_ids = sorted(int(p.stem) for p in Path("data/datasets/BAM_TiO2/images").glob("*.tif"))
    tsem_ids = [str(i) for i in all_ids if i % 2 == 1 and str(i) not in TSEM_HELDOUT]
    for stem in tsem_ids:
        jobs.append(("BAM_TiO2-TSEM", stem,
                      f"data/datasets/BAM_TiO2/prepared/images/{stem}.tif",
                      f"data/datasets/BAM_TiO2/masks/{stem}.tif",
                      bam_tio2_real_mask))

    with open("data/datasets/EMPS/train.txt") as f:
        emps_ids = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
    rng = np.random.default_rng(0)
    emps_subset = rng.choice(emps_ids, size=min(EMPS_SUBSET_SIZE, len(emps_ids)), replace=False)
    for stem in emps_subset:
        jobs.append(("EMPS", stem,
                      f"data/datasets/EMPS/images/{stem}.png",
                      f"data/datasets/EMPS/masks/{stem}.png",
                      emps_real_mask))
    return jobs


def main():
    jobs = collect_image_list()
    for tag in ["BAM_TiO2-SEM", "BAM_TiO2-TSEM", "EMPS"]:
        print(f"  {tag}: {sum(1 for j in jobs if j[0] == tag)} images")
    print(f"{len(jobs)} training images total; TSEM held out for eval: {sorted(TSEM_HELDOUT)}")

    sam = SamSegmenter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, "cpu")

    all_features = []
    all_labels = []
    all_sample_ids = []
    rng = np.random.default_rng(0)

    for i, (dataset, image_id, img_path, mask_path, mask_fn) in enumerate(jobs):
        image_raw = np.array(Image.open(img_path).convert("RGB"))
        image = preprocess(image_raw, denoise_on=True, equalize_on=True, contrast_on=False)
        gt = np.array(Image.open(mask_path))

        real_mask_full, n_particles, extra = mask_fn(gt)
        if real_mask_full is None:
            print(f"  [{i+1}/{len(jobs)}] {dataset}:{image_id}: no real particles, skipping")
            continue

        embedding = sam.encode_image(image)[0]  # (256, 64, 64)
        embedding_normed = embedding / (np.linalg.norm(embedding, axis=0, keepdims=True) + 1e-8)

        ref_mask_full = pick_self_reference_mask(real_mask_full, extra)
        ref_small = grid_cells_for_mask(ref_mask_full)
        ref_vec = embedding[:, ref_small].mean(axis=1)
        ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)
        sim_grid = np.tensordot(ref_vec, embedding_normed, axes=(0, 0))  # (64, 64)

        pos_small = grid_cells_for_mask(real_mask_full)
        pos_ys, pos_xs = np.where(pos_small)

        pos_dilated = cv2.dilate(pos_small.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        neg_candidates_ys, neg_candidates_xs = np.where(~pos_dilated)

        n_pos = len(pos_ys)
        n_neg = min(len(neg_candidates_ys), n_pos * NEG_PER_IMAGE_MULTIPLIER)
        neg_idx = rng.choice(len(neg_candidates_ys), size=n_neg, replace=False)

        sample_id = f"{dataset}:{image_id}"
        for y, x in zip(pos_ys, pos_xs):
            feat = np.concatenate([embedding[:, y, x], [sim_grid[y, x]]])
            all_features.append(feat)
            all_labels.append(1)
            all_sample_ids.append(sample_id)
        for idx in neg_idx:
            y, x = neg_candidates_ys[idx], neg_candidates_xs[idx]
            feat = np.concatenate([embedding[:, y, x], [sim_grid[y, x]]])
            all_features.append(feat)
            all_labels.append(0)
            all_sample_ids.append(sample_id)

        print(f"  [{i+1}/{len(jobs)}] {sample_id}: {n_particles} particles -> {n_pos} pos, {n_neg} neg cells")

    features = np.stack(all_features).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    sample_ids_arr = np.array(all_sample_ids)
    print(f"\ntotal samples: {len(labels)} ({labels.sum()} positive, {(labels==0).sum()} negative), "
          f"feature dim: {features.shape[1]}")
    np.savez(OUT_PATH, features=features, labels=labels, image_ids=sample_ids_arr)
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
