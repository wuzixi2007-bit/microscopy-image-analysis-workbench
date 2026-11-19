"""Formal strict one-to-one-matching evaluation (IoU>=0.5, Hungarian
assignment via scipy.optimize.linear_sum_assignment), replacing the various
ad-hoc benchmark scripts used to produce the numbers in README.md's "评估口径
改成严格一对一匹配" section -- same methodology, now a reusable script.

Two things this evaluates:
  --model instance_seg : the Mask R-CNN trained by
      scripts/train_instance_segmenter.py (EMPS-only training, motivated by
      the recall ceiling of the SAM+classifier "穷举检测" path).
  --model few_shot : the existing FewShotDetector.detect_all() + classifier
      baseline already shipped in app.py, using the same per-image
      self-reference methodology (3 references at small/median/large
      percentile) documented in README.md's "v2" section and
      scripts/build_particle_classifier_dataset.py's pick_self_reference_mask.

Two datasets:
  --dataset emps     : EMPS's 99 official test.txt images -- same domain the
      instance segmenter trained on.
  --dataset bam_tio2 : BAM_TiO2's 8 official test.txt SEM images -- out of
      domain for the instance segmenter (never trained on BAM_TiO2, see
      train_instance_segmenter.py's module docstring), and the SAME held-out
      set the few-shot baseline's numbers in README.md were measured on, so
      --model few_shot --dataset bam_tio2 reproduces that baseline exactly
      and --model instance_seg --dataset bam_tio2 is the apples-to-apples
      comparison on identical images.

Usage:
    python scripts/evaluate_detector.py --model instance_seg --dataset emps
    python scripts/evaluate_detector.py --model instance_seg --dataset bam_tio2
    python scripts/evaluate_detector.py --model few_shot --dataset bam_tio2
"""

import argparse

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops

import config
from core.detectors.few_shot import FewShotDetector
from core.detectors.particle_classifier import ParticleClassifier
from core.device import resolve_device
from core.preprocessing import preprocess
from core.reliability import FEATURE_GRID_SIZE, MIN_GRID_CELLS
from core.segmentation.sam_wrapper import SamSegmenter
from scripts.train_instance_segmenter import build_model

# Matches core/measurement.py's default noise floor and
# scripts/build_particle_classifier_dataset.py's MIN_AREA_PX -- keeps this
# evaluation's GT-instance extraction consistent with the rest of the project
# and with the prior ad-hoc benchmarks it's meant to formalize.
MIN_AREA_PX = 25


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union > 0 else 0.0


def match_strict(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float = 0.5):
    """Hungarian one-to-one matching on the full pred x gt IoU matrix; a pair
    only counts as a hit if its IoU also clears iou_threshold (the optimal
    assignment can still pick a pair below threshold when nothing better is
    available, so this check can't be dropped)."""
    if not pred_masks or not gt_masks:
        return 0, 0.0
    iou_mat = np.array([[iou(p, g) for g in gt_masks] for p in pred_masks])
    row_idx, col_idx = linear_sum_assignment(-iou_mat)
    matched_ious = [iou_mat[r, c] for r, c in zip(row_idx, col_idx) if iou_mat[r, c] >= iou_threshold]
    return len(matched_ious), (float(np.mean(matched_ious)) if matched_ious else 0.0)


def emps_gt_instances(stem: str) -> list[np.ndarray]:
    mask = np.array(Image.open(f"data/datasets/EMPS/masks/{stem}.png"))
    ids = np.unique(mask)
    ids = ids[ids != 0]
    return [mask == i for i in ids if (mask == i).sum() >= MIN_AREA_PX]


def bam_tio2_gt_instances(filename: str) -> list[np.ndarray]:
    """BAM_TiO2 masks are binary (foreground/background), not per-instance --
    connected components are the best available GT here. This has the same
    touching-particles-merge-into-one-instance blind spot noted in the
    training plan for why BAM_TiO2 isn't used for TRAINING; it applies
    equally to this GT, so recall on this dataset is a slight overestimate
    wherever particles touch (a merged pair only needs one correct detection
    to count as fully matched)."""
    gt = np.array(Image.open(f"data/datasets/BAM_TiO2/masks/{filename}"))
    if gt.ndim == 3:
        gt = gt[..., 0]
    lbl = label(gt > 0)
    return [lbl == r.label for r in regionprops(lbl) if r.area >= MIN_AREA_PX]


def pick_reference_masks(gt_instances: list[np.ndarray], image_shape, n: int = 3) -> list[np.ndarray]:
    """Small/median/large percentile particles from this image's own GT, as
    self-references -- the same methodology behind README.md's "3 参考图"
    few-shot benchmarks (see build_particle_classifier_dataset.py's
    pick_self_reference_mask), so the few_shot baseline's numbers here are
    directly comparable to the ones already documented.

    Filters out candidates below core.reliability's own grid-cell floor
    first: FewShotDetector.set_reference() downsamples a reference mask to
    SAM's 64x64 feature grid with nearest-neighbor interpolation, which can
    zero out a real small particle entirely on BAM_TiO2's higher-resolution
    images (a real particle isn't necessarily an unreliable *reference*,
    exactly the failure check_reference_selection() was built to catch --
    reused here instead of duplicating the threshold)."""
    h, w = image_shape[:2]
    cell_area = (min(h, w) / FEATURE_GRID_SIZE) ** 2
    usable = [m for m in gt_instances if m.sum() / cell_area >= MIN_GRID_CELLS]
    if not usable:
        usable = gt_instances  # smallest fallback: let set_reference() raise if truly nothing usable
    areas = [m.sum() for m in usable]
    order = np.argsort(areas)
    n = min(n, len(usable))
    pick_idx = sorted({order[int(p * (len(order) - 1))] for p in np.linspace(0, 1, n)})
    return [usable[i] for i in pick_idx]


def load_instance_segmenter(device: str, detections_per_img: int | None = None):
    model = build_model().to(device)
    state = torch.load(config.MODELS_DIR / "instance_segmenter.pth", map_location=device)
    model.load_state_dict(state)
    if detections_per_img is not None:
        # torchvision's maskrcnn_resnet50_fpn_v2 defaults roi_heads.detections_per_img
        # to 100 -- a final top-N-by-score cutoff applied AFTER scoring/NMS, not a
        # trained weight, so it's safe to change post-hoc without retraining. Real
        # BAM_TiO2 eval images with >200 GT particles were seen hitting this cutoff
        # exactly (pred count == 100 on the nose), capping recall on those images
        # regardless of how many more correct detections the model actually had
        # ranked just below the cutoff.
        model.roi_heads.detections_per_img = detections_per_img
    model.eval()
    return model


def instance_segmenter_predict(model, image: np.ndarray, device: str, score_threshold=0.5, mask_threshold=0.5):
    image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        pred = model([image_t.to(device)])[0]
    masks = []
    for score, mask in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
        if score >= score_threshold:
            masks.append(mask[0] >= mask_threshold)
    return masks


def run_eval(stems, gt_fn, predict_fn, label_str: str):
    total_gt = total_pred = total_matched = 0
    matched_ious = []
    for stem in stems:
        gt_instances = gt_fn(stem)
        pred_masks = predict_fn(stem, gt_instances)
        n_matched, mean_iou = match_strict(pred_masks, gt_instances)
        total_gt += len(gt_instances)
        total_pred += len(pred_masks)
        total_matched += n_matched
        if mean_iou > 0:
            matched_ious.append(mean_iou)
        print(f"  {stem}: gt={len(gt_instances)} pred={len(pred_masks)} matched={n_matched}")

    recall = total_matched / total_gt if total_gt else 0.0
    precision = total_matched / total_pred if total_pred else 0.0
    print(
        f"\n[{label_str}] n_images={len(stems)} total_gt={total_gt} total_pred={total_pred} "
        f"matched={total_matched}\nrecall={recall:.1%}  precision={precision:.1%}  "
        f"mean_matched_iou={np.mean(matched_ious) if matched_ious else 0:.3f}"
    )
    return recall, precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["instance_seg", "few_shot"], required=True)
    parser.add_argument("--dataset", choices=["emps", "bam_tio2"], required=True)
    parser.add_argument("--n-images", type=int, default=None, help="evaluate only the first N images (debugging)")
    parser.add_argument(
        "--detections-per-img",
        type=int,
        default=None,
        help="override roi_heads.detections_per_img (torchvision default 100) -- instance_seg only",
    )
    args = parser.parse_args()

    if args.dataset == "emps":
        with open("data/datasets/EMPS/test.txt") as f:
            stems = [line.strip().rsplit(".", 1)[0] for line in f if line.strip()]
        gt_fn = emps_gt_instances

        def load_raw(stem):
            return np.array(Image.open(f"data/datasets/EMPS/images/{stem}.png").convert("RGB"))
    else:
        with open("data/datasets/BAM_TiO2/test.txt") as f:
            stems = [line.strip() for line in f if line.strip()]  # already has .tif extension
        gt_fn = bam_tio2_gt_instances

        def load_raw(stem):
            return np.array(Image.open(f"data/datasets/BAM_TiO2/prepared/images/{stem}").convert("RGB"))

    if args.n_images:
        stems = stems[: args.n_images]

    device = resolve_device("cuda")
    print(f"device: {device}")
    print(f"evaluating --model {args.model} --dataset {args.dataset} ({len(stems)} images)\n")

    if args.model == "instance_seg":
        model = load_instance_segmenter(device, detections_per_img=args.detections_per_img)

        def predict_fn(stem, gt_instances):
            return instance_segmenter_predict(model, load_raw(stem), device)

        run_eval(stems, gt_fn, predict_fn, f"instance_seg / {args.dataset}")
    else:
        sam = SamSegmenter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, config.DEVICE)
        classifier = ParticleClassifier(str(config.MODELS_DIR / "particle_classifier.npz"))

        def predict_fn(stem, gt_instances):
            image = preprocess(load_raw(stem), denoise_on=True, equalize_on=True, contrast_on=False)
            detector = FewShotDetector(sam)
            for ref_mask in pick_reference_masks(gt_instances, image.shape):
                detector.set_reference(image, ref_mask)
            return [m for _, m in detector.detect_all(image, classifier)]

        run_eval(stems, gt_fn, predict_fn, f"few_shot / {args.dataset}")


if __name__ == "__main__":
    main()
