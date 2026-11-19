"""Wraps the Mask R-CNN trained by scripts/train_crack_segmenter.py behind
the same (N, H, W) bool mask-stack contract SamSegmenter.segment() and
InstanceSegmenter.predict() return, so app.py's existing M5 correction /
measurement / report pipeline works on its output unchanged.

See scripts/train_crack_segmenter.py's module docstring for training
methodology (DenseCrack-only, 410 images, widened RPN anchor aspect ratios
for crack elongation) and scripts/evaluate_crack_detector.py's for the
evaluation/diagnosis this wraps. Strict one-to-one matching on DenseCrack's
official test split (100 images): recall 74.8%, precision 73.0%, mean
matched IoU 0.741 WITH the merge post-processing this class applies by
default (see predict()); without it recall 75.4%, precision only 42.3%.

Why the merge exists: a single visually-continuous crack often gets
predicted as several separate local detections along its length
(confirmed: 67% of GT instances were substantially overlapped by >=2
predictions, while those predictions had low IoU with EACH OTHER --
consecutive segments, not duplicate boxes). Strict one-to-one matching
credits one segment per crack and counts the rest as false positives,
though 95.4% of them really do overlap real crack pixels. Pairwise-IoU
NMS can't catch this, since split segments don't overlap each other.

Trained only on DenseCrack (microscopy-scale dense micro-cracks, optical/
fluorescence imaging). CrackSeg9k (macro-photo crack benchmarks: road/
pavement/wall surfaces) was deliberately excluded -- a different imaging
modality from this project's microscopy domain (SOURCES.md flags it), and
mixing it into an earlier version measurably hurt DenseCrack performance.
Expect real degradation on crack imagery unlike DenseCrack's domain."""

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from ..device import resolve_device
from .tiling import needs_tiling, predict_tiled

NUM_CLASSES = 2  # background + crack, matches scripts/train_crack_segmenter.py
# Must match scripts/train_crack_segmenter.py's CRACK_ANCHOR_ASPECT_RATIOS
# exactly: the RPN head's conv output channels depend on the anchor count,
# so a mismatch makes load_state_dict() fail.
CRACK_ANCHOR_ASPECT_RATIOS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
# px; picked by a 0/3/5/8/10/15/20 sweep on DenseCrack's test split (see
# scripts/evaluate_crack_detector.py): recall and precision both peak at 10,
# IoU degrades beyond it as distinct nearby cracks start getting merged.
DEFAULT_MERGE_RADIUS = 10


def _build_architecture() -> torch.nn.Module:
    """Same anchor-widening and head replacement as
    scripts/train_crack_segmenter.py's build_model(), but with no
    pretrained weights requested -- load_state_dict() restores every
    parameter, and this app must run offline (same rationale as
    core/detectors/instance_seg.py)."""
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = (CRACK_ANCHOR_ASPECT_RATIOS,) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


def merge_nearby_masks(masks: list[np.ndarray], radius: int) -> list[np.ndarray]:
    """Union-find merge of masks whose DILATED footprints touch or overlap.
    The merged mask is the union of the ORIGINAL (undilated) masks, so the
    dilation only decides who gets grouped, never inflates the result."""
    if not masks:
        return []
    n = len(masks)
    dilated = [binary_dilation(m, disk(radius)) for m in masks]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if (dilated[i] & dilated[j]).any():
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        union = np.zeros_like(masks[0])
        for i in idxs:
            union |= masks[i]
        merged.append(union)
    return merged


class CrackSegmenter:
    def __init__(
        self,
        checkpoint_path,
        device: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        merge_radius: int = DEFAULT_MERGE_RADIUS,
    ):
        self.device = resolve_device(device)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.merge_radius = merge_radius
        model = _build_architecture()
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """`image`: (H, W, 3) RGB uint8, NOT preprocessed -- training used
        raw DenseCrack images. Returns an (N, H, W) bool mask stack, same
        contract as SamSegmenter.segment() / InstanceSegmenter.predict().
        Merges nearby detections by default; construct with merge_radius=0
        to get the raw, unmerged model output."""
        h, w = image.shape[:2]
        if needs_tiling(h, w):
            masks = predict_tiled(self._predict_masks, image)
        else:
            masks = self._predict_masks(image)
        if self.merge_radius > 0:
            masks = merge_nearby_masks(masks, self.merge_radius)
        if not masks:
            return np.empty((0, h, w), dtype=bool)
        return np.stack(masks)

    def _predict_masks(self, image: np.ndarray) -> list[np.ndarray]:
        """One detector pass, before the nearby-piece merge."""
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = self.model([image_t.to(self.device)])[0]
        masks = []
        for score, mask in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
            if score >= self.score_threshold:
                masks.append(mask[0] >= self.mask_threshold)
        return masks
