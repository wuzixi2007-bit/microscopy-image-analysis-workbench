"""Wraps the Mask R-CNN trained by scripts/train_instance_segmenter.py behind
the same (N, H, W) bool mask-stack contract SamSegmenter.segment() returns,
so app.py's existing M5 correction / measurement / report pipeline works on
its output unchanged. Unlike ZeroShotDetector/FewShotDetector, this doesn't
go through SAM at all -- the trained model outputs instance masks directly,
no point/box prompting step, no reference image, no text prompt.

See README.md "训练专门的实例分割模型" for the training data/methodology and
the evaluation numbers this wraps -- read them before trusting this on new
data: strict one-to-one-matching recall/precision was 85.8%/91.6% on EMPS
(same domain as training) but only 40.7%/78.4% on BAM_TiO2 (out of domain).
Trained only on EMPS; particle types/imaging modes far from that dataset
should be expected to regress further. There's no reliability check here
the way FewShotDetector has (core/reliability.py) -- Mask R-CNN's per-
instance score already reflects the model's own learned confidence, unlike
SAM's cosine-similarity score, which was shown to be untrustworthy on its
own (see README.md's "可靠性校验" section)."""

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..device import resolve_device
from .tiling import needs_tiling, predict_tiled

NUM_CLASSES = 2  # background + particle, matches scripts/train_instance_segmenter.py
# torchvision's default is 100 -- a final top-N-by-score cutoff applied AFTER
# scoring/NMS, not a trained weight. Real BAM_TiO2 eval images with >200
# particles hit it exactly (pred count == 100 on the nose for 5/8 held-out
# images), capping recall regardless of how many more correct detections
# ranked just below the cutoff. Raising it to 500 recovered +8.7pp recall on
# those same images with no measurable precision cost (see README.md); none
# of them needed more than 179, so 300 here is a safe middle-ground default.
DEFAULT_DETECTIONS_PER_IMG = 300


def _build_architecture() -> torch.nn.Module:
    """Same head-replacement as scripts/train_instance_segmenter.py's
    build_model(), but with NO pretrained weights requested anywhere --
    load_state_dict() below restores every parameter (including the
    backbone, saved as part of the checkpoint even though it was frozen
    during training) from the trained checkpoint, so there's nothing for a
    pretrained download to contribute here, and this app must run offline
    (see the top of app.py)."""
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)
    return model


class InstanceSegmenter:
    def __init__(
        self,
        checkpoint_path,
        device: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        detections_per_img: int = DEFAULT_DETECTIONS_PER_IMG,
    ):
        self.device = resolve_device(device)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        model = _build_architecture()
        state = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state)
        model.roi_heads.detections_per_img = detections_per_img
        model.to(self.device)
        model.eval()
        self.model = model

    def predict(self, image: np.ndarray) -> np.ndarray:
        """`image`: (H, W, 3) RGB uint8, NOT preprocessed -- training used
        raw EMPS images with no denoise/equalize/contrast step (see
        scripts/train_instance_segmenter.py's dataset loader), so feeding
        this the same preprocessing toggles the SAM-based modes use would
        shift the input away from what the model actually trained on.
        Returns an (N, H, W) bool mask stack, same contract as
        SamSegmenter.segment()."""
        h, w = image.shape[:2]
        if needs_tiling(h, w):
            masks = predict_tiled(self._predict_masks, image)
        else:
            masks = self._predict_masks(image)
        if not masks:
            return np.empty((0, h, w), dtype=bool)
        return np.stack(masks)

    def _predict_masks(self, image: np.ndarray) -> list[np.ndarray]:
        """One detector pass over `image` at the size torchvision gives it."""
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = self.model([image_t.to(self.device)])[0]
        masks = []
        for score, mask in zip(pred["scores"].cpu().numpy(), pred["masks"].cpu().numpy()):
            if score >= self.score_threshold:
                masks.append(mask[0] >= self.mask_threshold)
        return masks
