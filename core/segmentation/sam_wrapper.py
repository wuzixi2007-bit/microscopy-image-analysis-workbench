"""Single SAM instance shared by both ZeroShotDetector and FewShotDetector
prompts, and by FewShotDetector's own image encoder use in M1."""

import numpy as np
from scipy.ndimage import binary_fill_holes
from segment_anything import SamPredictor, sam_model_registry

from ..detectors.base import DetectionPrompt
from ..device import resolve_device


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Every target this app segments (particles, grains, cracks) is a
    solid blob, never a ring -- a hole in a mask is always a SAM artifact,
    not a real feature to preserve. Confirmed as a real, reproducible
    failure (not a one-off) on a BAM_TiO2 image: a particle with a bright
    SEM-imaging rim around its core made SAM's own highest-predicted-IoU
    candidate a ring excluding the core, across a whole neighborhood of
    nearby prompt points, not just one unlucky click (see README.md). A
    solid particle's mask should never have an interior hole; filling one
    when it's NOT an artifact (there is none in normal detections) is a
    no-op, so this is safe to apply unconditionally."""
    return binary_fill_holes(mask)


class SamSegmenter:
    def __init__(self, checkpoint_path, model_type: str, device: str):
        self.device = resolve_device(device)
        self._sam = self._load(checkpoint_path, model_type)
        self._predictor = SamPredictor(self._sam)
        self._last_image_id: int | None = None

    def _load(self, checkpoint_path, model_type: str):
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        return sam.to(self.device)

    def set_image(self, image: np.ndarray) -> None:
        """Skips re-running the encoder (~9s/image on CPU) if `image` is the
        same array object already encoded -- callers like FewShotDetector.detect()
        followed by segment() on the SAME `processed` array were paying for
        two full encodes of one image every single-target detection before
        this (detect_all()'s exhaustive path already avoided this by design,
        see decode_point()). Identity (id()), not value equality: comparing
        large arrays by content on every call would itself cost more than
        the encode it's trying to save."""
        if id(image) == self._last_image_id:
            return
        self._predictor.set_image(image)
        self._last_image_id = id(image)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Expose the image encoder output for FewShotDetector's feature
        matching (M1). (1, 256, 64, 64) for ViT-B."""
        self.set_image(image)
        return self._predictor.get_image_embedding().cpu().numpy()

    def decode_point(self, x: float, y: float) -> np.ndarray:
        """Single-point mask decode reusing the embedding cached by the most
        recent set_image()/encode_image() call on the SAME image -- SAM's
        encoder is the expensive step (~9s/image on CPU), the decoder is
        cheap (~0.1s) and designed for exactly this repeated-prompt reuse.
        Used by FewShotDetector.detect_all() to decode many candidate points
        without re-encoding for each one.

        multimask_output=True asks SAM for its 3 standard granularities
        (whole object / part / sub-part) instead of forcing one -- a single
        point is inherently ambiguous about extent (see README.md's
        over-segmentation section), and picking SAM's own highest-predicted-
        IoU mask among the 3 is cheaper and more principled than always
        taking whichever one multimask_output=False happens to pick."""
        masks, iou_scores, _ = self._predictor.predict(
            point_coords=np.array([[x, y]]), point_labels=np.array([1]), multimask_output=True
        )
        best = int(np.argmax(iou_scores))
        return _fill_holes(masks[best].astype(bool))

    def decode_box(self, box: tuple[float, float, float, float]) -> np.ndarray:
        """Box-prompted analog of decode_point(): reuses whatever embedding
        the most recent set_image()/encode_image() call cached, so the
        caller is responsible for pointing that at the right image first.

        Backs M5's "增加" (add) correction: instead of using the user's
        rough freehand stroke pixels directly as the added mask (imprecise,
        follows the mouse rather than the actual particle edge), treat the
        stroke's bounding box as a SAM box prompt and let SAM snap to the
        real boundary inside it. A tight box is also better than a lone
        point at respecting a touching neighbor just outside it -- exactly
        the "one detection accidentally spans two fused particles" failure
        mode this is meant to help fix (see README.md).

        multimask_output=True + picking SAM's own highest-predicted-IoU
        mask mirrors decode_point() for the same reason: a box still leaves
        some ambiguity about extent, and this was already measured to help
        (72.4%->88.2% precision) in decode_point()'s case."""
        masks, iou_scores, _ = self._predictor.predict(box=np.array(box), multimask_output=True)
        best = int(np.argmax(iou_scores))
        return _fill_holes(masks[best].astype(bool))

    def segment(self, image: np.ndarray, prompt: DetectionPrompt) -> np.ndarray:
        """Return a (N, H, W) bool mask stack. Each row of `prompt.boxes`
        yields its own mask (zero-shot mode, one box per target); if there
        are no boxes, all of `prompt.points`/`point_labels` are used together
        as a single prompt yielding one mask (few-shot mode's positive/
        negative point pair)."""
        self.set_image(image)

        masks = []
        if len(prompt.boxes) > 0:
            for box in prompt.boxes:
                mask, _, _ = self._predictor.predict(box=np.array(box), multimask_output=False)
                masks.append(_fill_holes(mask[0]))
        elif len(prompt.points) > 0:
            mask, _, _ = self._predictor.predict(
                point_coords=prompt.points, point_labels=prompt.point_labels, multimask_output=False
            )
            masks.append(_fill_holes(mask[0]))

        if not masks:
            return np.empty((0, image.shape[0], image.shape[1]), dtype=bool)
        return np.stack(masks).astype(bool)
