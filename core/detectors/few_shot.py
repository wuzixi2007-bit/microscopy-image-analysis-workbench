"""Mode B: PerSAM-style in-context matching. 1-3 reference (image, mask) pairs
-> mean-pooled target embedding -> cosine-similarity heatmap on new images ->
positive/negative point prompts for SAM. No gradient training."""

import cv2
import numpy as np

from ..reliability import check_few_shot_result
from .base import DetectionPrompt, Detector
from .particle_classifier import ParticleClassifier

FEATURE_SIZE = 64  # SAM ViT-B image embedding spatial resolution
MIN_EXHAUSTIVE_MASK_AREA = 20  # px; smaller decoded masks are near-certainly noise, not particles


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union > 0 else 0.0


class FewShotDetector(Detector):
    def __init__(self, sam_segmenter):
        # Reuses the same SAM image encoder as the segmentation backend,
        # so this detector depends on an already-loaded SamSegmenter
        # rather than owning its own model.
        self.sam_segmenter = sam_segmenter
        self._target_embedding: np.ndarray | None = None  # (256,) unit vector
        self._n_references = 0
        self._reference_areas: list[int] = []

    @property
    def n_references(self) -> int:
        return self._n_references

    @property
    def reference_mask_area(self) -> float:
        """Mean pixel area of the registered reference masks. Feeds
        core.reliability.check_few_shot_result, which flags a detection when
        its predicted area balloons far past this baseline."""
        if not self._reference_areas:
            raise RuntimeError("call set_reference() before reading reference_mask_area")
        return float(np.mean(self._reference_areas))

    def set_reference(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Register one (image, binary mask) pair; call up to 3 times. Later
        calls are folded into a running mean of the target embedding."""
        embedding = self.sam_segmenter.encode_image(image)[0]  # (256, 64, 64)
        mask_small = cv2.resize(
            mask.astype(np.uint8), (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if not mask_small.any():
            raise ValueError("reference mask has no positive pixels after downsampling to feature resolution")

        target_vec = embedding[:, mask_small].mean(axis=1)  # (256,)
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        self._reference_areas.append(int(mask.sum()))

        if self._target_embedding is None:
            self._target_embedding = target_vec
        else:
            merged = self._target_embedding * self._n_references + target_vec
            self._target_embedding = merged / (self._n_references + 1)
            self._target_embedding /= np.linalg.norm(self._target_embedding) + 1e-8
        self._n_references += 1

    def score_masks(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Cosine similarity between each candidate mask's mean-pooled SAM
        feature and the registered reference embedding -- same computation
        set_reference() does for a reference mask, applied to externally
        generated candidates instead (e.g. InstanceSegmenter's output).
        Lets a reference filter candidates from any mask source, not just
        this detector's own detect()/detect_all()."""
        if self._target_embedding is None:
            raise RuntimeError("call set_reference() before score_masks()")
        embedding = self.sam_segmenter.encode_image(image)[0]  # (256, 64, 64)
        scores = []
        for mask in masks:
            mask_small = cv2.resize(
                mask.astype(np.uint8), (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            if not mask_small.any():
                scores.append(0.0)
                continue
            vec = embedding[:, mask_small].mean(axis=1)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            scores.append(float(np.dot(vec, self._target_embedding)))
        return np.array(scores)

    def detect(self, image: np.ndarray, num_positive_points: int = 5, **kwargs) -> DetectionPrompt:
        """`num_positive_points` > 1 spreads several positive prompts along
        the high-similarity region via greedy NMS, instead of a single global
        argmax. A single point pools well for blob-shaped targets (particles,
        pores, grains) but under-specifies elongated ones (cracks, fibers),
        where one point tells SAM nothing about the target's extent."""
        if self._target_embedding is None:
            raise RuntimeError("call set_reference() before detect()")

        h, w = image.shape[:2]
        embedding = self.sam_segmenter.encode_image(image)[0]  # (256, 64, 64)
        embedding_normed = embedding / (np.linalg.norm(embedding, axis=0, keepdims=True) + 1e-8)

        sim = np.tensordot(self._target_embedding, embedding_normed, axes=(0, 0))  # (64, 64)
        sim_full = cv2.resize(sim, (w, h), interpolation=cv2.INTER_LINEAR)

        neg_y, neg_x = np.unravel_index(np.argmin(sim_full), sim_full.shape)
        min_distance = max(10, min(h, w) // 30)
        pos_points = self._select_positive_points(sim_full, num_positive_points, min_distance)

        points = np.vstack([pos_points, [[neg_x, neg_y]]])
        point_labels = np.array([1] * len(pos_points) + [0], dtype=int)
        scores = np.array([float(sim_full[int(y), int(x)]) for x, y in pos_points])
        return DetectionPrompt(points=points, point_labels=point_labels, scores=scores)

    def detect_all(
        self,
        image: np.ndarray,
        classifier: ParticleClassifier,
        clf_threshold: float = 0.5,
        max_peaks: int = 150,
        dedupe_iou: float = 0.3,
    ) -> list[tuple[float, np.ndarray]]:
        """Exhaustive variant of detect(): instead of collapsing the
        similarity heatmap into one merged target, treat every separated
        local maximum above `classifier`'s probability as its own candidate
        instance, decode each independently (cheap -- reuses the single
        image encoding below), and keep the ones that pass the existing
        reliability check. Returns a list of (score, mask), highest first,
        with near-duplicate masks (IoU > dedupe_iou, from overlapping peaks
        on the same instance) suppressed.

        Experimental (see README.md "训练轻量分类头"): on a 13-image
        held-out benchmark this recovers ~33% of real instances at ~71%
        precision -- much better than a fixed similarity threshold alone
        (~10% recall) but still far from exhaustive. detect() remains the
        default single-target path."""
        if self._target_embedding is None:
            raise RuntimeError("call set_reference() before detect_all()")

        h, w = image.shape[:2]
        embedding = self.sam_segmenter.encode_image(image)[0]  # (256, 64, 64); also caches for decode_point()
        embedding_normed = embedding / (np.linalg.norm(embedding, axis=0, keepdims=True) + 1e-8)
        sim = np.tensordot(self._target_embedding, embedding_normed, axes=(0, 0))  # (64, 64)

        feat_grid = embedding.reshape(256, -1).T  # (4096, 256)
        feat_grid = np.concatenate([feat_grid, sim.reshape(-1, 1)], axis=1)  # (4096, 257)
        clf_prob = classifier.predict_proba(feat_grid).reshape(FEATURE_SIZE, FEATURE_SIZE)

        # Tried two "more principled" replacements for this greedy loop --
        # both regressed 13-image recall badly and were reverted (see
        # README.md "峰值选择策略" for the numbers): (1) Gaussian-smoothing
        # clf_prob before peak-finding diluted isolated small particles'
        # peaks below threshold (most particles are only 1-2 grid cells,
        # surrounded by mostly-background neighbors); (2) swapping to
        # skimage.feature.peak_local_max's strict local-maxima test found
        # far fewer peaks than this greedy version even without smoothing --
        # clf_prob's plateaus (many near-tied cells, from a coarse 64x64
        # grid) don't fit peak_local_max's "strictly greater than neighbors"
        # assumption, while greedy argmax+suppression tolerates them fine.
        min_distance_px = max(10, min(h, w) // 30)  # same heuristic detect() uses for its negative-point spacing
        grid_min_distance = max(1, min_distance_px * FEATURE_SIZE / max(h, w))

        work = np.where(clf_prob >= clf_threshold, clf_prob, -np.inf)
        yy, xx = np.mgrid[0:FEATURE_SIZE, 0:FEATURE_SIZE]
        peaks = []
        for _ in range(max_peaks):
            gy, gx = np.unravel_index(np.argmax(work), work.shape)
            score = work[gy, gx]
            if not np.isfinite(score):
                break
            peaks.append((float((gx + 0.5) * w / FEATURE_SIZE), float((gy + 0.5) * h / FEATURE_SIZE), float(score)))
            dist_sq = (yy - gy) ** 2 + (xx - gx) ** 2
            work[dist_sq <= grid_min_distance**2] = -np.inf

        candidates = []
        for x, y, score in peaks:
            mask = self.sam_segmenter.decode_point(x, y)
            if mask.sum() < MIN_EXHAUSTIVE_MASK_AREA:
                continue
            if not check_few_shot_result(mask, self.reference_mask_area).is_reliable:
                continue
            candidates.append((score, mask))

        candidates.sort(key=lambda c: -c[0])
        kept: list[tuple[float, np.ndarray]] = []
        for score, mask in candidates:
            if any(_iou(mask, kept_mask) > dedupe_iou for _, kept_mask in kept):
                continue
            kept.append((score, mask))
        return kept

    @staticmethod
    def _select_positive_points(sim_full: np.ndarray, k: int, min_distance: int) -> np.ndarray:
        """Greedy non-max suppression: repeatedly take the highest remaining
        point, then suppress a disk around it, so points spread out along an
        elongated target instead of clustering at one peak."""
        sim = sim_full.copy()
        h, w = sim.shape
        yy, xx = np.mgrid[0:h, 0:w]
        points = []
        for _ in range(k):
            y, x = np.unravel_index(np.argmax(sim), sim.shape)
            if not np.isfinite(sim[y, x]):
                break
            points.append([float(x), float(y)])
            dist_sq = (yy - y) ** 2 + (xx - x) ** 2
            sim[dist_sq <= min_distance**2] = -np.inf
        return np.array(points)
