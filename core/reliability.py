"""Post-hoc sanity check for few-shot detection results, independent of the
raw cosine-similarity score.

Motivated by a real failure case (NIST lunar regolith particle SEM images):
a wrong match onto background noise scored a higher similarity (0.999-1.000)
than any genuine match measured elsewhere (0.85-0.96) -- the raw score alone
cannot be trusted to flag bad detections.

Calibrated against 5 known cases (3 genuine matches, 2 real failures across
particles/cracks): predicted mask area vs. the reference mask's area cleanly
separates them -- every success stayed at or below 1.16x the reference area,
every failure exceeded 3x. Predicted area being *smaller* than the reference
is normal (real targets vary in size) and is not flagged; only a large
*excess* is, since that pattern matches "the mask leaked into background."
"""

from dataclasses import dataclass, field

import numpy as np

MAX_AREA_RATIO = 3.0  # predicted area / reference area, upper bound before flagging


@dataclass
class ReliabilityCheck:
    area_ratio: float
    is_reliable: bool
    reasons: list = field(default_factory=list)


def check_few_shot_result(pred_mask: np.ndarray, reference_mask_area: int) -> ReliabilityCheck:
    area = int(pred_mask.sum())
    if area == 0:
        return ReliabilityCheck(area_ratio=0.0, is_reliable=False, reasons=["empty mask"])

    area_ratio = area / max(reference_mask_area, 1)
    reasons = []
    if area_ratio > MAX_AREA_RATIO:
        reasons.append(
            f"predicted area is {area_ratio:.1f}x the reference area (flag threshold {MAX_AREA_RATIO}x) "
            "-- likely matched background/noise rather than the target object"
        )
    return ReliabilityCheck(area_ratio=area_ratio, is_reliable=len(reasons) == 0, reasons=reasons)


# --- Pre-detection reference-selection guidance (BAM_TiO2 real-world lesson) ---
#
# SOURCES.md documents two ways a manually clicked few-shot reference has led
# to bad downstream matches on BAM_TiO2 (dense TiO2 nanoparticle SEM): a small
# particle from a sparse image matched background noise (predicted area 82%
# of the image); the largest connected region in a dense image turned out to
# be several particles fused into one cluster (IoU only 0.076).
#
# Two per-image auto-estimates were tried first and both rejected after
# measuring them against real BAM_TiO2 ground truth, not just reasoning
# about them:
#   1. Solidity as a "is this one particle or a fused cluster" shape
#      signal -- near-median-area GT regions (probably single particles)
#      had solidity mean/median 0.895/0.938; >=3x-median regions (probably
#      clusters) had 0.872/0.905. Heavily overlapping, not a usable
#      separator (same conclusion as the earlier shape-cascade attempt for
#      the exhaustive-detection classifier, see README.md).
#   2. A cheap per-image particle census (Otsu threshold + connected
#      components, no SAM) to compare a click against "this image's own
#      median particle size" or flag the whole image as sparse. Checked
#      against GT on all 40 usable BAM_TiO2 images: Otsu merges touching
#      particles unpredictably (undercounts by 1.4x-6x depending on how
#      cluttered the image is), misclassifying 17/40 (42%) of genuinely
#      dense images (GT >=40 particles) as "sparse", and flagging 5/8 (62%)
#      of real near-median GT particles as "oversized cluster" purely
#      because Otsu's own size estimate for that image was off. Both are
#      unacceptably high false-positive/miss rates for a warning that's
#      supposed to build trust, not cry wolf -- dropped entirely.
#
# What's left below relies on nothing estimated from a single arbitrary
# image: (a) a fixed, already-validated fact about SAM's own feature
# resolution (not something newly measured for this feature), and (b)
# comparing a new pick against reference sizes the user THEMSELVES already
# confirmed earlier in the same session, rather than an auto-estimate of
# "what's typical" -- ground truth we already have for free.
FEATURE_GRID_SIZE = 64  # SAM ViT-B image embedding spatial resolution, matches few_shot.py's FEATURE_SIZE
MIN_GRID_CELLS = 2  # below this many grid cells' worth of area, features risk mixing with background
PRIOR_REFERENCE_SIZE_RATIO = 2.0  # new pick vs. mean of already-confirmed picks, outside [1/ratio, ratio] -> warn


@dataclass
class ReferenceSelectionCheck:
    ref_area: int
    grid_cells: float
    warnings: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.warnings


def check_reference_selection(
    ref_mask: np.ndarray, image_shape: tuple[int, int], prior_reference_mean_area: float | None = None
) -> ReferenceSelectionCheck:
    """Warn (never block -- this is a heuristic hint, not a gate) right
    after a reference click, before the user commits to it with "确认此参
    考图".

    `image_shape` is (H, W) of the reference image -- used only for the
    grid-cell-size arithmetic below, not to re-derive some estimate of the
    image's own particle census (see the module-level comment on why that
    was tried and abandoned).

    `prior_reference_mean_area`, when given (from `FewShotDetector.
    reference_mask_area` once at least one reference is already confirmed
    in this session), flags a new pick that's wildly different in size from
    what the user themselves already confirmed -- few-shot matching
    mean-pools all references into one target embedding, so mixing e.g. a
    single small particle with a multi-particle cluster dilutes that
    embedding into something that resembles neither well."""
    h, w = image_shape[:2]
    cell_area = (min(h, w) / FEATURE_GRID_SIZE) ** 2
    ref_area = int(ref_mask.sum())
    warnings = []

    grid_cells = ref_area / cell_area if cell_area > 0 else float("inf")
    if grid_cells < MIN_GRID_CELLS:
        warnings.append(
            f"这个区域面积约 {ref_area}px，在 SAM 的 {FEATURE_GRID_SIZE}×{FEATURE_GRID_SIZE} 特征网格上只占约 "
            f"{grid_cells:.1f} 个格子——目标太小时，特征容易跟周围背景/纹理混叠（穷举检测漏检的根因分析也提到"
            "了同一个问题，见 README.md），参考特征的质量会打折扣，建议尽量选择更大、更典型的颗粒。"
        )

    if prior_reference_mean_area is not None and prior_reference_mean_area > 0:
        ratio = ref_area / prior_reference_mean_area
        if ratio > PRIOR_REFERENCE_SIZE_RATIO or ratio < 1 / PRIOR_REFERENCE_SIZE_RATIO:
            warnings.append(
                f"这次选的区域面积（约 {ref_area}px）跟你之前确认的参考图（平均约 "
                f"{prior_reference_mean_area:.0f}px）差异较大（{ratio:.1f}倍）——少样本匹配会把多张参考图"
                "的特征取平均当成同一类目标，大小差异太大容易稀释掉平均后的特征，确认这确实是同一类目标"
                "（而不是不小心选到了粘连在一起的颗粒团簇，或者选到了明显更小/更大的颗粒）。"
            )

    return ReferenceSelectionCheck(ref_area=ref_area, grid_cells=grid_cells, warnings=warnings)
