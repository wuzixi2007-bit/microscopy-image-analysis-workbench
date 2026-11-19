"""Per-particle morphology from a binary mask. Independent of the detection
models, so implemented now rather than left as an M3 stub."""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .scale import column_names


def _significant_regions(mask: np.ndarray, min_area_px: int, min_area_ratio: float, merge_disjoint: bool) -> list:
    """Regions of `mask` that survive the area floor. The reasoning for the
    two floors is in compute_particle_properties's docstring."""
    if merge_disjoint:
        labeled = mask.astype(np.uint8)
    else:
        labeled = label(mask.astype(bool))
    regions = regionprops(labeled)
    if not regions:
        return []
    area_floor = max(min_area_px, min_area_ratio * max(r.area for r in regions))
    return [r for r in regions if r.area >= area_floor]


def _centroid_px(region) -> tuple[float, float]:
    """(row, col) rounded to 0.1 px: the raw float repr ran to ~15 digits
    in the CSV and the report's detail table, far below pixel precision."""
    row, col = region.centroid
    return (round(float(row), 1), round(float(col), 1))


def compute_particle_properties(
    mask: np.ndarray,
    pixel_size_um: float | None = None,
    min_area_px: int = 25,
    min_area_ratio: float = 0.05,
    merge_disjoint: bool = False,
) -> list[dict]:
    """One dict per connected component in `mask`. `pixel_size_um` converts
    pixel measurements to micrometres when the image's scale is known; None
    (the default) means uncalibrated -- values stay in pixels and the
    column names say so (area_px2 / perimeter_px / equivalent_diameter_px,
    see core/scale.py for why the unit lives in the name). The old default
    of 1.0 produced pixel values under *_um names, which read as physical
    sizes. `centroid_px` is (row, col) in pixels in every case.

    `merge_disjoint=True` treats every foreground pixel as ONE region instead
    of auto-splitting by connectivity -- for a mask row that M5's "合并实例"
    op deliberately unioned from separate (possibly non-touching) instances,
    e.g. a crack broken into visually separate strokes but conceptually one
    object. Normal detector output is always a single connected blob per
    row already, so this is a no-op there; only use it for rows you know
    were explicitly merged, not detector output in general.

    Components smaller than `max(min_area_px, min_area_ratio * largest_component_area)`
    are dropped before measuring:

    - `min_area_px` is an absolute floor: for regions this small, skimage's
      perimeter estimator is unreliable (it can even report circularity > 1,
      which is impossible by the isoperimetric inequality). 25px is
      empirically where a compact blob's circularity estimate settles below
      1 (a filled circle of radius 3); below that (tested down to radius 1)
      the estimate is unusable regardless of dataset.
    - `min_area_ratio` is a *relative* floor, scaled to whatever this mask's
      actual particles look like: a fixed absolute threshold calibrated on
      one dataset (e.g. a few hundred px particles) is meaningless on
      another where real particles span tens of thousands of px -- there,
      the same 25px is background/edge noise, not a small real particle.
      Dropping anything under 5% of the largest component in the same mask
      adapts to the image's own scale instead."""
    regions = _significant_regions(mask, min_area_px, min_area_ratio, merge_disjoint)
    if not regions:
        return []

    names = column_names(pixel_size_um is not None)
    length_factor = pixel_size_um if pixel_size_um is not None else 1.0

    properties = []
    for region in regions:
        area_px = region.area
        perimeter_px = region.perimeter
        equiv_diameter_px = region.equivalent_diameter_area
        # circularity and aspect ratio are dimensionless -- unaffected by scale
        circularity = (4 * np.pi * area_px / perimeter_px**2) if perimeter_px > 0 else 0.0
        major = region.axis_major_length
        minor = region.axis_minor_length
        aspect_ratio = (major / minor) if minor > 0 else float("nan")

        properties.append(
            {
                "label": region.label,
                "centroid_px": _centroid_px(region),
                names["area"]: area_px * length_factor**2,
                names["perimeter"]: perimeter_px * length_factor,
                names["equivalent_diameter"]: equiv_diameter_px * length_factor,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
            }
        )
    return properties


# Proffitt & Rosen's chain-code weights. The naive weights (1 per orthogonal
# step, sqrt(2) per diagonal) overestimate a digitized line by ~5% on average
# and up to ~10% at oblique angles; these cut the mean bias to ~0 with a
# worst case of ~6% (checked on synthetic lines at 26 angles x 4 widths).
_ORTHOGONAL_STEP = 0.948
_DIAGONAL_STEP = 1.340


def skeleton_length_px(skeleton: np.ndarray) -> float:
    """Length of a 1px-wide 8-connected skeleton, in pixels, by weighted
    chain-code step counting. A diagonal step is counted only where the 2x2
    block has no orthogonal route between its two pixels, so an L-shaped
    corner isn't charged for both its sides and its diagonal."""
    s = skeleton.astype(bool)
    orthogonal = (s[:, :-1] & s[:, 1:]).sum() + (s[:-1] & s[1:]).sum()
    a, b, c, d = s[:-1, :-1], s[:-1, 1:], s[1:, :-1], s[1:, 1:]
    diagonal = (a & d & ~b & ~c).sum() + (b & c & ~a & ~d).sum()
    return float(orthogonal * _ORTHOGONAL_STEP + diagonal * _DIAGONAL_STEP)


def compute_crack_properties(
    mask: np.ndarray,
    pixel_size_um: float | None = None,
    min_area_px: int = 25,
    min_area_ratio: float = 0.05,
    merge_disjoint: bool = False,
) -> list[dict]:
    """Crack counterpart of compute_particle_properties: one dict per region,
    with length and mean width instead of equivalent diameter / circularity,
    which describe compact blobs and say nothing useful about a thin,
    elongated crack. Same units/scale/merge_disjoint/area-floor semantics.

    - `length`: skeleton (medial axis) length, so it follows the crack's
      curvature and branches rather than its bounding box. For a region
      merged from several non-touching pieces (merge_disjoint) it is the sum
      over the pieces; the gaps between them are NOT counted.
    - `mean_width` = area / length: the average thickness of the crack. The
      skeleton stops about half a width short of each tip, so very short,
      stubby regions read slightly wide.
    - Skeletonizing turns every ragged bit of the mask edge into a short
      spur, so strongly irregular masks measure somewhat long; a real
      branch is counted in full, as a branch is part of the crack.
    - `aspect_ratio` (length / mean_width) is dimensionless and unaffected
      by scale."""
    regions = _significant_regions(mask, min_area_px, min_area_ratio, merge_disjoint)
    names = column_names(pixel_size_um is not None)
    length_factor = pixel_size_um if pixel_size_um is not None else 1.0

    properties = []
    for region in regions:
        skeleton = skeletonize(region.image)
        # a compact blob skeletonizes to a single pixel (length 0); floor at
        # 1px rather than dropping the row, so every detected instance keeps
        # a row and the count stays consistent with the segmentation.
        length_px = max(skeleton_length_px(skeleton), 1.0)
        area_px = region.area
        width_px = area_px / length_px
        properties.append(
            {
                "label": region.label,
                "centroid_px": _centroid_px(region),
                names["area"]: area_px * length_factor**2,
                names["length"]: length_px * length_factor,
                names["mean_width"]: width_px * length_factor,
                "aspect_ratio": length_px / width_px,
            }
        )
    return properties
