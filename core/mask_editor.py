"""Pure helpers for M5 (network manual mask correction). Streamlit widget
orchestration lives in app.py; this module only converts what the canvas
returns into boolean masks at the original image resolution, so it can be
tested without a running Streamlit session."""

import cv2
import numpy as np
from skimage.measure import label, regionprops

MIN_INSTANCE_AREA_PX = 25  # matches core/measurement.py's noise floor


def canvas_layer_to_mask(image_data, target_shape: tuple[int, int]) -> np.ndarray:
    """`image_data` is the (canvas_h, canvas_w, 4) RGBA array from one
    st_canvas call; any drawn stroke has alpha > 0. Resized with nearest-
    neighbor to `target_shape` (original image's (H, W)) -- a drawn stroke
    is already a hard yes/no region, not something to blend."""
    if image_data is None:
        return np.zeros(target_shape, dtype=bool)
    alpha = np.asarray(image_data)[..., 3]
    stroke = (alpha > 0).astype(np.uint8)
    resized = cv2.resize(stroke, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def stroke_bounding_box(stroke_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) bounding box of a boolean stroke's True pixels, for
    feeding to SamSegmenter.decode_box() -- None for an empty stroke (user
    clicked 应用 without drawing anything, or drew but canvas_layer_to_mask's
    resize rounded it away)."""
    ys, xs = np.where(stroke_mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def masks_to_combined(masks: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Union an (N, H, W) mask stack into one (H, W) boolean mask -- the
    editor works on one combined region rather than per-instance, since
    manual correction is about "is this pixel a target or not", not which
    instance it belongs to (instances get re-split via connected components
    once editing is done, in combined_to_instance_masks)."""
    if masks.shape[0] == 0:
        return np.zeros(image_shape, dtype=bool)
    return np.any(masks, axis=0)


def combined_to_instance_masks(combined: np.ndarray) -> np.ndarray:
    """Split an edited combined mask back into one boolean mask per
    connected component, matching the (N, H, W) stack shape the rest of the
    app (overlay_masks, run_measurement_ui) already expects.

    Drops components under MIN_INSTANCE_AREA_PX before materializing them --
    a noisy detection (e.g. zero-shot's well-documented speckled failure
    mode) can label into hundreds of 1-2px fragments, and building a
    separate full-resolution boolean array per fragment is O(instances x
    image size): one real case measured 734 fragments taking 9+ seconds to
    split and overlay, which is what made the editor look frozen (see
    README.md). Real particles are never this small at this stage --
    reliability/area filters upstream already screen for that -- so this
    floor only ever removes noise, matching measurement.py's own floor."""
    labeled = label(combined)
    regions = [r for r in regionprops(labeled) if r.area >= MIN_INSTANCE_AREA_PX]
    if not regions:
        return np.empty((0, *combined.shape), dtype=bool)
    return np.stack([labeled == r.label for r in regions])


def apply_pinned_groups(
    combined: np.ndarray, pinned_groups: list[np.ndarray], kept_groups: list[np.ndarray] | tuple = ()
) -> tuple[np.ndarray, int]:
    """Turns the edited combined mask back into per-instance rows, in three
    stages. A pixel claimed by an earlier stage/group is never counted again.

    1. `pinned_groups`: unions the user (or the crack model's merge step)
       explicitly grouped -- unlike a plain connected component, the parts
       don't need to touch (e.g. a crack broken into separate strokes but
       one object; see README.md's over-segmentation notes). They skip the
       MIN_INSTANCE_AREA_PX noise floor because someone picked them
       deliberately. ORDER MATTERS: put a group before the groups it was
       built from, or its pixels are already claimed and the merge is
       silently a no-op (a bug the UI had: the original instances were
       processed before the user's merge of them).
    2. `kept_groups`: detector instances that must stay separate even when
       they touch (a dense particle image's instances usually do). Without
       this, the combined mask fuses touching neighbours and the connected-
       component split below hands back one blob -- 51 instances became 9 on
       a real BAM_TiO2 image. Each is clipped to the edited mask (so erasing
       or deleting still works) and to what earlier groups left, and dropped
       under the noise floor. Detector masks that overlap each other keep the
       overlap in whichever comes first.
    3. Whatever is left (strokes the user added) is split by connected
       components, as before.

    Returns (masks, n_pinned): n_pinned is how many LEADING rows are stage-1
    groups, which callers must measure whole (merge_disjoint=True) or
    compute_particle_properties's own connected-component call would
    silently re-split them. Stage-2/3 rows are ordinary single regions."""
    claimed = np.zeros_like(combined)
    pinned_masks = []
    for group in pinned_groups:
        g = group & combined & ~claimed
        if g.any():
            pinned_masks.append(g)
            claimed |= g

    kept_masks = []
    for group in kept_groups:
        g = group & combined & ~claimed
        if g.sum() >= MIN_INSTANCE_AREA_PX:
            kept_masks.append(g)
            claimed |= g

    remaining = combined & ~claimed
    normal_masks = combined_to_instance_masks(remaining)

    rows = pinned_masks + kept_masks + list(normal_masks)
    if not rows:
        return np.empty((0, *combined.shape), dtype=bool), 0
    return np.stack(rows), len(pinned_masks)
