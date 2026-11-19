"""Tiled inference for the Mask R-CNN wrappers (InstanceSegmenter,
CrackSegmenter).

Why: torchvision's detection transform rescales every input so its short
side is 800 px (long side capped at 1333), whatever the real size. A big
SEM image therefore shrinks, and particles that were comfortably detectable
become a few pixels wide and vanish -- silently. Measured with the particle
model on the real BAM_TiO2 image (51 instances at 1024x768) tiled up to
bigger canvases: 2048x1536 gave 49 instances where ~204 were expected,
and 4096x3072 gave ZERO. Nothing in the output hints at the loss.

So images that would be shrunk noticeably are cut into overlapping tiles
sized so each tile is processed at its native scale (the regime the models
were evaluated in), and the per-tile instances are stitched back together.

Stitching keeps an instance only in the tile whose "core" contains its
centroid. The cores partition the image (neighbouring cores meet in the
middle of the tiles' overlap), so an object seen by two tiles is reported
once. An object that reaches more than about overlap/2 past its core edge
can be truncated; the 160 px overlap covers anything up to ~160 px across,
larger than the particles these models are evaluated on. A long crack that crosses tiles
comes back as several pieces touching at the seams -- the crack wrapper's
nearby-piece merge joins them afterwards.

Memory: the (N, H, W) bool contract is N full-size masks, so a dense image
can ask for gigabytes. predict_tiled() refuses beyond a budget with a clear
message instead of letting the machine swap.
"""

import numpy as np

# torchvision GeneralizedRCNNTransform defaults the wrappers rely on
MODEL_MIN_SIZE = 800
MODEL_MAX_SIZE = 1333
# Tile when the model would shrink the image below this fraction of its
# real size. 0.75 leaves ordinary 1024x1024 SEM frames (scale 0.78) alone.
MIN_NATIVE_SCALE = 0.75

TILE_SIZE = MODEL_MIN_SIZE
TILE_OVERLAP = 160
MASK_BUDGET_BYTES = 3_000_000_000  # (N, H, W) bool stack


def model_scale(height: int, width: int) -> float:
    """The factor torchvision resizes an (H, W) input by before detection."""
    return min(MODEL_MIN_SIZE / min(height, width), MODEL_MAX_SIZE / max(height, width))


def needs_tiling(height: int, width: int) -> bool:
    return model_scale(height, width) < MIN_NATIVE_SCALE


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)  # last tile reaches the edge instead of being a sliver
    return starts


def _cores(starts: list[int], length: int, tile: int) -> list[tuple[int, int]]:
    """Per-tile core interval along one axis. Neighbouring cores meet at the
    midpoint of the two tiles' overlap, so the cores partition [0, length)
    exactly -- an object's centroid belongs to one core only. (The last tile
    is pushed back to reach the edge, so its overlap with the previous tile
    can exceed the nominal one; cutting at fixed offsets from each tile edge
    would then leave the cores overlapping and report objects twice.)"""
    cuts = [(starts[i] + tile + starts[i + 1]) // 2 for i in range(len(starts) - 1)]
    bounds = [0] + cuts + [length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(starts))]


def predict_tiled(predict_fn, image: np.ndarray, tile: int = TILE_SIZE, overlap: int = TILE_OVERLAP) -> list[np.ndarray]:
    """Run `predict_fn(tile_image) -> iterable of (th, tw) bool masks` over
    overlapping tiles and return full-size (H, W) bool masks, one per
    instance. See the module docstring for the stitching rule."""
    h, w = image.shape[:2]
    ys, xs = _starts(h, tile, overlap), _starts(w, tile, overlap)
    core_ys, core_xs = _cores(ys, h, tile), _cores(xs, w, tile)
    pieces: list[tuple[np.ndarray, int, int]] = []
    for y0, core_y in zip(ys, core_ys):
        for x0, core_x in zip(xs, core_xs):
            for mask in predict_fn(image[y0 : y0 + tile, x0 : x0 + tile]):
                my, mx = np.nonzero(mask)
                if my.size == 0:
                    continue
                cy, cx = my.mean() + y0, mx.mean() + x0
                if core_y[0] <= cy < core_y[1] and core_x[0] <= cx < core_x[1]:
                    pieces.append((np.asarray(mask, dtype=bool), y0, x0))

    needed = len(pieces) * h * w
    if needed > MASK_BUDGET_BYTES:
        raise ValueError(
            f"图像 {w}×{h} 上检测到 {len(pieces)} 个实例，逐实例保存掩膜需要约 {needed / 1e9:.1f} GB 内存，"
            f"超过 {MASK_BUDGET_BYTES / 1e9:.0f} GB 上限。请裁剪出感兴趣的区域后再分析。"
        )
    full = []
    for mask, y0, x0 in pieces:
        out = np.zeros((h, w), dtype=bool)
        out[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
        full.append(out)
    return full
