"""Decoding an uploaded file into the (H, W, 3) uint8 RGB array every other
module expects.

`Image.open(...).convert("RGB")` was used directly and silently mangles the
formats microscopes actually write: a 16-bit TIFF (mode I;16) converts with
CLIPPING, so everything above 255 saturates -- a ramp up to 60000 came out
100% white -- and a float TIFF (0..1) comes out nearly black. Both look like
"the model found nothing", with no hint the picture itself was wrong. Such
data is therefore stretched to 8 bits by percentile instead (robust to a few
hot pixels, and 16-bit detector data rarely fills its range)."""

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_PIXELS = 100_000_000  # beyond this the mask stacks and previews are not workable anyway
_STRETCH_PERCENTILES = (0.5, 99.5)
_HIGH_DEPTH_MODES = {"I", "I;16", "I;16L", "I;16B", "I;16N", "F"}


class ImageLoadError(ValueError):
    """The file can't be turned into an image; the message is user-facing."""


def _stretch_to_uint8(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float64)
    lo, hi = np.percentile(data, _STRETCH_PERCENTILES)
    if hi <= lo:  # percentiles collapse on a mostly-flat image; fall back to the full range
        lo, hi = float(data.min()), float(data.max())
    if hi <= lo:
        return np.zeros(data.shape, np.uint8)
    return np.clip((data - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_rgb(data: bytes) -> np.ndarray:
    """Decode the first frame of an image file to (H, W, 3) uint8 RGB.
    Multi-page TIFFs use their first page only. Raises ImageLoadError with a
    readable message for anything that isn't a usable image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"无法读取为图像文件（{type(exc).__name__}）：文件可能已损坏，或不是 PNG/JPG/TIFF 图像") from exc

    if image.width * image.height > MAX_PIXELS:
        raise ImageLoadError(f"图像 {image.width}×{image.height} 超过 {MAX_PIXELS // 1_000_000} 百万像素上限，请先裁剪或缩小")

    if image.mode in _HIGH_DEPTH_MODES:
        gray = _stretch_to_uint8(np.array(image))
        return np.stack([gray] * 3, axis=-1)
    return np.array(image.convert("RGB"))
