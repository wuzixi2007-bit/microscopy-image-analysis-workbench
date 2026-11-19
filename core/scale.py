"""Image scale (um per pixel) and everything unit-related that follows from
it. Pure logic, no Streamlit -- app.py's calibration widgets build a Scale
from user input, and measurement/stats/report only ever see this object.

Why this exists as its own module: before it, compute_particle_properties()
defaulted to pixel_size_um=1.0 and still named its columns area_um2 /
perimeter_um / equivalent_diameter_um, and report.py hardcoded "(µm)" -- so
an uncalibrated run labelled raw PIXEL counts as micrometres. Anyone reading
the CSV or report would take pixel values for physical sizes. Fixing only
the labels would just move the lie around, so units now travel with the
data: the unit suffix is in the column NAME (self-describing even when a CSV
is opened elsewhere), and the scale's provenance string is exported
alongside the numbers.

Uncalibrated is a first-class state, not an error: no scale means pixel
units everywhere (px / px²), never a silent fallback to 1.0 µm/px.
"""

import math
import numbers
from dataclasses import dataclass

# Everything is stored internally as micrometres per pixel. The unit a user
# types for a known-length bar is converted once, here.
_UNIT_TO_UM = {"nm": 1e-3, "µm": 1.0, "mm": 1e3}
LENGTH_UNITS = tuple(_UNIT_TO_UM)


@dataclass(frozen=True)
class Scale:
    um_per_px: float | None  # None = uncalibrated: report in pixels
    source: str  # human-readable provenance, exported with the results

    @property
    def calibrated(self) -> bool:
        return self.um_per_px is not None

    @property
    def length_unit(self) -> str:
        return "µm" if self.calibrated else "px"

    @property
    def area_unit(self) -> str:
        return "µm²" if self.calibrated else "px²"


UNCALIBRATED = Scale(None, "未标定（像素单位）")


def column_names(calibrated: bool) -> dict[str, str]:
    """Measurement column names, with the unit baked into the name. `length`
    and `mean_width` are the crack-specific metrics (core/measurement.py's
    compute_crack_properties)."""
    if calibrated:
        return {
            "area": "area_um2",
            "perimeter": "perimeter_um",
            "equivalent_diameter": "equivalent_diameter_um",
            "length": "length_um",
            "mean_width": "mean_width_um",
        }
    return {
        "area": "area_px2",
        "perimeter": "perimeter_px",
        "equivalent_diameter": "equivalent_diameter_px",
        "length": "length_px",
        "mean_width": "mean_width_px",
    }


def _require_positive_finite(value: float, what: str) -> float:
    # numbers.Real (not int/float) so numpy scalars from pixel measurements
    # are accepted -- the first version rejected np.int64 bar lengths read
    # straight off an image. bool is a Real too, but True as a "length" is
    # always a caller bug, so it's rejected explicitly.
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what}必须是大于0的有限数值，收到：{value!r}")
    return float(value)


def scale_from_um_per_px(um_per_px: float, origin: str = "手动输入") -> Scale:
    value = _require_positive_finite(um_per_px, "每像素对应的实际长度（µm/px）")
    return Scale(value, f"{origin} {value:.6g} µm/px")


def scale_from_known_length(pixel_length: float, real_length: float, unit: str, origin: str = "已知长度") -> Scale:
    """`pixel_length` px on the image correspond to `real_length` `unit`
    (e.g. a 200 nm scale bar spanning 27 px)."""
    if unit not in _UNIT_TO_UM:
        raise ValueError(f"不支持的长度单位 {unit!r}，可选：{', '.join(LENGTH_UNITS)}")
    px = _require_positive_finite(pixel_length, "标尺的像素长度")
    real = _require_positive_finite(real_length, "标尺的实际长度")
    um_per_px = real * _UNIT_TO_UM[unit] / px
    return Scale(um_per_px, f"{origin}：{px:.1f} px = {real:g} {unit}（{um_per_px:.6g} µm/px）")


def scale_from_field_width(field_width: float, unit: str, image_width_px: int, origin: str = "视野宽度（整幅图宽）") -> Scale:
    """The image's full horizontal field of view is `field_width` `unit`
    (SEM info bands usually print it, e.g. BAM_TiO2's "Width = 7.520 µm").
    The most reliable calibration when available: the pixel side is the
    image's own exact width, so there is no reading error -- unlike a
    scale bar, where +-1 px on a ~27 px bar is already a 3.7% error (see
    the BAM_TiO2/1908312.tif cross-check in this module's tests: bar
    outer-edge-to-outer-edge vs. tick-centre-to-tick-centre differ by 2 px,
    which is 7%). Only valid for the full original frame: a cropped or
    resized image no longer spans the stated field width."""
    return scale_from_known_length(image_width_px, field_width, unit, origin=origin)
