import math
import unittest

import numpy as np

from core.scale import (
    UNCALIBRATED,
    column_names,
    scale_from_field_width,
    scale_from_known_length,
    scale_from_um_per_px,
)


class ScaleConstructionTest(unittest.TestCase):
    def test_direct_value(self):
        s = scale_from_um_per_px(0.25)
        self.assertTrue(s.calibrated)
        self.assertEqual(s.um_per_px, 0.25)
        self.assertEqual((s.length_unit, s.area_unit), ("µm", "µm²"))

    def test_known_length_converts_units(self):
        # a 200 nm bar spanning 27 px
        self.assertAlmostEqual(scale_from_known_length(27, 200, "nm").um_per_px, 0.2 / 27)
        self.assertAlmostEqual(scale_from_known_length(100, 5, "µm").um_per_px, 0.05)
        self.assertAlmostEqual(scale_from_known_length(100, 1, "mm").um_per_px, 10.0)

    def test_field_width_matches_bam_tio2_banner(self):
        # BAM_TiO2/1908312.tif: "Width = 7.520 µm" over a 1024 px wide image
        self.assertAlmostEqual(scale_from_field_width(7.52, "µm", 1024).um_per_px, 0.00734375)

    def test_source_is_recorded(self):
        self.assertIn("27.0 px = 200 nm", scale_from_known_length(27, 200, "nm").source)

    def test_uncalibrated_is_pixel_units(self):
        self.assertFalse(UNCALIBRATED.calibrated)
        self.assertIsNone(UNCALIBRATED.um_per_px)
        self.assertEqual((UNCALIBRATED.length_unit, UNCALIBRATED.area_unit), ("px", "px²"))


class ScaleValidationTest(unittest.TestCase):
    def test_rejects_non_positive_and_non_finite(self):
        for bad in (0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    scale_from_um_per_px(bad)
                with self.assertRaises(ValueError):
                    scale_from_known_length(bad, 200, "nm")
                with self.assertRaises(ValueError):
                    scale_from_known_length(27, bad, "nm")

    def test_rejects_non_numbers(self):
        for bad in (None, "0.5", True, False, [1.0]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    scale_from_um_per_px(bad)

    def test_accepts_numpy_scalars(self):
        # regression: np.int64 bar lengths read off an image were rejected
        self.assertAlmostEqual(scale_from_known_length(np.int64(27), np.float64(200), "nm").um_per_px, 0.2 / 27)

    def test_rejects_unknown_unit(self):
        with self.assertRaises(ValueError):
            scale_from_known_length(27, 200, "inch")


class ColumnNamesTest(unittest.TestCase):
    def test_unit_is_in_the_name(self):
        px, um = column_names(False), column_names(True)
        self.assertEqual(px["area"], "area_px2")
        self.assertEqual(um["area"], "area_um2")
        self.assertEqual(px["equivalent_diameter"], "equivalent_diameter_px")
        self.assertEqual(um["perimeter"], "perimeter_um")

    def test_every_measure_has_both_variants(self):
        px, um = column_names(False), column_names(True)
        self.assertEqual(px.keys(), um.keys())
        self.assertTrue(all(name.endswith(("_px", "_px2")) for name in px.values()))
        self.assertTrue(all(name.endswith(("_um", "_um2")) for name in um.values()))


if __name__ == "__main__":
    unittest.main()
