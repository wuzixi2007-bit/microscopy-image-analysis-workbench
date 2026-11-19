import math
import unittest

import numpy as np

from core.measurement import compute_crack_properties, compute_particle_properties, skeleton_length_px


def square(size=20, offset=(5, 5), shape=(80, 80)):
    mask = np.zeros(shape, bool)
    r, c = offset
    mask[r : r + size, c : c + size] = True
    return mask


class ParticlePropertiesTest(unittest.TestCase):
    def test_uncalibrated_stays_in_pixels_and_says_so(self):
        (row,) = compute_particle_properties(square(20))
        self.assertEqual(row["area_px2"], 400)
        self.assertAlmostEqual(row["equivalent_diameter_px"], math.sqrt(4 * 400 / math.pi))
        self.assertFalse([k for k in row if k.endswith("_um") or k.endswith("_um2")])

    def test_calibrated_scales_lengths_linearly_and_area_quadratically(self):
        (px,) = compute_particle_properties(square(20))
        (um,) = compute_particle_properties(square(20), pixel_size_um=0.5)
        self.assertAlmostEqual(um["area_um2"], px["area_px2"] * 0.25)
        self.assertAlmostEqual(um["perimeter_um"], px["perimeter_px"] * 0.5)
        self.assertAlmostEqual(um["equivalent_diameter_um"], px["equivalent_diameter_px"] * 0.5)
        self.assertFalse([k for k in um if k.endswith("_px2") or k in ("perimeter_px", "equivalent_diameter_px")])

    def test_dimensionless_values_ignore_scale(self):
        (a,) = compute_particle_properties(square(20))
        (b,) = compute_particle_properties(square(20), pixel_size_um=3.0)
        self.assertEqual((a["circularity"], a["aspect_ratio"]), (b["circularity"], b["aspect_ratio"]))
        self.assertAlmostEqual(a["aspect_ratio"], 1.0, places=3)

    def test_centroid_is_row_col_in_pixels_rounded_to_a_tenth(self):
        (row,) = compute_particle_properties(square(20, offset=(10, 30)), pixel_size_um=0.5)
        self.assertEqual(row["centroid_px"], (19.5, 39.5))

    def test_empty_mask(self):
        self.assertEqual(compute_particle_properties(np.zeros((40, 40), bool)), [])

    def test_small_noise_components_are_dropped(self):
        mask = square(30)
        mask[70:73, 70:73] = True  # 9 px speck: under the 25 px floor
        self.assertEqual(len(compute_particle_properties(mask)), 1)

    def test_relative_floor_drops_fragments_next_to_a_large_particle(self):
        mask = np.zeros((300, 300), bool)
        mask[10:210, 10:210] = True  # 40000 px -> relative floor 2000 px
        mask[250:270, 250:270] = True  # 400 px: above 25 px, below 5% of the largest
        self.assertEqual(len(compute_particle_properties(mask)), 1)

    def test_disjoint_pieces_split_unless_merge_disjoint(self):
        mask = square(15, (5, 5)) | square(15, (50, 50))
        self.assertEqual(len(compute_particle_properties(mask)), 2)
        (merged,) = compute_particle_properties(mask, merge_disjoint=True)
        self.assertEqual(merged["area_px2"], 450)


class SkeletonLengthTest(unittest.TestCase):
    def test_straight_line_counts_steps_not_pixels(self):
        line = np.zeros((5, 30), bool)
        line[2, 3:24] = True  # 21 pixels = 20 orthogonal steps
        self.assertAlmostEqual(skeleton_length_px(line), 20 * 0.948)

    def test_diagonal_line(self):
        diag = np.eye(11, dtype=bool)  # 10 diagonal steps
        self.assertAlmostEqual(skeleton_length_px(diag), 10 * 1.340)

    def test_l_corner_is_not_charged_for_its_diagonal(self):
        corner = np.zeros((3, 3), bool)
        corner[0, 0] = corner[0, 1] = corner[1, 1] = True
        self.assertAlmostEqual(skeleton_length_px(corner), 2 * 0.948)

    def test_empty_and_single_pixel(self):
        self.assertEqual(skeleton_length_px(np.zeros((4, 4), bool)), 0.0)
        one = np.zeros((4, 4), bool)
        one[1, 1] = True
        self.assertEqual(skeleton_length_px(one), 0.0)


class CrackPropertiesTest(unittest.TestCase):
    @staticmethod
    def bar(length=100, width=4):
        mask = np.zeros((60, 160), bool)
        mask[28 : 28 + width, 20 : 20 + length] = True
        return mask

    def test_length_and_width_of_a_straight_crack(self):
        (row,) = compute_crack_properties(self.bar(100, 4))
        # the skeleton stops about half a width short of each tip
        self.assertTrue(90 <= row["length_px"] <= 100, row["length_px"])
        self.assertAlmostEqual(row["mean_width_px"], 400 / row["length_px"])
        self.assertAlmostEqual(row["aspect_ratio"], row["length_px"] / row["mean_width_px"])

    def test_calibrated_names_and_scaling(self):
        (px,) = compute_crack_properties(self.bar())
        (um,) = compute_crack_properties(self.bar(), pixel_size_um=0.1)
        self.assertAlmostEqual(um["length_um"], px["length_px"] * 0.1)
        self.assertAlmostEqual(um["mean_width_um"], px["mean_width_px"] * 0.1)
        self.assertAlmostEqual(um["area_um2"], px["area_px2"] * 0.01)
        self.assertAlmostEqual(um["aspect_ratio"], px["aspect_ratio"])

    def test_merged_disjoint_pieces_sum_their_lengths(self):
        two = np.zeros((60, 260), bool)
        two[10:14, 10:70] = True
        two[40:44, 100:160] = True
        (single,) = compute_crack_properties(two[:, :90])
        (merged,) = compute_crack_properties(two, merge_disjoint=True)
        self.assertAlmostEqual(merged["length_px"], 2 * single["length_px"], delta=1.0)
        self.assertEqual(len(compute_crack_properties(two)), 2)

    def test_compact_blob_keeps_a_row_with_floored_length(self):
        (row,) = compute_crack_properties(square(6, (3, 3), (20, 20)))
        self.assertGreaterEqual(row["length_px"], 1.0)


if __name__ == "__main__":
    unittest.main()
