import unittest

import numpy as np

from core.mask_editor import (
    apply_pinned_groups,
    canvas_layer_to_mask,
    combined_to_instance_masks,
    masks_to_combined,
    stroke_bounding_box,
)
from core.reliability import MAX_AREA_RATIO, check_few_shot_result, check_reference_selection


def block(r, c, size, shape=(100, 100)):
    m = np.zeros(shape, bool)
    m[r : r + size, c : c + size] = True
    return m


class MaskEditorTest(unittest.TestCase):
    def test_canvas_layer_none_is_empty(self):
        self.assertFalse(canvas_layer_to_mask(None, (30, 40)).any())

    def test_canvas_layer_resizes_strokes_with_nearest_neighbour(self):
        rgba = np.zeros((10, 10, 4), np.uint8)
        rgba[2:4, 5:8, 3] = 255
        mask = canvas_layer_to_mask(rgba, (20, 40))
        self.assertEqual(mask.shape, (20, 40))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(stroke_bounding_box(mask), (20, 4, 31, 7))

    def test_stroke_bounding_box_empty(self):
        self.assertIsNone(stroke_bounding_box(np.zeros((5, 5), bool)))

    def test_combined_union_and_empty_stack(self):
        masks = np.stack([block(0, 0, 10), block(50, 50, 10)])
        self.assertEqual(masks_to_combined(masks, (100, 100)).sum(), 200)
        self.assertFalse(masks_to_combined(np.empty((0, 100, 100), bool), (100, 100)).any())

    def test_split_drops_noise_and_keeps_components(self):
        combined = block(0, 0, 10) | block(50, 50, 10)
        combined[90:92, 90:92] = True  # 4 px of noise
        self.assertEqual(combined_to_instance_masks(combined).shape[0], 2)
        self.assertEqual(combined_to_instance_masks(np.zeros((9, 9), bool)).shape, (0, 9, 9))

    def test_pinned_group_survives_as_one_row_across_a_gap(self):
        a, b, c = block(0, 0, 10), block(50, 50, 10), block(80, 0, 10)
        masks, n_pinned = apply_pinned_groups(a | b | c, [a | b])
        self.assertEqual(n_pinned, 1)
        self.assertEqual(masks.shape[0], 2)  # the pinned a+b, then c on its own
        self.assertEqual(masks[0].sum(), 200)

    def test_pixel_claimed_by_a_pinned_group_is_not_counted_twice(self):
        a, b = block(0, 0, 10), block(50, 50, 10)
        masks, n_pinned = apply_pinned_groups(a | b, [a, a | b])
        self.assertEqual(n_pinned, 2)
        self.assertEqual(masks.sum(), (a | b).sum())

    def test_pinned_group_outside_the_edited_mask_is_ignored(self):
        masks, n_pinned = apply_pinned_groups(block(0, 0, 10), [block(60, 60, 10)])
        self.assertEqual(n_pinned, 0)
        self.assertEqual(masks.shape[0], 1)


class InstanceStagesTest(unittest.TestCase):
    """apply_pinned_groups's kept-instance stage and merge ordering."""

    def test_touching_instances_stay_separate_when_kept(self):
        a, b = block(10, 10, 20), block(10, 30, 20)  # share an edge
        combined = a | b
        self.assertEqual(apply_pinned_groups(combined, [])[0].shape[0], 1)  # old behaviour: fused
        masks, n_pinned = apply_pinned_groups(combined, [], [a, b])
        self.assertEqual((masks.shape[0], n_pinned), (2, 0))
        self.assertEqual(sorted(int(m.sum()) for m in masks), [400, 400])

    def test_deleted_or_erased_kept_instances_follow_the_combined_mask(self):
        a, b = block(10, 10, 20), block(10, 40, 20)
        masks, _ = apply_pinned_groups(a, [], [a, b])  # b deleted from the combined mask
        self.assertEqual(masks.shape[0], 1)
        eroded = a.copy()
        eroded[:, 20:] = False
        masks, _ = apply_pinned_groups(eroded, [], [a])
        self.assertEqual(int(masks[0].sum()), int(eroded.sum()))

    def test_added_stroke_becomes_its_own_row_even_next_to_a_kept_instance(self):
        a, added = block(10, 10, 20), block(10, 30, 20)
        masks, _ = apply_pinned_groups(a | added, [], [a])
        self.assertEqual(masks.shape[0], 2)

    def test_kept_fragment_under_the_noise_floor_is_dropped(self):
        a = block(10, 10, 20)
        sliver = np.zeros_like(a)
        sliver[50, 50:53] = True
        masks, _ = apply_pinned_groups(a | sliver, [], [a, sliver])
        self.assertEqual(masks.shape[0], 1)

    def test_n_pinned_counts_only_the_whole_measured_groups(self):
        a, b, c = block(0, 0, 10), block(50, 50, 10), block(80, 0, 10)
        masks, n_pinned = apply_pinned_groups(a | b | c, [a | b], [c])
        self.assertEqual((masks.shape[0], n_pinned), (2, 1))

    def test_merge_listed_before_its_parts_wins_but_after_is_a_noop(self):
        a, b = block(0, 0, 10), block(50, 50, 10)
        union = a | b
        self.assertEqual(apply_pinned_groups(union, [union, a, b])[0].shape[0], 1)
        # the order the UI used to pass (originals first): the merge claimed nothing
        self.assertEqual(apply_pinned_groups(union, [a, b, union])[0].shape[0], 2)

    def test_nested_merge_needs_latest_first(self):
        a, b, c = block(0, 0, 10), block(40, 40, 10), block(80, 0, 10)
        u1, u2 = a | b, a | b | c
        masks, n_pinned = apply_pinned_groups(u2, [u2, u1, a, b, c])
        self.assertEqual((masks.shape[0], n_pinned), (1, 1))


class ReliabilityTest(unittest.TestCase):
    def test_empty_mask_is_unreliable(self):
        r = check_few_shot_result(np.zeros((10, 10), bool), 100)
        self.assertFalse(r.is_reliable)

    def test_large_excess_is_flagged_but_smaller_is_fine(self):
        ref = 100
        big = np.zeros((100, 100), bool)
        big.flat[: int(ref * MAX_AREA_RATIO) + 10] = True
        self.assertFalse(check_few_shot_result(big, ref).is_reliable)
        small = np.zeros((100, 100), bool)
        small.flat[:20] = True
        self.assertTrue(check_few_shot_result(small, ref).is_reliable)

    def test_reference_selection_warns_on_tiny_regions_and_size_mismatch(self):
        tiny = block(0, 0, 2, (512, 512))
        self.assertFalse(check_reference_selection(tiny, (512, 512)).is_clean)
        ok = block(0, 0, 40, (512, 512))
        self.assertTrue(check_reference_selection(ok, (512, 512)).is_clean)
        self.assertFalse(check_reference_selection(ok, (512, 512), prior_reference_mean_area=100.0).is_clean)
        self.assertTrue(check_reference_selection(ok, (512, 512), prior_reference_mean_area=1500.0).is_clean)


if __name__ == "__main__":
    unittest.main()
