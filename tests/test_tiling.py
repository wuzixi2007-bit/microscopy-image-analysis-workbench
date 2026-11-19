import unittest
from unittest import mock

import numpy as np
from skimage.measure import label

from core.detectors import tiling
from core.detectors.tiling import _cores, _starts, needs_tiling, predict_tiled


def fake_detector(crop):
    """Stand-in for the model: every bright connected blob in the crop is one
    instance. Blobs cut by the crop border are returned truncated, like a
    real detector would."""
    lab = label(crop[..., 0] > 128)
    return [lab == i for i in range(1, lab.max() + 1)]


def scene(h, w, size=30, pitch=97):
    """Grid of bright squares on black; returns (image, number of squares)."""
    img = np.zeros((h, w, 3), np.uint8)
    n = 0
    for y in range(20, h - size - 10, pitch):
        for x in range(20, w - size - 10, pitch):
            img[y : y + size, x : x + size] = 255
            n += 1
    return img, n


class TilingDecisionTest(unittest.TestCase):
    def test_which_sizes_get_tiled(self):
        self.assertFalse(needs_tiling(512, 767))  # EMPS, upscaled by the model
        self.assertFalse(needs_tiling(768, 1024))  # BAM_TiO2, the evaluated regime
        self.assertFalse(needs_tiling(1024, 1024))  # common square SEM frame, scale 0.78
        self.assertTrue(needs_tiling(1536, 2048))
        self.assertTrue(needs_tiling(3072, 4096))
        self.assertTrue(needs_tiling(700, 2000))  # long side alone forces a big shrink


class TileGeometryTest(unittest.TestCase):
    def test_tiles_reach_the_edge_and_never_exceed_the_image(self):
        for length in (800, 801, 1000, 1600, 2048, 4097):
            starts = _starts(length, 800, 160)
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + 800 if length > 800 else 800, max(length, 800))
            self.assertTrue(all(s + 800 <= max(length, 800) for s in starts))
            self.assertEqual(len(starts), len(set(starts)))

    def test_cores_partition_the_axis_even_with_an_oversized_last_overlap(self):
        for length in (1000, 1700, 2048, 3333):
            starts = _starts(length, 800, 160)
            cores = _cores(starts, length, 800)
            self.assertEqual(cores[0][0], 0)
            self.assertEqual(cores[-1][1], length)
            for (_, end), (start, _) in zip(cores, cores[1:]):
                self.assertEqual(end, start)  # contiguous: no gap, no double cover
            for s, (c0, c1) in zip(starts, cores):  # each core lies inside its own tile
                self.assertTrue(s <= c0 < c1 <= s + 800)


class PredictTiledTest(unittest.TestCase):
    def test_every_object_is_reported_exactly_once(self):
        for h, w in ((1000, 1000), (1536, 2048), (900, 2500)):
            img, n = scene(h, w)
            masks = predict_tiled(fake_detector, img)
            self.assertEqual(len(masks), n, (h, w))
            union = np.sum([m.astype(int) for m in masks], axis=0)
            self.assertEqual(int(union.max()), 1)  # no pixel claimed twice
            self.assertTrue(np.array_equal(union > 0, img[..., 0] > 128))  # nothing lost either

    def test_masks_come_back_full_size_and_in_place(self):
        img, _ = scene(1000, 1000)
        for m in predict_tiled(fake_detector, img):
            self.assertEqual(m.shape, (1000, 1000))
            self.assertEqual(m.dtype, bool)
            self.assertEqual(int(m.sum()), 900)

    def test_object_on_a_tile_seam_is_not_duplicated(self):
        img = np.zeros((1000, 1000, 3), np.uint8)
        img[480:520, 380:420] = 255  # centred on the first x-seam region of an 800 px grid
        self.assertEqual(len(predict_tiled(fake_detector, img)), 1)
        img = np.zeros((1000, 1000, 3), np.uint8)
        img[300:340, 780:820] = 255  # straddles the 800 px tile edge
        self.assertEqual(len(predict_tiled(fake_detector, img)), 1)

    def test_empty_image(self):
        self.assertEqual(predict_tiled(fake_detector, np.zeros((900, 900, 3), np.uint8)), [])

    def test_refuses_instead_of_exhausting_memory(self):
        img, n = scene(1000, 1000)
        with mock.patch.object(tiling, "MASK_BUDGET_BYTES", n * 1000 * 1000 - 1):
            with self.assertRaises(ValueError) as ctx:
                predict_tiled(fake_detector, img)
        self.assertIn("裁剪", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
