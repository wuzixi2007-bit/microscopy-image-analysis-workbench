import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from core import image_io
from core.image_io import ImageLoadError, load_rgb


def encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


class StandardModesTest(unittest.TestCase):
    def test_rgb_png_roundtrips_exactly(self):
        rgb = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        out = load_rgb(encode(Image.fromarray(rgb), "PNG"))
        self.assertTrue(np.array_equal(out, rgb))

    def test_grayscale_becomes_three_equal_channels(self):
        gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = load_rgb(encode(Image.fromarray(gray), "PNG"))
        self.assertEqual(out.shape, (16, 16, 3))
        self.assertTrue(all(np.array_equal(out[..., c], gray) for c in range(3)))

    def test_rgba_palette_and_bilevel(self):
        rgba = Image.new("RGBA", (8, 8), (10, 20, 30, 128))
        self.assertEqual(load_rgb(encode(rgba, "PNG")).shape, (8, 8, 3))
        self.assertEqual(load_rgb(encode(Image.new("P", (8, 8)), "PNG")).shape, (8, 8, 3))
        self.assertEqual(load_rgb(encode(Image.new("1", (8, 8)), "PNG")).shape, (8, 8, 3))

    def test_jpeg(self):
        self.assertEqual(load_rgb(encode(Image.new("RGB", (40, 30), (200, 100, 50)), "JPEG")).shape, (30, 40, 3))


class HighBitDepthTest(unittest.TestCase):
    def test_16bit_tiff_is_stretched_not_clipped(self):
        ramp = np.linspace(0, 60000, 64 * 64).reshape(64, 64).astype(np.uint16)
        out = load_rgb(encode(Image.fromarray(ramp), "TIFF"))
        self.assertEqual((out.dtype, out.shape), (np.uint8, (64, 64, 3)))
        gray = out[..., 0]
        self.assertLess(float((gray == 255).mean()), 0.05)  # plain convert('RGB') made this 100% white
        self.assertGreater(int(gray.max()) - int(gray.min()), 200)
        self.assertTrue(np.all(np.diff(gray.reshape(-1).astype(int)) >= 0))  # order preserved

    def test_16bit_image_that_only_uses_a_low_range_is_still_visible(self):
        dim = (np.linspace(100, 900, 64 * 64).reshape(64, 64)).astype(np.uint16)  # a dim detector frame
        out = load_rgb(encode(Image.fromarray(dim), "TIFF"))
        self.assertGreater(int(out.max()) - int(out.min()), 200)

    def test_float_tiff(self):
        f = np.linspace(0, 1, 64 * 64).reshape(64, 64).astype(np.float32)
        out = load_rgb(encode(Image.fromarray(f, mode="F"), "TIFF"))
        self.assertGreater(int(out.max()) - int(out.min()), 200)

    def test_flat_high_depth_image_is_black_not_an_error(self):
        flat = np.full((16, 16), 4000, np.uint16)
        self.assertEqual(int(load_rgb(encode(Image.fromarray(flat), "TIFF")).max()), 0)

    def test_a_few_hot_pixels_do_not_crush_the_contrast(self):
        img = np.linspace(1000, 3000, 100 * 100).reshape(100, 100).astype(np.uint16)
        img[0, :5] = 65535
        out = load_rgb(encode(Image.fromarray(img), "TIFF"))[..., 0]
        self.assertGreater(float(np.median(out)), 100)  # without the percentile stretch the median would be ~8


class BadInputTest(unittest.TestCase):
    def test_garbage_empty_and_truncated_files(self):
        valid = encode(Image.new("RGB", (64, 64), (1, 2, 3)), "PNG")
        for name, data in (("garbage", b"definitely not an image"), ("empty", b""), ("truncated", valid[: len(valid) // 2])):
            with self.subTest(name):
                with self.assertRaises(ImageLoadError) as ctx:
                    load_rgb(data)
                self.assertIn("无法读取", str(ctx.exception))

    def test_error_is_a_valueerror_for_callers_that_only_catch_that(self):
        self.assertTrue(issubclass(ImageLoadError, ValueError))

    def test_oversized_image_is_refused_with_a_message(self):
        with mock.patch.object(image_io, "MAX_PIXELS", 100):
            with self.assertRaises(ImageLoadError) as ctx:
                load_rgb(encode(Image.new("RGB", (20, 20)), "PNG"))
        self.assertIn("上限", str(ctx.exception))

    def test_multipage_tiff_uses_the_first_page(self):
        first, second = Image.new("RGB", (16, 16), (255, 0, 0)), Image.new("RGB", (16, 16), (0, 0, 255))
        data = encode(first, "TIFF", save_all=True, append_images=[second])
        self.assertEqual(tuple(load_rgb(data)[0, 0]), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
