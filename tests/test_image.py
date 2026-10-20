"""Tests for resizing, letterboxing and model-input preparation."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.config import CONFIG
from fruitvision.imageio import Image, encode_png
from fruitvision.utils.image import (
    letterbox,
    preprocess_upload,
    resize,
    to_model_input,
)


def solid(width: int, height: int, colour) -> Image:
    return Image(width, height, bytearray(bytes(colour) * (width * height)))


class ResizeTests(unittest.TestCase):
    def test_downscale_averages_the_source(self):
        # Left half black, right half white -> a 1x1 result is mid grey.
        pixels = bytearray()
        for _ in range(4):
            pixels += bytes((0, 0, 0)) * 2 + bytes((255, 255, 255)) * 2
        result = resize(Image(4, 4, pixels), 1, 1)

        self.assertEqual(result.size, (1, 1))
        self.assertEqual(result.getpixel(0, 0), (127, 127, 127))

    def test_solid_colour_survives_any_resize(self):
        source = solid(20, 12, (10, 200, 30))
        for target in ((5, 3), (40, 24), (7, 19)):
            with self.subTest(target=target):
                out = resize(source, *target)
                self.assertEqual(out.size, target)
                self.assertEqual(out.getpixel(0, 0), (10, 200, 30))

    def test_same_size_resize_returns_a_copy(self):
        source = solid(4, 4, (1, 2, 3))
        out = resize(source, 4, 4)
        self.assertIsNot(out.pixels, source.pixels)
        self.assertEqual(bytes(out.pixels), bytes(source.pixels))

    def test_zero_sized_target_is_rejected(self):
        with self.assertRaises(ValueError):
            resize(solid(4, 4, (0, 0, 0)), 0, 4)


class LetterboxTests(unittest.TestCase):
    def test_output_is_square(self):
        result = letterbox(solid(40, 10, (0, 0, 255)), 32)
        self.assertEqual(result.size, (32, 32))

    def test_aspect_ratio_is_preserved_and_padded_with_white(self):
        # A wide image should end up as a band with white above and below.
        result = letterbox(solid(40, 10, (0, 0, 255)), 32)

        self.assertEqual(result.getpixel(16, 0), (255, 255, 255))   # top padding
        self.assertEqual(result.getpixel(16, 31), (255, 255, 255))  # bottom padding
        self.assertEqual(result.getpixel(16, 16), (0, 0, 255))      # the image itself

    def test_square_input_is_not_padded(self):
        result = letterbox(solid(50, 50, (7, 8, 9)), 16)
        self.assertEqual(result.getpixel(0, 0), (7, 8, 9))
        self.assertEqual(result.getpixel(15, 15), (7, 8, 9))


class ModelInputTests(unittest.TestCase):
    def test_vector_shape_and_range(self):
        vector = to_model_input(solid(60, 45, (128, 64, 32)), CONFIG)

        expected = CONFIG.channels * CONFIG.img_size * CONFIG.img_size
        self.assertEqual(len(vector), expected)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in vector))

    def test_channels_are_stored_in_planes(self):
        size = CONFIG.img_size
        area = size * size
        vector = to_model_input(solid(size, size, (255, 0, 0)), CONFIG)

        self.assertAlmostEqual(vector[0], 1.0)                 # red plane
        self.assertAlmostEqual(vector[area], 0.0)              # green plane
        self.assertAlmostEqual(vector[2 * area], 0.0)          # blue plane

    def test_preprocess_upload_reports_the_source_size(self):
        blob = encode_png(solid(48, 24, (200, 30, 30)))
        vector, size = preprocess_upload(blob, CONFIG)

        self.assertEqual(size, (48, 24))
        self.assertEqual(len(vector), CONFIG.channels * CONFIG.img_size ** 2)


if __name__ == "__main__":
    unittest.main()
