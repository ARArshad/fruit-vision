"""Tests for the pure-Python image codecs."""

from __future__ import annotations

import base64
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.imageio import (
    Image,
    UnsupportedImageError,
    decode_image,
    decode_jpeg,
    decode_png,
    encode_png,
    probe_jpeg_size,
    sniff_format,
)

# 16x16 crops of a dataset photo, embedded so the JPEG paths stay covered
# without committing a binary fixture.
BASELINE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQ"
    "FxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMK"
    "ChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCgoKCj/wAARCAAQABADAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQF"
    "BgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEI"
    "I0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNk"
    "ZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLD"
    "xMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEB"
    "AQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZH"
    "SElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaan"
    "qKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oA"
    "DAMBAAIRAxEAPwD1r4ifE46Cb2DStOurt7Ngk9wABEjEcjPJyMg9MZGK5KmIWsY7rqe1"
    "hskq1oRqSaXNsutu5xnwa+JOj3Pib7Ffw6mmq6k2xbm5uvORm6hccbc+uPyrLCU40225"
    "Nt9zDFZRUwqcr3Rk/HDwxrel6vqVxYWtzc6Tqb+cxtwzYfqVdR78ilUwz53JdT6DLs3p"
    "PDwpzaUoK2v6GH8DPAerav4wstUvbGey03TpVnaSeMxmR15VFB5POMnpitqVJp6nFmWZ"
    "wqU3GLu2f//Z"
)

PROGRESSIVE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQ"
    "FxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMK"
    "ChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCgoKCj/wgARCAAQABADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABAUG/8QA"
    "FQEBAQAAAAAAAAAAAAAAAAAAAwX/2gAMAwEAAhADEAAAAdbFIEKH/8QAGRAAAwADAAAA"
    "AAAAAAAAAAAAAwQFAAIT/9oACAEBAAEFAqNPhkakHZm4sYRYaBSuf//EABsRAAEEAwAA"
    "AAAAAAAAAAAAAAEAAgMEERMh/9oACAEDAQE/AXRV9eQer//EABcRAAMBAAAAAAAAAAAA"
    "AAAAAAABIQL/2gAIAQIBAT8Br1T/xAAfEAACAgIBBQAAAAAAAAAAAAACAwERABITISIx"
    "QUL/2gAIAQEABj8COFLItPJes0OGcrPoivGEAkSmde3AaYSC1zdzFXn/xAAbEAEAAwAD"
    "AQAAAAAAAAAAAAABABEhMUFRYf/aAAgBAQABPyHWmq6jKC60Ua+fIts76d8SFho9QcBP"
    "/9oADAMBAAIAAwAAABB//8QAGhEAAQUBAAAAAAAAAAAAAAAAAQARMUFRcf/aAAgBAwEB"
    "PxAYYjbbOr//xAAXEQEAAwAAAAAAAAAAAAAAAAABABEx/9oACAECAQE/EAuDJ//EABsQ"
    "AQEBAAMBAQAAAAAAAAAAAAERIQAxYVGR/9oACAEBAAE/EER7DAONLrSj1KTnutyyEZr7"
    "PzlaWwQ7ovdOMq2ktQnXZXqc/9k="
)

class SniffTests(unittest.TestCase):
    def test_recognises_known_formats(self):
        self.assertEqual(sniff_format(b"\x89PNG\r\n\x1a\n" + b"rest"), "png")
        self.assertEqual(sniff_format(b"\xff\xd8\xff\xe0"), "jpeg")
        self.assertEqual(sniff_format(b"BM\x00\x00"), "bmp")
        self.assertEqual(sniff_format(b"not an image"), "unknown")

    def test_unknown_format_is_rejected_clearly(self):
        with self.assertRaises(UnsupportedImageError):
            decode_image(b"this is plain text, not a picture")


class PngTests(unittest.TestCase):
    def test_encode_decode_round_trip_is_lossless(self):
        rng = random.Random(0)
        width, height = 37, 23
        pixels = bytearray(rng.randrange(256) for _ in range(width * height * 3))
        blob = encode_png(Image(width, height, pixels))

        decoded = decode_png(blob)

        self.assertEqual(decoded.size, (width, height))
        self.assertEqual(bytes(decoded.pixels), bytes(pixels))

    def test_dataset_images_decode(self):
        for path in sorted((ROOT / "data" / "raw" / "custom").glob("*/*.png")):
            with self.subTest(image=path.name):
                image = decode_png(path.read_bytes())
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)
                self.assertEqual(len(image.pixels), image.width * image.height * 3)

    def test_pixel_buffer_length_is_validated(self):
        with self.assertRaises(ValueError):
            Image(4, 4, bytearray(10))


class JpegTests(unittest.TestCase):
    def _mean_rgb(self, image):
        px = image.pixels
        count = image.width * image.height
        return tuple(sum(px[c::3]) / count for c in range(3))

    def test_baseline_jpeg_decodes(self):
        image = decode_jpeg(base64.b64decode(BASELINE_JPEG_B64))
        self.assertEqual(image.size, (16, 16))
        red, green, blue = self._mean_rgb(image)
        # The crop is the top of a red apple on white.
        self.assertGreater(red, green + 30)
        self.assertGreater(red, blue + 30)

    def test_progressive_jpeg_decodes_like_baseline(self):
        baseline = decode_jpeg(base64.b64decode(BASELINE_JPEG_B64))
        progressive = decode_jpeg(base64.b64decode(PROGRESSIVE_JPEG_B64))

        self.assertEqual(progressive.size, baseline.size)
        for a, b in zip(self._mean_rgb(baseline), self._mean_rgb(progressive)):
            self.assertLess(abs(a - b), 12.0)

    def test_dc_only_decode_is_one_eighth_scale(self):
        image = decode_jpeg(base64.b64decode(BASELINE_JPEG_B64), dc_only=True)
        self.assertEqual(image.size, (2, 2))

    def test_decode_image_dispatches_to_jpeg(self):
        image = decode_image(base64.b64decode(BASELINE_JPEG_B64))
        self.assertEqual(image.size, (16, 16))

    def test_size_can_be_probed_without_decoding(self):
        self.assertEqual(probe_jpeg_size(base64.b64decode(BASELINE_JPEG_B64)), (16, 16))
        self.assertEqual(probe_jpeg_size(base64.b64decode(PROGRESSIVE_JPEG_B64)), (16, 16))
        self.assertIsNone(probe_jpeg_size(b"\xff\xd8not really a jpeg"))

    def test_fast_decode_is_skipped_for_small_images(self):
        # A 16x16 JPEG would collapse to 2x2 at 1/8 scale, so decode_image
        # should quietly fall back to a full decode.
        blob = base64.b64decode(BASELINE_JPEG_B64)
        self.assertEqual(decode_image(blob, fast=True).size, (16, 16))


if __name__ == "__main__":
    unittest.main()
