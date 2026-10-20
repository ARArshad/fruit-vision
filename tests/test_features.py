"""Tests for the colour/shape/texture descriptor and its classifier."""

from __future__ import annotations

import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.config import CONFIG
from fruitvision.features import (
    DESCRIPTOR_LENGTH,
    HUE_BINS,
    describe,
    describe_image,
    rgb_to_hsv,
)
from fruitvision.imageio import Image
from fruitvision.models.descriptor import DescriptorNet, standardiser


def solid(size, colour):
    return bytearray(bytes(colour) * (size * size))


def disc(size, fruit, background, radius_ratio=0.35):
    """A filled circle of one colour on a background of another."""

    pixels = solid(size, background)
    centre = (size - 1) / 2.0
    radius = size * radius_ratio
    for y in range(size):
        for x in range(size):
            if (x - centre) ** 2 + (y - centre) ** 2 <= radius * radius:
                i = (y * size + x) * 3
                pixels[i : i + 3] = bytes(fruit)
    return pixels


class ColourConversionTests(unittest.TestCase):
    def test_primary_colours_map_to_the_expected_hues(self):
        for colour, expected in (((1, 0, 0), 0.0), ((0, 1, 0), 1 / 3), ((0, 0, 1), 2 / 3)):
            hue, saturation, value = rgb_to_hsv(*colour)
            self.assertAlmostEqual(hue, expected, places=6)
            self.assertAlmostEqual(saturation, 1.0)
            self.assertAlmostEqual(value, 1.0)

    def test_grey_has_no_hue_and_no_saturation(self):
        hue, saturation, value = rgb_to_hsv(0.5, 0.5, 0.5)
        self.assertEqual(hue, 0.0)
        self.assertEqual(saturation, 0.0)
        self.assertAlmostEqual(value, 0.5)


class DescriptorTests(unittest.TestCase):
    SIZE = 48

    def test_length_is_fixed(self):
        vector = describe(disc(self.SIZE, (220, 40, 40), (255, 255, 255)), self.SIZE)
        self.assertEqual(len(vector), DESCRIPTOR_LENGTH)

    def test_is_deterministic(self):
        pixels = disc(self.SIZE, (220, 40, 40), (255, 255, 255))
        self.assertEqual(describe(pixels, self.SIZE), describe(pixels, self.SIZE))

    def test_values_are_finite_and_bounded(self):
        for vector in (
            describe(solid(self.SIZE, (255, 255, 255)), self.SIZE),
            describe(solid(self.SIZE, (0, 0, 0)), self.SIZE),
            describe(disc(self.SIZE, (30, 200, 60), (12, 12, 12)), self.SIZE),
        ):
            self.assertEqual(len(vector), DESCRIPTOR_LENGTH)
            self.assertTrue(all(-10.0 <= v <= 10.0 for v in vector))

    def test_the_background_is_excluded(self):
        """The same fruit on white and on blue must describe almost the same."""

        on_white = describe(disc(self.SIZE, (220, 40, 40), (255, 255, 255)), self.SIZE)
        on_blue = describe(disc(self.SIZE, (220, 40, 40), (40, 60, 200)), self.SIZE)

        hue_white = on_white[:HUE_BINS]
        hue_blue = on_blue[:HUE_BINS]
        self.assertEqual(
            max(range(HUE_BINS), key=lambda i: hue_white[i]),
            max(range(HUE_BINS), key=lambda i: hue_blue[i]),
            "the dominant hue changed with the backdrop",
        )
        difference = sum(abs(a - b) for a, b in zip(hue_white, hue_blue))
        self.assertLess(difference, 0.2)

    def test_different_colours_describe_differently(self):
        red = describe(disc(self.SIZE, (220, 40, 40), (255, 255, 255)), self.SIZE)
        yellow = describe(disc(self.SIZE, (240, 220, 40), (255, 255, 255)), self.SIZE)
        self.assertNotEqual(
            max(range(HUE_BINS), key=lambda i: red[i]),
            max(range(HUE_BINS), key=lambda i: yellow[i]),
        )

    def test_shape_separates_round_from_elongated(self):
        size = self.SIZE
        round_fruit = disc(size, (220, 40, 40), (255, 255, 255))
        long_fruit = solid(size, (255, 255, 255))
        for y in range(size // 2 - 3, size // 2 + 3):
            for x in range(4, size - 4):
                i = (y * size + x) * 3
                long_fruit[i : i + 3] = bytes((220, 40, 40))

        # Feature 1 of the shape block is min(aspect, 1/aspect): 1.0 for a
        # square-ish outline, small for a long thin one.
        offset = HUE_BINS + 5 + 5 + 6
        self.assertGreater(describe(round_fruit, size)[offset + 1],
                           describe(long_fruit, size)[offset + 1])

    def test_describe_image_accepts_any_aspect_ratio(self):
        image = Image(20, 40, bytearray(bytes((200, 30, 30)) * (20 * 40)))
        self.assertEqual(len(describe_image(image, CONFIG)), DESCRIPTOR_LENGTH)


class StandardiserTests(unittest.TestCase):
    def test_centres_and_scales_each_feature(self):
        rows = [[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]]
        mean, std = standardiser(rows)
        self.assertAlmostEqual(mean[0], 3.0)
        self.assertAlmostEqual(std[0], (8.0 / 3.0) ** 0.5)
        # A constant feature keeps a scale of 1 rather than dividing by zero.
        self.assertAlmostEqual(std[1], 1.0)

    def test_rejects_an_empty_set(self):
        with self.assertRaises(ValueError):
            standardiser([])


class DescriptorNetTests(unittest.TestCase):
    def make_net(self):
        net = DescriptorNet(CONFIG, rng=random.Random(0))
        net.set_standardisation([0.1] * DESCRIPTOR_LENGTH, [0.5] * DESCRIPTOR_LENGTH)
        return net

    def test_probabilities_sum_to_one(self):
        net = self.make_net()
        probs = net.predict_proba([0.2] * DESCRIPTOR_LENGTH)
        self.assertEqual(len(probs), len(CONFIG.class_names))
        self.assertAlmostEqual(sum(probs), 1.0, places=9)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_state_round_trip_preserves_predictions(self):
        net = self.make_net()
        vector = [0.3] * DESCRIPTOR_LENGTH
        expected = net.predict_proba(vector)

        restored = DescriptorNet(CONFIG, rng=random.Random(99))
        restored.load_state_dict(net.state_dict())
        self.assertEqual(restored.predict_proba(vector), expected)

    def test_rejects_a_descriptor_of_the_wrong_size(self):
        net = self.make_net()
        state = net.state_dict()
        state["features"] = DESCRIPTOR_LENGTH + 1
        with self.assertRaises(ValueError):
            DescriptorNet(CONFIG).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
