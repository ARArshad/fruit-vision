"""Tests for dataset discovery, splitting and augmentation."""

from __future__ import annotations

import hashlib
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.config import CONFIG
from fruitvision.data.dataset import (
    build_eval_set,
    build_training_set,
    discover_images,
    split_dataset,
)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.items = discover_images(CONFIG)

    def test_every_class_has_images(self):
        found = {label for _, label in self.items}
        self.assertEqual(found, set(range(len(CONFIG.class_names))))

    def test_labels_match_the_folder_they_came_from(self):
        for path, label in self.items:
            with self.subTest(image=path.name):
                self.assertEqual(path.parent.name.lower(),
                                 CONFIG.class_names[label].lower())

    def test_no_duplicate_files(self):
        paths = [p for p, _ in self.items]
        self.assertEqual(len(paths), len(set(paths)))

    def test_no_duplicate_image_content(self):
        # An earlier version of this project shipped the same photos in both
        # the training and validation folders, which made the reported accuracy
        # meaningless. This guards against that coming back.
        digests = [hashlib.md5(path.read_bytes()).hexdigest() for path, _ in self.items]
        self.assertEqual(len(digests), len(set(digests)),
                         "the dataset contains duplicate images")


class HeldOutSetTests(unittest.TestCase):
    """The separate test folder is what makes the reported accuracy honest."""

    def setUp(self):
        if not CONFIG.test_dir.exists():
            self.skipTest("no separate held-out directory")
        self.train = discover_images(CONFIG)
        self.test = discover_images(CONFIG, CONFIG.test_dir)

    def test_held_out_set_covers_every_class(self):
        found = {label for _, label in self.test}
        self.assertEqual(found, set(range(len(CONFIG.class_names))))

    def test_no_image_appears_in_both_training_and_held_out(self):
        # Consecutive Fruits-360 images are one physical fruit rotated, so a
        # leak here would inflate the reported accuracy dramatically.
        train_digests = {hashlib.md5(p.read_bytes()).hexdigest() for p, _ in self.train}
        test_digests = {hashlib.md5(p.read_bytes()).hexdigest() for p, _ in self.test}
        overlap = train_digests & test_digests

        self.assertEqual(overlap, set(),
                         "%d image(s) appear in both the training and held-out sets"
                         % len(overlap))

    def test_held_out_paths_are_distinct_from_training_paths(self):
        self.assertFalse({p for p, _ in self.train} & {p for p, _ in self.test})


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.items = discover_images(CONFIG)
        self.train, self.test = split_dataset(self.items, CONFIG)

    def test_split_covers_every_image_exactly_once(self):
        self.assertEqual(len(self.train) + len(self.test), len(self.items))
        self.assertEqual(
            sorted(p for p, _ in self.train + self.test),
            sorted(p for p, _ in self.items),
        )

    def test_train_and_test_never_overlap(self):
        train_paths = {p for p, _ in self.train}
        test_paths = {p for p, _ in self.test}
        self.assertFalse(train_paths & test_paths)

    def test_both_splits_are_non_empty_and_stratified(self):
        self.assertTrue(self.train)
        self.assertTrue(self.test)
        for label in range(len(CONFIG.class_names)):
            self.assertTrue(any(l == label for _, l in self.train))
            self.assertTrue(any(l == label for _, l in self.test))

    def test_split_is_reproducible(self):
        again_train, again_test = split_dataset(self.items, CONFIG)
        self.assertEqual([p for p, _ in again_train], [p for p, _ in self.train])
        self.assertEqual([p for p, _ in again_test], [p for p, _ in self.test])


class AugmentationTests(unittest.TestCase):
    def test_augmentation_expands_the_dataset(self):
        items = discover_images(CONFIG)[:1]
        samples = build_training_set(items, CONFIG, rng=random.Random(0))
        self.assertEqual(len(samples), CONFIG.augmentations_per_image + 1)

    def test_augmented_vectors_have_the_right_shape_and_range(self):
        items = discover_images(CONFIG)[:1]
        expected = CONFIG.channels * CONFIG.img_size ** 2
        for vector, label in build_training_set(items, CONFIG, rng=random.Random(1)):
            self.assertEqual(len(vector), expected)
            self.assertEqual(label, 0)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in vector))

    def test_augmented_variants_actually_differ(self):
        items = discover_images(CONFIG)[:1]
        samples = build_training_set(items, CONFIG, rng=random.Random(2))
        unique = {tuple(round(v, 4) for v in vector) for vector, _ in samples}
        self.assertGreater(len(unique), len(samples) * 0.9)

    def test_eval_set_is_not_augmented(self):
        items = discover_images(CONFIG)[:2]
        first = build_eval_set(items, CONFIG)
        second = build_eval_set(items, CONFIG)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
