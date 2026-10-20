"""Tests for model loading and end-to-end prediction on the real dataset."""

from __future__ import annotations

import json
import pathlib
import random
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.config import CONFIG
from fruitvision.data.dataset import discover_images
from fruitvision.inference import (
    ModelNotTrainedError,
    get_default_model_path,
    load_class_names,
    load_model,
    predict_image_bytes,
    top_prediction,
)

MODEL_AVAILABLE = get_default_model_path().exists()
requires_model = unittest.skipUnless(
    MODEL_AVAILABLE,
    "No trained model. Run: python train.py",
)


class TopPredictionTests(unittest.TestCase):
    def test_returns_the_highest_scoring_class(self):
        label, confidence = top_prediction({"Apple": 0.82, "Banana": 0.18})
        self.assertEqual(label, "Apple")
        self.assertAlmostEqual(confidence, 0.82)

    def test_handles_close_scores(self):
        label, _ = top_prediction({"Apple": 0.49, "Banana": 0.51})
        self.assertEqual(label, "Banana")


class MissingModelTests(unittest.TestCase):
    def test_a_missing_model_gives_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = pathlib.Path(tmp) / "nope.json"
            with self.assertRaises(ModelNotTrainedError) as caught:
                load_model(missing)
            self.assertIn("train.py", str(caught.exception))


@requires_model
class LoadedModelTests(unittest.TestCase):
    def test_model_file_declares_its_architecture(self):
        with get_default_model_path().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertEqual(payload["format"], "fruitvision-model/2")
        self.assertEqual(payload["class_names"], CONFIG.class_names)
        for key in ("img_size", "channels", "conv1_filters", "conv2_filters",
                    "kernel_size", "dense_units"):
            self.assertIn(key, payload["architecture"])

    def test_model_file_carries_both_streams(self):
        with get_default_model_path().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertIn("layers", payload, "the pixel CNN stream is missing")
        descriptor = payload.get("descriptor")
        self.assertIsNotNone(descriptor, "the descriptor stream is missing")
        for key in ("size", "hidden", "features", "mean", "std", "layers"):
            self.assertIn(key, descriptor)
        self.assertEqual(len(descriptor["mean"]), descriptor["features"])
        self.assertEqual(len(descriptor["std"]), descriptor["features"])

    def test_class_names_come_from_the_model(self):
        self.assertEqual(load_class_names(), CONFIG.class_names)

    def test_model_is_cached_between_calls(self):
        self.assertIs(load_model()[0], load_model()[0])

    def test_model_is_reloaded_when_the_file_changes(self):
        """Retraining must take effect without restarting the server."""

        import os
        import time

        path = get_default_model_path()
        first = load_model()[0]
        # Touch the file the way a retrain would, then confirm we do not keep
        # serving the stale in-memory model.
        stat = path.stat()
        os.utime(path, (stat.st_atime, stat.st_mtime + 10))
        try:
            second = load_model()[0]
            self.assertIsNot(first, second)
        finally:
            os.utime(path, (stat.st_atime, stat.st_mtime))


# Accuracy floors. Deliberately below what the model actually scores, so the
# suite catches a genuinely broken model rather than failing on normal
# run-to-run variation after retraining.
MIN_TRAINING_ACCURACY = 0.85
MIN_HELD_OUT_ACCURACY = 0.70


def sample_images(items, count, seed):
    """A reproducible spread of images across all classes."""

    by_class = {}
    for path, label in items:
        by_class.setdefault(label, []).append((path, label))
    picked = []
    rng = random.Random(seed)
    per_class = max(1, count // max(1, len(by_class)))
    for label in sorted(by_class):
        group = sorted(by_class[label], key=lambda item: item[0].name)
        picked.extend(rng.sample(group, min(per_class, len(group))))
    return picked


def measure_accuracy(items):
    correct = 0
    for path, label in items:
        scores, _ = predict_image_bytes(path.read_bytes())
        predicted, _ = top_prediction(scores)
        correct += predicted == CONFIG.class_names[label]
    return correct / len(items)


@requires_model
class PredictionTests(unittest.TestCase):
    """The model must actually classify fruit, not just return valid JSON."""

    def test_training_images_are_classified_correctly(self):
        picked = sample_images(discover_images(CONFIG), 32, seed=1)
        accuracy = measure_accuracy(picked)

        self.assertGreaterEqual(
            accuracy, MIN_TRAINING_ACCURACY,
            "only %.0f%% of %d training images were classified correctly"
            % (accuracy * 100, len(picked)))

    def test_held_out_images_are_classified_correctly(self):
        if not CONFIG.test_dir.exists():
            self.skipTest("no separate held-out directory")
        picked = sample_images(discover_images(CONFIG, CONFIG.test_dir), 48, seed=2)
        accuracy = measure_accuracy(picked)

        self.assertGreaterEqual(
            accuracy, MIN_HELD_OUT_ACCURACY,
            "only %.0f%% of %d held-out images were classified correctly"
            % (accuracy * 100, len(picked)))

    # The nine photographs collected for the project before Fruits-360 and the
    # web-sourced training photos were added -- apple_01..04.png and
    # banana_01..05.png. Later PNGs (double-digit numbering) are imports and
    # are not held to this per-image guarantee.
    _ORIGINAL_OWN_PHOTOS = {
        "apple_01.png", "apple_02.png", "apple_03.png", "apple_04.png",
        "banana_01.png", "banana_02.png", "banana_03.png", "banana_04.png", "banana_05.png",
    }

    def test_own_photographs_are_classified_correctly(self):
        """The project's original hand-taken photos, which have real backgrounds."""

        own = [(p, l) for p, l in discover_images(CONFIG)
               if p.name in self._ORIGINAL_OWN_PHOTOS]
        if not own:
            self.skipTest("no hand-collected photographs in the dataset")
        for path, label in own:
            with self.subTest(image=path.name):
                scores, _ = predict_image_bytes(path.read_bytes())
                predicted, _ = top_prediction(scores)
                self.assertEqual(predicted, CONFIG.class_names[label])

    def test_scores_form_a_probability_distribution(self):
        path = discover_images(CONFIG)[0][0]
        scores, _ = predict_image_bytes(path.read_bytes())

        self.assertEqual(set(scores), set(CONFIG.class_names))
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))

    def test_prediction_is_deterministic(self):
        path = discover_images(CONFIG)[0][0]
        blob = path.read_bytes()
        self.assertEqual(predict_image_bytes(blob)[0], predict_image_bytes(blob)[0])


if __name__ == "__main__":
    unittest.main()
