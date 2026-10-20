"""Tests for the hand-written neural network: shapes, gradients, persistence."""

from __future__ import annotations

import math
import pathlib
import random
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.models.cnn import (
    Adam,
    Conv2D,
    Dense,
    MaxPool2D,
    Network,
    ReLU,
    softmax,
    softmax_cross_entropy,
)


class LayerTests(unittest.TestCase):
    def test_conv_output_shape(self):
        conv = Conv2D(3, 4, 3, 10, 10, rng=random.Random(0))
        out = conv.forward([0.0] * (3 * 10 * 10))
        self.assertEqual(conv.output_shape, (4, 8, 8))
        self.assertEqual(len(out), 4 * 8 * 8)

    def test_conv_with_a_known_kernel(self):
        # A single 1-in/1-out filter of all ones sums each 2x2 window.
        conv = Conv2D(1, 1, 2, 3, 3, rng=random.Random(0))
        conv.weights = [1.0] * 4
        conv.biases = [0.0]
        out = conv.forward([1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0])
        self.assertEqual(out, [12.0, 16.0, 24.0, 28.0])

    def test_maxpool_picks_the_largest_value(self):
        pool = MaxPool2D(1, 4, 4)
        out = pool.forward([
            1, 2, 5, 6,
            3, 4, 7, 8,
            9, 10, 13, 14,
            11, 12, 15, 16,
        ])
        self.assertEqual(pool.output_shape, (1, 2, 2))
        self.assertEqual(out, [4, 8, 12, 16])

    def test_relu_clips_negatives(self):
        self.assertEqual(ReLU().forward([-2.0, -0.1, 0.0, 3.0]), [0.0, 0.0, 0.0, 3.0])

    def test_dense_computes_the_expected_product(self):
        dense = Dense(3, 2, rng=random.Random(0))
        dense.weights = [1.0, 0.0, 0.0, 0.0, 2.0, 3.0]
        dense.biases = [0.5, -1.0]
        self.assertEqual(dense.forward([4.0, 5.0, 6.0]), [4.5, 27.0])


class SoftmaxTests(unittest.TestCase):
    def test_softmax_is_a_probability_distribution(self):
        probs = softmax([2.0, 1.0, 0.1])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(all(p > 0 for p in probs))
        self.assertEqual(max(range(3), key=lambda i: probs[i]), 0)

    def test_softmax_is_numerically_stable(self):
        probs = softmax([1000.0, 1001.0])
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertTrue(all(math.isfinite(p) for p in probs))

    def test_cross_entropy_gradient_points_at_the_target(self):
        loss, grad = softmax_cross_entropy([0.0, 0.0], 1)
        self.assertAlmostEqual(loss, math.log(2.0), places=6)
        self.assertAlmostEqual(sum(grad), 0.0, places=9)
        self.assertLess(grad[1], 0.0)  # push the true class logit up


class GradientTests(unittest.TestCase):
    """Verify backpropagation against numerical derivatives."""

    def test_analytic_gradients_match_finite_differences(self):
        config = TrainingConfig(
            img_size=12, conv1_filters=3, conv2_filters=4, dense_units=5, dropout=0.0
        )
        network = Network(config, rng=random.Random(7))
        rng = random.Random(1)
        x = [rng.random() for _ in range(config.channels * 12 * 12)]
        target = 1

        network.zero_grad()
        _, grad = softmax_cross_entropy(network.forward(x), target)
        network.backward(grad)

        epsilon = 1e-5
        worst = 0.0
        for index, layer in enumerate(network.layers):
            for values, grads in layer.parameters():
                picks = random.Random(index).sample(range(len(values)), min(8, len(values)))
                for i in picks:
                    original = values[i]
                    values[i] = original + epsilon
                    plus = softmax_cross_entropy(network.forward(x), target)[0]
                    values[i] = original - epsilon
                    minus = softmax_cross_entropy(network.forward(x), target)[0]
                    values[i] = original

                    numeric = (plus - minus) / (2 * epsilon)
                    analytic = grads[i]
                    scale = max(1e-8, abs(numeric) + abs(analytic))
                    worst = max(worst, abs(numeric - analytic) / scale)

        self.assertLess(worst, 1e-4, "backpropagation disagrees with numerical gradients")


class TrainingTests(unittest.TestCase):
    def test_network_can_learn_a_trivial_pattern(self):
        # Red images are class 0, green images are class 1.
        config = TrainingConfig(img_size=16, conv1_filters=4, conv2_filters=4,
                                dense_units=6, dropout=0.0)
        area = 16 * 16
        red = [1.0] * area + [0.0] * area + [0.0] * area
        green = [0.0] * area + [1.0] * area + [0.0] * area

        network = Network(config, rng=random.Random(3))
        optimiser = Adam(network.parameters(), lr=0.02)
        for _ in range(25):
            network.zero_grad()
            for vector, label in ((red, 0), (green, 1)):
                _, grad = softmax_cross_entropy(network.forward(vector, training=True), label)
                network.backward(grad)
            optimiser.step(scale=0.5)

        self.assertEqual(max(range(2), key=lambda i: network.predict_proba(red)[i]), 0)
        self.assertEqual(max(range(2), key=lambda i: network.predict_proba(green)[i]), 1)


class PersistenceTests(unittest.TestCase):
    def test_state_round_trip_reproduces_predictions(self):
        source = Network(CONFIG, rng=random.Random(11))
        target = Network(CONFIG, rng=random.Random(99))
        vector = [random.Random(5).random() for _ in range(CONFIG.channels * CONFIG.img_size ** 2)]

        self.assertNotEqual(source.predict_proba(vector), target.predict_proba(vector))
        target.load_state_dict(source.state_dict())
        self.assertEqual(source.predict_proba(vector), target.predict_proba(vector))

    def test_loading_a_mismatched_model_is_rejected(self):
        network = Network(CONFIG)
        with self.assertRaises(ValueError):
            network.load_state_dict([{}])

    def test_state_dict_is_a_snapshot_not_a_live_view(self):
        # Early stopping saves the best epoch's weights and restores them at the
        # end. If the snapshot aliased the live lists it would keep changing
        # with training and the restore would do nothing.
        network = Network(CONFIG, rng=random.Random(1))
        snapshot = network.state_dict()
        original = list(snapshot[0]["weights"][:5])

        for i in range(len(network.layers[0].weights)):
            network.layers[0].weights[i] += 1.0

        self.assertEqual(snapshot[0]["weights"][:5], original)

    def test_restoring_a_snapshot_recovers_the_earlier_model(self):
        network = Network(CONFIG, rng=random.Random(2))
        vector = [0.3] * (CONFIG.channels * CONFIG.img_size ** 2)
        before = network.predict_proba(vector)
        snapshot = network.state_dict()

        for i in range(len(network.layers[0].weights)):
            network.layers[0].weights[i] += 0.5
        self.assertNotEqual(network.predict_proba(vector), before)

        network.load_state_dict(snapshot)
        self.assertEqual(network.predict_proba(vector), before)

    def test_loading_keeps_the_optimisers_parameter_references_valid(self):
        network = Network(CONFIG, rng=random.Random(3))
        optimiser = Adam(network.parameters(), lr=0.01)
        network.load_state_dict(network.state_dict())

        self.assertIs(optimiser.params[0][0], network.layers[0].weights)

    def test_wrong_sized_weights_are_rejected(self):
        network = Network(CONFIG)
        broken = network.state_dict()
        broken[0]["weights"] = broken[0]["weights"][:-1]
        with self.assertRaises(ValueError):
            network.load_state_dict(broken)


class ArchitectureValidationTests(unittest.TestCase):
    def test_too_small_an_input_is_reported_clearly(self):
        with self.assertRaises(ValueError) as caught:
            Network(TrainingConfig(img_size=8))
        self.assertIn("too small", str(caught.exception))


class DropoutTests(unittest.TestCase):
    def test_dropout_is_disabled_at_inference_time(self):
        network = Network(CONFIG, rng=random.Random(4))
        vector = [0.5] * (CONFIG.channels * CONFIG.img_size ** 2)
        self.assertEqual(network.predict_proba(vector), network.predict_proba(vector))


if __name__ == "__main__":
    unittest.main()
