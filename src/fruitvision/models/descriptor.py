"""The descriptor classifier: a small fully connected network.

This is the second half of the model.  Where :mod:`fruitvision.models.cnn`
looks at raw pixels, this network is fed the measured colour/shape/texture
vector produced by :mod:`fruitvision.features`, which is a far easier problem
to learn from a small dataset.

It is built out of the very same hand-written layers as the CNN -- there is
still no library doing the maths anywhere in this project.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

from fruitvision.features import DESCRIPTOR_LENGTH
from fruitvision.models.cnn import Dense, Dropout, Layer, ReLU, softmax

__all__ = ["DescriptorNet", "standardiser"]

Vector = List[float]


def standardiser(vectors: Sequence[Sequence[float]]):
    """Return the per-feature ``(mean, standard deviation)`` of a training set.

    The descriptor mixes histogram fractions, pixel counts and edge energies,
    which live on wildly different scales.  Centring and rescaling each feature
    lets one learning rate suit all of them.
    """

    count = len(vectors)
    if not count:
        raise ValueError("Cannot standardise an empty set of feature vectors.")
    width = len(vectors[0])

    mean = [0.0] * width
    for row in vectors:
        for i, value in enumerate(row):
            mean[i] += value
    mean = [m / count for m in mean]

    variance = [0.0] * width
    for row in vectors:
        for i, value in enumerate(row):
            delta = value - mean[i]
            variance[i] += delta * delta
    # A constant feature would divide by zero, so its scale is left at 1.
    std = [math.sqrt(v / count) for v in variance]
    std = [s if s > 1e-6 else 1.0 for s in std]
    return mean, std


class DescriptorNet:
    """``descriptor -> standardise -> dense -> ReLU -> dropout -> dense``."""

    def __init__(self, config, rng: Optional[random.Random] = None) -> None:
        rng = rng or random.Random(config.seed)
        width = DESCRIPTOR_LENGTH
        hidden = config.descriptor_hidden

        self.config = config
        self.mean: Vector = [0.0] * width
        self.std: Vector = [1.0] * width
        self.layers: List[Layer] = [
            Dense(width, hidden, rng=rng),
            ReLU(),
            Dropout(config.descriptor_dropout, rng=rng),
            Dense(hidden, len(config.class_names), rng=rng),
        ]

    # -- preprocessing -----------------------------------------------------

    def set_standardisation(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = [float(v) for v in mean]
        self.std = [float(v) if abs(float(v)) > 1e-6 else 1.0 for v in std]

    def normalise(self, vector: Sequence[float]) -> Vector:
        return [(v - m) / s for v, m, s in zip(vector, self.mean, self.std)]

    # -- inference ---------------------------------------------------------

    def forward(self, vector: Sequence[float], training: bool = False) -> Vector:
        x: Vector = list(vector)
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def predict_proba(self, descriptor: Sequence[float]) -> Vector:
        """Class probabilities for a raw (un-normalised) descriptor."""

        return softmax(self.forward(self.normalise(descriptor)))

    # -- training ----------------------------------------------------------

    def backward(self, grad: Vector) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def num_parameters(self) -> int:
        return sum(len(values) for values, _ in self.parameters())

    # -- persistence -------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {
            "size": self.config.descriptor_size,
            "hidden": self.config.descriptor_hidden,
            "features": DESCRIPTOR_LENGTH,
            "mean": list(self.mean),
            "std": list(self.std),
            "layers": [layer.state_dict() for layer in self.layers],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        saved = int(state.get("features", DESCRIPTOR_LENGTH))
        if saved != DESCRIPTOR_LENGTH:
            raise ValueError(
                "The saved descriptor has %d features but this build produces %d. "
                "Retrain with 'python train.py' after changing fruitvision.features."
                % (saved, DESCRIPTOR_LENGTH)
            )
        self.set_standardisation(state["mean"], state["std"])
        for layer, layer_state in zip(self.layers, state["layers"]):
            if layer_state:
                layer.load_state_dict(layer_state)
