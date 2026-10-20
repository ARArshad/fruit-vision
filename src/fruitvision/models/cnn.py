"""A convolutional neural network written from scratch in pure Python.

This is a complete, trainable CNN -- forward pass, backpropagation and an Adam
optimiser -- implemented with nothing but the standard library.  It replaces
TensorFlow/Keras so the project can be unzipped and run anywhere without a
single ``pip install``.

Tensor layout
-------------
Feature maps are flat :class:`list` objects of floats in channel-major order::

    index = channel * height * width + row * width + column

Performance
-----------
A scalar Python loop over every multiply-accumulate would be far too slow.
Instead the convolutions are expressed as *shifted slice accumulations*: for
each of the nine kernel taps, an entire output row is updated in one list
comprehension over a slice of the input row.  The arithmetic then runs at the
interpreter's C-level list speed rather than one bytecode dispatch per
multiply, which is roughly an order of magnitude faster.
"""

from __future__ import annotations

import math
import random
from operator import mul
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Conv2D", "MaxPool2D", "Dense", "ReLU", "Dropout",
    "Network", "Adam", "softmax", "softmax_cross_entropy",
]

Tensor = List[float]
Shape = Tuple[int, int, int]  # (channels, height, width)


def _load_into(target: List[float], values: Sequence[float], what: str) -> None:
    """Copy saved values into an existing parameter list, in place.

    Assigning a fresh list would silently orphan any optimiser state already
    holding a reference to the old one, so the contents are replaced instead.
    """

    if len(values) != len(target):
        raise ValueError(
            "%s has %d values but the model expects %d -- the saved model does "
            "not match this architecture." % (what, len(values), len(target))
        )
    target[:] = [float(v) for v in values]


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class Layer:
    """Base class: a layer transforms a tensor and can backpropagate through it."""

    trainable: bool = False

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> List[Tuple[List[float], List[float]]]:
        """Return ``(values, gradients)`` pairs for the optimiser."""

        return []

    def zero_grad(self) -> None:
        for values, grads in self.parameters():
            for i in range(len(grads)):
                grads[i] = 0.0

    def state_dict(self) -> Dict[str, object]:
        return {}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        pass


class Conv2D(Layer):
    """2-D convolution with a square kernel, unit stride and no padding."""

    trainable = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        in_h: int,
        in_w: int,
        rng: Optional[random.Random] = None,
        needs_input_grad: bool = True,
    ) -> None:
        self.in_c = in_channels
        self.out_c = out_channels
        self.k = kernel_size
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = in_h - kernel_size + 1
        self.out_w = in_w - kernel_size + 1
        self.needs_input_grad = needs_input_grad

        # He initialisation keeps activations well-scaled through ReLUs.
        fan_in = in_channels * kernel_size * kernel_size
        std = math.sqrt(2.0 / fan_in)
        rng = rng or random.Random(0)
        n_weights = out_channels * fan_in
        self.weights = [rng.gauss(0.0, std) for _ in range(n_weights)]
        self.biases = [0.0] * out_channels
        self.grad_weights = [0.0] * n_weights
        self.grad_biases = [0.0] * out_channels
        self._input: Tensor = []

    @property
    def output_shape(self) -> Shape:
        return (self.out_c, self.out_h, self.out_w)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        self._input = x
        in_w, k = self.in_w, self.k
        out_h, out_w = self.out_h, self.out_w
        plane_in = self.in_h * in_w
        taps = self.in_c * k * k
        weights = self.weights

        out: Tensor = []
        for co in range(self.out_c):
            bias = self.biases[co]
            rows = [[bias] * out_w for _ in range(out_h)]
            wbase = co * taps
            for ci in range(self.in_c):
                offset = ci * plane_in
                for ky in range(k):
                    row_base = offset + ky * in_w
                    for kx in range(k):
                        weight = weights[wbase + (ci * k + ky) * k + kx]
                        if weight == 0.0:
                            continue
                        for oy in range(out_h):
                            src = row_base + oy * in_w + kx
                            window = x[src : src + out_w]
                            row = rows[oy]
                            rows[oy] = [
                                a + weight * b for a, b in zip(row, window)
                            ]
            for row in rows:
                out.extend(row)
        return out

    def backward(self, grad: Tensor) -> Tensor:
        x = self._input
        in_w, k = self.in_w, self.k
        out_h, out_w = self.out_h, self.out_w
        plane_in = self.in_h * in_w
        plane_out = out_h * out_w
        taps = self.in_c * k * k

        grad_x = [0.0] * (self.in_c * plane_in) if self.needs_input_grad else []

        for co in range(self.out_c):
            gbase = co * plane_out
            wbase = co * taps
            self.grad_biases[co] += sum(grad[gbase : gbase + plane_out])

            for ci in range(self.in_c):
                offset = ci * plane_in
                for ky in range(k):
                    row_base = offset + ky * in_w
                    for kx in range(k):
                        widx = wbase + (ci * k + ky) * k + kx
                        weight = self.weights[widx]
                        total = 0.0
                        for oy in range(out_h):
                            src = row_base + oy * in_w + kx
                            dst = gbase + oy * out_w
                            g_row = grad[dst : dst + out_w]
                            # dL/dW: correlate the incoming gradient with the input.
                            # map(mul, ...) keeps this loop entirely in C.
                            total += sum(map(mul, g_row, x[src : src + out_w]))

                            if self.needs_input_grad and weight != 0.0:
                                # dL/dx: scatter the gradient back through the tap.
                                target = grad_x[src : src + out_w]
                                grad_x[src : src + out_w] = [
                                    t + weight * g for t, g in zip(target, g_row)
                                ]
                        self.grad_weights[widx] += total
        return grad_x

    def parameters(self):
        return [(self.weights, self.grad_weights), (self.biases, self.grad_biases)]

    def state_dict(self):
        # Copies, not references: a snapshot taken mid-training (for example to
        # restore the best epoch afterwards) must not keep changing with it.
        return {"weights": list(self.weights), "biases": list(self.biases)}

    def load_state_dict(self, state):
        _load_into(self.weights, state["weights"], "Conv2D weights")
        _load_into(self.biases, state["biases"], "Conv2D biases")


class MaxPool2D(Layer):
    """2x2 max pooling with stride 2 (odd trailing rows/columns are dropped)."""

    def __init__(self, channels: int, in_h: int, in_w: int) -> None:
        self.c = channels
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = in_h // 2
        self.out_w = in_w // 2
        self._argmax: List[int] = []

    @property
    def output_shape(self) -> Shape:
        return (self.c, self.out_h, self.out_w)

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        in_w = self.in_w
        out: Tensor = [0.0] * (self.c * self.out_h * self.out_w)
        argmax = [0] * len(out)
        pos = 0
        for ch in range(self.c):
            plane = ch * self.in_h * in_w
            for oy in range(self.out_h):
                base = plane + (oy * 2) * in_w
                for ox in range(self.out_w):
                    i = base + ox * 2
                    best = i
                    value = x[i]
                    for cand in (i + 1, i + in_w, i + in_w + 1):
                        if x[cand] > value:
                            value = x[cand]
                            best = cand
                    out[pos] = value
                    argmax[pos] = best
                    pos += 1
        self._argmax = argmax
        return out

    def backward(self, grad: Tensor) -> Tensor:
        grad_x = [0.0] * (self.c * self.in_h * self.in_w)
        argmax = self._argmax
        for i, g in enumerate(grad):
            if g:
                grad_x[argmax[i]] += g
        return grad_x


class ReLU(Layer):
    def __init__(self) -> None:
        self._mask: Tensor = []

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        out = [v if v > 0.0 else 0.0 for v in x]
        self._mask = out
        return out

    def backward(self, grad: Tensor) -> Tensor:
        mask = self._mask
        return [g if mask[i] > 0.0 else 0.0 for i, g in enumerate(grad)]


class Dropout(Layer):
    """Inverted dropout: active during training, a no-op at inference time."""

    def __init__(self, rate: float, rng: Optional[random.Random] = None) -> None:
        self.rate = rate
        self.rng = rng or random.Random(0)
        self._mask: Optional[List[float]] = None

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        if not training or self.rate <= 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.rate
        scale = 1.0 / keep
        random_ = self.rng.random
        mask = [scale if random_() < keep else 0.0 for _ in x]
        self._mask = mask
        return [v * m for v, m in zip(x, mask)]

    def backward(self, grad: Tensor) -> Tensor:
        if self._mask is None:
            return grad
        return [g * m for g, m in zip(grad, self._mask)]


class Dense(Layer):
    """Fully connected layer."""

    trainable = True

    def __init__(
        self, in_features: int, out_features: int, rng: Optional[random.Random] = None
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / in_features)
        rng = rng or random.Random(0)
        self.weights = [rng.gauss(0.0, std) for _ in range(in_features * out_features)]
        self.biases = [0.0] * out_features
        self.grad_weights = [0.0] * (in_features * out_features)
        self.grad_biases = [0.0] * out_features
        self._input: Tensor = []

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        self._input = x
        n = self.in_features
        weights = self.weights
        out = [0.0] * self.out_features
        for o in range(self.out_features):
            base = o * n
            out[o] = sum(map(mul, weights[base : base + n], x)) + self.biases[o]
        return out

    def backward(self, grad: Tensor) -> Tensor:
        x = self._input
        n = self.in_features
        grad_x = [0.0] * n
        for o in range(self.out_features):
            g = grad[o]
            self.grad_biases[o] += g
            if g == 0.0:
                continue
            base = o * n
            gw = self.grad_weights
            gw[base : base + n] = [
                acc + g * xv for acc, xv in zip(gw[base : base + n], x)
            ]
            row = self.weights[base : base + n]
            grad_x = [gx + g * w for gx, w in zip(grad_x, row)]
        return grad_x

    def parameters(self):
        return [(self.weights, self.grad_weights), (self.biases, self.grad_biases)]

    def state_dict(self):
        return {"weights": list(self.weights), "biases": list(self.biases)}

    def load_state_dict(self, state):
        _load_into(self.weights, state["weights"], "Dense weights")
        _load_into(self.biases, state["biases"], "Dense biases")


# ---------------------------------------------------------------------------
# Loss and optimiser
# ---------------------------------------------------------------------------


def softmax(logits: Sequence[float]) -> List[float]:
    """Numerically stable softmax."""

    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax_cross_entropy(
    logits: Sequence[float], target: int
) -> Tuple[float, List[float]]:
    """Return the loss and the gradient with respect to the logits."""

    probs = softmax(logits)
    loss = -math.log(max(probs[target], 1e-12))
    grad = list(probs)
    grad[target] -= 1.0
    return loss, grad


class Adam:
    """Adam optimiser with decoupled weight decay (AdamW-style)."""

    def __init__(
        self,
        params: List[Tuple[List[float], List[float]]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [[0.0] * len(values) for values, _ in params]
        self.v = [[0.0] * len(values) for values, _ in params]

    def step(self, scale: float = 1.0) -> None:
        """Apply one update. ``scale`` divides gradients (e.g. by batch size)."""

        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        bias1 = 1.0 - b1 ** self.step_count
        bias2 = 1.0 - b2 ** self.step_count
        lr = self.lr

        for idx, (values, grads) in enumerate(self.params):
            m, v = self.m[idx], self.v[idx]
            decay = self.weight_decay
            for i in range(len(values)):
                g = grads[i] * scale
                if decay:
                    values[i] -= lr * decay * values[i]
                m[i] = b1 * m[i] + (1.0 - b1) * g
                v[i] = b2 * v[i] + (1.0 - b2) * g * g
                values[i] -= lr * (m[i] / bias1) / (math.sqrt(v[i] / bias2) + self.eps)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class Network:
    """The FruitVision CNN: conv -> pool -> conv -> pool -> dense -> softmax."""

    def __init__(self, config, rng: Optional[random.Random] = None) -> None:
        rng = rng or random.Random(config.seed)
        size = config.img_size
        k = config.kernel_size

        conv1 = Conv2D(
            config.channels, config.conv1_filters, k, size, size,
            rng=rng, needs_input_grad=False,
        )
        pool1 = MaxPool2D(*conv1.output_shape)
        conv2 = Conv2D(
            pool1.output_shape[0], config.conv2_filters, k,
            pool1.output_shape[1], pool1.output_shape[2], rng=rng,
        )
        pool2 = MaxPool2D(*conv2.output_shape)
        flat = pool2.output_shape[0] * pool2.output_shape[1] * pool2.output_shape[2]

        if pool2.output_shape[1] < 1 or pool2.output_shape[2] < 1:
            raise ValueError(
                "img_size=%d is too small for this architecture: the feature map "
                "collapses to nothing after two %dx%d convolutions and two pooling "
                "layers. Use img_size >= %d."
                % (size, k, k, 2 * (2 * 1 + k - 1) + k - 1)
            )

        self.layers: List[Layer] = [
            conv1, ReLU(), pool1,
            conv2, ReLU(), pool2,
            Dense(flat, config.dense_units, rng=rng), ReLU(),
            Dropout(config.dropout, rng=rng),
            Dense(config.dense_units, len(config.class_names), rng=rng),
        ]
        self.config = config
        self.flat_features = flat

    # -- inference ---------------------------------------------------------

    def forward(self, x: Tensor, training: bool = False) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def predict_proba(self, x: Tensor) -> List[float]:
        return softmax(self.forward(x, training=False))

    # -- training ----------------------------------------------------------

    def backward(self, grad: Tensor) -> None:
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

    def state_dict(self) -> List[Dict[str, object]]:
        return [layer.state_dict() for layer in self.layers]

    def load_state_dict(self, states: List[Dict[str, object]]) -> None:
        if len(states) != len(self.layers):
            raise ValueError(
                "Saved model has %d layers but this architecture has %d. "
                "The model file does not match the current configuration."
                % (len(states), len(self.layers))
            )
        for layer, state in zip(self.layers, states):
            if state:
                layer.load_state_dict(state)
