"""Image preprocessing: resizing, letterboxing and model-input preparation.

The exact same functions are used when building the training set and when
serving a prediction, so a photo is treated identically in both paths.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.imageio import Image, decode_image

__all__ = [
    "resize",
    "letterbox",
    "to_model_input",
    "preprocess_upload",
    "load_image_from_bytes",
]

WHITE = (255, 255, 255)


def resize(image: Image, out_w: int, out_h: int) -> Image:
    """Resize using box (area-average) sampling.

    Averaging every source pixel that falls inside an output pixel is the right
    filter for the large downscales this project performs -- it keeps colour
    fidelity, which is the dominant signal for telling an apple from a banana.
    """

    if out_w == image.width and out_h == image.height:
        return Image(image.width, image.height, bytearray(image.pixels))
    if out_w < 1 or out_h < 1:
        raise ValueError("resize target must be at least 1x1")

    src = image.pixels
    w, h = image.width, image.height
    out = bytearray(out_w * out_h * 3)

    # Pre-compute the source span covered by each output column.
    x_spans = []
    for col in range(out_w):
        x0 = col * w // out_w
        x1 = max(x0 + 1, (col + 1) * w // out_w)
        x_spans.append((x0, min(x1, w)))

    pos = 0
    for row in range(out_h):
        y0 = row * h // out_h
        y1 = min(max(y0 + 1, (row + 1) * h // out_h), h)
        for x0, x1 in x_spans:
            r = g = b = 0
            count = 0
            for y in range(y0, y1):
                base = (y * w + x0) * 3
                for _ in range(x0, x1):
                    r += src[base]
                    g += src[base + 1]
                    b += src[base + 2]
                    base += 3
                    count += 1
            out[pos] = r // count
            out[pos + 1] = g // count
            out[pos + 2] = b // count
            pos += 3

    return Image(out_w, out_h, out)


def letterbox(image: Image, size: int, background: Sequence[int] = WHITE) -> Image:
    """Scale to fit a ``size`` x ``size`` square, padding to keep the aspect ratio.

    Aspect ratio carries real information here: a banana is elongated and an
    apple is round.  Squashing every upload into a square would throw that cue
    away, so the image is scaled to fit and the remainder is padded instead.
    """

    scale = size / float(max(image.width, image.height))
    new_w = max(1, min(size, int(round(image.width * scale))))
    new_h = max(1, min(size, int(round(image.height * scale))))
    scaled = resize(image, new_w, new_h)

    canvas = bytearray(bytes(background) * (size * size))
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    row_bytes = new_w * 3
    for row in range(new_h):
        dst = ((off_y + row) * size + off_x) * 3
        src = row * row_bytes
        canvas[dst : dst + row_bytes] = scaled.pixels[src : src + row_bytes]

    return Image(size, size, canvas)


def to_model_input(image: Image, config: TrainingConfig = CONFIG) -> List[float]:
    """Convert an image into the flat float vector the network consumes.

    The layout is channel-major (all red, then all green, then all blue), which
    is what the convolution loops in :mod:`fruitvision.models.cnn` expect.
    """

    size = config.img_size
    square = letterbox(image, size)
    px = square.pixels
    area = size * size

    planes = [0.0] * (3 * area)
    for i in range(area):
        j = i * 3
        planes[i] = px[j] / 255.0
        planes[area + i] = px[j + 1] / 255.0
        planes[2 * area + i] = px[j + 2] / 255.0
    return planes


def load_image_from_bytes(image_bytes: bytes, fast: bool = True) -> Image:
    """Decode raw uploaded bytes into an :class:`Image`."""

    return decode_image(image_bytes, fast=fast)


def preprocess_upload(
    image_bytes: bytes,
    config: TrainingConfig = CONFIG,
) -> Tuple[List[float], Tuple[int, int]]:
    """Prepare an uploaded image for prediction.

    Returns the model-ready vector plus the decoded source dimensions.
    """

    image = load_image_from_bytes(image_bytes)
    return to_model_input(image, config), image.size
