"""A hand-written colour, shape and texture descriptor for a fruit photo.

Why this exists
---------------
The convolutional network in :mod:`fruitvision.models.cnn` has to learn what a
fruit looks like from roughly a thousand photographs, in pure Python, at
32x32 pixels.  That is not much to go on, and it shows: it confuses apples with
peaches and pears with mangoes.

The things that actually separate one fruit from another -- the colour of the
skin, how round or elongated it is, how rough the surface looks -- can be
measured directly instead of learned from scratch.  This module turns an image
into a compact vector of exactly those measurements, and a small classifier
trained on that vector is far more accurate than the raw-pixel network on this
size of dataset.

The descriptor
--------------
Everything is measured over the *fruit* rather than the whole frame: the
background is estimated from the border of the picture and excluded, so a
photograph on a white studio backdrop and the same fruit on a wooden table
produce similar numbers.

======================  =====  ==================================================
Block                   Size   What it captures
======================  =====  ==================================================
Hue histogram             16   the fruit's colours (red apple vs yellow banana)
Saturation histogram       5   vivid (lime) vs muted (avocado)
Value histogram            5   dark (pomegranate) vs bright (lemon)
RGB mean and spread        6   overall colour and how uniform it is
Shape                      6   how much of the frame is fruit, roundness, aspect
Grid summary              80   a 4x4 map of colour and coverage: stripes, layout
Texture                    2   edge energy: smooth apple vs rough pineapple
Colour coherence           4   how single-coloured the fruit is overall
======================  =====  ==================================================

That is :data:`DESCRIPTOR_LENGTH` numbers per image.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.imageio import Image
from fruitvision.utils.image import letterbox

__all__ = [
    "HUE_BINS",
    "GRID",
    "DESCRIPTOR_LENGTH",
    "describe",
    "describe_image",
    "rgb_to_hsv",
]

HUE_BINS = 16
GRID = 4
DESCRIPTOR_LENGTH = HUE_BINS + 5 + 5 + 6 + 6 + GRID * GRID * 5 + 2 + 4

Vector = List[float]


def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert one RGB triple in ``[0, 1]`` to hue, saturation and value.

    Hue is returned in ``[0, 1)`` rather than degrees so every feature in the
    descriptor lives on a comparable scale.
    """

    mx = r if r > g else g
    if b > mx:
        mx = b
    mn = r if r < g else g
    if b < mn:
        mn = b

    delta = mx - mn
    saturation = 0.0 if mx <= 0.0 else delta / mx
    if delta <= 1e-9:
        hue = 0.0
    elif mx == r:
        hue = ((g - b) / delta) % 6.0
    elif mx == g:
        hue = (b - r) / delta + 2.0
    else:
        hue = (r - g) / delta + 4.0
    return hue / 6.0, saturation, mx


def _hsv_planes(pixels: Sequence[int], size: int):
    """Convert a square RGB byte buffer into three flat HSV planes."""

    count = size * size
    hue = [0.0] * count
    sat = [0.0] * count
    val = [0.0] * count
    j = 0
    for i in range(count):
        h, s, v = rgb_to_hsv(pixels[j] / 255.0, pixels[j + 1] / 255.0, pixels[j + 2] / 255.0)
        hue[i] = h
        sat[i] = s
        val[i] = v
        j += 3
    return hue, sat, val


def _foreground_mask(hue, sat, val, size: int) -> Tuple[List[int], int]:
    """Mark which pixels belong to the fruit rather than the background.

    The colour of the backdrop is read off the border of the picture -- whatever
    dominates the outer frame is assumed to be background, which covers a white
    studio sweep, a wooden worktop or a plain wall equally well.  Pixels close
    to that colour are dropped.

    If the estimate is useless (almost everything or almost nothing looks like
    background, which happens on a busy close-up) the mask falls back to a
    central oval, because a photograph of a fruit almost always has the fruit in
    the middle.
    """

    count = size * size
    band = max(1, size // 12)
    border = [
        y * size + x
        for y in range(size)
        for x in range(size)
        if y < band or y >= size - band or x < band or x >= size - band
    ]

    sats = sorted(sat[i] for i in border)
    vals = sorted(val[i] for i in border)
    back_sat = sats[len(sats) // 2]
    back_val = vals[len(vals) // 2]
    # Hues are angles, so they are averaged as unit vectors weighted by
    # saturation -- a grey border has no meaningful hue to average.
    cx = sum(math.cos(2.0 * math.pi * hue[i]) * sat[i] for i in border)
    cy = sum(math.sin(2.0 * math.pi * hue[i]) * sat[i] for i in border)
    back_hue = (math.atan2(cy, cx) / (2.0 * math.pi)) % 1.0

    mask = [1] * count
    foreground = count
    for i in range(count):
        if abs(val[i] - back_val) >= 0.16 or abs(sat[i] - back_sat) >= 0.16:
            continue
        if back_sat >= 0.12:
            gap = abs(hue[i] - back_hue)
            if min(gap, 1.0 - gap) >= 0.08:
                continue
        mask[i] = 0
        foreground -= 1

    if foreground < 0.06 * count or foreground > 0.97 * count:
        mask = [0] * count
        foreground = 0
        radius = size * 0.42
        centre = (size - 1) / 2.0
        for y in range(size):
            for x in range(size):
                if (x - centre) ** 2 + (y - centre) ** 2 <= radius * radius:
                    mask[y * size + x] = 1
                    foreground += 1

    return mask, foreground


def describe(pixels: Sequence[int], size: int) -> Vector:
    """Measure a square RGB image, returning :data:`DESCRIPTOR_LENGTH` numbers."""

    count = size * size
    hue, sat, val = _hsv_planes(pixels, size)
    mask, foreground = _foreground_mask(hue, sat, val, size)
    inv = 1.0 / max(foreground, 1)

    hue_hist = [0.0] * HUE_BINS
    sat_hist = [0.0] * 5
    val_hist = [0.0] * 5
    sum_r = sum_g = sum_b = 0.0
    sum_r2 = sum_g2 = sum_b2 = 0.0
    hue_x = hue_y = 0.0
    x0, y0, x1, y1 = size, size, -1, -1

    for i in range(count):
        if not mask[i]:
            continue
        s = sat[i]
        v = val[i]
        # Weighting the hue histogram by saturation and brightness stops a few
        # dark or washed-out pixels from voting for a colour that is not there.
        hue_hist[int(hue[i] * HUE_BINS) % HUE_BINS] += s * v
        sat_hist[min(4, int(s * 5))] += 1.0
        val_hist[min(4, int(v * 5))] += 1.0
        angle = 2.0 * math.pi * hue[i]
        hue_x += math.cos(angle) * s
        hue_y += math.sin(angle) * s

        j = i * 3
        r = pixels[j] / 255.0
        g = pixels[j + 1] / 255.0
        b = pixels[j + 2] / 255.0
        sum_r += r
        sum_g += g
        sum_b += b
        sum_r2 += r * r
        sum_g2 += g * g
        sum_b2 += b * b

        y, x = divmod(i, size)
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y

    total_hue = sum(hue_hist) or 1.0
    hue_hist = [h / total_hue for h in hue_hist]
    sat_hist = [c * inv for c in sat_hist]
    val_hist = [c * inv for c in val_hist]

    mean_r, mean_g, mean_b = sum_r * inv, sum_g * inv, sum_b * inv
    std_r = math.sqrt(max(0.0, sum_r2 * inv - mean_r * mean_r))
    std_g = math.sqrt(max(0.0, sum_g2 * inv - mean_g * mean_g))
    std_b = math.sqrt(max(0.0, sum_b2 * inv - mean_b * mean_b))

    if x1 < x0:  # empty mask: treat the whole frame as the fruit
        x0, y0, x1, y1 = 0, 0, size - 1, size - 1
    box_w = x1 - x0 + 1
    box_h = y1 - y0 + 1
    aspect = box_w / float(box_h)
    shape = [
        foreground / float(count),          # how much of the frame is fruit
        min(aspect, 1.0 / aspect),          # elongation, regardless of direction
        1.0 if aspect >= 1.0 else 0.0,      # ...and which way round it lies
        foreground / float(box_w * box_h),  # how completely it fills its box
        box_w / float(size),
        box_h / float(size),
    ]

    # A coarse 4x4 map over the fruit's bounding box.  Averaged colour per cell
    # is what tells a striped watermelon from a plain lime, and the coverage of
    # each cell sketches the silhouette (a banana leaves its corners empty).
    grid: Vector = []
    for gy in range(GRID):
        ay0 = y0 + box_h * gy // GRID
        ay1 = max(ay0 + 1, y0 + box_h * (gy + 1) // GRID)
        for gx in range(GRID):
            ax0 = x0 + box_w * gx // GRID
            ax1 = max(ax0 + 1, x0 + box_w * (gx + 1) // GRID)
            cells = 0
            cell_x = cell_y = cell_s = cell_v = 0.0
            for y in range(ay0, ay1):
                base = y * size
                for x in range(ax0, ax1):
                    i = base + x
                    if not mask[i]:
                        continue
                    cells += 1
                    angle = 2.0 * math.pi * hue[i]
                    cell_x += math.cos(angle) * sat[i]
                    cell_y += math.sin(angle) * sat[i]
                    cell_s += sat[i]
                    cell_v += val[i]
            if cells:
                k = 1.0 / cells
                area = float((ay1 - ay0) * (ax1 - ax0))
                grid.extend([cell_x * k, cell_y * k, cell_s * k, cell_v * k, cells / area])
            else:
                grid.extend([0.0, 0.0, 0.0, 0.0, 0.0])

    # Texture: how much the brightness changes from pixel to pixel inside the
    # fruit.  A pineapple or a strawberry is busy, an apple is smooth.
    edge_sum = 0.0
    edge_count = 0
    strong_edges = 0
    for y in range(1, size - 1):
        base = y * size
        for x in range(1, size - 1):
            i = base + x
            if not mask[i]:
                continue
            magnitude = abs(val[i + 1] - val[i - 1]) + abs(val[i + size] - val[i - size])
            edge_sum += magnitude
            edge_count += 1
            if magnitude > 0.25:
                strong_edges += 1
    texture = [edge_sum / max(edge_count, 1), strong_edges / float(max(edge_count, 1))]

    # How tightly the hues cluster: 1.0 for a single-coloured orange, lower for
    # a two-tone peach or a striped watermelon.
    coherence = math.sqrt(hue_x * hue_x + hue_y * hue_y) * inv
    mean_sat = sum(sat[i] for i in range(count) if mask[i]) * inv
    mean_val = sum(val[i] for i in range(count) if mask[i]) * inv

    return (
        hue_hist + sat_hist + val_hist
        + [mean_r, mean_g, mean_b, std_r, std_g, std_b]
        + shape + grid + texture
        + [coherence, mean_sat, mean_val, std_r + std_g + std_b]
    )


def describe_image(image: Image, config: TrainingConfig = CONFIG) -> Vector:
    """Letterbox an image to the descriptor's working size and measure it."""

    size = config.descriptor_size
    return describe(letterbox(image, size).pixels, size)
