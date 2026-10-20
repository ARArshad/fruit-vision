"""Dataset discovery, splitting and augmentation.

The captured dataset is deliberately small, so almost all of the model's
robustness comes from augmentation: every source photo is expanded into many
randomly warped and re-coloured variants.  The transform is a single affine
warp (rotation + zoom + translation + horizontal flip) sampled with bilinear
interpolation, followed by brightness/colour jitter and a little noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.features import describe
from fruitvision.imageio import Image, decode_image
from fruitvision.utils.image import letterbox, to_model_input

__all__ = [
    "Sample",
    "_replace_background",
    "discover_images",
    "split_dataset",
    "build_eval_set",
    "build_training_set",
    "build_descriptor_set",
]

Vector = List[float]
Sample = Tuple[Vector, int]

# The cached square each augmented variant is warped out of.  Keeping it a few
# times larger than the network input means rotations and zooms still have real
# detail to draw on.
CACHE_SCALE = 4

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def discover_images(
    config: TrainingConfig = CONFIG, data_dir: Optional[Path] = None
) -> List[Tuple[Path, int]]:
    """Find every labelled image, returning ``(path, class_index)`` pairs.

    Class folders are named after :attr:`TrainingConfig.class_names` and are
    matched case-insensitively.  ``data_dir`` overrides the configured location,
    which is how the separate held-out test set is loaded.
    """

    if data_dir is not None:
        config = replace(config, data_dir=Path(data_dir))
    if not config.data_dir.exists():
        raise FileNotFoundError(
            "Dataset directory '%s' not found. It should contain one folder per "
            "class (%s)." % (config.data_dir, ", ".join(config.class_names))
        )

    items: List[Tuple[Path, int]] = []
    for index, class_name in enumerate(config.class_names):
        folder = config.data_dir / class_name
        if not folder.is_dir():
            matches = [
                p for p in config.data_dir.iterdir()
                if p.is_dir() and p.name.lower() == class_name.lower()
            ]
            if not matches:
                raise FileNotFoundError(
                    "No folder for class '%s' inside '%s'." % (class_name, config.data_dir)
                )
            folder = matches[0]

        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in _IMAGE_SUFFIXES and not p.name.startswith(".")
        )
        if not files:
            raise FileNotFoundError("No images found in '%s'." % folder)
        items.extend((path, index) for path in files)

    return items


def load_source(path: Path) -> Image:
    """Decode one dataset image from disk."""

    return decode_image(path.read_bytes(), fast=True)


def split_dataset(
    items: Sequence[Tuple[Path, int]],
    config: TrainingConfig = CONFIG,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Split into train/test, stratified per class and reproducible by seed.

    At least one image per class is held out whenever a class has more than one
    image, and at least one is always kept for training.
    """

    rng = random.Random(config.seed if seed is None else seed)
    by_class: Dict[int, List[Path]] = {}
    for path, label in items:
        by_class.setdefault(label, []).append(path)

    train: List[Tuple[Path, int]] = []
    test: List[Tuple[Path, int]] = []
    for label in sorted(by_class):
        paths = sorted(by_class[label])
        rng.shuffle(paths)
        n_test = int(round(len(paths) * config.test_fraction))
        n_test = max(1, n_test) if len(paths) > 1 else 0
        n_test = min(n_test, len(paths) - 1)
        test.extend((p, label) for p in paths[:n_test])
        train.extend((p, label) for p in paths[n_test:])

    return train, test


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _to_planes(image: Image) -> List[float]:
    """Flatten an image into channel-major floats in ``[0, 1]``."""

    px = image.pixels
    area = image.width * image.height
    planes = [0.0] * (3 * area)
    for i in range(area):
        j = i * 3
        planes[i] = px[j] / 255.0
        planes[area + i] = px[j + 1] / 255.0
        planes[2 * area + i] = px[j + 2] / 255.0
    return planes


def _warp(
    planes: Sequence[float],
    src_size: int,
    out_size: int,
    angle: float,
    zoom: float,
    dx: float,
    dy: float,
    flip: bool,
) -> List[float]:
    """Affine-warp a cached square into the network input size.

    Sampling is done by inverse mapping: for every output pixel we work out
    where it came from in the source and interpolate.  Anything that falls
    outside the source reads as white, matching the padding used elsewhere.
    """

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    area_src = src_size * src_size
    area_out = out_size * out_size
    out = [1.0] * (3 * area_out)
    limit = src_size - 1

    for oy in range(out_size):
        v = (2.0 * oy + 1.0) / out_size - 1.0
        for ox in range(out_size):
            u = (2.0 * ox + 1.0) / out_size - 1.0
            if flip:
                u = -u
            su = (u * cos_a + v * sin_a) / zoom + dx
            sv = (-u * sin_a + v * cos_a) / zoom + dy

            sx = (su + 1.0) * 0.5 * limit
            sy = (sv + 1.0) * 0.5 * limit
            if sx < 0.0 or sy < 0.0 or sx > limit or sy > limit:
                continue  # leave the white default

            x0 = int(sx)
            y0 = int(sy)
            x1 = x0 + 1 if x0 < limit else x0
            y1 = y0 + 1 if y0 < limit else y0
            fx = sx - x0
            fy = sy - y0
            w00 = (1.0 - fx) * (1.0 - fy)
            w10 = fx * (1.0 - fy)
            w01 = (1.0 - fx) * fy
            w11 = fx * fy

            i00 = y0 * src_size + x0
            i10 = y0 * src_size + x1
            i01 = y1 * src_size + x0
            i11 = y1 * src_size + x1
            dst = oy * out_size + ox

            for c in range(3):
                base = c * area_src
                out[c * area_out + dst] = (
                    planes[base + i00] * w00
                    + planes[base + i10] * w10
                    + planes[base + i01] * w01
                    + planes[base + i11] * w11
                )
    return out


def _replace_background(
    vector: List[float], size: int, rng: random.Random, threshold: float = 0.86
) -> List[float]:
    """Swap the plain studio background for a random one.

    Almost every Fruits-360 image is a single fruit on white, so a model trained
    on them learns "white surroundings" as part of each class and copes badly
    with real photographs. Repainting the background teaches it to rely on the
    fruit itself instead.

    Pixels are treated as background when all three channels are bright and
    close to neutral, which is exactly what the studio backdrop looks like and
    is rare inside the fruit itself.
    """

    area = size * size
    base = [rng.random(), rng.random(), rng.random()]
    # A gentle gradient plus noise stands in for a cluttered real-world scene.
    gradient = rng.uniform(-0.25, 0.25)
    noise = rng.uniform(0.0, 0.10)

    for i in range(area):
        r = vector[i]
        g = vector[area + i]
        b = vector[2 * area + i]
        if r < threshold or g < threshold or b < threshold:
            continue
        if max(r, g, b) - min(r, g, b) > 0.10:
            continue  # bright but coloured: part of the fruit, not the backdrop

        row = (i // size) / float(size)
        for c in range(3):
            value = base[c] + gradient * (row - 0.5)
            if noise:
                value += rng.gauss(0.0, noise)
            vector[c * area + i] = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
    return vector


def _jitter(
    vector: List[float], config: TrainingConfig, rng: random.Random
) -> List[float]:
    """Apply brightness, per-channel colour and noise jitter in place."""

    area = len(vector) // 3
    brightness = rng.uniform(-config.max_brightness, config.max_brightness)
    noise = config.noise_std

    for c in range(3):
        gain = 1.0 + rng.uniform(-config.max_colour_jitter, config.max_colour_jitter)
        base = c * area
        for i in range(base, base + area):
            value = vector[i] * gain + brightness
            if noise:
                value += rng.gauss(0.0, noise)
            vector[i] = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
    return vector


def build_eval_set(
    items: Sequence[Tuple[Path, int]],
    config: TrainingConfig = CONFIG,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Sample]:
    """Preprocess images exactly the way the API does -- no augmentation."""

    samples = []
    for index, (path, label) in enumerate(items, 1):
        samples.append((to_model_input(load_source(path), config), label))
        if progress is not None:
            progress(index, len(items))
    return samples


def build_training_set(
    items: Sequence[Tuple[Path, int]],
    config: TrainingConfig = CONFIG,
    rng: Optional[random.Random] = None,
    include_original: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Sample]:
    """Expand each source image into many augmented training samples.

    Variants are generated once up front rather than per epoch: with a dataset
    this small the extra diversity of re-sampling every epoch is not worth
    paying the augmentation cost repeatedly in pure Python.
    """

    rng = rng or random.Random(config.seed)
    cache_size = config.img_size * CACHE_SCALE
    samples: List[Sample] = []

    for index, (path, label) in enumerate(items, 1):
        if progress is not None:
            progress(index, len(items))
        source = load_source(path)
        if include_original:
            samples.append((to_model_input(source, config), label))

        planes = _to_planes(letterbox(source, cache_size))
        for _ in range(config.augmentations_per_image):
            angle = math.radians(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg))
            zoom = 1.0 + rng.uniform(-config.max_zoom, config.max_zoom)
            dx = rng.uniform(-config.max_shift, config.max_shift)
            dy = rng.uniform(-config.max_shift, config.max_shift)
            flip = rng.random() < 0.5
            vector = _warp(
                planes, cache_size, config.img_size, angle, zoom, dx, dy, flip
            )
            if rng.random() < config.background_swap_probability:
                vector = _replace_background(vector, config.img_size, rng)
            samples.append((_jitter(vector, config, rng), label))

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Descriptor stream
# ---------------------------------------------------------------------------

# Augmentation for the descriptor is deliberately mild.  Its features already
# ignore where the fruit sits and which way up it is, so the only variation
# worth simulating is the camera: a mirrored view and a different white
# balance or exposure.
DESCRIPTOR_GAIN = 0.14
DESCRIPTOR_BRIGHTNESS = 0.12


def _flip_horizontal(image: Image) -> Image:
    """Mirror an image left-to-right."""

    width, height = image.width, image.height
    src = image.pixels
    out = bytearray(len(src))
    for y in range(height):
        row = y * width * 3
        for x in range(width):
            s = row + (width - 1 - x) * 3
            d = row + x * 3
            out[d : d + 3] = src[s : s + 3]
    return Image(width, height, out)


def _jitter_pixels(image: Image, rng: random.Random) -> Image:
    """Re-expose and re-tint an image, standing in for a different camera."""

    gains = [1.0 + rng.uniform(-DESCRIPTOR_GAIN, DESCRIPTOR_GAIN) for _ in range(3)]
    shift = rng.uniform(-DESCRIPTOR_BRIGHTNESS, DESCRIPTOR_BRIGHTNESS) * 255.0
    src = image.pixels
    out = bytearray(len(src))
    for i in range(0, len(src), 3):
        for c in range(3):
            value = src[i + c] * gains[c] + shift
            out[i + c] = 0 if value < 0.0 else (255 if value > 255.0 else int(value))
    return Image(image.width, image.height, out)


def build_descriptor_set(
    items: Sequence[Tuple[Path, int]],
    config: TrainingConfig = CONFIG,
    rng: Optional[random.Random] = None,
    augment: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Sample]:
    """Turn labelled images into ``(descriptor, label)`` pairs.

    Each source photo contributes its own descriptor plus, when ``augment`` is
    set, a mirrored copy and a few re-exposed variants.
    """

    rng = rng or random.Random(config.seed)
    size = config.descriptor_size
    samples: List[Sample] = []

    for index, (path, label) in enumerate(items, 1):
        if progress is not None:
            progress(index, len(items))
        square = letterbox(load_source(path), size)
        samples.append((describe(square.pixels, size), label))
        if not augment:
            continue

        # PNG here means a photograph added by tools/prepare_dataset.py; the
        # bundled studio dataset is JPEG. Real-world scenes are rare and worth
        # much more than another turntable frame, so they get more variants.
        variants = (config.descriptor_photo_augmentations
                    if path.suffix.lower() == ".png"
                    else config.descriptor_augmentations)

        samples.append((describe(_flip_horizontal(square).pixels, size), label))
        for _ in range(max(0, variants - 1)):
            variant = _jitter_pixels(square, rng)
            if rng.random() < 0.5:
                variant = _flip_horizontal(variant)
            samples.append((describe(variant.pixels, size), label))

    return samples
