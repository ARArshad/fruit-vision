"""Configuration for the FruitVision classifier.

Every tunable value the training and serving code needs lives here, so the
model architecture, the preprocessing and the API all stay in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Repository root, derived from this file's location so the project works no
# matter which directory it is launched from.
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainingConfig:
    """Hyper-parameters, architecture and paths for the fruit classifier."""

    # --- Data -------------------------------------------------------------
    data_dir: Path = ROOT / "data" / "raw" / "custom"
    # Optional held-out set. When this directory exists its images are used as
    # the test set instead of carving one out of the training data -- important
    # for Fruits-360, where Training and Test contain *different physical
    # fruits*, so a random split would leak near-identical rotations.
    test_dir: Path = ROOT / "data" / "raw" / "test"
    output_dir: Path = ROOT / "artifacts"
    class_names: List[str] = field(
        default_factory=lambda: [
            "Apple", "Avocado", "Banana", "Cherry",
            "Grape", "Kiwi", "Lemon", "Lime",
            "Mango", "Orange", "Peach", "Pear",
            "Pineapple", "Pomegranate", "Strawberry", "Watermelon",
        ]
    )

    # --- Preprocessing ----------------------------------------------------
    img_size: int = 32          # network input is img_size x img_size RGB
    channels: int = 3

    # --- Architecture -----------------------------------------------------
    conv1_filters: int = 8
    conv2_filters: int = 24
    kernel_size: int = 3
    dense_units: int = 64
    dropout: float = 0.30

    # --- Descriptor classifier -------------------------------------------
    # The second stream: a hand-written colour/shape/texture descriptor
    # (fruitvision.features) fed to a small fully connected network. It carries
    # most of the accuracy, because measured colour and shape beat raw 32x32
    # pixels when there are only about a thousand training photographs.
    descriptor_size: int = 48       # the descriptor is measured at this resolution
    descriptor_hidden: int = 64
    descriptor_epochs: int = 35
    descriptor_lr: float = 0.02
    descriptor_weight_decay: float = 1e-4
    descriptor_dropout: float = 0.10
    descriptor_batch_size: int = 16
    # Variants generated per source image (flips and colour jitter).
    descriptor_augmentations: int = 3
    # Photographs added with tools/prepare_dataset.py -- always written as PNG,
    # unlike the JPEG studio dataset -- are the only real-world scenes in the
    # training data and there are very few of them, so they are repeated far
    # more often. Without this they are 1% of the samples and get ignored.
    descriptor_photo_augmentations: int = 12

    # How much the pixel CNN counts when the two streams are combined. The
    # descriptor is much the stronger of the two, so the CNN acts as a
    # tie-breaker: it barely changes which fruit wins, but it keeps the
    # reported confidence honest instead of pinned at 99%.
    cnn_weight: float = 0.25

    # --- Optimisation -----------------------------------------------------
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 42

    # --- Augmentation -----------------------------------------------------
    augmentations_per_image: int = 1
    max_rotation_deg: float = 18.0
    max_zoom: float = 0.35
    max_shift: float = 0.20
    max_brightness: float = 0.25
    max_colour_jitter: float = 0.12
    noise_std: float = 0.02
    # Fraction of augmented variants that get their plain studio background
    # repainted, so the model does not learn "white surroundings" as a feature.
    background_swap_probability: float = 0.65

    # Fraction of the (tiny) dataset held out for the honest test split.
    test_fraction: float = 0.25

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.channels, self.img_size, self.img_size)

    @property
    def model_path(self) -> Path:
        return self.output_dir / "model.json"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "training_metrics.json"


CONFIG = TrainingConfig()
