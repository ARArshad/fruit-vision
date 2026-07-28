"""Configuration definitions for the FruitVision project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    """Hyperparameters and paths required to train the classifier."""

    data_dir: Path = Path("data/raw/fruit-360")
    output_dir: Path = Path("artifacts")
    img_height: int = 100
    img_width: int = 100
    channels: int = 3
    batch_size: int = 32
    epochs: int = 15
    validation_split: float = 0.2
    learning_rate: float = 1e-3
    seed: int = 42
    class_names: List[str] = field(
        default_factory=lambda: [
            "Apple Red 1",
            "Banana",
            "Orange",
            "Mango",
            "Pear",
            "Pineapple",
            "Strawberry",
            "Watermelon",
        ]
    )


CONFIG = TrainingConfig()
