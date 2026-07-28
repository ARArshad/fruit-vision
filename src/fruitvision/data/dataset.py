"""Dataset loading utilities for FruitVision."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

from fruitvision.config import TrainingConfig

AUTOTUNE = tf.data.AUTOTUNE


def _assert_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory '{data_dir}' not found. Please download and extract the Fruit-360 dataset."
        )


def load_datasets(config: TrainingConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load training and validation datasets from directory structure.

    Expects the Fruit-360 dataset to be organized with subfolders per class:

    ```
    data/raw/fruit-360/
        Training/
            Apple Red 1/
                image_1.jpg
                ...
        Validation/
            Apple Red 1/
                image_1.jpg
                ...
    ```
    """

    training_path = config.data_dir / "Training"
    validation_path = config.data_dir / "Validation"

    _assert_data_dir(training_path)
    _assert_data_dir(validation_path)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        training_path,
        labels="inferred",
        label_mode="categorical",
        seed=config.seed,
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
    )

    train_class_names = getattr(train_ds, "class_names", None)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,
        labels="inferred",
        label_mode="categorical",
        seed=config.seed,
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
    )

    val_class_names = getattr(val_ds, "class_names", None)

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if train_class_names is not None:
        setattr(train_ds, "class_names", train_class_names)
    if val_class_names is not None:
        setattr(val_ds, "class_names", val_class_names)

    return train_ds, val_ds
