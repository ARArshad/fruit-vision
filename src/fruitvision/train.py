"""Training entry point for the FruitVision classifier."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.data.dataset import load_datasets
from fruitvision.models.cnn import build_model


def _prepare_output_dirs(output_dir: Path) -> Tuple[Path, Path]:
    """Ensure output directories for models and logs exist."""

    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, logs_dir


def train_model(
    config: TrainingConfig = CONFIG,
) -> Dict[str, float]:
    """Train the FruitVision CNN classifier using provided configuration."""

    train_ds, val_ds = load_datasets(config)

    class_names = getattr(train_ds, "class_names", config.class_names)
    num_classes = train_ds.element_spec[1].shape[-1]

    input_shape = (config.img_height, config.img_width, config.channels)
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    models_dir, logs_dir = _prepare_output_dirs(config.output_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    checkpoint_path = models_dir / f"fruitvision-{timestamp}.keras"
    log_dir = logs_dir / f"train-{timestamp}"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "num_classes": int(num_classes),
        "class_names": class_names,
    }

    metrics_path = config.output_dir / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    model.save(models_dir / "fruitvision-latest.keras")

    return metrics


if __name__ == "__main__":
    train_metrics = train_model()
    print("Training complete:", json.dumps(train_metrics, indent=2))
