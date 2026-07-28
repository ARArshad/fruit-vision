"""Inference helpers for FruitVision model deployment."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from fruitvision.config import CONFIG, TrainingConfig

MODEL_FILENAME = "fruitvision-latest.keras"


def get_default_model_path(config: TrainingConfig = CONFIG) -> Path:
    return config.output_dir / "models" / MODEL_FILENAME


@lru_cache(maxsize=1)
def load_model(model_path: Path | None = None) -> tf.keras.Model:
    """Load and cache the trained Keras model for inference."""

    path = model_path or get_default_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file '{path}' not found. Train the model or provide a valid path."
        )
    return tf.keras.models.load_model(path)


@lru_cache(maxsize=1)
def load_class_names() -> List[str]:
    """Retrieve class names from saved training metrics."""

    metrics_path = CONFIG.output_dir / "training_metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        class_names = data.get("class_names")
        if isinstance(class_names, list) and all(
            isinstance(name, str) for name in class_names
        ):
            return class_names

    return CONFIG.class_names


def predict(
    image_batch: np.ndarray,
    config: TrainingConfig = CONFIG,
    model_path: Path | None = None,
) -> Dict[str, float]:
    """Run model inference on a preprocessed image batch.

    Returns a mapping of class name to confidence score.
    """

    model = load_model(model_path=model_path)
    class_names = load_class_names()
    preds = model.predict(image_batch, verbose=0)[0]

    if len(preds) != len(class_names):
        raise ValueError(
            "Mismatch between model output size and configured class names."
        )

    return {cls: float(score) for cls, score in zip(class_names, preds)}


def top_prediction(scores: Dict[str, float]) -> Tuple[str, float]:
    """Return the class with highest confidence."""

    return max(scores.items(), key=lambda item: item[1])
