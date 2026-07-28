"""Image preprocessing utilities for inference."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from fruitvision.config import CONFIG, TrainingConfig


def load_image_from_bytes(
    image_bytes: bytes,
    config: TrainingConfig = CONFIG,
) -> np.ndarray:
    """Load an image from raw bytes and resize to model input shape."""

    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize((config.img_width, config.img_height))
        return np.array(img, dtype="float32")


def preprocess_upload(
    image_bytes: bytes,
    config: TrainingConfig = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare an uploaded image for model prediction.

    Returns both the model-ready batch array and the original image array.
    """

    processed = load_image_from_bytes(image_bytes, config)
    batch = np.expand_dims(processed, axis=0)
    return batch, processed
