"""Loading the trained model and turning images into predictions.

The saved model holds two streams -- the pixel CNN and the colour/shape
descriptor classifier -- and a prediction is their weighted average.  See
:mod:`fruitvision.features` for why the second stream exists.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.features import describe_image
from fruitvision.imageio import Image
from fruitvision.models.cnn import Network
from fruitvision.models.descriptor import DescriptorNet
from fruitvision.utils.image import load_image_from_bytes, to_model_input

__all__ = [
    "load_model",
    "load_descriptor",
    "load_class_names",
    "predict",
    "predict_image",
    "predict_image_bytes",
    "top_prediction",
    "get_default_model_path",
    "ModelNotTrainedError",
]


class ModelNotTrainedError(FileNotFoundError):
    """Raised when no trained model file is available."""


# Cached so the weights are parsed once rather than once per request.  The
# modification time is part of the cache entry: retraining rewrites model.json,
# and a long-running server should pick the new model up instead of serving the
# old one until it is restarted.
_MODEL_CACHE: Dict[str, Tuple[dict, float]] = {}


def get_default_model_path(config: TrainingConfig = CONFIG) -> Path:
    return config.model_path


def _load_payload(model_path: Optional[Path], config: TrainingConfig) -> Tuple[dict, str]:
    """Read and cache the raw JSON model file."""

    path = Path(model_path) if model_path else get_default_model_path(config)
    key = str(path)

    try:
        mtime = path.stat().st_mtime
    except OSError:
        raise ModelNotTrainedError(
            "No trained model at '%s'. Run 'python train.py' to create one." % path
        )

    cached = _MODEL_CACHE.get(key)
    if cached is not None and cached[1] == mtime:
        return cached[0], key

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_built"] = {}
    _MODEL_CACHE[key] = (payload, mtime)
    return payload, key


def _model_config(payload: dict, config: TrainingConfig) -> TrainingConfig:
    """Rebuild the configuration the saved weights belong to.

    An older model file still loads after the defaults in ``config.py`` change,
    because the shapes come from the file rather than from the current defaults.
    """

    architecture = payload.get("architecture", {})
    descriptor = payload.get("descriptor", {})
    class_names = payload.get("class_names") or list(config.class_names)

    return replace(
        config,
        class_names=list(class_names),
        img_size=int(architecture.get("img_size", config.img_size)),
        channels=int(architecture.get("channels", config.channels)),
        conv1_filters=int(architecture.get("conv1_filters", config.conv1_filters)),
        conv2_filters=int(architecture.get("conv2_filters", config.conv2_filters)),
        kernel_size=int(architecture.get("kernel_size", config.kernel_size)),
        dense_units=int(architecture.get("dense_units", config.dense_units)),
        descriptor_size=int(descriptor.get("size", config.descriptor_size)),
        descriptor_hidden=int(descriptor.get("hidden", config.descriptor_hidden)),
        cnn_weight=float(payload.get("cnn_weight", config.cnn_weight)),
    )


def load_model(
    model_path: Optional[Path] = None, config: TrainingConfig = CONFIG
) -> Tuple[Network, List[str]]:
    """Load the pixel CNN and its class names, caching the result."""

    payload, key = _load_payload(model_path, config)
    built = payload["_built"]
    if "cnn" not in built:
        model_config = _model_config(payload, config)
        if "layers" not in payload:
            raise ModelNotTrainedError(
                "The saved model has no CNN stream. Run 'python train.py'."
            )
        network = Network(model_config)
        network.load_state_dict(payload["layers"])
        network.config = model_config
        built["cnn"] = network
    return built["cnn"], list(payload.get("class_names") or config.class_names)


def load_descriptor(
    model_path: Optional[Path] = None, config: TrainingConfig = CONFIG
) -> Optional[DescriptorNet]:
    """Load the descriptor stream, or ``None`` if the model has not got one.

    A model file saved before the descriptor existed still works: the CNN then
    answers on its own.
    """

    payload, _ = _load_payload(model_path, config)
    built = payload["_built"]
    if "descriptor" not in built:
        state = payload.get("descriptor")
        if not state:
            built["descriptor"] = None
        else:
            model_config = _model_config(payload, config)
            net = DescriptorNet(model_config)
            net.load_state_dict(state)
            built["descriptor"] = net
    return built["descriptor"]


def load_class_names(config: TrainingConfig = CONFIG) -> List[str]:
    """Class names from the trained model, falling back to the configuration."""

    try:
        return load_model(config=config)[1]
    except ModelNotTrainedError:
        return list(config.class_names)


def _blend(
    cnn_probs: Optional[Sequence[float]],
    descriptor_probs: Optional[Sequence[float]],
    weight: float,
) -> List[float]:
    """Weighted average of the two streams, whichever of them are present."""

    if descriptor_probs is None:
        return list(cnn_probs or [])
    if cnn_probs is None:
        return list(descriptor_probs)
    return [(1.0 - weight) * d + weight * c for d, c in zip(descriptor_probs, cnn_probs)]


def _as_scores(probabilities: Sequence[float], class_names: Sequence[str]) -> Dict[str, float]:
    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model outputs %d classes but %d class names are configured."
            % (len(probabilities), len(class_names))
        )
    return {name: float(p) for name, p in zip(class_names, probabilities)}


def predict(
    vector: List[float],
    config: TrainingConfig = CONFIG,
    model_path: Optional[Path] = None,
    descriptor_vector: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Score a preprocessed image, returning ``{class_name: probability}``.

    ``vector`` is the CNN's pixel input.  Pass ``descriptor_vector`` as well --
    or use :func:`predict_image`, which prepares both -- to get the full model
    rather than the pixel stream on its own.
    """

    network, class_names = load_model(model_path, config)
    descriptor = load_descriptor(model_path, config)

    cnn_probs = network.predict_proba(vector)
    descriptor_probs = (
        descriptor.predict_proba(descriptor_vector)
        if descriptor is not None and descriptor_vector is not None
        else None
    )
    weight = network.config.cnn_weight
    return _as_scores(_blend(cnn_probs, descriptor_probs, weight), class_names)


def predict_image(
    image: Image,
    config: TrainingConfig = CONFIG,
    model_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Classify a decoded image with both streams of the model."""

    network, class_names = load_model(model_path, config)
    descriptor = load_descriptor(model_path, config)
    model_config = network.config

    cnn_probs = network.predict_proba(to_model_input(image, model_config))
    descriptor_probs = (
        descriptor.predict_proba(describe_image(image, model_config))
        if descriptor is not None
        else None
    )
    return _as_scores(
        _blend(cnn_probs, descriptor_probs, model_config.cnn_weight), class_names
    )


def predict_image_bytes(
    image_bytes: bytes,
    config: TrainingConfig = CONFIG,
    model_path: Optional[Path] = None,
) -> Tuple[Dict[str, float], Tuple[int, int]]:
    """Decode, preprocess and classify raw image bytes in one step."""

    image = load_image_from_bytes(image_bytes)
    return predict_image(image, config, model_path), image.size


def top_prediction(scores: Dict[str, float]) -> Tuple[str, float]:
    """Return the highest-scoring ``(class_name, confidence)`` pair."""

    return max(scores.items(), key=lambda item: item[1])
