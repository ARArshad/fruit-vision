"""Train the FruitVision classifier.

Run it from the project root via the top-level shim::

    python train.py              # normal training run
    python train.py --loocv      # leave-one-out cross-validation

Training is entirely optional: a trained ``artifacts/model.json`` ships with
the project, so the application runs straight out of the box.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # allow "python src/fruitvision/train.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitvision.config import CONFIG, TrainingConfig
from fruitvision.data.dataset import (
    Sample,
    build_descriptor_set,
    build_eval_set,
    build_training_set,
    discover_images,
    split_dataset,
)
from fruitvision.models.cnn import Adam, Network, softmax, softmax_cross_entropy
from fruitvision.models.descriptor import DescriptorNet, standardiser

__all__ = ["train_model", "evaluate", "cross_validate", "train_descriptor"]


def _make_reporter(label: str, verbose: bool):
    """Return a progress callback that keeps long preparation steps visible.

    Decoding a thousand JPEGs in pure Python takes a while; without this the
    command looks frozen before the first epoch starts.
    """

    if not verbose:
        return None

    def report(done: int, total: int) -> None:
        if done == total or done % 100 == 0:
            end = "\n" if done == total else "\r"
            print("  %s: %d/%d" % (label, done, total), end=end, flush=True)

    return report


def evaluate(network: Network, samples: Sequence[Sample]) -> Tuple[float, float]:
    """Return ``(accuracy, mean_loss)`` over a set of samples."""

    if not samples:
        return (0.0, 0.0)
    correct = 0
    total_loss = 0.0
    for vector, label in samples:
        logits = network.forward(vector, training=False)
        loss, _ = softmax_cross_entropy(logits, label)
        total_loss += loss
        if max(range(len(logits)), key=lambda i: logits[i]) == label:
            correct += 1
    return (correct / len(samples), total_loss / len(samples))


def _run_epochs(
    network: Network,
    fit: List[Sample],
    val: List[Sample],
    config: TrainingConfig,
    rng: random.Random,
    verbose: bool = True,
    patience: int = 4,
    monitor: Optional[List[Sample]] = None,
) -> Dict[str, object]:
    """Mini-batch training loop with early stopping on the validation loss."""

    optimiser = Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_loss = float("inf")
    best_state: Optional[List[Dict[str, object]]] = None
    best_epoch = 0
    stale = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        rng.shuffle(fit)
        running_loss = 0.0
        correct = 0

        for start in range(0, len(fit), config.batch_size):
            batch = fit[start : start + config.batch_size]
            network.zero_grad()
            for vector, label in batch:
                logits = network.forward(vector, training=True)
                loss, grad = softmax_cross_entropy(logits, label)
                running_loss += loss
                if max(range(len(logits)), key=lambda i: logits[i]) == label:
                    correct += 1
                network.backward(grad)
            optimiser.step(scale=1.0 / len(batch))

        train_acc = correct / len(fit)
        train_loss = running_loss / len(fit)
        val_acc, val_loss = evaluate(network, val)
        history.append(
            {
                "epoch": epoch,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }
        )

        if verbose:
            extra = ""
            if monitor:
                # Reported for visibility only: early stopping below still uses
                # the validation slice, so the held-out set is never tuned on.
                monitor_acc, _ = evaluate(network, monitor)
                extra = "  |  held-out %.3f" % monitor_acc
            print(
                "  epoch %2d/%d  loss %.4f  acc %.3f  |  val_loss %.4f  val_acc %.3f%s  (%.1fs)"
                % (epoch, config.epochs, train_loss, train_acc, val_loss, val_acc,
                   extra, time.time() - started),
                flush=True,
            )

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = network.state_dict()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print("  early stopping: validation loss stopped improving")
                break

    if best_state is not None:
        network.load_state_dict(best_state)

    return {"history": history, "best_epoch": best_epoch}


def _round_state(states: List[Dict[str, object]], places: int = 6):
    """Round weights before saving -- it roughly halves the model file."""

    out = []
    for state in states:
        if not state:
            out.append({})
            continue
        out.append({key: [round(float(v), places) for v in values]
                    for key, values in state.items()})
    return out


def save_model(
    network: Optional[Network] = None,
    config: TrainingConfig = CONFIG,
    path: Optional[Path] = None,
    descriptor: Optional[DescriptorNet] = None,
) -> Path:
    """Persist the trained streams and the architecture they belong to as JSON.

    Either stream may be omitted, in which case whatever is already on disk for
    it is kept -- that is what makes ``python train.py --stream descriptor``
    (a few minutes) possible without redoing the slow CNN run.
    """

    path = path or config.model_path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            payload = {}

    payload["format"] = "fruitvision-model/2"
    payload["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["class_names"] = list(config.class_names)
    payload["cnn_weight"] = config.cnn_weight

    if network is not None:
        payload["architecture"] = {
            "img_size": config.img_size,
            "channels": config.channels,
            "conv1_filters": config.conv1_filters,
            "conv2_filters": config.conv2_filters,
            "kernel_size": config.kernel_size,
            "dense_units": config.dense_units,
        }
        payload["parameters"] = network.num_parameters()
        payload["layers"] = _round_state(network.state_dict())

    if descriptor is not None:
        state = descriptor.state_dict()
        state["layers"] = _round_state(state["layers"])
        # Standardisation constants are kept at full precision: some features
        # have a spread of a thousandth, and rounding those would shift every
        # descriptor the model ever sees.
        state["mean"] = [float(v) for v in state["mean"]]
        state["std"] = [float(v) for v in state["std"]]
        payload["descriptor"] = state

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return path


def evaluate_descriptor(net: DescriptorNet, samples: Sequence[Sample]) -> Tuple[float, float]:
    """Return ``(accuracy, mean_loss)`` for the descriptor stream."""

    if not samples:
        return (0.0, 0.0)
    correct = 0
    total_loss = 0.0
    for vector, label in samples:
        logits = net.forward(net.normalise(vector))
        loss, _ = softmax_cross_entropy(logits, label)
        total_loss += loss
        if max(range(len(logits)), key=lambda i: logits[i]) == label:
            correct += 1
    return (correct / len(samples), total_loss / len(samples))


def train_descriptor(
    fit: List[Sample],
    val: List[Sample],
    config: TrainingConfig = CONFIG,
    rng: Optional[random.Random] = None,
    verbose: bool = True,
) -> Tuple[DescriptorNet, Dict[str, object]]:
    """Train the colour/shape/texture classifier on prepared descriptors.

    Unlike the CNN this converges in a couple of minutes, so it simply runs the
    full schedule and keeps whichever epoch scored best on the validation
    images rather than stopping early.
    """

    rng = rng or random.Random(config.seed)
    net = DescriptorNet(config, rng=random.Random(config.seed))
    net.set_standardisation(*standardiser([vector for vector, _ in fit]))

    optimiser = Adam(
        net.parameters(),
        lr=config.descriptor_lr,
        weight_decay=config.descriptor_weight_decay,
    )

    prepared = [(net.normalise(vector), label) for vector, label in fit]
    best_state = None
    best_acc = -1.0
    best_epoch = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.descriptor_epochs + 1):
        started = time.time()
        rng.shuffle(prepared)
        running_loss = 0.0
        correct = 0

        for start in range(0, len(prepared), config.descriptor_batch_size):
            batch = prepared[start : start + config.descriptor_batch_size]
            net.zero_grad()
            for vector, label in batch:
                logits = net.forward(vector, training=True)
                loss, grad = softmax_cross_entropy(logits, label)
                running_loss += loss
                if max(range(len(logits)), key=lambda i: logits[i]) == label:
                    correct += 1
                net.backward(grad)
            optimiser.step(scale=1.0 / len(batch))

        train_acc = correct / len(prepared)
        train_loss = running_loss / len(prepared)
        val_acc, val_loss = evaluate_descriptor(net, val)
        history.append({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        })
        if verbose:
            print("  epoch %2d/%d  loss %.4f  acc %.3f  |  val_loss %.4f  val_acc %.3f  (%.1fs)"
                  % (epoch, config.descriptor_epochs, train_loss, train_acc,
                     val_loss, val_acc, time.time() - started), flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = [layer.state_dict() for layer in net.layers]

    if best_state is not None:
        for layer, state in zip(net.layers, best_state):
            if state:
                layer.load_state_dict(state)

    return net, {"history": history, "best_epoch": best_epoch, "val_accuracy": best_acc}


def ensemble_probabilities(
    cnn_probs: Optional[Sequence[float]],
    descriptor_probs: Optional[Sequence[float]],
    weight: float,
) -> List[float]:
    """Blend the two streams into one set of class probabilities.

    ``weight`` is how much the pixel CNN counts. With either stream missing the
    other one is used on its own, so a half-trained project still predicts.
    """

    if descriptor_probs is None:
        return list(cnn_probs or [])
    if cnn_probs is None:
        return list(descriptor_probs)
    return [(1.0 - weight) * d + weight * c for d, c in zip(descriptor_probs, cnn_probs)]


def _predict_all(
    network: Optional[Network],
    descriptor: Optional[DescriptorNet],
    cnn_samples: Sequence[Sample],
    descriptor_samples: Sequence[Sample],
    weight: float,
) -> List[List[float]]:
    """Ensemble probabilities for a whole evaluation set, in order."""

    count = len(descriptor_samples) if descriptor else len(cnn_samples)
    out = []
    for i in range(count):
        cnn_probs = network.predict_proba(cnn_samples[i][0]) if network else None
        desc_probs = descriptor.predict_proba(descriptor_samples[i][0]) if descriptor else None
        out.append(ensemble_probabilities(cnn_probs, desc_probs, weight))
    return out


def _accuracy_of(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    correct = 0
    for probs, label in zip(probabilities, labels):
        if max(range(len(probs)), key=lambda i: probs[i]) == label:
            correct += 1
    return correct / len(labels)


def train_model(
    config: TrainingConfig = CONFIG,
    verbose: bool = True,
    save: bool = True,
    streams: Sequence[str] = ("cnn", "descriptor"),
) -> Dict[str, object]:
    """Train the model and, unless told otherwise, save the result.

    The model has two streams -- the pixel CNN and the descriptor classifier --
    and ``streams`` selects which of them this run retrains.  Whichever stream
    is not retrained is loaded from the existing ``model.json`` so the final
    accuracy figures still describe the complete model.
    """

    started = time.time()
    rng = random.Random(config.seed)

    items = discover_images(config)

    # Prefer a dedicated held-out directory when one exists: it contains
    # different physical fruits, which a random split of the training set
    # cannot give us (consecutive training images are rotations of one fruit).
    if config.test_dir.exists():
        train_items = items
        test_items = discover_images(config, config.test_dir)
        split_note = "held-out set from %s" % config.test_dir.name
    else:
        train_items, test_items = split_dataset(items, config)
        split_note = "held out from the training images"

    if verbose:
        print("Dataset: %d training images across %d classes, %d test images (%s)"
              % (len(train_items), len(config.class_names), len(test_items), split_note), flush=True)
        print("Streams to train: %s" % ", ".join(streams), flush=True)

    metrics: Dict[str, object] = {}
    network: Optional[Network] = None
    descriptor: Optional[DescriptorNet] = None

    # -- stream 1: the pixel CNN ------------------------------------------
    if "cnn" in streams:
        if verbose:
            print("\n[1/2] Pixel CNN", flush=True)
        reporter = _make_reporter("Preparing training images", verbose)
        samples = build_training_set(
            train_items, config, rng=rng, progress=reporter,
            # Fruits-360 is entirely single fruits on plain white. Training on
            # the untouched originals as well would reinforce that backdrop as
            # a feature, which is exactly what breaks real-world photographs.
            include_original=not config.background_swap_probability,
        )
        cut = max(1, int(len(samples) * 0.15))
        cnn_val, cnn_fit = samples[:cut], samples[cut:]

        network = Network(config, rng=random.Random(config.seed))
        if verbose:
            print("Model: %d parameters, input %dx%dx%d"
                  % (network.num_parameters(), config.channels, config.img_size, config.img_size),
                  flush=True)
            print("Training on %d samples (%d held back for validation)..."
                  % (len(cnn_fit), len(cnn_val)), flush=True)
        result = _run_epochs(network, cnn_fit, cnn_val, config, rng, verbose=verbose)

        train_acc, train_loss = evaluate(network, cnn_fit)
        val_acc, val_loss = evaluate(network, cnn_val)
        metrics["cnn"] = {
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "parameters": network.num_parameters(),
            "epochs_run": len(result["history"]),
            "best_epoch": result["best_epoch"],
            "training_samples": len(cnn_fit),
            "history": result["history"],
        }
    else:
        network = _load_existing_cnn(config, verbose)

    # -- stream 2: the descriptor classifier -------------------------------
    if "descriptor" in streams:
        if verbose:
            print("\n[2/2] Colour/shape descriptor", flush=True)
        # The split is over *source images*, not over descriptors: the mirrored
        # and re-exposed copies of one photo must never straddle it.
        fit_items, val_items = split_dataset(
            train_items, replace(config, test_fraction=0.15), seed=config.seed + 1
        )
        fit = build_descriptor_set(
            fit_items, config, rng=rng,
            progress=_make_reporter("Measuring training images", verbose),
        )
        val = build_descriptor_set(val_items, config, augment=False)
        if verbose:
            print("Descriptors: %d from %d images (%d validation images)"
                  % (len(fit), len(fit_items), len(val_items)), flush=True)
        descriptor, desc_result = train_descriptor(fit, val, config, rng=rng, verbose=verbose)

        train_acc, train_loss = evaluate_descriptor(descriptor, fit)
        val_acc, val_loss = evaluate_descriptor(descriptor, val)
        metrics["descriptor"] = {
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "parameters": descriptor.num_parameters(),
            "features": len(fit[0][0]),
            "epochs_run": len(desc_result["history"]),
            "best_epoch": desc_result["best_epoch"],
            "training_samples": len(fit),
            "history": desc_result["history"],
        }
    else:
        descriptor = _load_existing_descriptor(config, verbose)

    if network is None and descriptor is None:
        raise SystemExit("Nothing to evaluate: neither stream was trained or loaded.")

    # -- how well does the finished model do? ------------------------------
    if verbose:
        print("\nEvaluating on the held-out images...", flush=True)
    labels = [label for _, label in test_items]
    cnn_test = build_eval_set(test_items, config) if network else []
    desc_test = build_descriptor_set(test_items, config, augment=False) if descriptor else []

    cnn_only = [network.predict_proba(v) for v, _ in cnn_test] if network else []
    desc_only = [descriptor.predict_proba(v) for v, _ in desc_test] if descriptor else []
    combined = [
        ensemble_probabilities(
            cnn_only[i] if cnn_only else None,
            desc_only[i] if desc_only else None,
            config.cnn_weight,
        )
        for i in range(len(labels))
    ]

    test_acc = _accuracy_of(combined, labels)
    metrics.update({
        "test_accuracy": test_acc,
        "cnn_test_accuracy": _accuracy_of(cnn_only, labels) if cnn_only else None,
        "descriptor_test_accuracy": _accuracy_of(desc_only, labels) if desc_only else None,
        "cnn_weight": config.cnn_weight,
        "num_classes": len(config.class_names),
        "class_names": list(config.class_names),
        "parameters": (network.num_parameters() if network else 0)
                      + (descriptor.num_parameters() if descriptor else 0),
        "source_images": len(items),
        "test_images": len(test_items),
        "held_out_images": [p.name for p, _ in test_items],
        "duration_seconds": round(time.time() - started, 1),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Top-level train/validation numbers describe the stream that decides the
    # answer, which is the descriptor whenever it is present.
    primary = metrics.get("descriptor") or metrics.get("cnn") or {}
    for key in ("train_accuracy", "train_loss", "val_accuracy", "val_loss",
                "epochs_run", "best_epoch", "training_samples"):
        if key in primary:
            metrics[key] = primary[key]

    if verbose:
        per_class: Dict[int, Tuple[int, int]] = {}
        misses = []
        for (path, label), probs in zip(test_items, combined):
            best = max(range(len(probs)), key=lambda i: probs[i])
            hit = best == label
            correct, total = per_class.get(label, (0, 0))
            per_class[label] = (correct + hit, total + 1)
            if not hit:
                misses.append((path.name, config.class_names[label],
                               config.class_names[best], probs[best]))

        print("\nHeld-out accuracy per class:")
        for label in sorted(per_class):
            correct, total = per_class[label]
            print("  %-14s %3d/%-3d  %5.1f%%"
                  % (config.class_names[label], correct, total, 100.0 * correct / total))
        if misses:
            print("\nMisclassified (%d):" % len(misses))
            for name, true, predicted, confidence in misses[:15]:
                print("  %-22s true=%-12s predicted=%-12s (%.0f%%)"
                      % (name, true, predicted, confidence * 100))
            if len(misses) > 15:
                print("  ... and %d more" % (len(misses) - 15))

        metrics["per_class_test_accuracy"] = {
            config.class_names[l]: {"correct": c, "total": t}
            for l, (c, t) in per_class.items()
        }

    if save:
        model_path = save_model(
            network if "cnn" in streams else None,
            config,
            descriptor=descriptor if "descriptor" in streams else None,
        )
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with config.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        metrics["model_path"] = str(model_path)
        if verbose:
            size_kb = model_path.stat().st_size / 1024
            print("\nSaved %s (%.0f KB) and %s"
                  % (model_path.name, size_kb, config.metrics_path.name))

    return metrics


def _load_existing_cnn(config: TrainingConfig, verbose: bool) -> Optional[Network]:
    """Load the CNN stream from the saved model, for runs that skip it."""

    from fruitvision.inference import ModelNotTrainedError, load_model

    try:
        network, _ = load_model(config=config)
    except (ModelNotTrainedError, KeyError, ValueError) as exc:
        if verbose:
            print("Keeping no CNN stream (%s)" % exc, flush=True)
        return None
    if verbose:
        print("Reusing the saved CNN stream (%d parameters)"
              % network.num_parameters(), flush=True)
    return network


def _load_existing_descriptor(config: TrainingConfig, verbose: bool) -> Optional[DescriptorNet]:
    """Load the descriptor stream from the saved model, for runs that skip it."""

    from fruitvision.inference import load_descriptor

    descriptor = load_descriptor(config=config)
    if verbose:
        print("Reusing the saved descriptor stream"
              if descriptor else "No saved descriptor stream to reuse", flush=True)
    return descriptor


def cross_validate(config: TrainingConfig = CONFIG, verbose: bool = True) -> Dict[str, object]:
    """Leave-one-out cross-validation.

    With only a handful of source images a single held-out split says very
    little.  Training once per image and predicting the one left out uses every
    photo as an unseen test case, which is a far more honest measure.
    """

    items = discover_images(config)
    if len(items) > 40:
        raise SystemExit(
            "Leave-one-out cross-validation would train %d separate models, which "
            "is only practical for a very small dataset. With %d images, use the "
            "held-out test set instead: python train.py" % (len(items), len(items))
        )
    results = []
    correct = 0

    for index, (path, label) in enumerate(items):
        rest = [item for i, item in enumerate(items) if i != index]
        rng = random.Random(config.seed + index)
        samples = build_training_set(rest, config, rng=rng)
        cut = max(1, int(len(samples) * 0.15))
        val, fit = samples[:cut], samples[cut:]

        network = Network(config, rng=random.Random(config.seed))
        _run_epochs(network, fit, val, config, rng, verbose=False)

        probs = network.predict_proba(build_eval_set([(path, label)], config)[0][0])
        best = max(range(len(probs)), key=lambda i: probs[i])
        hit = best == label
        correct += hit
        results.append(
            {
                "image": path.name,
                "true": config.class_names[label],
                "predicted": config.class_names[best],
                "confidence": probs[best],
                "correct": hit,
            }
        )
        if verbose:
            print("  [%s] %-18s true=%-7s predicted=%-7s (%.1f%%)"
                  % ("OK " if hit else "MISS", path.name, config.class_names[label],
                     config.class_names[best], probs[best] * 100), flush=True)

    accuracy = correct / len(items)
    if verbose:
        print("\nLeave-one-out accuracy: %d/%d = %.1f%%"
              % (correct, len(items), accuracy * 100))
    return {"loocv_accuracy": accuracy, "folds": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the FruitVision classifier.")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument("--augment", type=int, help="augmented variants per image")
    parser.add_argument("--loocv", action="store_true",
                        help="run leave-one-out cross-validation instead of training")
    parser.add_argument("--no-save", action="store_true",
                        help="train but do not overwrite the saved model")
    parser.add_argument("--stream", choices=("both", "cnn", "descriptor"), default="both",
                        help="which stream to retrain: the pixel CNN (slow, ~45 min), "
                             "the colour/shape descriptor (fast, ~5 min) or both")
    args = parser.parse_args(argv)

    config = CONFIG
    if args.epochs:
        config.epochs = args.epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.augment is not None:
        config.augmentations_per_image = args.augment

    if args.loocv:
        print("Leave-one-out cross-validation over the whole dataset")
        summary = cross_validate(config)
        out = config.output_dir / "cross_validation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print("Wrote %s" % out.name)
        return 0

    streams = ("cnn", "descriptor") if args.stream == "both" else (args.stream,)
    metrics = train_model(config, save=not args.no_save, streams=streams)
    print("\nFinal: train %.1f%% | validation %.1f%% | held-out images %.1f%%"
          % (metrics["train_accuracy"] * 100, metrics["val_accuracy"] * 100,
             metrics["test_accuracy"] * 100))
    for stream, label in (("cnn_test_accuracy", "pixel CNN alone"),
                          ("descriptor_test_accuracy", "descriptor alone")):
        if metrics.get(stream) is not None:
            print("  %-20s %.1f%%" % (label, metrics[stream] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
