"""Assemble the static, server-free version of FruitVision in ./web.

    python build_web.py

The page in web/ classifies photos in the browser with web/model.js, a port
of the Python inference code.  This script copies in what that page needs from
the trained project: the model weights, a little metadata for the footer, and
a handful of sample photos to try.  The result is plain files that any static
host (GitHub Pages, Netlify, an S3 bucket) can serve as they are.

Re-run it after retraining, then check the port still agrees with Python:

    python -m unittest tests.test_web_parity
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ARTIFACTS = ROOT / "artifacts"
DEMO = ROOT / "demo_images"

SAMPLES = [
    "apple_4.jpg",
    "banana_1.jpg",
    "strawberry_1.jpg",
    "pineapple_1.jpg",
    "watermelon_1.jpg",
    "lemon_2.jpg",
]


def main() -> None:
    model_path = ARTIFACTS / "model.json"
    if not model_path.exists():
        raise SystemExit("No trained model at %s. Run 'python train.py' first." % model_path)

    shutil.copyfile(model_path, WEB / "model.json")
    model = json.loads(model_path.read_text(encoding="utf-8"))

    info = {
        "classes": model.get("class_names", []),
        "parameters": model.get("parameters"),
    }
    metrics_path = ARTIFACTS / "training_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        info["test_accuracy"] = metrics.get("test_accuracy")
        info["test_images"] = metrics.get("test_images")
        info["parameters"] = metrics.get("parameters", info["parameters"])
    (WEB / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    samples_dir = WEB / "samples"
    samples_dir.mkdir(exist_ok=True)
    for name in SAMPLES:
        shutil.copyfile(DEMO / name, samples_dir / name)

    print("Built %s: model.json, info.json and %d sample photos." % (WEB, len(SAMPLES)))


if __name__ == "__main__":
    main()
