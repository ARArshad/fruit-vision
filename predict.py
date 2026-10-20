#!/usr/bin/env python3
"""Classify fruit images from the command line.

    python predict.py photo.jpg
    python predict.py data/raw/custom/Apple/apple_01.png --json

Useful when you want a prediction without starting the web interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fruitvision.imageio import UnsupportedImageError  # noqa: E402
from fruitvision.inference import (  # noqa: E402
    ModelNotTrainedError,
    predict_image_bytes,
    top_prediction,
)


def classify(path: Path) -> dict:
    scores, size = predict_image_bytes(path.read_bytes())
    label, confidence = top_prediction(scores)
    return {
        "file": str(path),
        "prediction": label,
        "confidence": confidence,
        "scores": scores,
        "source_size": list(size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify fruit images.")
    parser.add_argument("images", nargs="+", type=Path, help="image files to classify")
    parser.add_argument("--json", action="store_true", help="print raw JSON output")
    args = parser.parse_args()

    results = []
    failed = False

    for path in args.images:
        if not path.is_file():
            print("  %-30s ERROR  file not found" % path.name, file=sys.stderr)
            failed = True
            continue
        try:
            result = classify(path)
        except ModelNotTrainedError as exc:
            print(exc, file=sys.stderr)
            return 2
        except UnsupportedImageError as exc:
            print("  %-30s ERROR  %s" % (path.name, exc), file=sys.stderr)
            failed = True
            continue

        results.append(result)
        if not args.json:
            # One fruit per image: the full distribution is only in --json.
            print("  %-30s -> %-12s %5.1f%% confident"
                  % (path.name, result["prediction"], result["confidence"] * 100))

    if args.json:
        print(json.dumps(results, indent=2))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
