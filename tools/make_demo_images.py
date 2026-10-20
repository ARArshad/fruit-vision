#!/usr/bin/env python3
"""Build and check the ``demo_images/`` folder used for showing the project off.

The folder holds a small set of photographs -- by default two per fruit -- that
the trained model is known to get right, so a live demo cannot go wrong.

    python tools/make_demo_images.py --check                 # verify the folder
    python tools/make_demo_images.py --from downloads/       # rebuild from photos
    python tools/make_demo_images.py --from downloads/ --per-class 3

``--from`` expects one folder per fruit::

    downloads/
      Apple/       any jpg/png photographs of apples
      Watermelon/  ...

Every candidate is classified, and only the ones the model names correctly are
copied in, most confident first.  Anything it gets wrong is left out and
reported, so the folder always matches what the model can actually do.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitvision.config import CONFIG  # noqa: E402
from fruitvision.inference import predict_image_bytes, top_prediction  # noqa: E402

OUTPUT_DIR = ROOT / "demo_images"
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def classify(path: Path):
    scores, _ = predict_image_bytes(path.read_bytes())
    return top_prediction(scores)


def fruit_of(name: str) -> str:
    """The fruit a demo file claims to be, from its filename (``apple_1.jpg``)."""

    stem = name.split("_")[0].lower()
    for fruit in CONFIG.class_names:
        if fruit.lower() == stem:
            return fruit
    return ""


def check() -> int:
    files = sorted(p for p in OUTPUT_DIR.iterdir()
                   if p.suffix.lower() in SUFFIXES and not p.name.startswith("."))
    if not files:
        print("No images in %s" % OUTPUT_DIR)
        return 1

    wrong = []
    for path in files:
        expected = fruit_of(path.name)
        predicted, confidence = classify(path)
        ok = predicted == expected
        print("  %-22s %-12s %5.1f%%  %s"
              % (path.name, predicted, confidence * 100, "ok" if ok else "WRONG, expected " + expected))
        if not ok:
            wrong.append(path.name)

    print("\n%d/%d correct" % (len(files) - len(wrong), len(files)))
    if wrong:
        print("Not recognised: %s" % ", ".join(wrong))
        return 1
    return 0


def rebuild(source: Path, per_class: int) -> int:
    if not source.is_dir():
        raise SystemExit("'%s' is not a folder. Expected one sub-folder per fruit." % source)

    kept, rejected, missing = [], [], []
    for fruit in CONFIG.class_names:
        folder = source / fruit
        if not folder.is_dir():
            matches = [p for p in source.iterdir() if p.is_dir() and p.name.lower() == fruit.lower()]
            folder = matches[0] if matches else None
        if folder is None:
            missing.append(fruit)
            continue

        scored = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in SUFFIXES or path.name.startswith("."):
                continue
            predicted, confidence = classify(path)
            if predicted == fruit:
                scored.append((confidence, path))
            else:
                rejected.append((path.name, fruit, predicted, confidence))

        scored.sort(key=lambda item: -item[0])
        if not scored:
            missing.append(fruit)
            continue
        for rank, (confidence, path) in enumerate(scored[:per_class], 1):
            kept.append({"fruit": fruit, "rank": rank, "confidence": round(confidence, 4),
                         "source": path})

    if not kept:
        raise SystemExit("The model recognised none of those images -- nothing to write.")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    manifest = []
    for entry in kept:
        path = entry.pop("source")
        target = OUTPUT_DIR / ("%s_%d%s" % (entry["fruit"].lower(), entry["rank"], path.suffix.lower()))
        shutil.copyfile(path, target)
        manifest.append({"file": target.name, **entry, "original": path.name})
        print("  %-20s %-12s %5.1f%%" % (target.name, entry["fruit"], entry["confidence"] * 100))

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\nWrote %d images to %s" % (len(manifest), OUTPUT_DIR))
    if rejected:
        print("Left out %d image(s) the model gets wrong:" % len(rejected))
        for name, expected, predicted, confidence in rejected[:20]:
            print("  %-28s %s -> %s (%.0f%%)" % (name, expected, predicted, confidence * 100))
        if len(rejected) > 20:
            print("  ... and %d more" % (len(rejected) - 20))
    if missing:
        print("No usable image for: %s" % ", ".join(missing))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or check the demo image folder.")
    parser.add_argument("--from", dest="source", type=Path,
                        help="folder of candidate photos, one sub-folder per fruit")
    parser.add_argument("--per-class", type=int, default=2,
                        help="how many images to keep per fruit (default: 2)")
    parser.add_argument("--check", action="store_true",
                        help="only verify the images already in demo_images/")
    args = parser.parse_args()

    if args.check or not args.source:
        return check()
    return rebuild(args.source, args.per_class)


if __name__ == "__main__":
    raise SystemExit(main())
