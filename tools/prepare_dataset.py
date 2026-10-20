#!/usr/bin/env python3
"""Add new photos to the training set.

Drop JPG/PNG/BMP files into a folder per class and point this script at it::

    python tools/prepare_dataset.py incoming/

    incoming/
      Apple/   photo1.jpg  photo2.png
      Banana/  photo3.jpg

Each image is decoded, shrunk to a small PNG and copied into
``data/raw/custom/<Class>/``.  Storing the dataset as PNG keeps it decodable by
the project's own pure-Python reader, and keeps the repository small.

Retrain afterwards with:  python train.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fruitvision.config import CONFIG  # noqa: E402
from fruitvision.imageio import UnsupportedImageError, decode_image, encode_png  # noqa: E402
from fruitvision.utils.image import resize  # noqa: E402

MAX_EDGE = 160
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def image_files(folder: Path) -> list:
    """Every image already stored in a class folder, in any supported format."""

    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUFFIXES and not p.name.startswith(".")
    )


def existing_hashes(folder: Path) -> set:
    """Hash the images already stored so duplicates are not added twice."""

    return {hashlib.md5(p.read_bytes()).hexdigest() for p in image_files(folder)}


def convert(source: Path, destination: Path) -> tuple:
    """Decode, shrink and write one image as PNG. Returns (width, height, bytes)."""

    image = decode_image(source.read_bytes(), fast=True)
    width, height = image.size
    scale = MAX_EDGE / float(max(width, height))
    if scale < 1.0:
        image = resize(image, max(1, round(width * scale)), max(1, round(height * scale)))
    blob = encode_png(image)
    destination.write_bytes(blob)
    return image.width, image.height, len(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add images to the FruitVision dataset.")
    parser.add_argument("source", type=Path,
                        help="folder containing one sub-folder per class")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen")
    args = parser.parse_args()

    if not args.source.is_dir():
        print("Not a folder: %s" % args.source, file=sys.stderr)
        return 1

    added = 0
    skipped_classes = []
    for class_name in CONFIG.class_names:
        incoming = args.source / class_name
        if not incoming.is_dir():
            skipped_classes.append(class_name)
            continue

        target_dir = CONFIG.data_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        seen = existing_hashes(target_dir)
        index = len(image_files(target_dir))

        for path in sorted(incoming.iterdir()):
            if path.suffix.lower() not in SUFFIXES or path.name.startswith("."):
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen:
                print("  skipped (duplicate): %s" % path.name)
                continue

            index += 1
            destination = target_dir / ("%s_%03d.png" % (class_name.lower(), index))
            if args.dry_run:
                print("  would add: %s -> %s" % (path.name, destination))
                added += 1
                continue

            try:
                width, height, size = convert(path, destination)
            except UnsupportedImageError as exc:
                print("  failed: %s (%s)" % (path.name, exc), file=sys.stderr)
                index -= 1
                continue

            seen.add(digest)
            added += 1
            print("  added: %-34s -> %s  %dx%d  %.1f KB"
                  % (path.name, destination.name, width, height, size / 1024))

    if skipped_classes:
        print("\nNo folder supplied for %d of the %d classes (%s)."
              % (len(skipped_classes), len(CONFIG.class_names),
                 ", ".join(skipped_classes[:6])
                 + (", ..." if len(skipped_classes) > 6 else "")))

    print("\n%d image(s) %s." % (added, "would be added" if args.dry_run else "added"))
    if added and not args.dry_run:
        print("Retrain with:  python train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
