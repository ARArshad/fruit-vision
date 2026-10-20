#!/usr/bin/env python3
"""Package the project into a zip file that runs anywhere.

    python tools/make_zip.py

Writes ``dist/FruitVision.zip`` containing everything needed to run the app --
source, dataset, trained model and documentation -- and nothing that would
bloat it: no git history, no caches, no virtual environment.

Add ``--lean`` to leave out the reference PDFs and screenshots.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Directories skipped wherever they appear.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".idea", ".vscode", ".claude", "dist", "node_modules",
}

# Files skipped by exact name or by suffix.
SKIP_NAMES = {".DS_Store", "Thumbs.db", ".gitignore"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".swp"}

# Extra directories left out by --lean.
LEAN_SKIP = {"docs"}


def should_skip(path: Path, lean: bool) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts

    if any(part in SKIP_DIRS for part in parts):
        return True
    if lean and parts[0] in LEAN_SKIP:
        return True
    if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def collect(lean: bool):
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and not should_skip(path, lean):
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package FruitVision for distribution.")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "FruitVision.zip",
                        help="where to write the archive")
    parser.add_argument("--lean", action="store_true",
                        help="exclude the docs folder (PDFs and screenshots)")
    parser.add_argument("--list", action="store_true",
                        help="list the files that would be included, then exit")
    args = parser.parse_args()

    files = list(collect(args.lean))
    total = sum(path.stat().st_size for path in files)

    if args.list:
        for path in files:
            print("  %8.1f KB  %s" % (path.stat().st_size / 1024, path.relative_to(ROOT)))
        print("\n%d files, %.1f MB uncompressed" % (len(files), total / 1024 / 1024))
        return 0

    if not (ROOT / "artifacts" / "model.json").exists():
        print("  ! artifacts/model.json is missing -- the package will not be able to")
        print("    make predictions. Run 'python train.py' first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path("FruitVision") / path.relative_to(ROOT))

    packed = args.output.stat().st_size
    print("Created %s" % args.output)
    print("  %d files, %.1f MB uncompressed -> %.1f MB zipped"
          % (len(files), total / 1024 / 1024, packed / 1024 / 1024))
    print("\nSend that file to anyone. They unzip it and run:  python run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
