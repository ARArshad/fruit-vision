"""FruitVision test suite.

Run every test with the standard library only::

    python -m unittest discover -s tests -t .

No third-party test runner is needed; the suite uses only :mod:`unittest`.
"""

import pathlib
import sys

# Make both the repository root (for `backend`) and `src` (for `fruitvision`)
# importable however the suite is launched.
ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
