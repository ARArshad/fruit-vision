#!/usr/bin/env python3
"""Train the FruitVision classifier.

    python train.py                 # train and save artifacts/model.json
    python train.py --loocv         # leave-one-out cross-validation
    python train.py --epochs 20     # override a hyper-parameter

Training is optional -- a trained model already ships with the project. This
takes a few minutes, because the network is plain Python rather than a
compiled library.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from fruitvision.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
