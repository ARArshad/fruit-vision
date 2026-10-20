# FruitVision

A CNN-based fruit classification system. Upload a photo, get back the fruit
class and a confidence score. It recognises **16 fruits**: Apple, Avocado, Banana, Cherry, Grape, Kiwi, Lemon, Lime, Mango,
Orange, Peach, Pear, Pineapple, Pomegranate, Strawberry and Watermelon.

**It runs with zero installation.** No `pip install`, no virtual environment,
no internet connection, no GPU. Unzip the folder, run one command, done:

```bash
python run.py
```

Or just double-click **`start.bat`** (Windows) / **`start.command`** (macOS,
Linux) — no terminal needed.

New to this? Follow **[RUN_GUIDE.md](RUN_GUIDE.md)** — a step-by-step guide
written for someone who has never used a terminal.

---

## Why there are no dependencies

The whole system — the convolutional network, its training loop, the image
decoders and the web server — is written from scratch against the Python
standard library.

| Layer | Normally | Here |
|---|---|---|
| Neural network + training | TensorFlow / Keras | [`src/fruitvision/models/cnn.py`](src/fruitvision/models/cnn.py) |
| Image decoding | Pillow | [`src/fruitvision/imageio.py`](src/fruitvision/imageio.py) |
| Numerics | NumPy | plain Python lists |
| Web server | FastAPI + Uvicorn | [`backend/api.py`](backend/api.py) (`http.server`) |
| Tests | pytest | `unittest` |

That is the entire reason this folder is a few megabytes and runs on any
machine with Python 3.8+, rather than needing a ~700 MB platform-specific
TensorFlow install pinned to particular Python versions.

---

## How it works

```
 Browser                        Python server                      Model
┌──────────────────┐   PNG   ┌──────────────────────┐  32×32 px  ┌──────────────────┐
│ pick / drop file │────────▶│ decode → pad to a    │───────────▶│ pixel CNN        │
│ decode + shrink  │         │ square               │            │ 58,376 params    │
│ to ≤256 px       │         │                      │            └────────┬─────────┘
│                  │         │                      │  48×48 px  ┌────────▼─────────┐
│                  │         │ measure colour,      │───────────▶│ descriptor net   │
│                  │  JSON   │ shape and texture    │  124 nums  │ 9,040 params     │
│  one fruit name  │◀────────│ blend → one answer   │◀───────────┴──────────────────┘
└──────────────────┘         └──────────────────────┘
```

The browser decodes and shrinks the photo before uploading. That is deliberate:
the server's JPEG decoder is pure Python, so handing it a 12-megapixel phone
snapshot would take seconds. The browser has a fast native decoder, so it does
that part and uploads a small PNG instead. (The server can still decode full
JPEGs itself — `predict.py` relies on it — it is just slower.)

### Preprocessing

Images are **padded to a square, not squashed** into one. Aspect ratio is real
information here: a banana is elongated and an apple is round. Squashing every
upload into a square would throw that cue away.

### Two streams, one answer

The model looks at each photo in two different ways and averages the verdicts.

**1. The pixel CNN** — the classic approach, learned end to end from raw pixels:

```
input 3×32×32
  → Conv2D  8 filters 3×3  → ReLU → MaxPool 2×2      (8×15×15)
  → Conv2D 24 filters 3×3  → ReLU → MaxPool 2×2      (24×6×6)
  → Dense 64 → ReLU → Dropout 0.30
  → Dense 16 → Softmax
```

58,376 parameters. Backpropagation is verified against numerical finite
differences in the test suite (`tests/test_cnn.py`), which is what proves the
hand-written gradients are correct.

**2. The colour/shape descriptor** — the measurements a person would actually
use to name a fruit ([`src/fruitvision/features.py`](src/fruitvision/features.py)).
The background is estimated from the border of the picture and excluded, then
124 numbers are measured over the fruit itself: a hue histogram, saturation and
brightness histograms, RGB statistics, how round or elongated the outline is, a
4×4 map of colour and coverage (this is what sees a watermelon's stripes), and
how rough the surface looks. A small network
([`models/descriptor.py`](src/fruitvision/models/descriptor.py)) classifies that
vector:

```
descriptor 124 → standardise → Dense 64 → ReLU → Dropout 0.10 → Dense 16 → Softmax
```

**Why both?** With around a thousand training photographs, a from-scratch CNN
at 32×32 simply cannot learn what separates an apple from a peach — it scored
65% on the held-out set. The measured descriptor scores in the nineties on the
same images, because colour and shape are handed to it directly instead of
having to be discovered. The two are blended `0.75 × descriptor + 0.25 × CNN`
(`cnn_weight` in `config.py`): the descriptor decides the answer, and the CNN
keeps the reported confidence honest rather than pinned at 99%.

---

## Results

Held-out accuracy (`data/raw/test/`, 367 images the model never trains on):

| Stream | Accuracy |
|---|---|
| Pixel CNN alone | 65.1% |
| Colour/shape descriptor alone | 91.3% |
| **Combined (what the app uses)** | **91.6%** |

The training set is 1,217 images: the Fruits-360 studio photos plus 148 real
photographs (fruit in a hand, on a table, in a market crate) downloaded from
Wikimedia Commons -- see [`data/WEB_IMAGE_SOURCES.md`](data/WEB_IMAGE_SOURCES.md)
for the licence of every one. Without those real photos the model had only
ever seen fruit on a plain white studio backdrop and got about 8% of ordinary
internet photos right; with them it gets the great majority right.

### Honest limitations

- **Look-alikes still get confused.** Pears, bananas, avocados and peaches
  share a similar rounded/elongated shape and muted colour, so the descriptor
  occasionally swaps between them -- see `per_class_test_accuracy` in
  `artifacts/training_metrics.json` for exactly where.
- **The pixel CNN alone is weak** (65%) at 32x32 in pure Python; it is kept
  mainly as a confidence check on the stronger descriptor stream (see "Two
  streams, one answer" above).
- **A demo needs a decent photo.** A single fruit, filling a reasonable part
  of the frame, works far better than a cluttered scene or a tiny fruit in the
  corner. Use `demo_images/` for a guaranteed-correct set to show off with.

To improve any of this, add more photos (`tools/prepare_dataset.py`) and
retrain (`python train.py --stream descriptor` for a ~10 minute turnaround).

## Project layout

```
FruitVision/
├── run.py                      Start the web app          ← the main entry point
├── start.bat / start.command   Double-click launchers (Windows / macOS & Linux)
├── train.py                    Retrain the model (optional)
├── predict.py                  Classify images from the command line
├── RUN_GUIDE.md                Step-by-step instructions
│
├── frontend/                   Web interface (HTML / CSS / JS, no frameworks)
├── backend/
│   └── api.py                  HTTP server and REST endpoints
│
├── src/fruitvision/
│   ├── config.py               All settings and hyper-parameters
│   ├── imageio.py              PNG / JPEG / BMP decoders, PNG encoder
│   ├── features.py             The colour / shape / texture descriptor
│   ├── models/cnn.py           The neural network, backprop and Adam
│   ├── models/descriptor.py    The classifier that reads the descriptor
│   ├── data/dataset.py         Loading, splitting and augmentation
│   ├── utils/image.py          Resizing, letterboxing, preprocessing
│   ├── train.py                Training loop and cross-validation
│   └── inference.py            Model loading and prediction
│
├── data/
│   ├── README.md               Dataset description, sources and licence
│   ├── raw/custom/             Training images, one folder per fruit
│   └── raw/test/               Held-out images, never used for training
├── demo_images/                Ready-made demo photos the model gets right
├── artifacts/
│   ├── model.json              Trained weights (both streams)
│   └── training_metrics.json   Accuracy and loss from the last run
│
├── tests/                      80 tests, standard-library unittest
├── tools/
│   ├── prepare_dataset.py      Add new photos to the dataset
│   └── make_zip.py             Package the project for distribution
└── docs/                       Design documents and screenshots
```

---

## Commands

| Command | What it does |
|---|---|
| `python run.py` | Start the web app (opens the browser) |
| `python run.py --port 9000` | Use a specific port |
| `python run.py --no-browser` | Do not open a browser |
| `python predict.py photo.jpg` | Classify from the command line |
| `python -m unittest discover -s tests -t .` | Run all tests |
| `python train.py` | Retrain both streams (~50 min) |
| `python train.py --stream descriptor` | Retrain just the descriptor (~10 min) |
| `python train.py --loocv` | Leave-one-out cross-validation (small datasets only) |
| `python tools/prepare_dataset.py incoming/` | Add photos to the dataset |
| `python tools/make_demo_images.py` | Rebuild `demo_images/` after retraining |
| `python tools/make_zip.py` | Build a distributable zip |

---

## API

The server exposes a small REST API on the same port as the UI.

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /api/info` | Model metadata: classes, parameter count, accuracy |
| `POST /predict` | Classify an image |

`POST /predict` accepts a `multipart/form-data` upload under the field `file`,
a JSON body `{"image": "data:image/png;base64,..."}`, or raw image bytes with
an `image/*` content type.

```bash
curl -F "file=@data/raw/custom/Apple/apple_01.png" http://localhost:8000/predict
```

```json
{
  "prediction": "Apple",
  "confidence": 0.9998,
  "scores": {"Apple": 0.9998, "Peach": 0.0001, "...": 0.0},
  "source_size": [137, 160],
  "input_size": [32, 32]
}
```

---

## Configuration

Every tunable value lives in [`src/fruitvision/config.py`](src/fruitvision/config.py):
input size, filter counts, dropout, learning rate, augmentation strength and the
class list. Adding another fruit is a matter of adding its name to `class_names`,
dropping photos into `data/raw/custom/<Name>/`, and retraining — the output
layer, the API and the interface all size themselves from that list.
