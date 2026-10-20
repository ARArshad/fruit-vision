# Documentation

## Design documents

| File | Contents |
|---|---|
| `FruitVision_SDD.pdf` | Software Design Document, chapters 1–5 |
| `FruitVision_SDD_full.pdf` | Full SDD including the supplementary chapters |
| `FruitVision_Context_Implementation_Testing.pdf` | Chapters 6 (System Context), 7 (Implementation), 8 (Testing) |

> **Note on the technology stack.** These PDFs describe the original
> implementation, which used TensorFlow/Keras, FastAPI, Uvicorn, Pillow and
> NumPy. The project has since been rewritten to use **only the Python standard
> library**, so that it runs anywhere with no installation. The system design —
> layered architecture, the CNN, the REST endpoints, the preprocessing pipeline —
> is unchanged; only the libraries differ.
>
> If you are updating the documents, the table below is the replacement for
> "Table 7.1 — Technology stack by layer":
>
> | Layer | Technology | Purpose |
> |---|---|---|
> | Presentation | HTML, CSS, JavaScript | Image upload UI and result display in the browser. |
> | Application | Python `http.server` | REST endpoints (`/health`, `/api/info`, `/predict`) and static hosting. |
> | Processing | `fruitvision.imageio`, `fruitvision.utils.image` | Decode PNG/JPEG/BMP, pad to square, resize to 32×32, scale to 0–1. |
> | Intelligence | `fruitvision.models.cnn` | Custom CNN (forward pass, backpropagation, Adam) in pure Python. |
> | Intelligence | `fruitvision.features`, `fruitvision.models.descriptor` | Hand-written colour/shape/texture descriptor and the small network that classifies it. Its verdict is averaged with the CNN's. |
> | Testing | `unittest` | 80 automated unit and integration tests. |
>
> And for "Table 6.2 — Operating environment", the server row becomes:
> *Python 3.8 or newer. No third-party packages.*

## Screenshots

| File | Status |
|---|---|
| `context-diagram.png` | Current — the system context is unchanged. |
| `arch-diagram.png` | Current — the layered architecture is unchanged. |
| `ui-home.png` | **Outdated** — the interface was redesigned (drag-and-drop upload, per-class score bars). Retake it. |
| `ui-prediction-apple.png` | **Outdated** — same reason. The confidence figure also changed: the old model reported 51.74%, the retrained one reports 100%. |
| `ui-prediction-banana.png` | **Outdated** — same reason. |
| `test-results.png` | **Outdated** — shows the old pytest run. The suite is now 80 `unittest` tests. |

The interface keeps the project's original green (`#2b8a3e`), so it still
matches the colour scheme in the diagrams, but the layout was modernised:
drag-and-drop upload, an image preview, a confidence bar per class, and a dark
mode that follows the operating system.

To retake the UI screenshots, run `python run.py`, open
<http://localhost:8000>, and capture the page before and after a prediction.
To retake the test screenshot, run:

```bash
python -m unittest discover -s tests -t .
```
