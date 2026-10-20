# FruitVision — How to Run It

**You do not need to install anything.** No `pip install`, no virtual
environment, no internet connection. If your computer has Python, you can run
this project.

---

## The quick way: double-click

Unzip the folder, then double-click:

| System | File |
|---|---|
| **Windows** | `start.bat` |
| **macOS / Linux** | `start.command` — on macOS, right-click it and choose **Open** the first time |

Your browser opens with FruitVision ready to use. If that works, you are done —
skip to [Step 4](#step-4--identify-a-fruit).

If nothing happens, or a window flashes and disappears, follow the numbered
steps below instead.

---

## Step 1 — Check you have Python

Open a terminal:

| System | How to open a terminal |
|---|---|
| **Windows** | Press `Win`, type `cmd`, press Enter |
| **macOS** | Press `Cmd + Space`, type `Terminal`, press Enter |
| **Linux** | Press `Ctrl + Alt + T` |

Type this and press Enter:

```
python --version
```

You should see something like `Python 3.11.4`. **Any version 3.8 or newer works.**

<details>
<summary>What if that command fails?</summary>

- Try `python3 --version` instead. If that works, use `python3` everywhere
  below instead of `python`.
- If neither works, Python is not installed. Get it free from
  [python.org/downloads](https://www.python.org/downloads/).
  **On Windows, tick "Add Python to PATH"** on the first screen of the
  installer, then close and reopen your terminal.
</details>

---

## Step 2 — Go into the project folder

Unzip the project somewhere you can find it, then in the terminal type `cd `
(with a space) and drag the unzipped **FruitVision** folder onto the terminal
window. Press Enter.

Or type the path yourself:

```
cd path/to/FruitVision
```

To confirm you are in the right place, run `dir` (Windows) or `ls` (macOS /
Linux). You should see `run.py` listed.

---

## Step 3 — Start it

```
python run.py
```

You will see:

```
  FruitVision is running
  ----------------------
  Open:  http://localhost:8000/

  Press Ctrl+C to stop.
```

Your browser opens automatically. If it does not, open it yourself and go to
**http://localhost:8000**

---

## Step 4 — Identify a fruit

1. Click **Choose an image** and pick a fruit photo.
2. Click **Predict**.
3. The system shows the fruit name and confidence percentage.

You get one answer: **Prediction: Apple** and **Confidence: 78.90%**.

It recognises 16 fruits: Apple, Avocado, Banana, Cherry, Grape, Kiwi, Lemon, Lime, Mango,
Orange, Peach, Pear, Pineapple, Pomegranate, Strawberry and Watermelon.

**For a demo, use the `demo_images/` folder.** Every picture in it is a real
photograph (not a plain studio shot) that the trained model names correctly,
so nothing goes wrong in front of an audience. `demo_images/README.md` lists
what each file should say and where the photo came from.

More pictures to try are in `data/raw/test/` — one folder per fruit, also never
used for training.

---

## Step 5 — Stop it

Click the terminal window and press **Ctrl + C**.

---

## Optional extras

Everything below is optional — the app works without any of it.

### Classify from the command line

```
python predict.py my_photo.jpg
```

```
  my_photo.jpg                   -> Banana    99.8%   (Banana 99.8%, Apple 0.2%)
```

### Run the tests

```
python -m unittest discover -s tests -t .
```

Expect `Ran 80 tests ... OK`.

### Retrain the model

The trained model already ships with the project, so you never have to do this.
If you want to reproduce it:

```
python train.py
```

This takes roughly **50 minutes** — the model is written in plain Python, which
is slower than a compiled library but needs nothing installed. It prints
progress as it goes and overwrites `artifacts/model.json`.

The model has two halves. Retraining only the fast one takes about 10 minutes:

```
python train.py --stream descriptor
```

After retraining, refresh the demo folder so it matches the new model:

```
python tools/make_demo_images.py
```

### Add your own training photos

```
python tools/prepare_dataset.py incoming/
```

where `incoming/` holds a folder per class:

```
incoming/
  Apple/    my_apple_1.jpg
  Banana/   my_banana_1.jpg
```

Then retrain with `python train.py`.

### Package it up to send to someone

```
python tools/make_zip.py
```

Creates `dist/FruitVision.zip`, ready to email or upload.

---

## If something goes wrong

| What you see | What to do |
|---|---|
| `python: command not found` | Use `python3` instead, or install Python (Step 1). |
| `'python' is not recognized...` (Windows) | Python is not on your PATH. Reinstall it and tick **"Add Python to PATH"**. |
| `FruitVision needs Python 3.8 or newer` | Your Python is too old. Install a current version from python.org. |
| Browser shows "can't connect" | Make sure the terminal still shows *FruitVision is running*. Check the exact address it printed. |
| Port 8000 is busy | The app moves to the next free port automatically and prints it. Or run `python run.py --port 9000`. |
| `No trained model found` | Run `python train.py` (about 35 minutes). |
| Browser cannot read the image | Use a normal JPG, PNG or BMP. HEIC photos from iPhones are not supported — export as JPEG first. |
| `start.command` will not open (macOS) | Right-click it and choose **Open**, then confirm. macOS blocks double-clicking downloaded scripts the first time. |
| `start.command` opens in a text editor | Right-click → **Open With** → **Terminal**. |
| Prediction seems wrong | It only knows the 16 fruits listed above, and most of its training images are studio shots on white backgrounds. A plain background and a single fruit filling the frame work best — see the honest limits in [README.md](README.md). |

---

## What is actually running?

Nothing is installed and nothing is downloaded. `run.py` starts a small web
server from Python's own `http.server` module, which serves the page in
`frontend/` and answers prediction requests using the neural network in
`src/fruitvision/`. Everything stays on your machine.
