"""The browser build (web/model.js) must agree with the Python model.

Every demo photo is decoded and shrunk once in Python, then classified twice
on exactly the same pixels: by ``predict_image`` and by the JavaScript port
running under Node.  The class probabilities must match to within floating
point noise, so a change to the Python preprocessing or layers that is not
mirrored in model.js fails here instead of silently on the live site.

Skipped when Node.js is not installed or no trained model exists.
"""

from __future__ import annotations

import base64
import json
import pathlib
import shutil
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fruitvision.imageio import decode_image
from fruitvision.inference import get_default_model_path, predict_image
from fruitvision.utils.image import resize

NODE = shutil.which("node")
MODEL_PATH = get_default_model_path()
DEMO_DIR = ROOT / "demo_images"
WEB_MODEL = ROOT / "web" / "model.js"

# What the web page does before classifying: shrink the longest edge to 256.
UPLOAD_MAX_EDGE = 256

NODE_SCRIPT = r"""
const fs = require("fs");
const fv = require(process.argv[1]);
const model = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
const out = cases.map((c) => {
  const pixels = Uint8Array.from(Buffer.from(c.pixels, "base64"));
  return fv.predictImage(model, { width: c.width, height: c.height, pixels }).scores;
});
process.stdout.write(JSON.stringify(out));
"""


def _shrink(image):
    scale = min(1.0, UPLOAD_MAX_EDGE / float(max(image.width, image.height)))
    return resize(
        image,
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )


@unittest.skipUnless(NODE, "Node.js is not installed")
@unittest.skipUnless(MODEL_PATH.exists(), "no trained model in artifacts/")
class WebParityTest(unittest.TestCase):
    def test_javascript_matches_python(self):
        photos = sorted(DEMO_DIR.glob("*.jpg"))
        self.assertTrue(photos, "no demo images to compare on")

        images = [_shrink(decode_image(p.read_bytes())) for p in photos]
        cases = [
            {
                "width": im.width,
                "height": im.height,
                "pixels": base64.b64encode(bytes(im.pixels)).decode("ascii"),
            }
            for im in images
        ]

        result = subprocess.run(
            [NODE, "-e", NODE_SCRIPT, str(WEB_MODEL), str(MODEL_PATH)],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        )
        js_scores = json.loads(result.stdout)

        worst = 0.0
        for photo, image, js in zip(photos, images, js_scores):
            py = predict_image(image)
            self.assertEqual(
                max(py, key=py.get), max(js, key=js.get),
                "%s: Python and the web build disagree on the fruit" % photo.name,
            )
            for name, p in py.items():
                worst = max(worst, abs(p - js[name]))
        self.assertLess(worst, 1e-9, "largest probability gap %.3g" % worst)


if __name__ == "__main__":
    unittest.main()
