"""Integration tests that drive the HTTP server over a real socket."""

from __future__ import annotations

import base64
import json
import pathlib
import sys
import threading
import unittest
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend.api import ApiError, create_server, decode_data_url, parse_multipart
from fruitvision.config import CONFIG
from fruitvision.data.dataset import discover_images
from fruitvision.imageio import Image, encode_png
from fruitvision.inference import get_default_model_path

MODEL_AVAILABLE = get_default_model_path().exists()
requires_model = unittest.skipUnless(
    MODEL_AVAILABLE, "No trained model. Run: python train.py"
)


def multipart_body(field: str, filename: str, payload: bytes, boundary: str) -> bytes:
    return (
        ("--%s\r\n" % boundary).encode()
        + ('Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
           % (field, filename)).encode()
        + b"Content-Type: image/png\r\n\r\n"
        + payload
        + ("\r\n--%s--\r\n" % boundary).encode()
    )


class BodyParsingTests(unittest.TestCase):
    def test_multipart_round_trip(self):
        boundary = "----test123"
        body = multipart_body("file", "fruit.png", b"\x89PNG-data", boundary)
        fields = parse_multipart(body, "multipart/form-data; boundary=" + boundary)

        self.assertIn("file", fields)
        self.assertEqual(fields["file"], ("fruit.png", b"\x89PNG-data"))

    def test_multipart_without_a_boundary_is_rejected(self):
        with self.assertRaises(ApiError) as caught:
            parse_multipart(b"whatever", "multipart/form-data")
        self.assertEqual(caught.exception.status, 400)

    def test_base64_data_url_is_decoded(self):
        payload = base64.b64encode(b"hello").decode()
        self.assertEqual(decode_data_url("data:image/png;base64," + payload), b"hello")

    def test_non_data_urls_are_rejected(self):
        with self.assertRaises(ApiError):
            decode_data_url("https://example.com/cat.png")


class ServerTestCase(unittest.TestCase):
    """Boots the real server on an ephemeral port for each test class."""

    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0)
        cls.server.RequestHandlerClass.quiet = True
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def url(self, path: str) -> str:
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def request(self, path, data=None, content_type=None, method=None):
        """Return ``(status, parsed_body_or_bytes)``, never raising on 4xx/5xx."""

        req = urllib.request.Request(self.url(path), data=data, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
                status = response.status
                header = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            header = exc.headers.get("Content-Type", "")
        if header.startswith("application/json"):
            return status, json.loads(raw.decode("utf-8"))
        return status, raw


class StaticAndHealthTests(ServerTestCase):
    def test_health_reports_ok(self):
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_root_serves_the_interface(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>FruitVision</title>", body)

    def test_static_assets_are_served(self):
        for path, needle in (("/styles.css", b".dropzone"), ("/app.js", b"predict")):
            with self.subTest(path=path):
                status, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn(needle, body)

    def test_unknown_paths_return_404(self):
        status, _ = self.request("/does-not-exist")
        self.assertEqual(status, 404)

    def test_path_traversal_is_blocked(self):
        status, _ = self.request("/../run.py")
        self.assertIn(status, (403, 404))

    def test_info_endpoint_describes_the_model(self):
        status, body = self.request("/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(body["name"], "FruitVision")
        self.assertIn("classes", body)


class PredictEndpointTests(ServerTestCase):
    def test_empty_body_is_rejected(self):
        status, body = self.request("/predict", data=b"", content_type="application/json",
                                    method="POST")
        self.assertEqual(status, 400)
        self.assertIn("detail", body)

    def test_non_image_upload_is_rejected(self):
        boundary = "----test999"
        payload = multipart_body("file", "notes.txt", b"not an image at all", boundary)
        status, body = self.request(
            "/predict", data=payload,
            content_type="multipart/form-data; boundary=" + boundary, method="POST",
        )
        self.assertEqual(status, 400)
        self.assertIn("detail", body)

    def test_unsupported_content_type_is_rejected(self):
        status, _ = self.request("/predict", data=b"abc", content_type="text/plain",
                                 method="POST")
        self.assertEqual(status, 415)

    def test_get_on_predict_is_not_found(self):
        status, _ = self.request("/predict")
        self.assertEqual(status, 404)

    def test_corrupt_images_produce_a_readable_error(self):
        cases = {
            "truncated png": (b"\x89PNG\r\n\x1a\n" + b"garbage" * 8, "image/png"),
            "truncated jpeg": (b"\xff\xd8\xff\xe0" + b"junk" * 16, "image/jpeg"),
        }
        for name, (payload, content_type) in cases.items():
            with self.subTest(case=name):
                status, body = self.request("/predict", data=payload,
                                            content_type=content_type, method="POST")
                self.assertEqual(status, 400)
                self.assertTrue(body["detail"])

    def test_bad_json_payloads_are_reported(self):
        cases = {
            "not json": (b"{{{", 400),
            "missing image key": (b'{"nope": 1}', 400),
            "bad base64": (b'{"image":"data:image/png;base64,!!!"}', 400),
            "not a data url": (b'{"image":"https://example.com/a.png"}', 400),
        }
        for name, (payload, expected) in cases.items():
            with self.subTest(case=name):
                status, body = self.request("/predict", data=payload,
                                            content_type="application/json", method="POST")
                self.assertEqual(status, expected)
                self.assertIn("detail", body)

    def test_server_stays_healthy_after_a_rejected_request(self):
        self.request("/predict", data=b"{{{", content_type="application/json", method="POST")
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    @requires_model
    def test_json_data_url_upload_returns_a_prediction(self):
        path, label = discover_images(CONFIG)[0]
        data_url = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        payload = json.dumps({"image": data_url}).encode()

        status, body = self.request("/predict", data=payload,
                                    content_type="application/json", method="POST")

        self.assertEqual(status, 200)
        self.assertEqual(body["prediction"], CONFIG.class_names[label])
        self.assertGreaterEqual(body["confidence"], 0.0)
        self.assertLessEqual(body["confidence"], 1.0)
        self.assertEqual(set(body["scores"]), set(CONFIG.class_names))

    @requires_model
    def test_multipart_upload_returns_a_prediction(self):
        path, label = discover_images(CONFIG)[-1]
        boundary = "----test555"
        payload = multipart_body("file", path.name, path.read_bytes(), boundary)

        status, body = self.request(
            "/predict", data=payload,
            content_type="multipart/form-data; boundary=" + boundary, method="POST",
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["prediction"], CONFIG.class_names[label])

    @requires_model
    def test_raw_image_post_is_accepted(self):
        blob = encode_png(Image(8, 8, bytearray(bytes((220, 30, 30)) * 64)))
        status, body = self.request("/predict", data=blob, content_type="image/png",
                                    method="POST")

        self.assertEqual(status, 200)
        self.assertIn(body["prediction"], CONFIG.class_names)


if __name__ == "__main__":
    unittest.main()
