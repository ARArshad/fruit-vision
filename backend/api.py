"""HTTP backend for FruitVision, built on the standard library only.

``http.server`` replaces FastAPI/Uvicorn so the project needs no installed
packages.  The same server hosts the web UI *and* the REST API on one port,
which also means the browser never makes a cross-origin request.

Endpoints
---------
``GET  /``          the web interface
``GET  /health``    liveness probe
``GET  /api/info``  model metadata (classes, input size, accuracy)
``POST /predict``   classify an image

``/predict`` accepts either a ``multipart/form-data`` upload under the field
name ``file``, or ``application/json`` of the form ``{"image": "<data-url>"}``,
which is what the browser sends after it has decoded and shrunk the photo.
"""

from __future__ import annotations

import base64
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fruitvision.config import CONFIG  # noqa: E402
from fruitvision.imageio import UnsupportedImageError  # noqa: E402
from fruitvision.inference import (  # noqa: E402
    ModelNotTrainedError,
    load_descriptor,
    load_model,
    predict_image,
    top_prediction,
)
from fruitvision.utils.image import load_image_from_bytes  # noqa: E402

FRONTEND_DIR = ROOT / "frontend"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# An over-sized body is read and discarded so we can still reply politely,
# but only up to this much -- beyond it we simply hang up.
DRAIN_LIMIT_BYTES = 200 * 1024 * 1024

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
}


class ApiError(Exception):
    """An error with an HTTP status code to report back to the client."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ---------------------------------------------------------------------------
# Request body parsing
# ---------------------------------------------------------------------------


def parse_multipart(body: bytes, content_type: str) -> Dict[str, Tuple[str, bytes]]:
    """Minimal ``multipart/form-data`` parser.

    Returns ``{field_name: (filename, raw_bytes)}``.  Python's ``cgi`` module
    did this for us but was removed in 3.13, and the format is simple enough to
    handle directly.
    """

    marker = "boundary="
    if marker not in content_type:
        raise ApiError(400, "Malformed upload: no multipart boundary.")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    delimiter = b"--" + boundary.encode("utf-8")

    fields: Dict[str, Tuple[str, bytes]] = {}
    for chunk in body.split(delimiter)[1:]:
        if chunk[:2] == b"--":  # closing delimiter
            break
        if chunk[:2] == b"\r\n":
            chunk = chunk[2:]
        head, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]

        name = filename = ""
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                for piece in line.split(";")[1:]:
                    key, _, value = piece.strip().partition("=")
                    value = value.strip('"')
                    if key == "name":
                        name = value
                    elif key == "filename":
                        filename = value
        if name:
            fields[name] = (filename, data)
    return fields


def decode_data_url(value: str) -> bytes:
    """Extract the bytes from a ``data:image/...;base64,...`` URL."""

    if not value.startswith("data:"):
        raise ApiError(400, "Expected a data URL for the image.")
    header, _, payload = value.partition(",")
    if not payload:
        raise ApiError(400, "Malformed data URL.")
    try:
        if "base64" in header:
            return base64.b64decode(payload)
        from urllib.parse import unquote_to_bytes

        return unquote_to_bytes(payload)
    except Exception as exc:  # pragma: no cover - defensive
        raise ApiError(400, "Could not decode the image data.") from exc


def extract_image(body: bytes, content_type: str) -> bytes:
    """Pull raw image bytes out of a request in any of the supported encodings."""

    lowered = content_type.lower()

    if lowered.startswith("multipart/form-data"):
        fields = parse_multipart(body, content_type)
        for key in ("file", "image"):
            if key in fields:
                return fields[key][1]
        raise ApiError(400, "No file field found in the upload.")

    if lowered.startswith("application/json"):
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "Request body is not valid JSON.") from exc
        image = payload.get("image")
        if not isinstance(image, str):
            raise ApiError(400, "JSON body must contain an 'image' data URL.")
        return decode_data_url(image)

    if lowered.startswith("image/"):
        return body  # a bare image POSTed directly

    raise ApiError(415, "Unsupported content type '%s'." % (content_type or "unknown"))


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


def handle_predict(body: bytes, content_type: str) -> Dict[str, object]:
    """Classify an uploaded image and describe the result."""

    if not body:
        raise ApiError(400, "The request body was empty.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "That image is larger than the 25 MB limit.")

    image_bytes = extract_image(body, content_type)
    if not image_bytes:
        raise ApiError(400, "No image data was received.")

    try:
        network, _ = load_model()
    except ModelNotTrainedError as exc:
        raise ApiError(503, str(exc)) from exc

    try:
        image = load_image_from_bytes(image_bytes)
    except UnsupportedImageError as exc:
        raise ApiError(400, str(exc)) from exc
    except Exception as exc:
        raise ApiError(400, "That file could not be read as an image (%s)." % exc) from exc

    scores = predict_image(image)
    label, confidence = top_prediction(scores)
    return {
        # The one answer the interface shows. The full distribution is still
        # reported for anyone calling the API directly.
        "prediction": label,
        "confidence": confidence,
        "scores": scores,
        "source_size": list(image.size),
        "input_size": [network.config.img_size, network.config.img_size],
        "descriptor_size": network.config.descriptor_size,
    }


def handle_info() -> Dict[str, object]:
    """Report what the server knows about its model."""

    info: Dict[str, object] = {
        "name": "FruitVision",
        "classes": list(CONFIG.class_names),
        "model_loaded": False,
    }
    try:
        network, class_names = load_model()
    except ModelNotTrainedError as exc:
        info["error"] = str(exc)
        return info

    descriptor = load_descriptor()
    info["model_loaded"] = True
    info["classes"] = class_names
    info["parameters"] = network.num_parameters() + (
        descriptor.num_parameters() if descriptor else 0
    )
    info["input_size"] = network.config.img_size
    info["streams"] = ["pixel CNN"] + (["colour/shape descriptor"] if descriptor else [])

    if CONFIG.metrics_path.exists():
        try:
            with CONFIG.metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            info["metrics"] = {
                key: metrics[key]
                for key in ("train_accuracy", "val_accuracy", "test_accuracy",
                            "trained_at", "source_images")
                if key in metrics
            }
        except (OSError, json.JSONDecodeError):
            pass
    return info


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class FruitVisionHandler(BaseHTTPRequestHandler):
    server_version = "FruitVision/1.0"
    protocol_version = "HTTP/1.1"
    quiet = False

    # -- helpers -----------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"detail": message, "error": message})

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (FRONTEND_DIR / relative).resolve()
        try:
            target.relative_to(FRONTEND_DIR.resolve())  # block path traversal
        except ValueError:
            self._send_error_json(403, "Forbidden.")
            return

        if not target.is_file():
            self._send_error_json(404, "Not found: %s" % path)
            return

        content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send(200, target.read_bytes(), content_type)

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        if length > MAX_UPLOAD_BYTES:
            # Read the body away before replying. If we answered immediately the
            # client would still be sending, and it would see a broken pipe
            # instead of the "too large" message we are trying to give it.
            self._drain(length)
            raise ApiError(413, "That image is larger than the 25 MB limit.")
        return self.rfile.read(length)

    def _drain(self, length: int) -> None:
        """Discard an over-sized request body, giving up if it is absurd."""

        if length > DRAIN_LIMIT_BYTES:
            self.close_connection = True
            return
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                self.close_connection = True
                return
            remaining -= len(chunk)

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        try:
            if path == "/health":
                self._send_json(200, {"status": "ok"})
            elif path == "/api/info":
                self._send_json(200, handle_info())
            else:
                self._serve_static(path)
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception:
            traceback.print_exc()
            self._send_error_json(500, "Internal server error.")

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path != "/predict":
                self._send_error_json(404, "Not found: %s" % path)
                return
            body = self._read_body()
            result = handle_predict(body, self.headers.get("Content-Type", ""))
            self._send_json(200, result)
        except ApiError as exc:
            self._send_error_json(exc.status, exc.message)
        except Exception:
            traceback.print_exc()
            self._send_error_json(500, "Internal server error while classifying.")

    def log_message(self, fmt: str, *args) -> None:
        if not self.quiet:
            sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create the HTTP server (call ``serve_forever`` to start it)."""

    server = ThreadingHTTPServer((host, port), FruitVisionHandler)
    server.daemon_threads = True
    return server
