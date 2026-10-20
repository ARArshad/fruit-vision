#!/usr/bin/env python3
"""FruitVision launcher -- start the web application.

    python run.py

That is the whole setup.  No virtual environment, no ``pip install``: this
project runs on a stock Python 3.8+ interpreter using only the standard
library.  The command starts a local web server and opens the interface in
your browser.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser
from pathlib import Path

MIN_PYTHON = (3, 8)

if sys.version_info < MIN_PYTHON:
    sys.stderr.write(
        "FruitVision needs Python %d.%d or newer -- this is Python %s.\n"
        % (MIN_PYTHON[0], MIN_PYTHON[1], ".".join(map(str, sys.version_info[:3])))
    )
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.api import create_server  # noqa: E402
from fruitvision.config import CONFIG  # noqa: E402


def find_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    """Return the first free port at or after ``preferred``."""

    for offset in range(attempts):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit(
        "Could not find a free port between %d and %d. Close some applications "
        "or pass --port with a different number." % (preferred, preferred + attempts - 1)
    )


def check_model() -> bool:
    """Warn -- but do not abort -- when the trained model is missing."""

    if CONFIG.model_path.exists():
        return True
    print("  ! No trained model found at %s" % CONFIG.model_path)
    print("    The interface will load, but predictions will fail.")
    print("    Create one with:  python train.py")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FruitVision web application.")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind (default: 127.0.0.1, this machine only)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--quiet", action="store_true", help="suppress per-request logging")
    args = parser.parse_args()

    port = find_free_port(args.host, args.port)
    server = create_server(args.host, port)
    server.RequestHandlerClass.quiet = args.quiet

    url = "http://%s:%d/" % ("localhost" if args.host == "127.0.0.1" else args.host, port)

    banner = ["", "  FruitVision is running", "  ----------------------", "  Open:  %s" % url]
    if port != args.port:
        banner.append("  (port %d was busy, using %d instead)" % (args.port, port))
    print("\n".join(banner))
    check_model()
    print("\n  Press Ctrl+C to stop.\n")
    # Flush explicitly: when output is redirected to a file it is block-buffered,
    # and the banner would otherwise not appear until the server stops.
    sys.stdout.flush()

    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down. Goodbye.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
