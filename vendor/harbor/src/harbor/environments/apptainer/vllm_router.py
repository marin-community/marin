#!/usr/bin/env python3
"""Tiny round-robin HTTP proxy for multiple vLLM backends.

Reads backend URLs from a file (one per line) and load-balances OpenAI-compatible
/v1/* requests across them using round-robin. Re-reads the backends file on
every dispatch, so new backends can be added on the fly by appending to it.

Usage:
    python vllm_router.py --port 8001 --backends backends.txt

backends.txt entries: full base URL incl. /v1 path, e.g.
    http://jrc0359:8000/v1
    http://jrc0360:8000/v1
"""

import argparse
import itertools
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

_BACKENDS = []  # list of base URLs
_BACKENDS_LOCK = threading.Lock()
_CYCLER = None
_BACKENDS_FILE = None
_LAST_LOAD = 0.0
_RELOAD_INTERVAL = 5.0  # seconds


def _load_backends_locked():
    global _BACKENDS, _CYCLER, _LAST_LOAD
    try:
        with open(_BACKENDS_FILE) as f:
            new_backends = [
                ln.strip().rstrip("/")
                for ln in f
                if ln.strip() and not ln.startswith("#")
            ]
        if new_backends != _BACKENDS:
            _BACKENDS = new_backends
            _CYCLER = itertools.cycle(_BACKENDS) if _BACKENDS else None
            print(f"[router] loaded {len(_BACKENDS)} backends: {_BACKENDS}", flush=True)
        _LAST_LOAD = time.time()
    except FileNotFoundError:
        pass


def _next_backend():
    with _BACKENDS_LOCK:
        if time.time() - _LAST_LOAD > _RELOAD_INTERVAL:
            _load_backends_locked()
        if not _BACKENDS:
            return None
        return next(_CYCLER)


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet logs
        pass

    def _proxy(self, body=None):
        backend = _next_backend()
        if not backend:
            self.send_error(503, "no backends available")
            return
        url = (
            backend + self.path
        )  # self.path includes leading slash (e.g. /v1/chat/completions)
        # Drop the /v1 prefix from backend if path already contains it
        # backend ends without trailing slash; self.path starts with /
        # if backend already ends with /v1 and path starts with /v1, fix
        if backend.endswith("/v1") and self.path.startswith("/v1"):
            url = backend[:-3] + self.path
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        req = urllib.request.Request(
            url, data=body, method=self.command, headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                resp_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "connection")
                }
                payload = resp.read()
                for k, v in resp_headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as e:
            payload = e.read()
            self.send_response(e.code)
            self.send_header(
                "Content-Type", e.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            err = json.dumps({"error": str(e), "backend": backend, "url": url}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_GET(self):
        if self.path == "/router/status":
            with _BACKENDS_LOCK:
                body = json.dumps({"backends": _BACKENDS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._proxy()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else None
        self._proxy(body)


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--backends", required=True, help="path to backends file (one URL per line)"
    )
    args = p.parse_args()

    global _BACKENDS_FILE
    _BACKENDS_FILE = args.backends
    _load_backends_locked()
    print(
        f"[router] listening on {args.host}:{args.port} (backends file: {_BACKENDS_FILE})",
        flush=True,
    )

    server = ThreadedServer((args.host, args.port), RouterHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
