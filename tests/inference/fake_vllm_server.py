# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake ``vllm serve`` child for test_vllm_server.py.

``VllmEnvironment`` appends ``serve <model> ... --port <p>``, so ``--port`` is read from argv.

Modes:
  serve                          Answer /v1/models with 200.
  hang <counter>                 Record the start, then sleep without becoming ready.
  stuck-fault <counter>          Log a streamer fault while the parent stays alive.
  exit                           Exit successfully without becoming ready.
"""

import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STREAMER_FAULT = "ValueError: Could not receive runai_response from libstreamer due to: b'File access error'"


def _record_start(counter: str) -> int:
    path = Path(counter)
    n = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(n))
    return n


def _serve() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data": [{"id": "fake-model"}]}')

        def log_message(self, *args) -> None:
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    mode = sys.argv[1]
    if mode == "serve":
        _serve()
    elif mode == "hang":
        _record_start(sys.argv[2])
        time.sleep(30)
    elif mode == "stuck-fault":
        _record_start(sys.argv[2])
        print(STREAMER_FAULT, file=sys.stderr, flush=True)
        time.sleep(30)
    elif mode == "exit":
        return
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
