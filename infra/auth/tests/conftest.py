# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def openfga_http(tmp_path):
    """Canned remote decisions test the adapter; prove.py exercises the real engine."""
    state = {"allowed": True, "status": 200, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(
                {"path": self.path, "payload": payload, "authorization": self.headers.get("Authorization")}
            )
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"allowed": state["allowed"]}).encode())

        def log_message(self, *_args):
            pass

    token_file = tmp_path / "openfga.token"
    token_file.write_text("synthetic-openfga-key")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield {
                "api_url": f"http://127.0.0.1:{server.server_port}",
                "token_file": token_file,
                "store_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "model_id": "01ARZ3NDEKTSV4RRFFQ69G5FAW",
                "state": state,
            }
        finally:
            server.shutdown()
            thread.join()
