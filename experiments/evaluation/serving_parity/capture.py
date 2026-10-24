# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record canary wire contracts without changing the evaluator's requests."""

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests


def echo_contract(request: dict, response: dict) -> None:
    """Detect the historical generation-cap override on loglikelihood requests."""
    if not request.get("echo"):
        return
    if request.get("max_tokens") != 1:
        raise ValueError("Loglikelihood request must generate exactly one unscored token")
    usage = response["usage"]
    for choice in response["choices"]:
        logs = choice["logprobs"]
        expected = usage["prompt_tokens"] + usage["completion_tokens"]
        if len(logs["tokens"]) != expected or len(logs["token_logprobs"]) != expected:
            raise ValueError("Echo array length does not match prompt plus generated suffix")
        if any(value is None for value in logs["token_logprobs"][1:]):
            raise ValueError("Missing noninitial echoed prompt score")


@contextlib.contextmanager
def capture_requests(endpoint: str, output: Path):
    """Forward local evaluator traffic; archive bodies, never the endpoint capability."""
    lock = threading.Lock()
    stream = output.open("x")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args):  # noqa: A002 - stdlib override parameter
            pass  # Base handler logs paths; capability-bearing upstream URLs must stay private.

        def do_GET(self):
            self.forward(None)

        def do_POST(self):
            self.forward(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def forward(self, payload):
            path = self.path.removeprefix("/v1")
            try:
                upstream = requests.request(self.command, endpoint.rstrip("/") + path, json=payload, timeout=180)
                body = upstream.content
                status = upstream.status_code
                if payload is not None:
                    with lock:
                        stream.write(
                            json.dumps({"request": payload, "status": status, "response": upstream.json()}) + "\n"
                        )
                        stream.flush()
                self.send_response(status)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except requests.RequestException:
                self.send_error(502, "Upstream transport failed")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        stream.close()
