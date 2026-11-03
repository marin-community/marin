# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class JudgeServer(ThreadingHTTPServer):
    """Record requests to a local OpenAI-compatible judge endpoint and answer each with ``reply``."""

    requests: list[dict]
    reply: str


class JudgeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server: JudgeServer = self.server  # pyrefly: ignore[bad-assignment]
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(request)
        body = json.dumps(
            {
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": server.reply}, "finish_reason": "stop"}
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


@pytest.fixture
def judge_server():
    server = JudgeServer(("127.0.0.1", 0), JudgeHandler)
    server.requests = []
    server.reply = "PASS"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
