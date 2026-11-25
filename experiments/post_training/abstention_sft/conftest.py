# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class ChatServer(ThreadingHTTPServer):
    """A local OpenAI-compatible endpoint that answers each request with the next scripted reply."""

    requests: list[dict]
    replies: list[str]


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server: ChatServer = self.server  # pyrefly: ignore[bad-assignment]
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(request)
        reply = server.replies[min(len(server.requests), len(server.replies)) - 1]
        body = json.dumps(
            {
                "id": "chatcmpl-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": request["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
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
def chat_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    server = ChatServer(("127.0.0.1", 0), ChatHandler)
    server.requests = []
    server.replies = ["A"]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
