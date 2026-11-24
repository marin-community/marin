# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaluation worker's model client against an inference proxy that sheds load."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from marin.evaluation.task_worker import model_client
from rolloutengine.chat import ChatModel
from rolloutengine.models import ModelRequest

COMPLETION = {
    "id": "served",
    "object": "chat.completion",
    "created": 0,
    "model": "solver",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ls"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
OVERLOADED = {"error": "too many pending proxy requests; back off and retry"}


@pytest.fixture
def overloaded_proxy(monkeypatch):
    """A chat endpoint that rejects its first request as the inference proxy does when full."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = (429, OVERLOADED) if len(requests) == 1 else (200, COMPLETION)
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
async def test_a_request_the_proxy_sheds_is_retried_rather_than_ending_the_attempt(overloaded_proxy):
    endpoint, requests = overloaded_proxy
    async with model_client(endpoint, "key", timeout=30) as client:
        turn = await ChatModel(client, "solver", max_tokens=None)(
            ModelRequest(({"role": "user", "content": "List the files."},), {})
        )

    assert turn.message["content"] == "ls"
    assert len(requests) == 2
