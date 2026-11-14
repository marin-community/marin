# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from marin.inference.openai_chat import OpenAIChatClient


@pytest.fixture
def chat_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["x-priority"], body))
            if body["model"] == "wrong_protocol":
                self.wfile.write(b"\x00\x00\x18\x04\x00\x00\x00\x00\x00")
                self.close_connection = True
                return
            status = 503 if body["model"] == "unavailable" else 200
            payload = json.dumps({"choices": [{"message": {"content": None, "tool_calls": []}}]}).encode()
            self.send_response(status)
            if body["model"] == "truncated_chunked":
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b'10\r\n{"choice')
                self.close_connection = True
                return
            if body["model"] == "truncated_length":
                self.send_header("Content-Length", str(len(payload) + 10))
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True
                return
            self.send_header("Content-Length", str(len(payload)))
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


def test_direct_chat_preserves_nullable_tool_response_and_sends_once(chat_server):
    url, requests = chat_server
    client = OpenAIChatClient(url, "fixture-token", timeout=2)
    body = {"model": "fixture", "messages": [{"role": "user", "content": "review"}], "max_tokens": 512}
    result = client.complete(body)
    assert result["choices"][0]["message"]["content"] is None
    assert requests == [("/v1/chat/completions", "bulk", body)]
    with pytest.raises(ConnectionError, match="HTTP 503"):
        client.complete({"model": "unavailable"})
    assert len(requests) == 2


@pytest.mark.parametrize("model", ["truncated_chunked", "truncated_length"])
def test_interrupted_response_is_a_retryable_transport_failure(chat_server, model):
    url, requests = chat_server
    client = OpenAIChatClient(url, "fixture-token", timeout=2)
    with pytest.raises(ConnectionError, match="response ended"):
        client.complete({"model": model})
    assert len(requests) == 1


def test_wrong_protocol_at_stale_port_is_a_retryable_transport_failure(chat_server):
    url, requests = chat_server
    client = OpenAIChatClient(url, "fixture-token", timeout=2)
    with pytest.raises(ConnectionError):
        client.complete({"model": "wrong_protocol"})
    assert len(requests) == 1
