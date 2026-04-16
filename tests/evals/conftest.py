# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fake OpenAI-compatible HTTP server for Tier B eval transport tests.

Answers `/v1/completions` and `/v1/chat/completions` with synthetic `logprobs`
shapes matching what lm-eval's `local-completions` / `local-chat-completions`
clients expect. Used to verify URL/payload encoding without launching a real
vLLM instance.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@dataclass
class FakeOpenAIServer:
    url: str
    received_requests: list[tuple[str, dict]] = field(default_factory=list)


def _build_fake_logprobs(prompt: str, *, n_choices: int = 1) -> list[dict]:
    """Shape lm-eval's `local-completions` expects for parse_logprobs.

    Each choice must have `logprobs.token_logprobs` and `logprobs.top_logprobs`
    aligned with the echoed prompt tokens.
    """
    # Treat each whitespace-split piece as a pseudo-token; lm-eval drives the
    # real tokenization on its side via `ctxlen`.
    tokens = prompt.split()
    token_logprobs = [-0.1 * (i + 1) for i in range(len(tokens))]
    top_logprobs = [{tok: lp, "other": lp - 1.0} for tok, lp in zip(tokens, token_logprobs, strict=True)]
    return [
        {
            "index": i,
            "text": "",
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": [0] * len(tokens),
            },
            "finish_reason": "length",
        }
        for i in range(n_choices)
    ]


def _build_fake_chat_message(content: str = "fake response") -> dict:
    return {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }


def _make_handler(received: list[tuple[str, dict]]):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence noisy default logging
            pass

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b"{}"
            return json.loads(body) if body else {}

        def _respond_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._respond_json({"object": "list", "data": [{"id": "fake", "object": "model"}]})
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            body = self._read_json()
            received.append((self.path, body))
            if self.path == "/v1/completions":
                prompt = body.get("prompt", "")
                prompts = prompt if isinstance(prompt, list) else [prompt]
                choices: list[dict] = []
                for i, p in enumerate(prompts):
                    fake_choice = _build_fake_logprobs(p if isinstance(p, str) else " ".join(map(str, p)))[0]
                    fake_choice["index"] = i
                    choices.append(fake_choice)
                self._respond_json(
                    {
                        "id": "cmpl-fake",
                        "object": "text_completion",
                        "model": body.get("model", "fake"),
                        "choices": choices,
                    }
                )
                return
            if self.path == "/v1/chat/completions":
                self._respond_json(
                    {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion",
                        "model": body.get("model", "fake"),
                        "choices": [_build_fake_chat_message()],
                    }
                )
                return
            self.send_response(404)
            self.end_headers()

    return _Handler


@pytest.fixture
def fake_openai() -> Iterator[FakeOpenAIServer]:
    received: list[tuple[str, dict]] = []
    httpd = HTTPServer(("127.0.0.1", 0), _make_handler(received))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield FakeOpenAIServer(url=f"http://127.0.0.1:{port}/v1", received_requests=received)
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
