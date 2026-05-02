# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import requests
from levanter.inference.openai_protocol import ChatCompletionRequest, CompletionRequest
from pydantic import ValidationError


@dataclass(frozen=True)
class OpenAIStubRequest:
    path: str
    payload: dict[str, object]


@dataclass
class OpenAIStubState:
    requests: list[OpenAIStubRequest] = field(default_factory=list)


@dataclass
class DeterministicOpenAIStub:
    base_url: str
    model: str
    state: OpenAIStubState

    def requests_for(self, path: str) -> list[OpenAIStubRequest]:
        return [request for request in self.state.requests if request.path == path]


class _DeterministicOpenAIServer(ThreadingHTTPServer):
    model: str
    state: OpenAIStubState


class _DeterministicOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self._write_json(404, {"error": "not found"})
            return
        self._write_json(200, {"object": "list", "data": [{"id": self._stub_server.model, "object": "model"}]})

    def do_POST(self) -> None:
        payload = self._read_json()
        if self.path == "/v1/completions":
            self._handle_completions(payload)
            return
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions(payload)
            return
        self._write_json(404, {"error": "not found"})

    @property
    def _stub_server(self) -> _DeterministicOpenAIServer:
        return cast(_DeterministicOpenAIServer, self.server)

    def _handle_completions(self, payload: dict[str, object]) -> None:
        self._stub_server.state.requests.append(OpenAIStubRequest(path=self.path, payload=payload))
        try:
            request = CompletionRequest.model_validate(payload)
        except ValidationError as exc:
            self._write_json(400, {"error": str(exc)})
            return
        if request.model != self._stub_server.model:
            self._write_json(400, {"error": "wrong model"})
            return

        prompt = request.prompt
        if not isinstance(prompt, str):
            self._write_json(400, {"error": "prompt must be a string"})
            return
        if request.echo is not True or request.logprobs is None:
            self._write_json(400, {"error": "scoring requests must set echo=true and logprobs"})
            return
        text = prompt
        if request.max_tokens > 0:
            text += " answer"
        tokens, offsets = _tokenize_with_offsets(text)
        token_logprobs = [_token_logprob(token) for token in tokens]
        self._write_json(
            200,
            {
                "id": "cmpl-stub",
                "object": "text_completion",
                "model": self._stub_server.model,
                "choices": [
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": token_logprobs,
                            "top_logprobs": [
                                {token: score} for token, score in zip(tokens, token_logprobs, strict=True)
                            ],
                            "text_offset": offsets,
                        },
                        "finish_reason": "length",
                    }
                ],
            },
        )

    def _handle_chat_completions(self, payload: dict[str, object]) -> None:
        self._stub_server.state.requests.append(OpenAIStubRequest(path=self.path, payload=payload))
        try:
            request = ChatCompletionRequest.model_validate(payload)
        except ValidationError as exc:
            self._write_json(400, {"error": str(exc)})
            return
        if request.model != self._stub_server.model:
            self._write_json(400, {"error": "wrong model"})
            return
        self._write_json(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": self._stub_server.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "stub response"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers["Content-Length"])
        return json.loads(self.rfile.read(content_length))

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def serve_deterministic_openai_stub(
    *,
    model: str = "gpt2",
) -> Iterator[DeterministicOpenAIStub]:
    state = OpenAIStubState()
    server = _DeterministicOpenAIServer(("127.0.0.1", 0), _DeterministicOpenAIHandler)
    server.model = model
    server.state = state
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield DeterministicOpenAIStub(base_url=f"http://127.0.0.1:{server.server_port}/v1", model=model, state=state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def assert_completions_scoring_contract(base_url: str, model: str) -> None:
    response = requests.post(
        f"{base_url}/completions",
        json={
            "model": model,
            "prompt": "Marin eval",
            "max_tokens": 1,
            "temperature": 0,
            "seed": 1,
            "echo": True,
            "logprobs": 1,
        },
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["text"] == "Marin eval answer"
    assert choice["index"] == 0
    assert choice["logprobs"]["tokens"] == ["Marin", " eval", " answer"]
    assert choice["logprobs"]["token_logprobs"] == [-0.0, -0.1, -0.3]
    assert choice["logprobs"]["top_logprobs"] == [{"Marin": -0.0}, {" eval": -0.1}, {" answer": -0.3}]
    assert choice["logprobs"]["text_offset"] == [0, 5, 10]


def assert_chat_generation_contract(base_url: str, model: str) -> None:
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0,
            "stop": None,
            "seed": 1,
        },
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["content"] == "stub response"


def _tokenize_with_offsets(text: str) -> tuple[list[str], list[int]]:
    if not text:
        return [], []
    pieces = text.split(" ")
    tokens = [pieces[0], *[f" {piece}" for piece in pieces[1:]]]
    offsets: list[int] = []
    offset = 0
    for token in tokens:
        offsets.append(offset)
        offset += len(token)
    return tokens, offsets


def _token_logprob(token: str) -> float:
    if token == " B":
        return -0.1
    if token == " C":
        return -2.0
    if token == " eval":
        return -0.1
    if token == " answer":
        return -0.3
    return -0.0
