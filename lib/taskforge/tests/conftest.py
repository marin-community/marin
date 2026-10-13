# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``fake_glm``: a scripted fake of the GLM router, run as a real local HTTP server.

``POST /v1/chat/completions`` streams queued responses and ``GET /health`` reports queued worker counts.
``other_fake_glm`` is a second, independent server, for an endpoint that moves.
``fast_provider`` is a provider config whose backoff is a millisecond, so retries and holds are quick.
"""

import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from taskforge.llm.policy import LLMProviderConfig

GLM_MODEL = "glm-5.3"


@pytest.fixture
def fast_provider() -> LLMProviderConfig:
    return LLMProviderConfig(
        model=GLM_MODEL,
        max_attempts=8,
        hold_timeout=60.0,
        backoff_initial=0.001,
        backoff_maximum=0.001,
        backoff_factor=1.0,
        stall_timeout=600.0,
        max_connections=8,
        structured_repairs=1,
    )


@dataclass
class ScriptedStream:
    events: list[dict | str]
    send_done: bool
    stall: threading.Event | None


@dataclass
class ScriptedStatus:
    code: int
    body: str
    headers: dict[str, str]


@dataclass
class FakeGlm:
    base_url: str = ""
    responses: deque = field(default_factory=deque)
    requests: list[dict] = field(default_factory=list)
    health_workers: deque = field(default_factory=lambda: deque([{"high": 1, "bulk": 1}]))
    health_polls: int = 0
    health_status: int = 200
    released: threading.Event = field(default_factory=threading.Event)

    def stream(
        self,
        content: str = "",
        reasoning: str = "",
        finish: str | None = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        tool_calls: tuple[tuple[str, str], ...] = (),
        send_done: bool = True,
        stall_after_first: bool = False,
        raw_payload: str | None = None,
    ) -> None:
        """Queue one streamed reply; ``tool_calls`` holds ``(name, arguments)`` pairs.

        ``raw_payload`` is sent verbatim as the second ``data:`` line, for malformed-stream tests.
        """
        deltas: list[dict] = [{"role": "assistant", "content": ""}]
        deltas += [{"reasoning": part} for part in _halves(reasoning)]
        deltas += [{"content": part} for part in _halves(content)]
        for index, (name, arguments) in enumerate(tool_calls):
            deltas.append(
                {"tool_calls": [{"index": index, "id": f"call-{index}", "type": "function", "function": {"name": name}}]}
            )
            deltas += [
                {"tool_calls": [{"index": index, "function": {"arguments": part}}]} for part in _halves(arguments)
            ]
        events: list[dict] = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
        events[-1]["choices"][0]["finish_reason"] = finish
        events.append(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        )
        payloads: list[dict | str] = list(events)
        if raw_payload is not None:
            payloads.insert(1, raw_payload)
        stall = self.released if stall_after_first else None
        self.responses.append(ScriptedStream(payloads, send_done, stall))

    def status(self, code: int, body: str, headers: dict[str, str] | None = None) -> None:
        self.responses.append(ScriptedStatus(code, body, headers or {}))

    def next_health(self) -> dict[str, int]:
        self.health_polls += 1
        return self.health_workers.popleft() if len(self.health_workers) > 1 else self.health_workers[0]


def _halves(text: str) -> list[str]:
    middle = len(text) // 2
    return [part for part in (text[:middle], text[middle:]) if part]


def _handler(fake: FakeGlm) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - BaseHTTPRequestHandler signature
            pass

        def _send(self, code: int, body: str, headers: dict[str, str]) -> None:
            self.send_response(code)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self) -> None:
            assert self.path == "/health"
            self._send(fake.health_status, json.dumps({"status": "ok", "workers": fake.next_health()}), {})

        def do_POST(self) -> None:
            assert self.path == "/v1/chat/completions"
            fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            response = fake.responses.popleft()
            if isinstance(response, ScriptedStatus):
                self._send(response.code, response.body, response.headers)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index, event in enumerate(response.events):
                payload = event if isinstance(event, str) else json.dumps(event)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                if index == 1 and response.stall is not None:
                    response.stall.wait()
                    return
            if response.send_done:
                self.wfile.write(b"data: [DONE]\n\n")

    return Handler


@contextmanager
def _serving() -> Iterator[FakeGlm]:
    fake = FakeGlm()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    fake.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield fake
    fake.released.set()
    server.shutdown()
    server.server_close()


@pytest.fixture
def fake_glm() -> Iterator[FakeGlm]:
    with _serving() as fake:
        yield fake


@pytest.fixture
def other_fake_glm() -> Iterator[FakeGlm]:
    with _serving() as fake:
        yield fake
