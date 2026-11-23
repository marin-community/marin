# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A scripted fake of the GLM router, served over real local HTTP for the transport tests."""

import json
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GLM_MODEL = "glm-5.3"
TRICKLE_INTERVAL = 0.005


@dataclass
class ScriptedStream:
    events: list[dict | str]
    send_done: bool
    stall: threading.Event | None
    interval: float
    head_interval: float = 0.0
    """The pause, in seconds, before each line of the response head (status line, header, blank line)."""


@dataclass
class ScriptedStatus:
    code: int
    body: str
    headers: dict[str, str]
    trickle: int = 0
    """Bytes of whitespace sent ``TRICKLE_INTERVAL`` apart before ``body``."""


@dataclass
class FakeGlm:
    base_url: str = ""
    responses: deque = field(default_factory=deque)
    requests: list[dict] = field(default_factory=list)
    request_headers: list[dict[str, str]] = field(default_factory=list)
    health_workers: deque = field(default_factory=lambda: deque([{"high": 1, "bulk": 1}]))
    health_polls: int = 0
    health_status: int = 200
    health_text: str | None = None
    """A raw ``/health`` body in place of the JSON worker report."""
    health_trickle: int = 0
    """Bytes of JSON whitespace sent ``TRICKLE_INTERVAL`` apart before the ``/health`` body."""
    released: threading.Event = field(default_factory=threading.Event)
    cut: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))
    """Released once for each trickled body whose connection the client closed before the body ended."""

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
        interval: float = 0.0,
        head_interval: float = 0.0,
    ) -> None:
        """Queue one streamed reply; ``tool_calls`` holds ``(name, arguments)`` pairs.

        ``raw_payload`` is sent verbatim as the second ``data:`` line, for malformed-stream tests.
        ``interval`` is the pause, in seconds, before each event after the first; ``head_interval`` is
        ``ScriptedStream.head_interval``.
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
        self.responses.append(ScriptedStream(payloads, send_done, stall, interval, head_interval))

    def status(self, code: int, body: str, headers: dict[str, str] | None = None, trickle: int = 0) -> None:
        self.responses.append(ScriptedStatus(code, body, headers or {}, trickle))

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

        def _send(self, code: int, body: str, headers: dict[str, str], trickle: int = 0) -> None:
            self.send_response(code)
            for key, value in headers.items():
                self.send_header(key, value)
            if trickle:
                self.send_header("Content-Length", str(trickle + len(body)))
            self.end_headers()
            try:
                for _ in range(trickle):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(TRICKLE_INTERVAL)
                self.wfile.write(body.encode())
            except (BrokenPipeError, ConnectionResetError):
                if not trickle:
                    raise
                fake.cut.release()

        def do_GET(self) -> None:
            assert self.path == "/health"
            report = json.dumps({"status": "ok", "workers": fake.next_health()})
            body = report if fake.health_text is None else fake.health_text
            self._send(fake.health_status, body, {}, fake.health_trickle)

        def do_POST(self) -> None:
            assert self.path == "/v1/chat/completions", self.path
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append(body)
            fake.request_headers.append({key.lower(): value for key, value in self.headers.items()})
            response = fake.responses.popleft()
            if isinstance(response, ScriptedStatus):
                self._send(response.code, response.body, response.headers, response.trickle)
                return
            for line in (f"{self.protocol_version} 200 OK", "Content-Type: text/event-stream", ""):
                time.sleep(response.head_interval)
                self.wfile.write(f"{line}\r\n".encode())
            for index, event in enumerate(response.events):
                if index:
                    time.sleep(response.interval)
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
def serving() -> Iterator[FakeGlm]:
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
