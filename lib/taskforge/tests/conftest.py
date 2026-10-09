# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for taskforge tests.

Live tests carry ``@pytest.mark.live_glm`` and request the ``glm_settings`` fixture, which skips
them unless the interactive GLM-5.3 endpoint is configured through the environment. Web tests take
``parallel_key``, which skips unless ``TASKFORGE_PARALLEL_KEY_FILE`` names the Parallel key file.
Live tests write raw evidence under ``evidence_root``, which is outside the checkout.

``fake_glm`` is a scripted fake of the GLM router: ``POST /v1/chat/completions`` streams queued
responses and ``GET /health`` reports queued worker counts. It runs as a real local HTTP server.
``ledger`` is an in-memory ``Ledger`` that keeps every recorded entry in ``entries``.
"""

import json
import os
import tempfile
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from taskforge.ledger.records import LedgerEntry
from taskforge.llm.endpoint import GLM_MODEL

BASE_URL_ENV = "TASKFORGE_GLM_BASE_URL"
TOKEN_FILE_ENV = "TASKFORGE_GLM_TOKEN_FILE"
TOKEN_KEY = "GLM_API_TOKEN"
PARALLEL_KEY_FILE_ENV = "TASKFORGE_PARALLEL_KEY_FILE"
PARALLEL_KEY = "PARALLEL_KEY"
EVIDENCE_DIR_ENV = "TASKFORGE_EVIDENCE_DIR"
DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "taskforge-evidence"


@dataclass(frozen=True)
class GlmSettings:
    base_url: str
    model: str
    token: str = field(repr=False)


@dataclass(frozen=True)
class ParallelKey:
    """Kept out of ``repr`` so a failing test's traceback, which prints its fixtures, omits the key."""

    value: str = field(repr=False)


def read_key_line(path: Path, key: str) -> str:
    """Return the value of the ``<key>=...`` line in ``path``; the value is never echoed."""
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key and value.strip():
            return value.strip()
    raise ValueError(f"{path} has no non-empty {key}= line")


@pytest.fixture(scope="session")
def glm_settings() -> GlmSettings:
    base_url = os.environ.get(BASE_URL_ENV)
    token_file = os.environ.get(TOKEN_FILE_ENV)
    if not base_url or not token_file:
        pytest.skip(f"live GLM test: set {BASE_URL_ENV} and {TOKEN_FILE_ENV} (see lib/taskforge/README.md)")
    return GlmSettings(
        base_url=base_url.rstrip("/"), model=GLM_MODEL, token=read_key_line(Path(token_file).expanduser(), TOKEN_KEY)
    )


@pytest.fixture(scope="session")
def parallel_key() -> ParallelKey:
    """The Parallel API key from the ``PARALLEL_KEY=...`` line of ``TASKFORGE_PARALLEL_KEY_FILE``."""
    key_file = os.environ.get(PARALLEL_KEY_FILE_ENV)
    if not key_file:
        pytest.skip(f"live web test: set {PARALLEL_KEY_FILE_ENV} to the Parallel key file")
    return ParallelKey(read_key_line(Path(key_file).expanduser(), PARALLEL_KEY))


@pytest.fixture(scope="session")
def evidence_root() -> Path:
    """Where live checks write raw evidence: ``TASKFORGE_EVIDENCE_DIR`` when set, else ``<tmp>/taskforge-evidence``.

    The default is the system temp directory, so evidence survives across runs on one machine and never
    lands in the checkout.
    """
    return Path(os.environ.get(EVIDENCE_DIR_ENV) or DEFAULT_EVIDENCE_ROOT).expanduser()


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def ledger() -> ListLedger:
    return ListLedger()


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
        prompt_token_ids: tuple[int, ...] = (),
        token_ids: tuple[int, ...] = (),
    ) -> None:
        """Queue one streamed reply; ``tool_calls`` holds ``(name, arguments)`` pairs.

        ``raw_payload`` is sent verbatim as the second ``data:`` line, for malformed-stream tests.
        ``prompt_token_ids`` and ``token_ids`` emulate vLLM's ``return_token_ids``: the first is sent on
        the first chunk, the second with one logprob per token on the chunk that carries the finish reason.
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
        events = [{"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
        events[-1]["choices"][0]["finish_reason"] = finish
        if prompt_token_ids:
            events[0]["prompt_token_ids"] = list(prompt_token_ids)
        if token_ids:
            events[-1]["choices"][0]["token_ids"] = list(token_ids)
            events[-1]["choices"][0]["logprobs"] = {"content": [{"token": str(t), "logprob": -0.5} for t in token_ids]}
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
        def log_message(self, *args: object) -> None:
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


@pytest.fixture
def fake_glm() -> Iterator[FakeGlm]:
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
