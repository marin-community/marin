# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory, SandboxUnavailable

from taskcompendium.models import CommandSemantics, EnvironmentRequirements


class JudgeServer(ThreadingHTTPServer):
    """Record requests to a local OpenAI-compatible judge endpoint."""

    requests: list[dict]
    reply: str
    status_code: int


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
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": server.reply},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()
        self.send_response(server.status_code)
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
    server.status_code = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def local_pipeline_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = str(tmp_path / "marin")
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setenv("MARIN_TEMP_PREFIX", prefix)


@pytest.fixture
def local_machine_environment(tmp_path: Path) -> tuple[EnvironmentRequirements, LocalMachineFactory]:
    lock = tmp_path / "requirements.lock"
    lock.write_bytes(b"")
    try:
        factory = LocalMachineFactory()
    except SandboxUnavailable as error:
        pytest.skip(str(error))
    return EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=str(lock)), factory
