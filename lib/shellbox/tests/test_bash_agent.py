# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the model and errors sent through BashAgent's OpenAI request."""

import asyncio
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from shellbox.agent import BashAgent


@dataclass
class _EndpointState:
    models: list[str] = field(default_factory=list)
    error_body: str | None = None


class _IdleShell:
    interactive = False


class _RequestEnvironment:
    async def open_bash_session(self) -> _IdleShell:
        return _IdleShell()


@pytest.fixture
def model_endpoint() -> Iterator[tuple[str, _EndpointState]]:
    state = _EndpointState()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.models.append(request["model"])
            if state.error_body is not None:
                body = state.error_body.encode()
                self.send_response(400)
            else:
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "Done."}}]}).encode()
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1", state
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize(
    ("model_name", "expected_model"),
    [
        ("hosted_vllm/served-grug", "served-grug"),
        ("served-grug", "served-grug"),
        ("openai/served-grug", "openai/served-grug"),
    ],
)
def test_bash_agent_sends_served_model_name(
    tmp_path: Path, model_endpoint: tuple[str, _EndpointState], model_name: str, expected_model: str
) -> None:
    base_url, state = model_endpoint
    agent = BashAgent(logs_dir=tmp_path, model_name=model_name, base_url=base_url)

    asyncio.run(agent.run("Solve task", cast(BaseEnvironment, _RequestEnvironment()), AgentContext()))

    assert state.models == [expected_model]


def test_bash_agent_http_error_includes_response_body(
    tmp_path: Path, model_endpoint: tuple[str, _EndpointState]
) -> None:
    base_url, state = model_endpoint
    state.error_body = "unknown model: served-grug"
    agent = BashAgent(logs_dir=tmp_path, model_name="hosted_vllm/served-grug", base_url=base_url)

    with pytest.raises(httpx.HTTPStatusError) as error:
        asyncio.run(agent.run("Solve task", cast(BaseEnvironment, _RequestEnvironment()), AgentContext()))

    assert error.value.response.status_code == 400
    assert "unknown model: served-grug" in str(error.value)
    assert state.models == ["served-grug"]
