# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from harbor.agents.installed.sotopia import SotopiaAgent
from harbor.environments.base import ExecResult
from harbor.llms import vllm_tokenization


@pytest.mark.asyncio
async def test_agent_install_restages_task_config(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    task_config = environment_dir / "task_config.json"
    task_config.write_text('{"environment_id": "env"}\n')
    environment = SimpleNamespace(
        environment_dir=environment_dir,
        upload_file=AsyncMock(),
        exec=AsyncMock(return_value=ExecResult(return_code=0)),
    )

    await SotopiaAgent(logs_dir=tmp_path).install(environment)

    environment.upload_file.assert_any_await(
        task_config,
        "/opt/sotopia/task_config.json",
    )


@pytest.mark.asyncio
async def test_dependency_install_failure_stops_setup(tmp_path: Path) -> None:
    environment = SimpleNamespace(
        environment_dir=tmp_path,
        upload_file=AsyncMock(),
        exec=AsyncMock(return_value=ExecResult(return_code=1, stderr="Package download failed")),
    )

    with pytest.raises(RuntimeError, match="Package download failed"):
        await SotopiaAgent(logs_dir=tmp_path).install(environment)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "connection", "protocol", 429, 503])
async def test_tokenizer_recovers_after_transient_failure(monkeypatch, failure):
    requests = []

    def tokenizer(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if failure == "connection":
                raise httpx.ConnectError("connection lost", request=request)
            if failure == "protocol":
                raise httpx.RemoteProtocolError("connection closed", request=request)
            return httpx.Response(failure)
        return httpx.Response(200, json={"tokens": [1, 2, 3]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(tokenizer)) as client:
        monkeypatch.setattr(vllm_tokenization, "shared_async_tokenize_client", lambda: client)
        tokens = await vllm_tokenization.tokenize_chat(
            api_base="http://model/v1",
            model="target",
            messages=[{"role": "user", "content": "hello"}],
            add_generation_prompt=True,
            template_options={},
        )

    assert tokens == [1, 2, 3]
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert requests[0].url == requests[1].url == "http://model/tokenize"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, attempts", [(503, 3), (400, 1)])
async def test_tokenizer_propagates_exhausted_or_permanent_errors(monkeypatch, status, attempts):
    requests = []

    def tokenizer(request):
        requests.append(request)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(tokenizer)) as client:
        monkeypatch.setattr(vllm_tokenization, "shared_async_tokenize_client", lambda: client)
        with pytest.raises(httpx.HTTPStatusError) as error:
            await vllm_tokenization.tokenize_chat(
                api_base="http://model/v1",
                model="target",
                messages=[],
                add_generation_prompt=True,
                template_options={},
            )

    assert error.value.response.status_code == status
    assert len(requests) == attempts
