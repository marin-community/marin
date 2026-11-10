# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from harbor.agents.installed.sotopia import SotopiaAgent
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
    )

    await SotopiaAgent(logs_dir=tmp_path).install(environment)

    environment.upload_file.assert_any_await(
        task_config,
        "/opt/sotopia/task_config.json",
    )


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


@pytest.mark.parametrize("model_alias, expected_model", [(None, "org/original"), ("hosted_vllm/resumed-endpoint", "resumed-endpoint")])
async def test_agent_routes_target_alias_and_preserves_model_identity(tmp_path, model_alias, expected_model):
    agent = SotopiaAgent(
        logs_dir=tmp_path,
        model_name="hosted_vllm/org/original",
        model_alias=model_alias,
        api_base="https://target.example/v1",
        partner_model="openai/partner",
        partner_api_base="https://partner.example/v1",
        evaluator_model="openai/judge",
        evaluator_api_base="https://judge.example/v1",
    )
    environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")))
    await agent.run("Start the episode", environment, object())
    env = environment.exec.await_args.kwargs["env"]
    assert env["SOTOPIA_TARGET_MODEL"] == f"custom/{expected_model}@https://target.example/v1"
    assert env["SOTOPIA_PARTNER_MODEL"] == "custom/partner@https://partner.example/v1"
    assert env["SOTOPIA_EVALUATOR_MODEL"] == "custom/judge@https://judge.example/v1"
    assert agent.to_agent_info().model_info.model_dump() == {"name": "org/original", "provider": "hosted_vllm"}
