import asyncio
import shlex
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError


@pytest.mark.asyncio
async def test_codex_server_timeout_is_reported_as_agent_timeout(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    commands = []

    async def exec_command(*, command, **kwargs):
        commands.append(command)
        argv = shlex.split(command.removeprefix("set -o pipefail; "))
        if argv[0] == "timeout":
            return ExecResult(return_code=124, stdout="", stderr="")
        return ExecResult(return_code=0, stdout="", stderr="")

    environment = AsyncMock(spec=BaseEnvironment)
    environment.agent_timeout_sec = 1800.0
    environment.exec.side_effect = exec_command
    agent = Codex(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(AgentTimeoutError):
        await agent.run("Write the result", environment, AgentContext())

    assert shlex.split(commands[-1].removeprefix("set -o pipefail; ")) == [
        "rm",
        "-rf",
        "/tmp/codex-secrets",
        "$CODEX_HOME",
    ]


@pytest.mark.asyncio
async def test_codex_cancellation_starts_no_new_remote_finalization(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    entered = asyncio.Event()
    blocked = asyncio.Event()
    commands = []

    async def exec_command(*, command, **kwargs):
        commands.append(command)
        argv = shlex.split(command.removeprefix("set -o pipefail; "))
        if argv[0] == "timeout":
            entered.set()
            await blocked.wait()
        return ExecResult(return_code=0, stdout="", stderr="")

    environment = AsyncMock(spec=BaseEnvironment)
    environment.agent_timeout_sec = 1800.0
    environment.exec.side_effect = exec_command
    agent = Codex(logs_dir=tmp_path, model_name="openai/test-model")
    task = asyncio.create_task(
        agent.run("Write the result", environment, AgentContext())
    )
    await entered.wait()
    requests_before_cancellation = len(commands)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(commands) == requests_before_cancellation


@pytest.mark.asyncio
async def test_codex_non_timeout_exit_retains_error_and_finalizes(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.delenv("CODEX_FORCE_AUTH_JSON", raising=False)
    commands = []

    async def exec_command(*, command, **kwargs):
        commands.append(command)
        argv = shlex.split(command.removeprefix("set -o pipefail; "))
        status = 1 if argv[0] == "timeout" else 0
        return ExecResult(return_code=status, stdout="", stderr="")

    environment = AsyncMock(spec=BaseEnvironment)
    environment.agent_timeout_sec = 1800.0
    environment.exec.side_effect = exec_command
    agent = Codex(logs_dir=tmp_path, model_name="openai/test-model")

    with pytest.raises(NonZeroAgentExitCodeError) as failure:
        await agent.run("Write the result", environment, AgentContext())

    assert failure.value.result.return_code == 1
    assert shlex.split(commands[-1].removeprefix("set -o pipefail; ")) == [
        "rm",
        "-rf",
        "/tmp/codex-secrets",
        "$CODEX_HOME",
    ]
