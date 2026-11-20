import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.mini_swe_agent import MiniSweAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_source", ["protected", "huggingface"])
async def test_failed_install_preserves_error_tail_without_credentials(
    tmp_path, caplog, credential_source
):
    secret = (
        "synthetic-protected-credential"
        if credential_source == "protected"
        else "hf_" + "Q" * 32
    )
    protected = frozenset({secret}) if credential_source == "protected" else frozenset()
    agent = MiniSweAgent(logs_dir=tmp_path, protected_values=protected)
    stderr = (
        "package installation preamble\n" * 100
        + secret
        + "\nterminal dependency failure"
    )
    environment = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=2, stdout="", stderr=stderr)
        )
    )
    with (
        caplog.at_level(logging.DEBUG),
        pytest.raises(NonZeroAgentExitCodeError) as caught,
    ):
        await agent.exec_as_agent(
            environment, command="false", env={"RUN_MODE": "synthetic-runtime-value"}
        )
    evidence = json.loads(
        next((tmp_path / "command-failures").glob("*.json")).read_text()
    )
    assert evidence["stderr"].endswith("terminal dependency failure")
    assert evidence["stderr"].startswith("package installation preamble")
    assert "terminal dependency failure" in str(caught.value)
    assert secret not in str(caught.value)
    assert secret not in json.dumps(evidence)
    logged = repr([vars(record) for record in caplog.records])
    assert "RUN_MODE" in logged
    assert "synthetic-runtime-value" not in logged
    assert secret not in logged


@pytest.mark.asyncio
async def test_evidence_write_failure_keeps_the_command_exit_as_primary_error(
    tmp_path, monkeypatch
):
    agent = MiniSweAgent(logs_dir=tmp_path)
    environment = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=2, stdout="", stderr="dependency failed"
            )
        )
    )

    def cannot_write(*args, **kwargs):
        raise PermissionError("read-only evidence directory")

    monkeypatch.setattr(type(tmp_path), "open", cannot_write)
    with pytest.raises(NonZeroAgentExitCodeError) as caught:
        await agent.exec_as_agent(environment, command="false")
    assert isinstance(caught.value.__cause__, PermissionError)
    assert caught.value.result.return_code == 2
    assert "dependency failed" in str(caught.value)


@pytest.mark.asyncio
async def test_command_failure_serialization_keeps_the_event_loop_responsive(
    tmp_path, monkeypatch
):
    agent = MiniSweAgent(logs_dir=tmp_path)
    environment = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                return_code=2, stdout="", stderr="dependency failed"
            )
        )
    )
    entered = threading.Event()
    release = threading.Event()
    original = json.dumps

    def slow_serialize(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test serialization was not released")
        return original(*args, **kwargs)

    monkeypatch.setattr("harbor.agents.installed.base.json.dumps", slow_serialize)
    task = asyncio.create_task(agent.exec_as_agent(environment, command="false"))
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(NonZeroAgentExitCodeError):
        await task
    assert len(list((tmp_path / "command-failures").glob("*.json"))) == 1
