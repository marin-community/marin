import logging
from types import SimpleNamespace

import pytest

from harbor.environments.base import ExecResult
from harbor.models.task.config import VerifierCollectConfig
from harbor.trial.single_step import SingleStepTrial


@pytest.mark.asyncio
async def test_collect_hook_failures_are_diagnostic_and_do_not_stop_later_hooks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    trial = object.__new__(SingleStepTrial)
    trial.logger = logging.getLogger("test.collect-hooks")
    calls: list[str] = []

    async def exec_hook(command: str, **kwargs) -> ExecResult:
        calls.append(command)
        if command == "nonzero":
            return ExecResult(
                stdout="partial output", stderr="bad state", return_code=7
            )
        if command == "provider-error":
            raise RuntimeError("sandbox disconnected")
        return ExecResult(stdout="done", stderr="", return_code=0)

    trial.agent_environment = SimpleNamespace(exec=exec_hook)
    hooks = [
        VerifierCollectConfig(command="nonzero"),
        VerifierCollectConfig(command="provider-error"),
        VerifierCollectConfig(command="success"),
    ]

    with caplog.at_level(logging.WARNING, logger="test.collect-hooks"):
        await trial._run_collect_hooks(hooks)

    assert calls == ["nonzero", "provider-error", "success"]
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "command='nonzero'" in message
        and "return_code=7" in message
        and "partial output" in message
        and "bad state" in message
        for message in messages
    )
    provider_record = next(
        record for record in caplog.records if "provider-error" in record.getMessage()
    )
    assert provider_record.exc_info is not None
    assert "sandbox disconnected" in str(provider_record.exc_info[1])
