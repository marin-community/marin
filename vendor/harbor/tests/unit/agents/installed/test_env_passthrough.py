"""Unit tests for extra_env environment variable passthrough in BaseInstalledAgent."""

import logging
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode


class TestExtraEnvInRun:
    """Test that extra env vars are merged into exec calls during run()."""

    @pytest.mark.asyncio
    async def test_extra_env_overrides_agent_defaults(self, temp_dir):
        agent = ClaudeCode(
            logs_dir=temp_dir,
            extra_env={"IS_SANDBOX": "0"},
        )

        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        mock_context = AsyncMock()

        await agent.run("do something", mock_env, mock_context)

        for call in mock_env.exec.call_args_list:
            env = call.kwargs.get("env") or call.kwargs.get("env", {})
            assert env["IS_SANDBOX"] == "0"

    @pytest.mark.asyncio
    async def test_no_extra_env_passes_original(self, temp_dir):
        agent = ClaudeCode(logs_dir=temp_dir)

        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        mock_context = AsyncMock()

        await agent.run("do something", mock_env, mock_context)

        # With no extra env, IS_SANDBOX should still be "1" from ClaudeCode defaults
        for call in mock_env.exec.call_args_list:
            env = call.kwargs.get("env") or call.kwargs.get("env", {})
            if env:
                assert env.get("IS_SANDBOX") == "1"


class TestNonZeroExitCode:
    """Test that non-zero exit codes raise RuntimeError."""

    @pytest.mark.asyncio
    async def test_non_zero_exit_code_raises(self, temp_dir):
        agent = ClaudeCode(logs_dir=temp_dir)

        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(
            return_code=1, stdout="partial output", stderr="provider error"
        )
        mock_context = AsyncMock()

        with pytest.raises(NonZeroAgentExitCodeError, match="exit 1") as error:
            await agent.run("do something", mock_env, mock_context)

        assert error.value.result is not None
        assert error.value.result.return_code == 1
        assert error.value.result.stdout == "partial output"
        assert error.value.result.stderr == "provider error"


class TestExecSetupLogging:
    """Test debug logging behavior for setup command execution."""

    @pytest.mark.asyncio
    async def test_exec_setup_logs_success(self, temp_dir, caplog):
        agent = ClaudeCode(logs_dir=temp_dir)
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="ok", stderr="")
        caplog.set_level(logging.DEBUG)

        await agent.exec_as_root(environment=mock_env, command="echo ok", env={})

        assert any(
            rec.getMessage() == "Running command: echo ok"
            and rec.__dict__["user"] == "root"
            and rec.__dict__["env"] == {}
            for rec in caplog.records
        )
        assert any(
            rec.getMessage() == "Command outputs captured"
            and rec.__dict__["stdout"] == "ok"
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_exec_setup_logs_failure(self, temp_dir, caplog):
        agent = ClaudeCode(logs_dir=temp_dir)
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(
            return_code=7, stdout="bad", stderr="err"
        )
        caplog.set_level(logging.DEBUG)

        with pytest.raises(RuntimeError, match="Command failed \\(exit 7\\)"):
            await agent.exec_as_agent(
                environment=mock_env,
                command="badcmd",
                env={"FOO": "bar"},
            )

        assert any(rec.getMessage() == "Command failed" for rec in caplog.records)
        assert any(
            rec.__dict__.get("return_code") == 7 and rec.__dict__.get("stdout") == "bad"
            for rec in caplog.records
        )
        assert any(
            rec.getMessage() == "Running command: badcmd"
            and rec.__dict__.get("env") == {"FOO": "bar"}
            for rec in caplog.records
        )
