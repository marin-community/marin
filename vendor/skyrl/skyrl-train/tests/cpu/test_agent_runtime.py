"""Installed-agent runtime admission through the real Harbor trial boundary."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from harbor.environments import daytona
from harbor.host_capabilities import DaytonaConnection, HostCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig
from omegaconf import OmegaConf
from shellbox.machine import ExitReason, Result
from taskcompendium.models import Source

from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts import agent_runtime
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from tests.cpu.task_specs import lowered_task


class ProviderUnavailable(RuntimeError):
    pass


async def unexpected_model_request(*args):
    pytest.fail("Provider startup failure must precede model inference")


class UnusedDecoder:
    def decode(self, token_ids):
        pytest.fail("Provider startup failure must precede token decoding")


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_original_trial_uses_captured_phase_connection_at_provider_startup(tmp_path, monkeypatch, phase):
    package = tmp_path / "package"
    package.mkdir()
    (package / "environment").mkdir()
    (package / "tests").mkdir()
    (package / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (package / "instruction.md").write_text("Write an answer.")
    (package / "task.toml").write_text('[environment]\ndocker_image = "fixture/image:latest"\n')
    task = harbor_trial_task(package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    lowered = lowered_task(task, "harbor_trial", attempt_timeout=15, total_turn_timeout=5)
    settings = InstalledAgentSettings(
        agent=AgentConfig(name="pi", kwargs={"thinking_format": "qwen-chat-template"}),
        output_path=str(tmp_path / "attempts"),
        setup_timeout=5,
        run_timeout=5,
        cleanup_timeout=5,
        service_startup_timeout=5,
        service_shutdown_timeout=5,
    )
    connections = {
        name: DaytonaConnection(api_key=f"{name}-key", api_url=f"https://{name}.invalid/api", target=f"{name}-target")
        for name in ("train", "eval")
    }
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=connections["train"],
        eval_daytona_connection=connections["eval"],
    )
    harness = runtime.original_harness(lowered, "session", HarborTaskSettings.from_config(OmegaConf.create({})), phase)
    monkeypatch.setenv("DAYTONA_API_KEY", "ambient-key")
    monkeypatch.setenv("DAYTONA_API_URL", "https://ambient.invalid/api")
    monkeypatch.setenv("DAYTONA_TARGET", "ambient-target")
    # Replace only SDK allocation and external endpoint registration. Harbor still
    # constructs the real agent, environment and trial, and owns failure cleanup.
    attempted = []

    class UnavailableDaytona:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            attempted.append(self.config)
            raise ProviderUnavailable("fixture provider unavailable")

        async def __aexit__(self, *args):
            pass

    class LocalEndpoint:
        async def api_base(self, *, required_lifetime):
            return "http://127.0.0.1:1/v1"

    @asynccontextmanager
    async def registered_endpoint(port, *, request_timeout):
        yield LocalEndpoint()

    monkeypatch.setattr(daytona, "AsyncDaytona", UnavailableDaytona)
    monkeypatch.setattr(agent_runtime, "registered_agent_endpoint", registered_endpoint)
    await runtime.startup()
    try:
        result = await harness.run_trial(unexpected_model_request)
    finally:
        await runtime.shutdown()

    assert result.exception_info.exception_type == "ProviderUnavailable"
    assert result.exception_info.exception_message == "fixture provider unavailable"
    assert len(attempted) == 1
    selected = connections[phase]
    assert (attempted[0].api_key, attempted[0].api_url, attempted[0].target) == (
        selected.api_key,
        selected.api_url,
        selected.target,
    )
    attempts = list((tmp_path / "attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "identity.json").is_file()
    assert list((attempts[0] / "trials").glob("*/result.json"))


@pytest.mark.asyncio
async def test_borrowed_machine_commands_use_explicit_environment_and_reject_host_values(tmp_path, monkeypatch):
    class ProcessTransport:
        async def run(self, command):
            process = await asyncio.create_subprocess_exec(
                *command.argv,
                env=command.env,
                cwd=command.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)

    monkeypatch.setenv("AMBIENT_ONLY", "must-not-enter-command")
    environment = BorrowedMachineEnvironment(
        ProcessTransport(),
        tmp_path,
        "borrowed",
        EnvironmentConfig(env={"VISIBLE": "task-value"}),
        host_capabilities=HostCapabilities(protected_values=frozenset({"host-secret"})),
    )
    result = await environment.exec('printf "%s:%s" "$VISIBLE" "${AMBIENT_ONLY-unset}"')
    assert result.return_code == 0
    assert result.stdout == "task-value:unset"
    with pytest.raises(ValueError, match="host credential material"):
        await environment.exec("true", env={"DISGUISED": "host-secret"})
