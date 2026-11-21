"""Installed-agent runtime admission through the real Harbor trial boundary."""

import asyncio
import json
import shutil
from contextlib import asynccontextmanager

import daytona
import pytest
from harbor.host_capabilities import HostCapabilities
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.environments.factory import EnvironmentFactory
from shellbox.backends.daytona.connection import DAYTONA_API_URL, DAYTONA_TARGET, DaytonaConnection
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRolloutInterrupted
from shellbox.machine import ExitReason, Result
from taskcompendium.models import Source

from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts import agent_runtime
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_trial_session import run_harbor_trial
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from tests.cpu.task_specs import lowered_task


class ProviderUnavailable(RuntimeError):
    pass


async def unexpected_model_request(*args):
    pytest.fail("Provider startup failure must precede model inference")


class UnusedDecoder:
    def decode(self, token_ids):
        pytest.fail("Provider startup failure must precede token decoding")


class ProcessTransport:
    def __init__(self):
        self.users = []

    async def run(self, command):
        self.users.append(command.user)
        executable = shutil.which(command.argv[0])
        assert executable is not None
        process = await asyncio.create_subprocess_exec(
            executable,
            *command.argv[1:],
            env=command.env,
            cwd=command.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)


@pytest.fixture
def harbor_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "environment").mkdir()
    (package / "tests").mkdir()
    (package / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (package / "instruction.md").write_text("Write an answer.")
    (package / "task.toml").write_text('[environment]\ndocker_image = "fixture/image:latest"\n')
    return package


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_original_trial_uses_captured_phase_connection_at_provider_startup(
    tmp_path, monkeypatch, phase, harbor_package
):
    package = harbor_package
    task = harbor_trial_task(package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    task = task.model_copy(
        update={
            "environment_requirements": task.environment_requirements.model_copy(
                update={"setup_commands": ("exit 23",), "working_directory": "/workspace"}
            )
        }
    )
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
    connections = {name: DaytonaConnection(api_key=f"{name}-key") for name in ("train", "eval")}
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=connections["train"],
        eval_daytona_connection=connections["eval"],
    )
    harbor = HarborTaskSettings.from_config(OmegaConf.create({}))
    worker = TaskRolloutWorker(
        OmegaConf.create({}),
        WholeTaskProjection(OmegaConf.create({}), UnusedDecoder()),
        None,
        {},
        harbor=harbor,
        shutdown_timeout=5,
        installed_agents=runtime,
    )
    configured, harnesses = worker._configured_task_and_harnesses(lowered, harbor, "session", phase)
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
        with pytest.raises(GradedRolloutInterrupted) as interrupted:
            await RolloutEngine({}, harnesses).run_task(configured, unexpected_model_request)
        assert interrupted.value.grade.reward is None
    finally:
        await runtime.shutdown()

    assert len(attempted) == 1
    selected = connections[phase]
    # The ambient DAYTONA_* variables set above must not reach the client.
    assert attempted[0].api_key == selected.api_key
    assert attempted[0].api_url == DAYTONA_API_URL == "https://app.daytona.io/api"
    assert attempted[0].target == DAYTONA_TARGET == "us"
    attempts = list((tmp_path / "attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "identity.json").is_file()
    [result_path] = list((attempts[0] / "trials").glob("*/result.json"))
    result = TrialResult.model_validate_json(result_path.read_text())
    assert result.exception_info.exception_type == "ProviderUnavailable"


@pytest.mark.asyncio
async def test_borrowed_machine_commands_use_explicit_environment_and_reject_host_values(tmp_path, monkeypatch):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 23])
async def test_original_trial_runs_setup_before_installation_and_retains_failures(
    tmp_path, monkeypatch, harbor_package, exit_code
):
    machine = ProcessTransport()

    class LocalEnvironment(BorrowedMachineEnvironment):
        async def start(self, force_build, *, service_tasks):
            pass

        async def stop(self, delete):
            pass

        async def download_dir(self, source_dir, target_dir):
            target_dir.mkdir(parents=True, exist_ok=True)

    class SetupCheckingAgent(Pi):
        async def setup(self, environment):
            assert (tmp_path / "prepared").read_text() == "task-value"
            (tmp_path / "installed").touch()

        async def run(self, instruction, environment, context):
            (tmp_path / "agent-ran").touch()

    environment = LocalEnvironment(
        machine,
        tmp_path,
        "declared-setup",
        EnvironmentConfig(env={"VISIBLE": "task-value"}),
        host_capabilities=HostCapabilities(),
    )
    agent = SetupCheckingAgent(
        logs_dir=tmp_path / "agent",
        model_name="openai/fixture",
        api_base="https://model.example.test/synthetic-only/v1",
        thinking_format="chat-template",
        collect_rollout_details=True,
    )
    monkeypatch.setattr(AgentFactory, "create_agent_from_config", lambda *args, **kwargs: agent)
    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", lambda **kwargs: environment)
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=harbor_package),
            trials_dir=str(tmp_path / "trials"),
            agent=AgentConfig(name="pi"),
            verifier={"disable": True},
        )
    )
    commands = (
        f'printf "%s" "$VISIBLE" > prepared; printf first; printf diagnostic >&2; exit {exit_code}',
        'read value < prepared; test "$value" = task-value && printf second',
    )
    evidence = tmp_path / "setup-evidence"
    backend = AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder())
    result = await run_harbor_trial(
        trial, backend, unexpected_model_request, evidence,
        setup_commands=commands, setup_cwd=str(tmp_path), setup_timeout=5,
    )
    assert (tmp_path / "prepared").read_text() == "task-value"
    evidence = evidence / "task-setup"
    assert (evidence / "0.stdout").read_text() == "first"
    assert (evidence / "0.stderr").read_text() == "diagnostic"
    receipt = json.loads((evidence / "receipt.json").read_text())
    assert receipt["completed"] is (exit_code == 0)
    assert (tmp_path / "installed").exists() is (exit_code == 0)
    assert (tmp_path / "agent-ran").exists() is (exit_code == 0)
    assert set(machine.users) == {"root"}
    if exit_code:
        assert result.exception_info.exception_type == "RuntimeError"
        assert not (evidence / "1.stdout").exists()
    else:
        assert (evidence / "1.stdout").read_text() == "second"
