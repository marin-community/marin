"""Installed-agent runtime admission through the real Harbor trial boundary."""

import asyncio
import json
import shutil
from contextlib import asynccontextmanager

import daytona
import pytest
from harbor.host_capabilities import HostCapabilities, MODEL_RELAY_PORT
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.environments.factory import EnvironmentFactory
from harbor.environments.capabilities import EnvironmentCapabilities
from shellbox.backends.daytona.connection import DAYTONA_API_URL, DAYTONA_TARGET, DaytonaConnection
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from harbor.verifier.verifier import Verifier
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRolloutInterrupted
from shellbox.machine import ExitReason, Result
from shellbox.machine import Backend, MachineSpec, RegistryImage
from taskcompendium.models import Source
from verifyit.modes.grade_judge import JudgeConnection

from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts import agent_runtime
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts import harbor_tasks
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_trial_session import run_harbor_trial
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, TaskRolloutWorkerSpec
from skyrl_train.rollouts.machines import TaskMachineError
from skyrl_train.rollouts.workers import WorkerShard
from tests.cpu.task_specs import lowered_task, text_task


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
@pytest.mark.parametrize("model_caller", [None, "environment", "verifier"])
@pytest.mark.parametrize(
    ("provisioning", "eval_provisioning"), [(None, None), ("declarative", None), ("snapshot", "declarative")]
)
async def test_original_trial_uses_captured_phase_connection_at_provider_startup(
    tmp_path, monkeypatch, phase, provisioning, eval_provisioning, model_caller, harbor_package
):
    package = harbor_package
    if model_caller is not None:
        with (package / "task.toml").open("a") as config:
            config.write(f'\n[{model_caller}.env]\nOPENAI_BASE_URL = "http://task-model.invalid/v1"\n')
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
        judge_model=None if model_caller is None else "fixture-judge",
    )
    connections = {name: DaytonaConnection(api_key=f"{name}-key") for name in ("train", "eval")}
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=connections["train"],
        eval_daytona_connection=connections["eval"],
        judge=None if model_caller is None else JudgeConnection("https://host-judge.invalid/v1", "host-judge-secret"),
    )
    harbor_config = {}
    if provisioning is not None:
        harbor_config["daytona_provisioning"] = provisioning
    if eval_provisioning is not None:
        harbor_config["eval_daytona_provisioning"] = eval_provisioning
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": harbor_config}))
    declarative = (eval_provisioning if phase == "eval" and eval_provisioning else provisioning) == "declarative"
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
    allocations = []

    class UnavailableDaytona:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            attempted.append(self.config)
            if not declarative:
                raise ProviderUnavailable("fixture provider unavailable")
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, params, *, timeout):
            allocations.append(params)
            raise ProviderUnavailable("fixture allocation unavailable")

        async def get(self, name):
            raise daytona.DaytonaNotFoundError("fixture sandbox absent")

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
    if declarative:
        assert len(allocations) == 1
        assert isinstance(allocations[0], daytona.CreateSandboxFromImageParams)
        assert "host-judge-secret" not in allocations[0].env_vars.values()
    attempts = list((tmp_path / "attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "identity.json").is_file()
    [result_path] = list((attempts[0] / "trials").glob("*/result.json"))
    result = TrialResult.model_validate_json(result_path.read_text())
    assert result.exception_info.exception_type == "ProviderUnavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["${OPENAI_BASE_URL}", "https://declared-provider.invalid/v1"])
async def test_verifier_model_calls_receive_relay_values_only_during_verification(tmp_path, harbor_package, endpoint):
    with (harbor_package / "task.toml").open("a") as config:
        config.write(
            f'\n[verifier.env]\nOPENAI_BASE_URL = "{endpoint}"\n'
            'OPENAI_API_KEY = "${OPENAI_API_KEY}"\nMODEL_NAME = "${MODEL_NAME}"\n'
        )
    task = harbor_trial_task(harbor_package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"))
    settings = InstalledAgentSettings(
        agent=AgentConfig(name="pi"), output_path=str(tmp_path / "attempts"),
        setup_timeout=5, run_timeout=5, cleanup_timeout=5,
        service_startup_timeout=5, service_shutdown_timeout=5, judge_model="fixture-judge",
    )
    connection = DaytonaConnection(api_key="daytona-secret")
    runtime = InstalledAgentRuntime(
        settings, AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()), 128,
        daytona_connection=connection, eval_daytona_connection=connection,
        judge=JudgeConnection("https://host-judge.invalid/v1", "host-judge-secret"),
    )
    binding = runtime.original_trial_binding(
        lowered_task(task, "harbor_trial", attempt_timeout=15),
        HarborTaskSettings.from_config(OmegaConf.create({})), "eval", {},
    )
    paths = TrialPaths(tmp_path / "verifier-trial")
    paths.mkdir()
    commands = []

    class VerifierMachine:
        async def run(self, command):
            commands.append(command)
            paths.reward_text_path.write_text("1")
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

    class MountedEnvironment(BorrowedMachineEnvironment):
        @property
        def capabilities(self):
            return EnvironmentCapabilities(mounted=True)

    environment = MountedEnvironment(
        VerifierMachine(), tmp_path / "verifier-trial", "verifier-only", EnvironmentConfig(),
        host_capabilities=binding.host_capabilities,
    )
    await environment.exec("true")
    assert "OPENAI_BASE_URL" not in commands[0].env
    assert "OPENAI_API_KEY" not in commands[0].env
    result = await Verifier(
        task=Task(task_dir=harbor_package), trial_paths=paths, environment=environment,
        override_env=binding.verifier_environment, skip_tests_upload=True,
    ).verify()
    assert result.rewards == {"reward": 1.0}
    assert commands[-1].env["OPENAI_BASE_URL"] == f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1"
    assert commands[-1].env["OPENAI_API_KEY"] == "EMPTY"
    assert commands[-1].env["MODEL_NAME"] == "fixture-judge"
    assert "host-judge-secret" not in commands[-1].env.values()


@pytest.mark.asyncio
async def test_shellbox_eval_provisioning_override_reaches_provider_with_shared_credential(monkeypatch):
    monkeypatch.setenv("SHARED_DAYTONA_KEY", "shared-key")
    config = OmegaConf.create(
        {
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 5,
            },
            "trainer": {"policy": {"model": {"path": "fixture"}}, "step_wise_training": False},
            "environment": {"task_sessions": {"max_verifier_workers": 0}},
            "trajectory_runner": {"max_concurrent_tasks": 1, "shutdown_timeout": 5},
        }
    )
    harbor_config = OmegaConf.create(
        {
            "harbor": {
                "daytona_api_key_env": "SHARED_DAYTONA_KEY",
                "daytona_provisioning": "snapshot",
                "eval_daytona_provisioning": "declarative",
            }
        }
    )
    allocations = []

    class UnavailableDaytona:
        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, params, *, timeout):
            allocations.append((self.config, params))
            raise ProviderUnavailable("fixture allocation unavailable")

        async def get(self, name):
            raise daytona.DaytonaNotFoundError("fixture sandbox absent")

    monkeypatch.setattr(harbor_tasks, "AsyncDaytona", UnavailableDaytona)
    worker = TaskRolloutWorkerSpec.from_config(config, [], harbor_config=harbor_config).build(
        UnusedDecoder(), WorkerShard(0, 1)
    )
    spec = worker.harbor.configured_spec(lowered_task(text_task(), backend=Backend.DOCKER), phase="eval")
    try:
        with pytest.raises(TaskMachineError):
            await worker.factories[spec.task_machine.backend].create(
                MachineSpec(source=RegistryImage("fixture/image:latest"))
            )
    finally:
        await worker.shutdown()
    [(connection, params)] = allocations
    assert connection.api_key == "shared-key"
    assert isinstance(params, daytona.CreateSandboxFromImageParams)


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
