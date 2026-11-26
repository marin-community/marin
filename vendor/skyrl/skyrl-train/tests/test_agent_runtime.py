"""Installed-agent runtime admission through the real Harbor trial boundary."""

import asyncio
import json
import os
import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import daytona
import httpx
import pytest
from harbor import host_capabilities
from harbor.host_capabilities import HostCapabilities
from harbor.agents.factory import AgentFactory
from harbor.agents.installed.pi import Pi
from harbor.environments import model_relay
from harbor.environments.factory import EnvironmentFactory
from harbor.environments.model_relay import JudgeAccess, ModelRelay, task_model_relays
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.trial import Trial
from harbor.utils.env import validate_task_environment
from harbor.verifier.verifier import Verifier
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRolloutInterrupted
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox import machine as shellbox_machine
from shellbox.http_service import HttpRequest, serve_http
from shellbox.machine import ExitReason, Result
from shellbox.process_session import SubprocessSession
from taskcompendium.models import Source
from verifyit.modes.grade_judge import JudgeConnection

from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts import agent_runtime
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_trial_session import run_harbor_trial
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.config.utils import get_default_config
from tests.cpu.task_specs import lowered_task


HOST_JUDGE_KEY = "host-private-key"


class ProviderUnavailable(RuntimeError):
    pass


async def unexpected_model_request(*args):
    pytest.fail("Provider startup failure must precede model inference")


class UnusedDecoder:
    def decode(self, token_ids):
        pytest.fail("Provider startup failure must precede token decoding")


@pytest.mark.asyncio
async def test_original_trial_keeps_turn_limit_below_trajectory_budget(tmp_path, harbor_package):
    settings = InstalledAgentSettings(
        agent=AgentConfig(
            name="pi",
            kwargs={
                "thinking_format": "chat-template",
                "model_info": {"max_input_tokens": 131072, "max_output_tokens": 24576},
                "compaction_reserve_tokens": 24576,
            },
        ),
        output_path=str(tmp_path / "attempts"),
        setup_timeout=360,
        run_timeout=1800,
        cleanup_timeout=120,
        service_startup_timeout=60,
        service_shutdown_timeout=60,
    )
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend("fixture", 131072, unexpected_model_request, UnusedDecoder()),
        98304,
        daytona_connection=DaytonaConnection(api_key="training-key"),
        eval_daytona_connection=DaytonaConnection(api_key="evaluation-key"),
    )
    assert runtime.validate_agent()["model_info"]["max_output_tokens"] == 24576
    assert runtime.limits.max_output_tokens == 24576
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=harbor_package),
            agent=runtime.serving_agent_config(settings.agent, "http://127.0.0.1:1/v1"),
            environment={"type": "daytona"},
            trials_dir=tmp_path / "trials",
            trial_name="turn-budget",
        )
    )
    assert isinstance(trial.agent, Pi)
    assert trial.agent.rollout_correlation_id is not None

    settings.agent.kwargs["compaction_reserve_tokens"] = 16384
    with pytest.raises(ValueError, match="compaction_reserve_tokens must be at least max_output_tokens"):
        runtime.validate_agent()


@pytest.mark.asyncio
async def test_worker_spec_builds_installed_agent_with_separate_judge_and_phase_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    for name, value in {"TEST_JUDGE_KEY": "judge-key", "TEST_RL_KEY": "training-key", "TEST_EVAL_KEY": "evaluation-key"}.items():
        monkeypatch.delenv(name, raising=False)
        (tmp_path / name).write_text(value)
    config = OmegaConf.merge(
        get_default_config(),
        {
            "trainer": {"policy": {"model": {"path": "fixture/model"}}},
            "environment": {
                "task_sessions": {
                    "max_verifier_workers": 0,
                    "remote_judge": {"base_url": "https://judge.invalid", "api_key_env": "TEST_JUDGE_KEY"},
                    "daytona": {"api_key_env": "TEST_RL_KEY", "eval_api_key_env": "TEST_EVAL_KEY", "ttl_minutes": 90},
                }
            },
        },
    )
    harbor = OmegaConf.create(
        {
            "harbor": {
                "environment_type": "daytona",
                "daytona_api_key_env": "TEST_RL_KEY",
                "eval_daytona_api_key_env": "TEST_EVAL_KEY",
            },
            "installed_agent": {
                "agent": {"name": "pi"},
                "output_path": str(tmp_path),
                "setup_timeout": 5,
                "run_timeout": 5,
                "cleanup_timeout": 5,
                "service_startup_timeout": 5,
                "service_shutdown_timeout": 5,
            },
        }
    )
    worker = TaskRolloutWorkerSpec.from_config(config, [], harbor_config=harbor).build(
        UnusedDecoder(), WorkerShard(0, 1)
    )
    try:
        assert worker.judge_connection.base_url == "https://judge.invalid"
        assert worker.judge_connection.api_key == "judge-key"
        assert worker.installed_agents.daytona_connection.api_key == "training-key"
        assert worker.installed_agents.eval_daytona_connection.api_key == "evaluation-key"
    finally:
        await worker.shutdown()


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
@pytest.mark.parametrize("judge_declared", [False, True])
async def test_original_trial_uses_captured_phase_connection_at_provider_startup(
    tmp_path, monkeypatch, phase, judge_declared, harbor_package
):
    package = harbor_package
    with (package / "task.toml").open("a") as config_file:
        config_file.write("""
[verifier.env]
OPENAI_BASE_URL = "https://original-provider.invalid/v1"
""")
        if judge_declared:
            config_file.write('OPENAI_API_KEY = "${OPENAI_API_KEY}"\nMODEL_NAME = "fixture-judge"\n')
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
    harbor = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {"override_storage_mb": 8192, "eval_override_storage_mb": 16384},
            }
        )
    )
    worker = TaskRolloutWorker(
        OmegaConf.create({}),
        WholeTaskProjection(OmegaConf.create({}), UnusedDecoder()),
        None,
        {},
        harbor=harbor,
        inference_queue_timeout=2280, shutdown_timeout=5,
        installed_agents=runtime,
        judge_connection=JudgeConnection("https://judge.invalid/v1", "judge-secret") if judge_declared else None,
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
    assert (attempted[0].api_key, attempted[0].api_url, attempted[0].target) == (
        selected.api_key,
        selected.api_url,
        selected.target,
    )
    attempts = list((tmp_path / "attempts").iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "identity.json").is_file()
    [result_path] = list((attempts[0] / "trials").glob("*/result.json"))
    result = TrialResult.model_validate_json(result_path.read_text())
    assert result.exception_info.exception_type == "ProviderUnavailable"
    archived_config = TrialConfig.model_validate_json((result_path.parent / "config.json").read_text())
    assert archived_config.environment.override_storage_mb == (16384 if phase == "eval" else 8192)
    assert archived_config.verifier.env["OPENAI_BASE_URL"] == (
        "http://127.0.0.1:8787/v1" if judge_declared else "https://original-provider.invalid/v1"
    )
    if not judge_declared:
        assert "model_endpoint" not in archived_config.environment.kwargs
    assert "judge-secret" not in (result_path.parent / "config.json").read_text()


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
async def test_original_verifier_uses_host_authenticated_relay_without_guest_provider_key(
    tmp_path, monkeypatch, harbor_package
):
    (harbor_package / "task.toml").write_text("""
[environment]
docker_image = "fixture/image:latest"
[verifier.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
OPENAI_BASE_URL = "https://original-provider.invalid/v1"
MODEL_NAME = "fixture-judge"
""")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task = harbor_trial_task(
        harbor_package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    )
    lowered = lowered_task(task, "harbor_trial", attempt_timeout=15, total_turn_timeout=5).model_copy(
        update={"judge_connection": JudgeConnection("https://judge.invalid/v1", "host-only-key")}
    )
    runtime = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent=AgentConfig(name="pi"),
            output_path=str(tmp_path / "attempts"),
            setup_timeout=5,
            run_timeout=5,
            cleanup_timeout=5,
            service_startup_timeout=5,
            service_shutdown_timeout=5,
        ),
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=DaytonaConnection(api_key="training-key"),
        eval_daytona_connection=DaytonaConnection(api_key="evaluation-key"),
    )
    binding = runtime.original_trial_binding(lowered, HarborTaskSettings.from_config(OmegaConf.create({})), "eval", {})
    environment = BorrowedMachineEnvironment(
        ProcessTransport(),
        tmp_path,
        "verifier",
        EnvironmentConfig(),
        host_capabilities=binding.host_capabilities,
    )
    guest = environment.resolve_env_vars(binding.verifier_environment)
    result = await environment.exec('printf "%s:%s" "$OPENAI_API_KEY" "$OPENAI_BASE_URL"', env=guest)
    assert result.stdout == "EMPTY:http://127.0.0.1:8787/v1"
    assert "host-only-key" not in json.dumps({"environment": binding.environment.model_dump(), "guest": guest})
    with pytest.raises(ValueError, match="host credential material"):
        environment.resolve_env_vars({"LEAK": "host-only-key"})
    relay, candidate = task_model_relays(
        model_endpoint=binding.environment.kwargs["model_endpoint"],
        model_name=binding.environment.kwargs["model_name"],
        candidate_model_endpoint=None,
        candidate_model_name=None,
        authentication=binding.host_capabilities.model_authentication,
        judge_access=JudgeAccess.VERIFIER,
        environments=(),
    )
    assert relay is not None and candidate is None
    requests = []

    def provider(request):
        requests.append(request)
        return httpx.Response(200, json={"grade": 1})

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client(transport=httpx.MockTransport(provider), **kwargs)
    )
    response = await relay.handle(
        HttpRequest("POST", "/v1/chat/completions", json.dumps({"model": "guest-alias"}).encode())
    )
    assert response.status == 200
    assert requests[0].url == "https://judge.invalid/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer host-only-key"
    assert json.loads(requests[0].content)["model"] == "fixture-judge"


@pytest.mark.asyncio
async def test_original_trial_rebuild_keeps_placeholder_verifier_credentials(
    tmp_path, monkeypatch, harbor_package
):
    """The rebuilt trial config must carry the projected placeholders, not their redaction.

    ``run_original_trial`` rebuilds the trial config through ``model_dump()``. The
    sensitive-env redactor runs in python mode too, so the relay placeholder
    ``OPENAI_API_KEY="EMPTY"`` became the literal ``"****"`` and Harbor's
    task-environment boundary rejected it before the verifier ran — every judged
    Micro210 BixBench trial ended as unscored infrastructure.
    """
    package = harbor_package
    with (package / "task.toml").open("a") as config_file:
        config_file.write(
            '\n[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
            'OPENAI_BASE_URL = "https://provider.invalid/v1"\nMODEL_NAME = "fixture-judge"\n'
        )
    # The judged launches carry the real judge credential as OPENAI_API_KEY on the
    # worker (run_micro210 / run_judged), exactly like production.
    monkeypatch.setenv("OPENAI_API_KEY", HOST_JUDGE_KEY)
    task_spec = harbor_trial_task(
        package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    )
    lowered = lowered_task(task_spec, "harbor_trial", attempt_timeout=15, total_turn_timeout=5).model_copy(
        update={"judge_connection": JudgeConnection("https://judge.invalid/v1", HOST_JUDGE_KEY)}
    )
    runtime = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent=AgentConfig(name="pi"),
            output_path=str(tmp_path / "attempts"),
            setup_timeout=5,
            run_timeout=5,
            cleanup_timeout=5,
            service_startup_timeout=5,
            service_shutdown_timeout=5,
        ),
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=DaytonaConnection(api_key="training-key", api_url="https://train.invalid/api", target="train-target"),
        eval_daytona_connection=DaytonaConnection(api_key="evaluation-key", api_url="https://eval.invalid/api", target="eval-target"),
    )
    harness = runtime.original_harness(lowered, "session", HarborTaskSettings.from_config(OmegaConf.create({})), "eval")

    captured: dict = {}
    real_create = Trial.create

    async def capturing_create(trial_config, **kwargs):
        captured["trial_config"] = trial_config
        return await real_create(trial_config, **kwargs)

    async def scored_run_harbor_trial(trial, backend, complete, engine_dir, **kwargs):
        return TrialResult(
            trial_name=trial.config.trial_name,
            task_name=trial.task.name,
            task_id=trial.config.task.get_task_id(),
            config=trial.config,
            task_checksum=trial.task.checksum,
            trial_uri="file:///tmp/unused",
            agent_info=AgentInfo(name="pi", version="0"),
            verifier_result=VerifierResult(rewards={"reward": 1.0}),
        )

    @asynccontextmanager
    async def served_agent(trial_dir, run_timeout, *, capability_lifetime, agent=None):
        yield AgentConfig(name="pi")

    runtime.port = 1
    monkeypatch.setattr(agent_runtime.Trial, "create", capturing_create)
    monkeypatch.setattr(agent_runtime, "run_harbor_trial", scored_run_harbor_trial)
    monkeypatch.setattr(runtime, "_served_agent", served_agent)
    result = await harness.run_trial(unexpected_model_request)
    assert result.verifier_result.rewards == {"reward": 1.0}

    trial_config = captured["trial_config"]
    assert trial_config.verifier.env["OPENAI_API_KEY"] == "EMPTY"
    assert trial_config.verifier.env["OPENAI_BASE_URL"].startswith("http://127.0.0.1:")
    assert HOST_JUDGE_KEY not in trial_config.verifier.env.values()

    # Harbor's own task-environment boundary — the check that rejected the redacted
    # literal in production (verifier.verify -> resolve_env_vars) — must accept the
    # rebuilt verifier env merged with the package's declared env. The restored task
    # bytes are the fixture package's; publish_attempt removed the evidence copy.
    restored_task = Task(harbor_package)
    merged_env = {**restored_task.config.verifier.env, **trial_config.verifier.env}
    validate_task_environment(merged_env)
    assert merged_env["OPENAI_API_KEY"] == "EMPTY"
    assert merged_env["MODEL_NAME"] == "fixture-judge"


def test_original_verifier_declaring_model_rejects_missing_connection(tmp_path, harbor_package):
    with (harbor_package / "task.toml").open("a") as config_file:
        config_file.write('\n[verifier.env]\nMODEL_NAME = "fixture-judge"\n')
    task = harbor_trial_task(
        harbor_package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    )
    lowered = lowered_task(task, "harbor_trial", attempt_timeout=15, total_turn_timeout=5)
    runtime = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent=AgentConfig(name="pi"),
            output_path=str(tmp_path / "attempts"),
            setup_timeout=5,
            run_timeout=5,
            cleanup_timeout=5,
            service_startup_timeout=5,
            service_shutdown_timeout=5,
        ),
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=DaytonaConnection(api_key="training-key"),
        eval_daytona_connection=DaytonaConnection(api_key="evaluation-key"),
    )
    with pytest.raises(ValueError, match="declared Harbor verifier model requires a remote judge connection"):
        runtime.original_harness(lowered, "session", HarborTaskSettings.from_config(OmegaConf.create({})), "eval")


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
        trial,
        backend,
        unexpected_model_request,
        evidence,
        setup_commands=commands,
        setup_cwd=str(tmp_path),
        setup_timeout=5,
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


class RelayGuestMachine:
    """A local-process guest whose /workspace, /tests and /logs live under a host root."""

    def __init__(self, root):
        self.root = root
        self.path("/workspace").mkdir(parents=True, exist_ok=True)
        self.requests: list[tuple[str, str, bytes]] = []

    def path(self, value):
        return self.root / value.lstrip("/")

    def _map(self, value):
        for prefix in ("/workspace", "/tests", "/logs"):
            value = value.replace(prefix, str(self.path(prefix)))
        return value

    def _machine_command(self, command):
        # Exec strings carry guest paths (/tests, /logs); map every argv element too.
        argv = tuple(self._map(value) if isinstance(value, str) else value for value in command.argv)
        return shellbox_machine.Command(
            argv,
            cwd=self._map(command.cwd) if command.cwd else str(self.root),
            # Environment values pass through unmapped: override paths are host paths and
            # relay URLs are guest-absolute.
            env=command.env,
            timeout=command.timeout,
        )

    async def run(self, command):
        mapped = self._machine_command(command)
        process = await asyncio.create_subprocess_exec(
            *mapped.argv,
            cwd=mapped.cwd,
            env={**os.environ, **mapped.env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return Result(process.returncode, stdout, stderr, False, False, ExitReason.EXITED)

    async def open_process(self, command):
        mapped = self._machine_command(command)
        process = await asyncio.create_subprocess_exec(
            *mapped.argv,
            cwd=mapped.cwd,
            env={**os.environ, **mapped.env},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def stop():
            if process.returncode is None:
                process.kill()
            await process.wait()

        return SubprocessSession(process, stop=stop)

    async def upload(self, source, target):
        destination = self.path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, destination)

    async def download(self, source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.path(source).is_dir():
            shutil.copytree(self.path(source), target, dirs_exist_ok=True)
        else:
            shutil.copyfile(self.path(source), target)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_bixbench_verifier_grades_through_host_judge_relay(harbor_package, tmp_path, monkeypatch):
    """The original verifier's declared model calls reach the host judge through the relay.

    The guest sees only the local relay endpoint and a dummy key; the provider observes
    the host credential and the policy judge model; the scored reward crosses back.
    """
    package = harbor_package
    with (package / "task.toml").open("a") as config_file:
        config_file.write(
            '\n[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
            'OPENAI_BASE_URL = "https://provider.invalid/v1"\nMODEL_NAME = "fixture-judge"\n'
        )
    (package / "tests" / "test.sh").write_text(
        "#!/bin/sh\n"
        "python3 - <<'PY'\n"
        "import json, os, urllib.request\n"
        "assert os.environ['OPENAI_API_KEY'] == 'EMPTY'\n"
        f"assert {HOST_JUDGE_KEY!r} not in os.environ.values()\n"
        "request = urllib.request.Request(\n"
        "    os.environ['OPENAI_BASE_URL'] + '/chat/completions',\n"
        "    data=json.dumps({'model': os.environ['MODEL_NAME'], 'messages': [{'role': 'user', 'content': 'Grade'}]}).encode(),\n"
        "    headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': 'application/json'},\n"
        ")\n"
        "with urllib.request.urlopen(request) as response:\n"
        "    assert json.load(response)['choices'][0]['message']['content'] == 'CORRECT'\n"
        "with open(os.environ['REWARD_PATH'], 'w') as reward:\n"
        "    reward.write('1')\n"
        "PY\n"
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    monkeypatch.setattr(host_capabilities, "MODEL_RELAY_PORT", port)
    machine = RelayGuestMachine(tmp_path / "guest")
    machine.path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    task_spec = harbor_trial_task(
        package, source=Source(dataset="fixture", revision="1", row="0", importer_revision="1")
    )
    lowered = lowered_task(task_spec, "harbor_trial", attempt_timeout=15, total_turn_timeout=5).model_copy(
        update={"judge_connection": JudgeConnection("https://provider.invalid/v1", HOST_JUDGE_KEY)}
    )
    runtime = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent=AgentConfig(name="pi"),
            output_path=str(tmp_path / "attempts"),
            setup_timeout=5,
            run_timeout=5,
            cleanup_timeout=5,
            service_startup_timeout=5,
            service_shutdown_timeout=5,
        ),
        AgentInferenceBackend("fixture", 1024, unexpected_model_request, UnusedDecoder()),
        128,
        daytona_connection=DaytonaConnection(api_key="daytona-secret"),
        eval_daytona_connection=DaytonaConnection(api_key="daytona-secret"),
    )
    binding = runtime.original_trial_binding(
        lowered,
        HarborTaskSettings.from_config(OmegaConf.create({})),
        "eval",
        {"REWARD_PATH": str(machine.path("/logs/verifier/reward.txt"))},
    )
    assert binding.verifier_environment["OPENAI_API_KEY"] == "EMPTY"
    assert binding.verifier_environment["OPENAI_BASE_URL"] == f"http://127.0.0.1:{port}/v1"

    def provider(request):
        machine.requests.append((request.headers["authorization"], request.url.path, request.read()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "CORRECT"}}]})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        model_relay.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(provider), **kwargs)
    )
    relay = ModelRelay.from_endpoint(
        binding.environment.kwargs["model_endpoint"],
        binding.environment.kwargs["model_name"],
        authentication=binding.host_capabilities.model_authentication,
    )
    assert relay is not None
    task = Task(package)
    environment = BorrowedMachineEnvironment(
        machine, tmp_path / "trial", "bixbench", task.config.environment, host_capabilities=binding.host_capabilities
    )
    verifier = Verifier(task=task, trial_paths=paths, environment=environment, override_env=binding.verifier_environment)
    try:
        async with asyncio.TaskGroup() as service_tasks:
            async with serve_http(machine, relay.handle, port=port, task_group=service_tasks):
                result = await verifier.verify()
    finally:
        await machine.close()
    assert result.rewards == {"reward": 1.0}
    assert len(machine.requests) == 1
    assert machine.requests[0][0] == f"Bearer {HOST_JUDGE_KEY}"
    assert machine.requests[0][1] == "/v1/chat/completions"
    assert json.loads(machine.requests[0][2])["model"] == "fixture-judge"
