# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from daytona import DaytonaConnectionError
from omegaconf import OmegaConf
from rolloutengine.test_support import shell_call
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from taskcompendium.models import ResourceGroups, ScriptGrader
from taskcompendium.runtime.resources import inline_resource
from tests.cpu.task_specs import NATIVE_ENVIRONMENT
from hydra import compose, initialize_config_dir
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineSpec, Result
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from rigging.task_errors import RetrySettings
from daytona import DaytonaError
from skyrl_train.trajectory_runners.types import TaskRequestBatch
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task

from cloud.iris.tests.task_worker_fixtures import ScriptedClient, WorkerTokenizer, sampled_reply

RETRY_SETTINGS = RetrySettings(max_retries=2, min_wait_sec=1, wait_multiplier=2)

SETUP_ENVIRONMENT = SHELL_ENVIRONMENT.model_copy(update={"setup_commands": ("true",)})


class Reset(StrEnum):
    CREATE = "create"
    COMMAND = "command"
    UPLOAD = "upload"


@dataclass
class ResettingMachine:
    machine: Machine
    reset: bool
    operation: Reset

    async def run(self, command: Command) -> Result:
        if self.reset and self.operation is Reset.COMMAND:
            raise DaytonaConnectionError("Failed to execute command: [Errno 104] Connection reset by peer")
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        if self.reset and self.operation is Reset.UPLOAD:
            raise DaytonaConnectionError("Connection reset while uploading resources")
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass
class ResettingFactory:
    """Reset the connection of the first ``failures`` machines, as Daytona does."""

    failures: int
    reset: Reset = Reset.CREATE
    backend: Backend = Backend.SHELLSIM
    creations: int = 0

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        ShellSimMachineFactory().check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        self.creations += 1
        fails = self.creations <= self.failures
        if fails and self.reset == Reset.CREATE:
            raise DaytonaConnectionError("Failed to upload file: [Errno 104] Connection reset by peer")
        return ResettingMachine(await ShellSimMachineFactory().create(spec), reset=fails, operation=self.reset)


async def rollout(worker_config, factory, *, environment=SHELL_ENVIRONMENT, harbor=None, replies=None, retry=RETRY_SETTINGS, resources=None, grader=None, verifier_factory=None):
    waits = []

    async def wait(delay):
        waits.append(delay)

    task = text_task().model_copy(
        update={"environment_requirements": environment, "tags": () if harbor is None else ("harbor",)}
    )
    if resources is not None:
        task = task.model_copy(update={"resources": resources})
    if grader is not None:
        task = task.model_copy(update={"grader": grader})
    factories = {"shellsim": factory}
    if verifier_factory is not None:
        factories["verifier"] = verifier_factory
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient(replies or [sampled_reply("Done")]),
        factories,
        harbor=harbor,
        retry_wait=wait,
        retry=retry,
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(
            TaskRequestBatch(rows=(task_input(lowered_task(task, max_turns=2, backend="shellsim", verifier_backend="verifier" if grader else None)),))
        )
    finally:
        await worker.shutdown()
    return result, waits


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "recovers"),
    [(RETRY_SETTINGS.max_retries, True), (RETRY_SETTINGS.max_retries + 1, False)],
    ids=["recovers", "gives-up"],
)
async def test_lost_connection_at_machine_start_restarts_the_rollout(worker_config, failures, recovers):
    factory = ResettingFactory(failures)
    result, waits = await rollout(worker_config, factory)
    assert factory.creations == RETRY_SETTINGS.max_retries + 1
    assert (result.failure is None) == recovers
    # Each restart waits first, because the resets arrive in bursts.
    assert len(waits) == RETRY_SETTINGS.max_retries and all(wait > 0 for wait in waits)
    assert result.metrics["rollout_retries"] == RETRY_SETTINGS.max_retries


@pytest.mark.asyncio
async def test_lost_connection_in_the_setup_command_restarts_the_rollout(worker_config):
    factory = ResettingFactory(failures=1, reset=Reset.COMMAND)
    result, _ = await rollout(worker_config, factory, environment=SETUP_ENVIRONMENT)
    assert factory.creations == 2
    assert result.failure is None


@pytest.mark.asyncio
async def test_lost_connection_in_a_model_command_keeps_the_failure(worker_config):
    factory = ResettingFactory(failures=1, reset=Reset.COMMAND)
    replies = [sampled_reply(None, tool_calls=shell_call("true")["tool_calls"])]
    result, waits = await rollout(worker_config, factory, replies=replies)
    assert factory.creations == 1
    assert waits == []
    assert result.failure.exception_type == "TaskMachineError"


@pytest.mark.asyncio
async def test_harbor_task_obeys_the_same_zero_retry_budget(worker_config):
    factory = ResettingFactory(failures=1)
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {}}))
    result, waits = await rollout(worker_config, factory, harbor=harbor, retry=RetrySettings())
    assert factory.creations == 1
    assert waits == []
    assert result.failure.exception_type == "TaskMachineError"



@pytest.mark.asyncio
async def test_permanent_provider_failure_does_not_restart_the_machine(worker_config):
    class UnauthorizedFactory(ResettingFactory):
        async def create(self, _spec):
            self.creations += 1
            raise DaytonaError("unauthorized", status_code=401)

    factory = UnauthorizedFactory(failures=1)
    result, waits = await rollout(worker_config, factory)
    assert factory.creations == 1
    assert waits == []
    assert result.failure.cause_type == "DaytonaError"
    assert result.failure.provider_status_code == 401


@pytest.mark.asyncio
async def test_lost_connection_during_resource_upload_restarts_before_the_model_turn(worker_config):
    factory = ResettingFactory(failures=1, reset=Reset.UPLOAD)
    resources = ResourceGroups(worker=(inline_resource("input.txt", b"task input"),))
    result, waits = await rollout(worker_config, factory, resources=resources)
    assert factory.creations == 2
    assert len(waits) == 1
    assert result.failure is None


@pytest.mark.asyncio
async def test_lost_connection_during_grading_keeps_the_sampled_rollout(worker_config):
    class UnavailableVerifier(DockerMachineFactory):
        async def create(self, _spec):
            raise DaytonaConnectionError("Connection reset during grading")

    factory = ResettingFactory(failures=0)
    grader = ScriptGrader(argv=("true",), environment=NATIVE_ENVIRONMENT)
    result, waits = await rollout(worker_config, factory, grader=grader, verifier_factory=UnavailableVerifier())
    assert factory.creations == 1
    assert waits == []
    assert result.failure.exception_type == "TaskMachineError"
    assert result.failure.operation == "grade"
    assert result.steps


def test_harbor_preset_preserves_its_budget_in_the_common_task_settings():
    directory = Path(__file__).resolve().parents[3] / "skyrl-train/skyrl_train/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(config_name="ppo_base_config", overrides=["+terminal_bench_config=terminal_bench"])
    settings = RetrySettings.model_validate(OmegaConf.to_container(config.environment.task_sessions.retry, resolve=True))
    assert (settings.max_retries, settings.wait_multiplier) == (2, 2)
    assert "max_retries" not in config.terminal_bench_config.harbor
