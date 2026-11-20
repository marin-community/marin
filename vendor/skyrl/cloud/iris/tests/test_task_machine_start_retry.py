"""A rollout whose task machine loses its connection while starting restarts with a fresh machine."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from daytona import DaytonaConnectionError
from omegaconf import OmegaConf
from rolloutengine.test_support import shell_call
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineSpec, Result
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import MACHINE_START_RETRIES, TaskRolloutWorker
from skyrl_train.trajectory_runners.types import TaskRequestBatch
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task

from cloud.iris.tests.task_worker_fixtures import ScriptedClient, WorkerTokenizer, sampled_reply

SETUP_ENVIRONMENT = SHELL_ENVIRONMENT.model_copy(update={"setup_commands": ("true",)})


class Reset(StrEnum):
    """Where the Daytona connection resets."""

    CREATE = "create"
    COMMAND = "command"


@dataclass
class ResettingMachine:
    machine: Machine
    reset: bool

    async def run(self, command: Command) -> Result:
        if self.reset:
            raise DaytonaConnectionError("Failed to execute command: [Errno 104] Connection reset by peer")
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
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
        return ResettingMachine(await ShellSimMachineFactory().create(spec), reset=fails)


async def rollout(worker_config, factory, *, environment=SHELL_ENVIRONMENT, harbor=None, replies=None):
    """Run one shell task and return its result with the waits between restarts."""
    waits = []

    async def wait(delay):
        waits.append(delay)

    task = text_task().model_copy(
        update={"environment_requirements": environment, "tags": () if harbor is None else ("harbor",)}
    )
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient(replies or [sampled_reply("Done")]),
        {"shellsim": factory},
        harbor=harbor,
        retry_wait=wait,
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(
            TaskRequestBatch(rows=(task_input(lowered_task(task, max_turns=2, backend="shellsim")),))
        )
    finally:
        await worker.shutdown()
    return result, waits


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "recovers"),
    [(MACHINE_START_RETRIES, True), (MACHINE_START_RETRIES + 1, False)],
    ids=["recovers", "gives-up"],
)
async def test_lost_connection_at_machine_start_restarts_the_rollout(worker_config, failures, recovers):
    factory = ResettingFactory(failures)
    result, waits = await rollout(worker_config, factory)
    assert factory.creations == MACHINE_START_RETRIES + 1
    assert (result.failure is None) == recovers
    # Each restart waits first, because the resets arrive in bursts.
    assert len(waits) == MACHINE_START_RETRIES and all(wait > 0 for wait in waits)
    assert result.metrics["rollout_retries"] == MACHINE_START_RETRIES


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
async def test_harbor_task_restarts_only_under_its_own_retry_policy(worker_config):
    factory = ResettingFactory(failures=1)
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}}))
    result, waits = await rollout(worker_config, factory, harbor=harbor)
    assert factory.creations == 1
    assert waits == []
    assert result.failure.exception_type == "TaskMachineError"
