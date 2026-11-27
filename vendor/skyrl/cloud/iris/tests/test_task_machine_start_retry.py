# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
from daytona import DaytonaConnectionError
from rolloutengine.test_support import shell_call
from rolloutengine.models import CLEANUP_ERRORS_DIAGNOSTIC
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from taskcompendium.models import ResourceGroups, ScriptGrader
from taskcompendium.runtime.resources import inline_resource
from tests.cpu.task_specs import NATIVE_ENVIRONMENT
from hydra import compose, initialize_config_dir
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineSpec, Result
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, TaskRolloutWorkerSpec
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.rollouts.machines import OwnedMachineFactory
from rigging.task_errors import RetrySettings, RetryDisposition
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from daytona import DaytonaError
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryBatch
from skyrl_train.rollouts.training_record import TrainingRollout
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task

from cloud.iris.tests.task_worker_fixtures import ScriptedClient, WorkerTokenizer, sampled_reply

RETRY_SETTINGS = RetrySettings(jitter=0, max_retries=2, min_wait_sec=1, wait_multiplier=2)

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


@dataclass
class RolloutEvidence:
    result: TrainingRollout
    waits: list[float]
    batch: TrajectoryBatch


async def rollout_evidence(
    worker_config,
    factory,
    *,
    environment=SHELL_ENVIRONMENT,
    replies=None,
    retry=RETRY_SETTINGS,
    resources=None,
    grader=None,
    verifier_factory=None,
):
    waits = []

    async def wait(delay):
        waits.append(delay)

    task = text_task().model_copy(update={"environment_requirements": environment})
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
        retry_wait=wait,
        retry=retry,
        shutdown_timeout=5,
    )
    lowered = lowered_task(task, max_turns=2, backend="shellsim", verifier_backend="verifier" if grader else None)
    request = TaskRequestBatch(rows=(task_input(lowered),))
    try:
        (result,) = await worker.generate(request)
        batch = worker.projection.project([result], request)
    finally:
        await worker.shutdown()
    return RolloutEvidence(result, waits, batch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "recovers"),
    [(RETRY_SETTINGS.max_retries, True), (RETRY_SETTINGS.max_retries + 1, False)],
    ids=["recovers", "gives-up"],
)
async def test_lost_connection_at_machine_start_restarts_the_rollout(worker_config, failures, recovers):
    factory = ResettingFactory(failures)
    evidence = await rollout_evidence(worker_config, factory)
    assert factory.creations == RETRY_SETTINGS.max_retries + 1
    assert (evidence.result.failure is None) == recovers
    # Each restart waits first, because the resets arrive in bursts.
    assert len(evidence.waits) == RETRY_SETTINGS.max_retries and all(wait > 0 for wait in evidence.waits)
    assert evidence.result.metrics["rollout_retries"] == RETRY_SETTINGS.max_retries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation, environment, resources",
    [
        (Reset.COMMAND, SETUP_ENVIRONMENT, None),
        (Reset.UPLOAD, SHELL_ENVIRONMENT, ResourceGroups(worker=(inline_resource("input.txt", b"task input"),))),
    ],
    ids=["setup-command", "resource-upload"],
)
async def test_connection_loss_before_first_model_turn_restarts_machine(
    worker_config, operation, environment, resources
):
    factory = ResettingFactory(failures=1, reset=operation)
    evidence = await rollout_evidence(worker_config, factory, environment=environment, resources=resources)
    assert factory.creations == 2
    assert evidence.waits == [1]
    assert evidence.result.failure is None


@pytest.mark.asyncio
async def test_lost_connection_in_a_model_command_keeps_the_failure(worker_config):
    factory = ResettingFactory(failures=1, reset=Reset.COMMAND)
    replies = [sampled_reply(None, tool_calls=shell_call("true")["tool_calls"])]
    evidence = await rollout_evidence(worker_config, factory, replies=replies)
    assert factory.creations == 1
    assert evidence.waits == []
    assert evidence.result.failure.exception_type == "TaskMachineError"
    assert evidence.batch["error_treatments"] == ["mask"]
    assert evidence.batch["exclude_from_baseline"] == [True]
    assert not any(evidence.batch["loss_masks"][0])


@pytest.mark.asyncio
async def test_task_obeys_zero_retry_budget(worker_config):
    factory = ResettingFactory(failures=1)
    evidence = await rollout_evidence(
        worker_config,
        factory,
        retry=RetrySettings(
            jitter=0,
        ),
    )
    assert factory.creations == 1
    assert evidence.waits == []
    assert evidence.result.failure.exception_type == "TaskMachineError"


@pytest.mark.asyncio
async def test_permanent_provider_failure_does_not_restart_the_machine(worker_config):
    class UnauthorizedFactory(ResettingFactory):
        async def create(self, _spec):
            self.creations += 1
            raise DaytonaError("unauthorized", status_code=401)

    factory = UnauthorizedFactory(failures=1)
    evidence = await rollout_evidence(worker_config, factory)
    assert factory.creations == 1
    assert evidence.waits == []
    assert evidence.result.failure.cause_type == "DaytonaError"
    assert evidence.result.failure.provider_status_code == 401


@pytest.mark.asyncio
async def test_lost_connection_during_grading_keeps_the_sampled_rollout(worker_config):
    class UnavailableVerifier(DockerMachineFactory):
        async def create(self, _spec):
            raise DaytonaConnectionError("Connection reset during grading")

    factory = ResettingFactory(failures=0)
    grader = ScriptGrader(argv=("true",), placement={"kind": "grader_machine", "environment": NATIVE_ENVIRONMENT})
    evidence = await rollout_evidence(worker_config, factory, grader=grader, verifier_factory=UnavailableVerifier())
    assert factory.creations == 1
    assert evidence.waits == []
    assert evidence.result.failure.exception_type == "TaskMachineError"
    assert evidence.result.failure.operation == "grade"
    assert evidence.result.failure.retry_disposition is RetryDisposition.TRANSIENT
    assert evidence.result.steps
    assert evidence.batch["error_treatments"] == ["mask"]
    assert evidence.batch["exclude_from_baseline"] == [True]
    assert not any(evidence.batch["loss_masks"][0])


@pytest.mark.asyncio
async def test_yaml_spec_worker_retries_machine_start():
    directory = Path(__file__).resolve().parents[3] / "skyrl-train/skyrl_train/config"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        config = compose(
            config_name="ppo_base_config",
            overrides=[
                "+environment.task_sessions.retry.max_retries=2",
                "+environment.task_sessions.retry.wait_multiplier=2",
                "+environment.task_sessions.retry.jitter=0",
            ],
        )
    worker = TaskRolloutWorkerSpec.from_config(config, []).build(WorkerTokenizer(), WorkerShard(0, 1))
    factory = ResettingFactory(failures=2)
    worker.factories["shellsim"] = OwnedMachineFactory(factory, "shellsim")
    worker.model_client = ScriptedClient([sampled_reply("Done")])
    waits = []

    async def wait(delay):
        waits.append(delay)

    worker.retry_wait = wait
    task = text_task().model_copy(update={"environment_requirements": SHELL_ENVIRONMENT})
    request = TaskRequestBatch(rows=(task_input(lowered_task(task, max_turns=2, backend="shellsim")),))
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert result.failure is None
    assert factory.creations == 3
    assert waits == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [ConnectionError("connection reset"), ModelServerError("server_error", "fixture", 503)]
)
async def test_model_failure_after_sampled_work_does_not_restart(worker_config, failure):
    factory = ResettingFactory(failures=0)
    replies = [sampled_reply(None, tool_calls=shell_call("true")["tool_calls"]), failure]
    evidence = await rollout_evidence(worker_config, factory, replies=replies)
    assert factory.creations == 1
    assert evidence.waits == []
    assert evidence.result.steps
    assert evidence.result.failure.retry_disposition is RetryDisposition.TRANSIENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_code", "retries", "disposition"),
    [(2, 0, RetryDisposition.PERMANENT), (137, 1, RetryDisposition.TRANSIENT)],
    ids=["task-failure", "signal-exit"],
)
async def test_agent_setup_retries_signal_exit(worker_config, exit_code, retries, disposition):
    environment = SHELL_ENVIRONMENT.model_copy(update={"setup_commands": (f"exit {exit_code}",)})
    evidence = await rollout_evidence(
        worker_config,
        ResettingFactory(failures=0),
        environment=environment,
        retry=RetrySettings(max_retries=1, jitter=0),
    )
    assert evidence.result.failure is not None
    assert evidence.result.failure.diagnostics["setup_exit_code"] == exit_code
    assert evidence.result.failure.retry_disposition is disposition
    assert evidence.result.metrics["rollout_retries"] == retries
    assert len(evidence.waits) == retries


@pytest.mark.asyncio
async def test_unconfirmed_cleanup_keeps_a_transient_startup_failure_terminal(worker_config):
    close_error = ConnectionError("cleanup confirmation lost")

    class FailedCleanupMachine(ResettingMachine):
        async def close(self):
            await self.machine.close()
            raise close_error

    class FailedCleanupFactory(ResettingFactory):
        async def create(self, spec):
            self.creations += 1
            return FailedCleanupMachine(await ShellSimMachineFactory().create(spec), reset=True, operation=Reset.UPLOAD)

    factory = FailedCleanupFactory(failures=1)
    waits = []

    async def wait(delay):
        waits.append(delay)

    task = text_task().model_copy(
        update={
            "environment_requirements": SHELL_ENVIRONMENT,
            "resources": ResourceGroups(worker=(inline_resource("input.txt", b"task input"),)),
        }
    )
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient([sampled_reply()]),
        {"shellsim": factory},
        retry=RETRY_SETTINGS,
        retry_wait=wait,
        shutdown_timeout=5,
    )
    request = TaskRequestBatch(rows=(task_input(lowered_task(task, backend="shellsim")),))
    try:
        (result,) = await worker.generate(request)
    finally:
        with pytest.raises(ExceptionGroup) as shutdown:
            await worker.shutdown()
    assert shutdown.value.subgroup(lambda error: error is close_error) is not None
    assert factory.creations == 1
    assert waits == []
    assert not result.steps
    assert result.grade.diagnostics[CLEANUP_ERRORS_DIAGNOSTIC]
    assert result.failure.retry_disposition is RetryDisposition.TRANSIENT
