# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The preflight probes a local task machine before any trial and falls back to a container when it cannot start."""

from dataclasses import dataclass

from shellbox.backends.shellsim.machine import ShellSimMachine
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    ExitReason,
    MachineFactory,
    MachineSpec,
    PackageSpec,
    Result,
)
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.sandbox.factories import MachineHost
from taskforge.sandbox.images import GRADER_BASE_IMAGE
from taskforge.spec import draft
from taskforge.stage50_validate.outcome import Cause
from taskforge.stage50_validate.preflight import preflight
from taskforge.stage50_validate.trials import Deadlines, EngineSettings
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory

READ_ONLY = b"mkdir: cannot create directory '/opt/gen': Read-only file system\n"
DEADLINES = Deadlines(total_turn_timeout=20, attempt_timeout=60)


class ReadOnlyOptMachine:
    """A ShellSim machine whose shell refuses to write under /opt, as bubblewrap's read-only mounts do."""

    def __init__(self, inner: ShellSimMachine):
        self.inner = inner

    async def run(self, command: Command) -> Result:
        if command.argv[:2] == ("sh", "-c") and "/opt" in command.argv[2]:
            return Result(1, b"", READ_ONLY, False, False, ExitReason.EXITED)
        return await self.inner.run(command)

    async def upload(self, source, target) -> None:
        await self.inner.upload(source, target)

    async def download(self, source, target) -> None:
        await self.inner.download(source, target)

    async def close(self) -> None:
        await self.inner.close()


@dataclass(frozen=True)
class ReadOnlyLocalFactory:
    """Stands in for the local backend: ShellSim machines with /opt read-only."""

    backend: Backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        """Accept every environment: the test checks the preflight's fallback, not a backend's support."""

    async def create(self, spec: MachineSpec) -> ReadOnlyOptMachine:
        return ReadOnlyOptMachine(await PortlessShellSimFactory().create(MachineSpec(source=PackageSpec())))


def cpython_task(setup: tuple[str, ...]) -> TaskSpec:
    return draft.assemble(
        "opt-writer",
        "Read /workspace/numbers.txt and reply with the sum alone.",
        AnswerType.TEXT,
        PlainText(),
        draft.answer_grader(ExactSpec(expected=("42",))),
        Source(dataset="taskforge-validate-tests", revision="1", row="opt", importer_revision="1"),
        environment=draft.requirements(image=GRADER_BASE_IMAGE, setup=setup, workdir="/workspace"),
        files=(draft.file("workspace/numbers.txt", "40\n2\n"),),
    )


def settings(local: ReadOnlyLocalFactory | None) -> EngineSettings:
    factories: dict[str, MachineFactory] = {
        Backend.SHELLSIM.value: PortlessShellSimFactory(),
        Backend.DOCKER.value: FixtureImageFactory(),
    }
    if local is not None:
        factories[Backend.LOCAL.value] = local
    return EngineSettings(
        factories=factories,
        max_turns=2,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
    )


async def test_a_setup_command_writing_under_a_read_only_root_moves_the_task_machine_to_a_container():
    engine = settings(ReadOnlyLocalFactory())

    result = await preflight(
        cpython_task(("mkdir -p /workspace /opt/gen",)), MachineHost.LAPTOP, engine.factories, engine, DEADLINES
    )

    assert result.lowered.runtime.task_machine is not None
    assert result.lowered.runtime.task_machine.backend == Backend.DOCKER.value
    assert result.lowered.task.environment_requirements.docker_image == GRADER_BASE_IMAGE
    assert result.fallback is not None
    assert (result.fallback.from_backend, result.fallback.to_backend) == (Backend.LOCAL.value, Backend.DOCKER.value)
    assert result.fallback.cause is Cause.TASK_SETUP
    assert "Environment setup command failed" in result.fallback.detail


async def test_a_task_whose_setup_runs_locally_keeps_the_local_backend():
    engine = settings(ReadOnlyLocalFactory())

    result = await preflight(
        cpython_task(("mkdir -p /workspace/out",)), MachineHost.LAPTOP, engine.factories, engine, DEADLINES
    )

    assert result.fallback is None
    assert result.lowered.runtime.task_machine is not None
    assert result.lowered.runtime.task_machine.backend == Backend.LOCAL.value


async def test_a_host_without_the_local_backend_is_not_probed():
    engine = settings(None)

    result = await preflight(
        cpython_task(("mkdir -p /opt/gen",)), MachineHost.LAPTOP, engine.factories, engine, DEADLINES
    )

    assert result.fallback is None
    assert result.lowered.runtime.task_machine is not None
    assert result.lowered.runtime.task_machine.backend == Backend.DOCKER.value
