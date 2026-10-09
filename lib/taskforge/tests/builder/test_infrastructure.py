# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host failures inside a build raise ``BuildInfrastructureFailure``; program failures do not."""

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rolloutengine.contracts import RolloutInterrupted
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, Machine, MachineSpec, Result, UnsupportedMachineSpec
from taskcompendium.models import AnswerType, PlainText

from taskforge.builder.author import compile_program
from taskforge.builder.infrastructure import BuildInfrastructureFailure, InfrastructureCause
from taskforge.builder.run import item_id_for, run_build
from taskforge.builder.sdk import Build, BuildFailure, BuildServices
from taskforge.builder.step import StepCache
from taskforge.ledger.records import EntryKind
from taskforge.sandbox.images import DockerBuild
from taskforge.spec import draft

SHELLSIM = draft.requirements(image=None)
DOCKER = draft.requirements(image=f"registry.example/task@sha256:{'0' * 64}")
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText()
FULL_CREDIT = "print(1.0)\n"
GRADER = draft.python_grader(
    FULL_CREDIT, {}, environment=draft.grader_environment(None), answer_path=draft.ANSWER_PATH, timeout=60
)
UNAVAILABLE = ConnectError(Code.UNAVAILABLE, "controller connection refused")


class FailingFactory:
    """A factory whose ``create`` raises ``error``."""

    backend = Backend.SHELLSIM

    def __init__(self, error: Exception):
        self.error = error

    async def create(self, spec: MachineSpec) -> Machine:
        raise self.error


class StalledFactory:
    """A factory whose ``create`` never returns."""

    backend = Backend.SHELLSIM

    async def create(self, spec: MachineSpec) -> Machine:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


SCHEDULING_TIMEOUT = FailingFactory(TimeoutError("Iris sandbox did not start within 600 seconds"))


@dataclasses.dataclass
class DroppingMachine:
    """A ShellSim machine whose commands fail with ``error`` once the host drops."""

    machine: Machine
    error: Exception

    async def run(self, command: Command) -> Result:
        raise self.error

    async def upload(self, source, target) -> None:
        await self.machine.upload(source, target)

    async def download(self, source, target) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


class DroppingFactory:
    """A ShellSim factory whose machines lose their host before the first command."""

    backend = Backend.SHELLSIM

    def __init__(self, error: Exception):
        self.error = error

    async def create(self, spec: MachineSpec) -> Machine:
        return DroppingMachine(await ShellSimMachineFactory().create(spec), self.error)


@pytest.fixture
def build_with(proposal, tmp_path, services) -> Callable:
    """``async with build_with(factories) as b``: a build context over these machine factories."""

    @asynccontextmanager
    async def make(factories) -> AsyncIterator[Build]:
        cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
        async with services() as s:
            yield Build(proposal, cache.item_id, dataclasses.replace(s, factories=factories), cache, tmp_path, 0)

    return make


@pytest.mark.parametrize(
    ("factory", "cause"),
    [
        (SCHEDULING_TIMEOUT, "scheduling_timeout"),
        (FailingFactory(UNAVAILABLE), "host_unreachable"),
        (FailingFactory(ConnectionRefusedError("controller")), "host_unreachable"),
        (DroppingFactory(UNAVAILABLE), "host_unreachable"),
    ],
)
async def test_host_failures_in_a_machine_are_infrastructure(build_with, factory, cause):
    async with build_with({Backend.SHELLSIM.value: factory}) as b:
        with pytest.raises(BuildInfrastructureFailure) as machine_failure:
            async with b.machine(SHELLSIM, MACHINE) as machine:
                await machine.run(Command(argv=("true",)))
        with pytest.raises(BuildInfrastructureFailure) as grader_failure:
            # Capturing the output path is a command in the task machine.
            await b.try_grader(
                SHELLSIM, GRADER, AnswerType.TEXT, PLAIN, "question", "42", output_paths=("/workspace/answer",)
            )

    assert machine_failure.value.cause == grader_failure.value.cause == InfrastructureCause(cause)


async def test_a_machine_backend_the_host_lacks_is_infrastructure(build_with):
    async with build_with({Backend.SHELLSIM.value: ShellSimMachineFactory()}) as b:
        with pytest.raises(BuildInfrastructureFailure) as machine_failure:
            async with b.machine(DOCKER, MACHINE):
                pass
        with pytest.raises(BuildInfrastructureFailure) as grader_failure:
            await b.try_grader(DOCKER, GRADER, AnswerType.TEXT, PLAIN, "question", "42")

    assert machine_failure.value.cause == grader_failure.value.cause == InfrastructureCause.NO_FACTORY


async def test_a_host_without_an_image_builder_is_infrastructure(build_with):
    build = DockerBuild(files=(draft.file("Dockerfile", "FROM busybox\n"),))
    async with build_with({Backend.SHELLSIM.value: ShellSimMachineFactory()}) as b:
        with pytest.raises(BuildInfrastructureFailure) as failure:
            await b.publish_image(build)

    assert failure.value.cause == InfrastructureCause.NO_IMAGE_BUILDER


@pytest.mark.parametrize(
    "error",
    [
        # Shellbox's Docker backend when the program's Dockerfile does not build.
        RuntimeError("docker build failed: COPY failed: file not found"),
        UnsupportedMachineSpec("Docker task images require setsid for command cancellation"),
        # The controller rejected the request itself, such as a malformed image reference.
        ConnectError(Code.INVALID_ARGUMENT, "invalid task image"),
    ],
)
async def test_an_image_or_spec_the_program_chose_stays_the_programs_failure(build_with, error):
    async with build_with({Backend.SHELLSIM.value: FailingFactory(error)}) as b:
        with pytest.raises(type(error)):
            async with b.machine(SHELLSIM, MACHINE):
                pass
        with pytest.raises(RolloutInterrupted) as interrupted:
            await b.try_grader(SHELLSIM, GRADER, AnswerType.TEXT, PLAIN, "question", "42")

    assert interrupted.value.__cause__ is error


async def test_the_programs_own_startup_deadline_stays_the_programs_failure(build_with):
    async with build_with({Backend.SHELLSIM.value: StalledFactory()}) as b:
        with pytest.raises(TimeoutError):
            async with b.machine(SHELLSIM, draft.machine(startup_timeout=0.05)):
                pass


async def build_on(factory, source, proposal, tmp_path, services):
    async with services() as s:
        machines: BuildServices = dataclasses.replace(s, factories={Backend.SHELLSIM.value: factory})
        program = compile_program(source, proposal.digest)
        return await run_build(program, proposal, tmp_path / "item", tmp_path / "cache", machines)


PROTOTYPE = (
    "    reference = await b.try_grader("
    'env, package, AnswerType.TEXT, ANSWER_FORMAT, "question", "ANSWER = 42", files=FILES)\n'
)
WRAPPED = (
    "    try:\n"
    "        reference = await b.try_grader("
    'env, package, AnswerType.TEXT, ANSWER_FORMAT, "question", "ANSWER = 42", files=FILES)\n'
    "    except Exception as error:\n"
    '        raise b.failure(f"the grader did not run: {error}") from error\n'
)


@pytest.mark.parametrize("wrap", [False, True], ids=["propagated", "wrapped-by-the-program"])
async def test_run_build_raises_host_failures_as_infrastructure(
    proposal, program_source, tmp_path, services, ledger, wrap
):
    assert PROTOTYPE in program_source
    source = program_source.replace(PROTOTYPE, WRAPPED) if wrap else program_source

    with pytest.raises(BuildInfrastructureFailure) as failure:
        await build_on(SCHEDULING_TIMEOUT, source, proposal, tmp_path, services)

    assert failure.value.cause == InfrastructureCause.SCHEDULING_TIMEOUT
    stage = [entry for entry in ledger.entries if entry.kind == EntryKind.STAGE]
    assert [(entry.cause, entry.attrs["infrastructure"]) for entry in stage] == [
        ("BuildInfrastructureFailure", "scheduling_timeout")
    ]
    assert not (tmp_path / "item" / "draft").exists()


async def test_run_build_keeps_program_caused_machine_failures(proposal, program_source, tmp_path, services):
    image_failure = FailingFactory(RuntimeError("docker build failed: COPY failed"))
    wrapped = program_source.replace(PROTOTYPE, WRAPPED)

    with pytest.raises(BuildFailure, match="the grader did not run"):
        await build_on(image_failure, wrapped, proposal, tmp_path, services)
