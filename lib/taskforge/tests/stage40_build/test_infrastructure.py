# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host failures inside a build raise ``BuildInfrastructureFailure``; program failures do not."""

import asyncio
import dataclasses

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rolloutengine.contracts import RolloutInterrupted
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    Machine,
    MachineSpec,
    Result,
    UnsupportedMachineSpec,
)
from taskcompendium.models import AnswerType, PlainText, TaskSpec

from taskforge.ledger.records import EntryKind
from taskforge.sandbox.images import DockerBuild
from taskforge.spec import draft
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink
from taskforge.stage40_build.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build.infrastructure import InfrastructureCause
from taskforge.stage40_build.run import DRAFT_DIR, ResearchRequirement, run_build
from taskforge.stage40_build.sandbox_effects import build_machine, grade_candidate, publish_image
from tests.sandbox.fixture_images import shell_grader
from tests.stage40_build.conftest import FACTORIES, ITEM, SAMPLING

SHELLSIM = draft.requirements(image=None)
DOCKER = draft.requirements(image=f"registry.example/task@sha256:{'0' * 64}")
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText()
FULL_CREDIT = """printf '%s' '{"reward": 1.0}' > /logs/verifier/reward.json\n"""
GRADER = shell_grader(
    FULL_CREDIT, {}, environment=draft.grader_environment(None), answer_path=draft.ANSWER_PATH, timeout=60
)
UNAVAILABLE = ConnectError(Code.UNAVAILABLE, "controller connection refused")


class FailingFactory:
    """A factory whose ``create`` raises ``error``."""

    backend = Backend.SHELLSIM

    def __init__(self, error: Exception):
        self.error = error

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        ShellSimMachineFactory().check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise self.error


class StalledFactory:
    """A factory whose ``create`` never returns."""

    backend = Backend.SHELLSIM

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        ShellSimMachineFactory().check_environment(requirements, spec)

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

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        ShellSimMachineFactory().check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        return DroppingMachine(await ShellSimMachineFactory().create(spec), self.error)


@pytest.mark.parametrize(
    ("factory", "cause"),
    [
        (SCHEDULING_TIMEOUT, "scheduling_timeout"),
        (FailingFactory(UNAVAILABLE), "host_unreachable"),
        (FailingFactory(ConnectionRefusedError("controller")), "host_unreachable"),
        (DroppingFactory(UNAVAILABLE), "host_unreachable"),
    ],
)
async def test_host_failures_in_a_machine_are_infrastructure(context, factory, cause):
    on_host = context(factories={**FACTORIES, Backend.SHELLSIM.value: factory})
    with pytest.raises(BuildInfrastructureFailure) as machine_failure:
        async with build_machine(on_host.services, SHELLSIM, MACHINE) as prepared:
            await prepared.run(Command(argv=("true",)))
    with pytest.raises(BuildInfrastructureFailure) as grader_failure:
        # The candidate's workspace file is a shell command in the task machine.
        await grade_candidate(
            on_host,
            SHELLSIM,
            GRADER,
            AnswerType.TEXT,
            PLAIN,
            "question",
            "42",
            workspace=(draft.file("workspace/a", "1"),),
        )

    assert machine_failure.value.cause == grader_failure.value.cause == InfrastructureCause(cause)


async def test_a_machine_backend_the_host_lacks_is_infrastructure(context):
    on_host = context(factories={Backend.SHELLSIM.value: ShellSimMachineFactory()})
    with pytest.raises(BuildInfrastructureFailure) as machine_failure:
        async with build_machine(on_host.services, DOCKER, MACHINE):
            pass
    with pytest.raises(BuildInfrastructureFailure) as grader_failure:
        await grade_candidate(on_host, DOCKER, GRADER, AnswerType.TEXT, PLAIN, "question", "42")

    assert machine_failure.value.cause == grader_failure.value.cause == InfrastructureCause.NO_FACTORY


async def test_a_host_without_an_image_builder_is_infrastructure(context):
    build = DockerBuild(files=(draft.file("Dockerfile", "FROM busybox\n"),))
    with pytest.raises(BuildInfrastructureFailure) as failure:
        await publish_image(context(), build)

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
async def test_an_image_or_spec_the_program_chose_stays_the_programs_failure(context, error):
    on_host = context(factories={**FACTORIES, Backend.SHELLSIM.value: FailingFactory(error)})
    with pytest.raises(type(error)):
        async with build_machine(on_host.services, SHELLSIM, MACHINE):
            pass
    with pytest.raises(RolloutInterrupted) as interrupted:
        await grade_candidate(on_host, SHELLSIM, GRADER, AnswerType.TEXT, PLAIN, "question", "42")

    assert interrupted.value.__cause__ is error


async def test_the_programs_own_startup_deadline_stays_the_programs_failure(context):
    on_host = context(factories={Backend.SHELLSIM.value: StalledFactory()})
    with pytest.raises(TimeoutError):
        async with build_machine(on_host.services, SHELLSIM, draft.machine(startup_timeout=0.05)):
            pass


def grading_program(wrapped: bool):
    """A one-step program whose sink grades a candidate, letting a host failure propagate or wrapping it."""

    @rl_env_build_sink
    async def task(context: RLEnvBuildContext) -> TaskSpec:
        try:
            await grade_candidate(context, SHELLSIM, GRADER, AnswerType.TEXT, PLAIN, "question", "42")
        except Exception as error:
            if not wrapped:
                raise
            raise BuildFailure(f"the grader did not run: {error}") from error
        raise AssertionError("the factory fails every machine")

    return rl_env_build_program(sink=task)


async def build_on(factory, wrapped: bool, proposal, services, glm_client, tmp_path):
    return await run_build(
        grading_program(wrapped),
        proposal,
        ITEM,
        tmp_path / "round",
        str(tmp_path / "store"),
        glm_client,
        SAMPLING,
        dataclasses.replace(services, factories={**FACTORIES, Backend.SHELLSIM.value: factory}),
        0,
        research=ResearchRequirement.OFF,
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["propagated", "wrapped-by-the-program"])
async def test_run_build_raises_host_failures_as_infrastructure(
    proposal, services, glm_client, tmp_path, ledger, wrapped
):
    with pytest.raises(BuildInfrastructureFailure) as failure:
        await build_on(SCHEDULING_TIMEOUT, wrapped, proposal, services, glm_client, tmp_path)

    assert failure.value.cause == InfrastructureCause.SCHEDULING_TIMEOUT
    steps = [entry for entry in ledger.entries if entry.kind == EntryKind.STEP]
    assert [(entry.cause, entry.attrs["infrastructure"]) for entry in steps] == [
        ("BuildInfrastructureFailure", "scheduling_timeout")
    ]
    assert not (tmp_path / "round" / DRAFT_DIR).exists()


async def test_run_build_keeps_program_caused_machine_failures(proposal, services, glm_client, tmp_path):
    image_failure = FailingFactory(RuntimeError("docker build failed: COPY failed"))

    with pytest.raises(BuildFailure, match="task: the grader did not run"):
        await build_on(image_failure, True, proposal, services, glm_client, tmp_path)
