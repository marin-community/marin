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
    MachineTerminated,
    Result,
    UnsupportedMachineSpec,
)
from taskcompendium.models import AnswerType, PlainText, TaskSpec

from taskforge.ledger.records import EntryKind
from taskforge.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT
from taskforge.sandbox.images import DockerBuild
from taskforge.spec import draft
from taskforge.stage10_proposal.model import SolverMachine
from taskforge.stage30_author.program import rl_env_build_program, rl_env_build_sink
from taskforge.stage40_build_with_program import sandbox_effects
from taskforge.stage40_build_with_program.context import BuildFailure, BuildInfrastructureFailure, RLEnvBuildContext
from taskforge.stage40_build_with_program.infrastructure import InfrastructureCause, machine_available
from taskforge.stage40_build_with_program.run import DRAFT_DIR, ResearchRequirement, run_build
from taskforge.stage40_build_with_program.sandbox_effects import (
    GRADE_CANDIDATE_MACHINE,
    GRADE_CANDIDATE_VERIFIER_TIMEOUT,
    build_machine,
    grade_candidate,
    publish_image,
)
from taskforge.stage40_build_with_program.sdk import GRADER_TIMEOUT
from taskforge.stage50_validate.lowering import VERIFIER_TIMEOUT
from tests.sandbox.fixture_images import PortlessShellSimFactory, ShellGraderImageFactory, shell_grader
from tests.stage40_build_with_program.conftest import FACTORIES, WORKPIECE, SAMPLING, RecordingImageBuilder, declaring

SHELLSIM = draft.requirements(image=None)
DOCKER = draft.requirements(image=f"registry.example/task@sha256:{'0' * 64}")
MACHINE = draft.machine(startup_timeout=60)
PLAIN = PlainText()
FULL_CREDIT = """printf '%s' '{"reward": 1.0}' > /logs/verifier/reward.json\n"""
GRADER = shell_grader(
    FULL_CREDIT, {}, environment=draft.grader_environment(None), answer_path=draft.ANSWER_PATH, timeout=60
)
UNAVAILABLE = ConnectError(Code.UNAVAILABLE, "controller connection refused")
FILES = (draft.file("workspace/data.txt", "40 2\n"),)
DEADLINE = 0.3
"""Seconds a test's startup or verifier deadline allows; a stalled machine stalls for longer."""
PARENT_GONE = {
    # The Iris factory's _create_sync when the sandbox task ended before it ran.
    "sandbox-failed-before-running": RuntimeError("Iris sandbox task failed before running: Parent job terminated"),
    # The controller refusing a sandbox under a parent job a shard relaunch killed.
    "cannot-submit-job": ConnectError(
        Code.FAILED_PRECONDITION,
        "Cannot submit job: parent job /muchanem/gen1000-s1p has terminated (state=JOB_STATE_KILLED)",
    ),
}
LOST = {
    "machine-terminated": MachineTerminated("Iris sandbox task t/0 is TASK_STATE_KILLED: Parent job terminated"),
    # ShellSim closes a machine after one command timeout; every later call raises this.
    "shellsim-machine-closed": RuntimeError("Machine is closed"),
}


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
class StallingMachine:
    """A machine whose ``stalled`` operation (``run`` or ``upload``) never returns, as a sandbox that stops answering."""

    machine: Machine
    stalled: str

    async def run(self, command: Command) -> Result:
        if self.stalled == "run":
            await asyncio.Event().wait()
        return await self.machine.run(command)

    async def upload(self, source, target) -> None:
        if self.stalled == "upload":
            await asyncio.Event().wait()
        await self.machine.upload(source, target)

    async def download(self, source, target) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclasses.dataclass(frozen=True)
class StallingFactory:
    """The fixture container backend, whose machines never finish their ``stalled`` operation."""

    stalled: str
    images: ShellGraderImageFactory = dataclasses.field(default_factory=ShellGraderImageFactory)

    @property
    def backend(self) -> Backend:
        return self.images.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.images.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        return StallingMachine(await self.images.create(spec), self.stalled)


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
        return DroppingMachine(await PortlessShellSimFactory().create(spec), self.error)


@pytest.mark.parametrize(
    ("factory", "cause"),
    [
        (SCHEDULING_TIMEOUT, "scheduling_timeout"),
        (FailingFactory(UNAVAILABLE), "host_unreachable"),
        (FailingFactory(ConnectionRefusedError("controller")), "host_unreachable"),
        (DroppingFactory(UNAVAILABLE), "host_unreachable"),
        *((FailingFactory(error), "machine_create_failed") for error in PARENT_GONE.values()),
        *((DroppingFactory(error), "machine_lost") for error in LOST.values()),
    ],
    ids=[
        "scheduling-timeout",
        "controller-unavailable",
        "connection-refused",
        "dropped-under-a-command",
        *PARENT_GONE,
        *LOST,
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
    ("machine", "cause"),
    [(None, InfrastructureCause.NO_IMAGE_BUILDER), (SolverMachine.PACKAGES, InfrastructureCause.NO_FACTORY)],
    ids=["container-without-an-image-builder", "packages-without-the-local-backend"],
)
def test_a_declared_machine_the_host_cannot_provide_is_the_hosts_failure(services, proposal, machine, cause):
    """The host raises rather than sending the model a machine it can run, so it never chooses the solver's."""
    with pytest.raises(BuildInfrastructureFailure) as failure:
        machine_available(services, declaring(proposal, machine))

    assert failure.value.cause == cause


@pytest.mark.parametrize("machine", [None, *list(SolverMachine)])
def test_a_host_with_an_image_builder_and_the_local_backend_provides_every_machine(services, proposal, machine):
    able = dataclasses.replace(
        services,
        images=RecordingImageBuilder(),
        factories={**FACTORIES, Backend.LOCAL.value: ShellSimMachineFactory()},
    )

    machine_available(able, declaring(proposal, machine))


async def test_run_build_ends_a_build_whose_declared_machine_the_host_lacks_before_any_step(
    proposal, services, glm_client, tmp_path, ledger
):
    @rl_env_build_sink
    async def task(context: RLEnvBuildContext) -> TaskSpec:
        raise AssertionError("no step runs on a host that cannot provide the machine")

    with pytest.raises(BuildInfrastructureFailure) as failure:
        await run_build(
            rl_env_build_program(sink=task),
            declaring(proposal, SolverMachine.PACKAGES),
            WORKPIECE,
            tmp_path / "round",
            str(tmp_path / "store"),
            glm_client,
            SAMPLING,
            services,
            0,
            research=ResearchRequirement.OFF,
        )

    assert failure.value.cause == InfrastructureCause.NO_FACTORY
    assert not ledger.entries
    assert not (tmp_path / "round").exists()


@pytest.mark.parametrize(
    "error",
    [
        # Shellbox's Docker backend when the program's Dockerfile does not build.
        RuntimeError("docker build failed: COPY failed: file not found"),
        UnsupportedMachineSpec("Docker task images require setsid for command cancellation"),
        # The controller rejected the request itself, such as a malformed image reference.
        ConnectError(Code.INVALID_ARGUMENT, "invalid task image"),
        # A precondition of the request, not the parent job's end.
        ConnectError(Code.FAILED_PRECONDITION, "task image is not in the allowed registry"),
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


def test_verifier_deadlines_outlast_the_sandbox_scheduling_timeout():
    """A verifier deadline at or under the factory's scheduling timeout cancels the machine wait first.

    The factory's ``TimeoutError`` is then never raised, the engine reports a bare deadline, and a
    sandbox that never starts spends a program revision instead of failing as infrastructure. Both
    deadlines leave the factory its whole budget and the grader its own on top.
    """
    assert GRADE_CANDIDATE_VERIFIER_TIMEOUT >= SANDBOX_SCHEDULING_TIMEOUT + GRADER_TIMEOUT
    assert VERIFIER_TIMEOUT >= SANDBOX_SCHEDULING_TIMEOUT + GRADER_TIMEOUT
    assert GRADE_CANDIDATE_MACHINE.startup_timeout is not None
    assert GRADE_CANDIDATE_MACHINE.startup_timeout > SANDBOX_SCHEDULING_TIMEOUT


@pytest.mark.parametrize("stalled", ["run", "upload"], ids=["command", "upload"])
async def test_a_grading_deadline_on_a_stalled_verifier_is_infrastructure(context, monkeypatch, stalled):
    """The grader's own command has its own timeout, so a verifier deadline that expires is the sandbox's latency."""
    monkeypatch.setattr(sandbox_effects, "GRADE_CANDIDATE_VERIFIER_TIMEOUT", DEADLINE)
    on_host = context(factories={**FACTORIES, Backend.DOCKER.value: StallingFactory(stalled)})
    with pytest.raises(BuildInfrastructureFailure) as failure:
        await grade_candidate(on_host, None, GRADER, AnswerType.TEXT, PLAIN, "question", "42")

    assert failure.value.cause == InfrastructureCause.SANDBOX_DEADLINE


@pytest.mark.parametrize(
    ("factories", "requirements"),
    [
        ({Backend.SHELLSIM.value: StalledFactory()}, SHELLSIM),
        ({Backend.DOCKER.value: StallingFactory("upload")}, DOCKER),
    ],
    ids=["never-scheduled", "stalled-upload"],
)
async def test_a_startup_deadline_on_a_stalled_machine_is_infrastructure(context, factories, requirements):
    on_host = context(factories={**FACTORIES, **factories})
    settings = draft.machine(startup_timeout=DEADLINE)
    with pytest.raises(BuildInfrastructureFailure) as machine_failure:
        async with build_machine(on_host.services, requirements, settings, FILES):
            pass
    with pytest.raises(BuildInfrastructureFailure) as grader_failure:
        await grade_candidate(
            on_host, requirements, GRADER, AnswerType.TEXT, PLAIN, "question", "42", files=FILES, settings=settings
        )

    assert machine_failure.value.cause == grader_failure.value.cause == InfrastructureCause.SANDBOX_DEADLINE


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
        WORKPIECE,
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
