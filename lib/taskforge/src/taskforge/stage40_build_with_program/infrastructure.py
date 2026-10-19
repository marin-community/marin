# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host failures inside a build, told apart from failures of the builder program.

A build program uses the host through machines: ``sandbox_effects.build_machine`` and
``sandbox_effects.grade_candidate`` create them from ``RLEnvBuildServices.factories``.
``host_checked_factories`` wraps those factories so that an error the host raises while creating or
driving a machine becomes a ``BuildInfrastructureFailure``. ``run_program`` raises it instead of
handing the program's author a traceback, because no revision of the program can fix the host.

Only errors the program cannot have caused are infrastructure:

* ``NO_FACTORY``: the build needs a machine backend this host has no factory for (a laptop without
  Docker). Lowering raises it, since RolloutEngine checks each machine's environment with its
  factory before it acquires one. It is deterministic on the host (``HOST_REJECTIONS``): no retry here changes it, so the
  loop abandons the item at once without retrying the build.
* ``NO_IMAGE_BUILDER``: the build publishes a task image and this host has no ``ImageBuilder``.
  Also deterministic on the host.
* ``SCHEDULING_TIMEOUT``: the factory itself gave up waiting for a machine (Iris could not
  schedule the sandbox). The engine's verifier deadline also covers the verifier machine's start,
  so every such deadline exceeds ``SANDBOX_SCHEDULING_TIMEOUT``; a shorter one would cancel the
  wait first and the factory's timeout would never be raised.
* ``SANDBOX_DEADLINE``: a machine's startup deadline or the verifier deadline expired while a
  machine started, received files or ran a command (``deadline_failure``). The grader's own command
  runs under its own timeout and comes back as a ``TIMED_OUT`` result, not as a deadline, and the
  verifier deadline is the scheduling budget plus that timeout, so when a deadline still expires the
  time went to sandbox start, transfer or exec latency.
* ``MACHINE_CREATE_FAILED``: the host could not start a sandbox, as when the Iris sandbox task
  ended before it ran or the controller refused it under a terminated parent job.
* ``MACHINE_LOST``: a machine ended under a running command or transfer (``MachineTerminated``,
  or the closed-machine messages ``sandbox.host_failures`` shares with validation).
* ``HOST_UNREACHABLE``: a connection to the machine host failed, or the controller answered an RPC
  with a transport, capacity or credential error.
* ``BUILD_PROCESS``: the child process running the build (``build_process.py``) or loading the
  program (``stage30_author.load_process``) exited without writing its result, as when its
  services could not start or the kernel killed it for memory.

Everything else is the program's: an image it built or named that fails to build, pull or start,
a machine spec the backend refuses (``UnsupportedMachineSpec`` or ``ValueError``), an RPC the
controller refuses for the request itself (a code outside ``HOST_RPC_CODES``), and setup or grader
commands that fail. Shellbox raises a bare ``RuntimeError`` for both a failed Docker build and an
unreachable Docker daemon, so those stay the program's too.

The proposal pins the solver's machine, so the host never chooses it: ``machine_available`` raises
``NO_FACTORY`` for a package machine on a host without the local backend and ``NO_IMAGE_BUILDER`` for
a container proposal on a host without an image builder. ``run_build`` and the agentic author's
``author_round`` call it before anything runs, so such an item ends as the host's before its author
or its program spends a model call.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rolloutengine.contracts import RolloutInterrupted, RolloutOperation
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineFactory, MachineSpec, Result

from taskforge.sandbox.factories import MachineHost, container_backend
from taskforge.sandbox.host_failures import machine_create_failed, machine_lost
from taskforge.stage10_proposal.model import SolverMachine, TaskProposal
from taskforge.stage40_build_with_program.context import BuildInfrastructureFailure
from taskforge.stage40_build_with_program.services import RLEnvBuildServices

HOST_RPC_CODES = frozenset(
    {
        Code.UNAVAILABLE,
        Code.DEADLINE_EXCEEDED,
        Code.RESOURCE_EXHAUSTED,
        Code.ABORTED,
        Code.INTERNAL,
        Code.UNAUTHENTICATED,
        Code.PERMISSION_DENIED,
    }
)
"""Controller RPC codes that describe the host (transport, capacity, credentials), not the request."""


class InfrastructureCause(StrEnum):
    """Why the host, not the program, failed a build."""

    NO_FACTORY = "no_factory"
    NO_IMAGE_BUILDER = "no_image_builder"
    SCHEDULING_TIMEOUT = "scheduling_timeout"
    SANDBOX_DEADLINE = "sandbox_deadline"
    """A startup or verifier deadline expired while a machine started, received files or ran a command."""
    MACHINE_CREATE_FAILED = "machine_create_failed"
    """The host could not start a sandbox, as under a terminated parent job."""
    MACHINE_LOST = "machine_lost"
    """A machine ended under a running command or transfer."""
    HOST_UNREACHABLE = "host_unreachable"
    BUILD_PROCESS = "build_process"


HOST_REJECTIONS = frozenset({InfrastructureCause.NO_FACTORY, InfrastructureCause.NO_IMAGE_BUILDER})
"""Causes that hold for as long as the host is unchanged: the item is abandoned at once, not retried."""
DEADLINE_OPERATIONS = frozenset({RolloutOperation.START, RolloutOperation.GRADE})
"""The engine operations whose deadline only a slow sandbox can exhaust: machine start and grading."""


def machine_available(services: RLEnvBuildServices, proposal: TaskProposal) -> None:
    """Raise the host's failure when ``services`` cannot provide the solver machine ``proposal`` declares.

    Raises:
        BuildInfrastructureFailure: ``NO_IMAGE_BUILDER`` for a container proposal on a host without an
            image builder, ``NO_FACTORY`` for a package machine on a host without the local backend.
    """
    machine = proposal.header.solver_machine
    if machine is None and services.images is None:
        raise BuildInfrastructureFailure(
            InfrastructureCause.NO_IMAGE_BUILDER, "the proposal declares a container and this host cannot publish images"
        )
    if machine is SolverMachine.PACKAGES and Backend.LOCAL.value not in services.factories:
        raise BuildInfrastructureFailure(
            InfrastructureCause.NO_FACTORY,
            f"the proposal declares solver_machine {machine} and this host has no {Backend.LOCAL} machine factory",
        )


def _chain(error: BaseException) -> Iterator[BaseException]:
    """``error``, then each ``__cause__`` (or, absent one, ``__context__``), each link once."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def infrastructure_failure(error: BaseException) -> BuildInfrastructureFailure | None:
    """A fresh copy of the first ``BuildInfrastructureFailure`` in ``error``'s cause chain, or ``None``.

    The chain is ``error`` itself, then each ``__cause__`` (or, absent one, ``__context__``), so a
    host failure the engine wrapped in ``RolloutInterrupted``, or the program wrapped in a
    ``BuildFailure``, is found. Raise the copy ``from error`` to keep the whole chain.
    """
    for link in _chain(error):
        if isinstance(link, BuildInfrastructureFailure):
            return BuildInfrastructureFailure(link.cause, link.message)
    return None


def deadline_failure(error: RolloutInterrupted) -> BuildInfrastructureFailure | None:
    """The ``SANDBOX_DEADLINE`` failure of an interrupted machine start or grading that a deadline ended, or ``None``.

    A ``TimeoutError`` anywhere in the cause chain of a ``START`` or ``GRADE`` interruption is a
    deadline the engine held around the sandbox; any other interruption stays as it is.
    """
    if error.operation not in DEADLINE_OPERATIONS:
        return None
    if not any(isinstance(link, TimeoutError) for link in _chain(error)):
        return None
    return BuildInfrastructureFailure(
        InfrastructureCause.SANDBOX_DEADLINE,
        f"{error}: the sandbox did not answer within the {error.operation} deadline",
    )


def _host_failure(error: Exception) -> BuildInfrastructureFailure | None:
    if isinstance(error, ConnectError) and error.code in HOST_RPC_CODES:
        return BuildInfrastructureFailure(
            InfrastructureCause.HOST_UNREACHABLE, f"machine controller RPC failed ({error.code.value}): {error.message}"
        )
    if isinstance(error, ConnectionError):
        return BuildInfrastructureFailure(
            InfrastructureCause.HOST_UNREACHABLE, f"machine host connection failed: {error}"
        )
    return None


def _create_failure(error: Exception) -> BuildInfrastructureFailure | None:
    if machine_create_failed(error):
        return BuildInfrastructureFailure(
            InfrastructureCause.MACHINE_CREATE_FAILED, f"the host could not start the sandbox: {error}"
        )
    return _host_failure(error)


def _drive_failure(error: Exception) -> BuildInfrastructureFailure | None:
    if machine_lost(error):
        return BuildInfrastructureFailure(
            InfrastructureCause.MACHINE_LOST, f"the machine ended under the build: {error}"
        )
    return _host_failure(error)


@contextmanager
def _host_failures(classify: Callable[[Exception], BuildInfrastructureFailure | None]) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        failure = classify(error)
        if failure is None:
            raise
        raise failure from error


@dataclass(frozen=True)
class HostCheckedMachine:
    """A ``Machine`` whose host errors raise ``BuildInfrastructureFailure``."""

    machine: Machine

    async def run(self, command: Command) -> Result:
        with _host_failures(_drive_failure):
            return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        with _host_failures(_drive_failure):
            await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        with _host_failures(_drive_failure):
            await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass(frozen=True)
class HostCheckedFactory:
    """A ``MachineFactory`` whose host errors raise ``BuildInfrastructureFailure``."""

    factory: MachineFactory

    @property
    def backend(self) -> Backend:
        return self.factory.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> HostCheckedMachine:
        with _host_failures(_create_failure):
            try:
                machine = await self.factory.create(spec)
            except TimeoutError as error:
                raise BuildInfrastructureFailure(
                    InfrastructureCause.SCHEDULING_TIMEOUT, f"the machine factory gave up waiting: {error}"
                ) from error
        return HostCheckedMachine(machine)


@dataclass(frozen=True)
class AbsentFactory:
    """The factory for a machine backend this host cannot create.

    It raises the ``NO_FACTORY`` failure from ``check_environment`` as well as from ``create``, so
    lowering a task onto the backend fails before any of the task's machines starts.
    """

    absent: Backend

    @property
    def backend(self) -> Backend:
        return self.absent

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        raise self._missing()

    async def create(self, spec: MachineSpec) -> Machine:
        raise self._missing()

    def _missing(self) -> BuildInfrastructureFailure:
        return BuildInfrastructureFailure(
            InfrastructureCause.NO_FACTORY, f"this host has no {self.absent} machine factory"
        )


def host_backends(host: MachineHost) -> tuple[Backend, ...]:
    """The backends ``spec.draft.lower`` can choose on ``host``: ShellSim and the host's container backend."""
    return (Backend.SHELLSIM, container_backend(host))


def host_checked_factories(factories: Mapping[str, MachineFactory], host: MachineHost) -> dict[str, MachineFactory]:
    """``factories`` with host errors classified, and an ``AbsentFactory`` for every backend of
    ``host_backends(host)`` it lacks. Keys are ``Backend`` values, as ``machine_factories`` returns them.
    """
    checked: dict[str, MachineFactory] = {key: HostCheckedFactory(factory) for key, factory in factories.items()}
    for backend in host_backends(host):
        checked.setdefault(backend.value, AbsentFactory(backend))
    return checked
