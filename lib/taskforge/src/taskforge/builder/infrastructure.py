# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host failures inside a build, told apart from failures of the builder program.

A builder program reaches the host through machines: ``Build.machine`` and ``Build.try_grader``
create them from ``BuildServices.factories``. ``host_checked_factories`` wraps those factories so
that an error the host raises while creating or driving a machine becomes a
``BuildInfrastructureFailure``. ``run_build`` raises it instead of handing the program's author a
traceback, because no revision of the program can fix the host.

Only errors the program cannot have caused are infrastructure:

* ``NO_FACTORY``: the build needs a machine backend this host has no factory for (a laptop without
  Docker). It is deterministic on the host (``HOST_REJECTIONS``): no retry here changes it, so the
  loop abandons the item at once without retrying the build.
* ``NO_IMAGE_BUILDER``: the build publishes a task image and this host has no ``ImageBuilder``.
  Also deterministic on the host.
* ``SCHEDULING_TIMEOUT``: the factory itself gave up waiting for a machine (Iris could not
  schedule the sandbox). The deadline a program sets with ``spec.machine(startup_timeout=...)`` is
  enforced outside the factory and stays the program's.
* ``HOST_UNREACHABLE``: a connection to the machine host failed, or the controller answered an RPC
  with a transport, capacity or credential error.

Everything else is the program's: an image it built or named that fails to build, pull or start,
a machine spec the backend refuses (``UnsupportedMachineSpec``), and setup or grader commands that
fail. Shellbox raises a bare ``RuntimeError`` for both a failed Docker build and an
unreachable Docker daemon, so those stay the program's too.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from shellbox.machine import Backend, Command, Machine, MachineFactory, MachineSpec, Result

from taskforge.sandbox.factories import MachineHost, container_backend

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
    HOST_UNREACHABLE = "host_unreachable"


HOST_REJECTIONS = frozenset({InfrastructureCause.NO_FACTORY, InfrastructureCause.NO_IMAGE_BUILDER})
"""Causes that hold for as long as the host is unchanged: the item is abandoned at once, not retried."""


class BuildInfrastructureFailure(Exception):
    """The host failed while a build was running; the program is not at fault.

    Not a ``BuildFailure``. The original host error is the ``__cause__`` chain.
    """

    def __init__(self, cause: InfrastructureCause, message: str):
        super().__init__(f"{cause}: {message}")
        self.cause = cause
        self.message = message


def infrastructure_failure(error: BaseException) -> BuildInfrastructureFailure | None:
    """A fresh copy of the first ``BuildInfrastructureFailure`` in ``error``'s cause chain, or ``None``.

    The chain is ``error`` itself, then each ``__cause__`` (or, absent one, ``__context__``), so a
    host failure the engine wrapped in ``RolloutInterrupted``, or the program wrapped in a
    ``BuildFailure``, is found. Raise the copy ``from error`` to keep the whole chain.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        if isinstance(current, BuildInfrastructureFailure):
            return BuildInfrastructureFailure(current.cause, current.message)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


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


@contextmanager
def _host_failures() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        failure = _host_failure(error)
        if failure is None:
            raise
        raise failure from error


@dataclass(frozen=True)
class HostCheckedMachine:
    """A ``Machine`` whose host errors raise ``BuildInfrastructureFailure``."""

    machine: Machine

    async def run(self, command: Command) -> Result:
        with _host_failures():
            return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        with _host_failures():
            await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        with _host_failures():
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

    async def create(self, spec: MachineSpec) -> HostCheckedMachine:
        with _host_failures():
            try:
                machine = await self.factory.create(spec)
            except TimeoutError as error:
                raise BuildInfrastructureFailure(
                    InfrastructureCause.SCHEDULING_TIMEOUT, f"the machine factory gave up waiting: {error}"
                ) from error
        return HostCheckedMachine(machine)


@dataclass(frozen=True)
class AbsentFactory:
    """The factory for a machine backend this host cannot create."""

    absent: Backend

    @property
    def backend(self) -> Backend:
        return self.absent

    async def create(self, spec: MachineSpec) -> Machine:
        raise BuildInfrastructureFailure(
            InfrastructureCause.NO_FACTORY, f"this host has no {self.absent} machine factory"
        )


def host_backends(host: MachineHost) -> tuple[Backend, ...]:
    """The backends ``spec.lower`` can choose on ``host``: ShellSim and the host's container backend."""
    return (Backend.SHELLSIM, container_backend(host))


def host_checked_factories(factories: Mapping[str, MachineFactory], host: MachineHost) -> dict[str, MachineFactory]:
    """``factories`` with host errors classified, and an ``AbsentFactory`` for every backend of
    ``host_backends(host)`` it lacks. Keys are ``Backend`` values, as ``machine_factories`` returns them.
    """
    checked: dict[str, MachineFactory] = {key: HostCheckedFactory(factory) for key, factory in factories.items()}
    for backend in host_backends(host):
        checked.setdefault(backend.value, AbsentFactory(backend))
    return checked
