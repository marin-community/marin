# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, resource installation, and bounded cleanup."""

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from rigging.task_errors import RetryDisposition, setup_exit_is_task_failure
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    ExitReason,
    HostImage,
    Machine,
    MachineFactory,
    MachineSpec,
    PackageSpec,
    ProcessMachine,
    ProcessSession,
    Result,
    TaskRequirements,
    UnsupportedMachineSpec,
)
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskResource, TaskSpec
from taskcompendium.runtime.environment import prepare_machine_spec
from taskcompendium.runtime.resources import resource_bytes, write_build_context

from rolloutengine.cleanup import _Cleanup, _retain_task
from rolloutengine.models import MachineRuntimeSpec

SETUP_OUTPUT_CHARS = 2000
"""How much of a failed setup command's stdout and stderr the error keeps, from the end."""


@dataclass(frozen=True)
class _UserMachine:
    machine: Machine
    user: str

    def _command(self, command: Command) -> Command:
        return replace(command, user=self.user) if command.user is None else command

    async def run(self, command: Command) -> Result:
        return await self.machine.run(self._command(command))

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass(frozen=True)
class _UserProcessMachine(_UserMachine):
    machine: ProcessMachine

    async def open_process(self, command: Command) -> ProcessSession:
        return await self.machine.open_process(self._command(command))


def _verifier_view(machine: Machine) -> Machine:
    """Remove the agent's user override for the trusted verifier in a shared machine."""
    return machine.machine if isinstance(machine, _UserMachine) else machine


async def _install_resources(machine: Machine, resources: tuple[TaskResource, ...]) -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, resource in enumerate(resources):
            source = Path(directory) / str(index)
            source.write_bytes(resource_bytes(resource))
            source.chmod(0o644 if resource.mode is None else int(resource.mode, 8))
            if resource.mtime_ns is not None:
                os.utime(source, ns=(resource.mtime_ns, resource.mtime_ns))
            await machine.upload(source, f"/{resource.path}")


@contextmanager
def machine_spec(requirements: EnvironmentRequirements, runtime: MachineRuntimeSpec) -> Iterator[MachineSpec]:
    """The machine spec for ``requirements`` on the selected runtime.

    A Docker build context becomes a ``DockerfileSource`` in a temporary directory that exists until the
    context exits: the backend checks it there and builds the image when it creates the machine. An
    image machine, pulled or built, starts in the image's working directory unless the requirements
    name one.
    """
    with ExitStack() as stack:
        if requirements.docker_image is not None:
            source = RegistryImage(requirements.docker_image)
        elif requirements.docker_build is not None:
            context = Path(stack.enter_context(TemporaryDirectory(prefix="rollout-build-")))
            source = DockerfileSource(context, write_build_context(requirements.docker_build, context))
        elif requirements.packages_lock is not None:
            source = HostImage()
        elif requirements.package_spec is not None:
            source = requirements.package_spec
        elif requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR:
            source = PackageSpec()
        else:
            raise UnsupportedMachineSpec("Machine execution requires explicit semantics and native dependencies")
        workdir = requirements.working_directory
        if workdir is None:
            workdir = "" if isinstance(source, RegistryImage | DockerfileSource) else "/workspace"
        yield MachineSpec(
            source=source,
            workdir=workdir,
            network=runtime.network,
            memory_mb=runtime.memory_mb,
            cpus=runtime.cpus,
            storage_mb=runtime.storage_mb,
            gpus=runtime.gpus,
            startup_timeout=runtime.startup_timeout,
        )


async def _discard_machine(creation: asyncio.Task[Machine], cleanup: _Cleanup) -> None:
    machine = await creation
    await _Cleanup(cleanup.timeout).run("late_machine_close", machine.close)


async def _acquire_machine(
    runtime: MachineRuntimeSpec,
    spec: MachineSpec,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    owned: AsyncExitStack,
) -> Machine:
    """Create a machine closed by ``owned``, retaining ownership if creation outlives cancellation."""
    machine_cleanup = cleanup
    if runtime.cleanup_timeout is not None:
        machine_cleanup = _Cleanup(runtime.cleanup_timeout, cleanup.errors)
    factory = factories[runtime.backend]
    creation = asyncio.create_task(factory.create(spec))
    try:
        machine = await asyncio.shield(creation)
    except asyncio.CancelledError:
        _retain_task(asyncio.create_task(_discard_machine(creation, machine_cleanup)))
        raise
    owned.push_async_callback(machine_cleanup.run, "machine_close", machine.close)
    if runtime.user is None:
        return machine
    if isinstance(machine, ProcessMachine):
        return _UserProcessMachine(machine, runtime.user)
    return _UserMachine(machine, runtime.user)


def _output_tail(output: bytes) -> str:
    """The last ``SETUP_OUTPUT_CHARS`` characters of a command's output, for an error message."""
    text = output.decode(errors="replace").strip()
    if len(text) <= SETUP_OUTPUT_CHARS:
        return text
    return f"...{text[-SETUP_OUTPUT_CHARS:]}"


class MachineSetupError(RuntimeError):
    """An authored setup command returned without completing successfully."""

    def __init__(self, command: str, result: Result):
        self.command = command
        self.reason = result.reason
        self.exit_code = result.exit_code
        self.stdout = _output_tail(result.stdout)
        self.stderr = _output_tail(result.stderr)
        super().__init__(
            f"Environment setup command failed: {self.reason}, exit={self.exit_code}; "
            f"command: {self.command!r}; stdout: {self.stdout!r}; stderr: {self.stderr!r}"
        )

    def retry_disposition(self) -> RetryDisposition:
        """Classify the setup exit at its source: a signal exit or lost shell can
        pass on a fresh machine; a normal non-zero exit is the task's own failure."""
        if self.reason is ExitReason.TIMED_OUT or not setup_exit_is_task_failure(self.exit_code or 0):
            return RetryDisposition.TRANSIENT
        return RetryDisposition.PERMANENT


class MachineSetupTimeout(MachineSetupError, TimeoutError):
    """An authored setup command reported a timeout."""


async def _prepare_machine(
    requirements: EnvironmentRequirements,
    runtime: MachineRuntimeSpec | None,
    resources: tuple[TaskResource, ...],
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    owned: AsyncExitStack,
) -> Machine | None:
    """Prepare a task machine with its resources and setup commands."""
    if runtime is None:
        return None
    async with asyncio.timeout(runtime.startup_timeout):
        # A Docker build context exists only while the machine is created; the built machine no longer reads it.
        with machine_spec(requirements, runtime) as spec:
            prepared = await asyncio.to_thread(
                prepare_machine_spec, requirements, factories[runtime.backend], spec, dict(os.environ)
            )
            machine = await _acquire_machine(runtime, prepared, factories, cleanup, owned)
        await _install_resources(machine, resources)
        for command in requirements.setup_commands:
            result = await machine.run(Command(("sh", "-c", command), timeout=runtime.startup_timeout, user="0"))
            if result.reason == ExitReason.TIMED_OUT:
                raise MachineSetupTimeout(command, result)
            if result.exit_code != 0:
                raise MachineSetupError(command, result)
        if runtime.user is not None:
            result = await machine.run(Command(("true",), timeout=runtime.startup_timeout))
            if result.reason == ExitReason.TIMED_OUT:
                raise TimeoutError("Execution user startup probe timed out")
            if result.exit_code != 0:
                raise RuntimeError(f"Execution user startup probe failed: {result.reason}, exit={result.exit_code}")
    return machine


@dataclass(frozen=True)
class _OwnedMachine:
    """A machine the attempt closes; closing it early would bypass bounded cleanup."""

    machine: Machine

    async def run(self, command: Command) -> Result:
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        pass


@dataclass(frozen=True)
class _OwnedProcessMachine(_OwnedMachine):
    machine: ProcessMachine

    async def open_process(self, command: Command) -> ProcessSession:
        return await self.machine.open_process(command)


@dataclass(frozen=True)
class _AttemptMachineFactory:
    """Create grading machines under the attempt's startup deadline, user, and cleanup."""

    runtime: MachineRuntimeSpec
    factories: Mapping[str, MachineFactory]
    cleanup: _Cleanup
    owned: AsyncExitStack
    verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None

    @property
    def backend(self) -> Backend:
        return self.factories[self.runtime.backend].backend

    def check_environment(self, requirements: TaskRequirements, spec: MachineSpec) -> None:
        self.factories[self.runtime.backend].check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        async with asyncio.timeout(self.runtime.startup_timeout):
            if self.verifier is not None:
                machine = await self.verifier(spec)
            else:
                machine = await _acquire_machine(self.runtime, spec, self.factories, self.cleanup, self.owned)
        if isinstance(machine, ProcessMachine):
            return _OwnedProcessMachine(machine)
        return _OwnedMachine(machine)


@asynccontextmanager
async def task_machine(
    task: TaskSpec, factory: MachineFactory, runtime: MachineRuntimeSpec, *, cleanup_timeout: float
) -> AsyncIterator[Machine]:
    """Start a machine for ``task``: install its resources and run its setup commands. Close it on exit.

    The machine is for a harness that runs without the engine. ``runtime.cleanup_timeout``, when it is set,
    replaces ``cleanup_timeout`` for the machine's close. A close that fails or misses its deadline raises
    ``RuntimeError`` after the body returns. If the body raises, its exception propagates with a note that names
    the failed closes.
    """
    async with prepare_machine(
        task.environment_requirements,
        task.resources.all + task.resources.worker,
        factory,
        runtime,
        cleanup_timeout=cleanup_timeout,
    ) as machine:
        yield machine


@asynccontextmanager
async def prepare_machine(
    requirements: EnvironmentRequirements,
    resources: tuple[TaskResource, ...],
    factory: MachineFactory,
    runtime: MachineRuntimeSpec,
    *,
    cleanup_timeout: float,
) -> AsyncIterator[Machine]:
    """Install resources and run setup on a machine, closing it when the context exits.

    Cleanup failures propagate after a successful body, or are attached to its exception as a note.
    ``runtime.cleanup_timeout`` overrides the supplied cleanup deadline when set.
    """
    cleanup = _Cleanup(cleanup_timeout)
    try:
        async with AsyncExitStack() as owned:
            machine = await _prepare_machine(
                requirements, runtime, resources, {runtime.backend: factory}, cleanup, owned
            )
            assert machine is not None
            yield machine
    except BaseException as error:
        if cleanup.errors:
            error.add_note(f"Machine cleanup failed: {[failure.operation for failure in cleanup.errors]}")
        raise
    if cleanup.errors:
        raise RuntimeError(f"Machine cleanup failed: {[error.operation for error in cleanup.errors]}")
