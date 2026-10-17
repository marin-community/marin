# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, resource installation, and bounded cleanup."""

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    ExitReason,
    HostImage,
    Machine,
    MachineFactory,
    MachineSpec,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
)
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    TaskResource,
    TaskSpec,
    require_resolved_environment,
)
from taskcompendium.runtime.environment import prepare_machine_spec
from taskcompendium.runtime.resources import resource_bytes

from rolloutengine.cleanup import _Cleanup, _retain_task
from rolloutengine.spec import MachineRuntimeSpec

SETUP_OUTPUT_CHARS = 2000
"""How much of a failed setup command's stdout and stderr the error keeps, from the end."""


@dataclass(frozen=True)
class _UserMachine:
    machine: Machine
    user: str

    async def run(self, command: Command):
        return await self.machine.run(replace(command, user=self.user) if command.user is None else command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


async def _install_resources(machine: Machine, resources: tuple[TaskResource, ...]) -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, resource in enumerate(resources):
            source = Path(directory) / str(index)
            source.write_bytes(resource_bytes(resource))
            source.chmod(0o644 if resource.mode is None else int(resource.mode, 8))
            if resource.mtime_ns is not None:
                os.utime(source, ns=(resource.mtime_ns, resource.mtime_ns))
            await machine.upload(source, f"/{resource.path}")


def _machine_spec(requirements: EnvironmentRequirements, runtime: MachineRuntimeSpec) -> MachineSpec:
    require_resolved_environment(requirements)
    if requirements.docker_image is not None:
        source = RegistryImage(requirements.docker_image)
    elif requirements.packages_lock is not None:
        source = HostImage()
    elif requirements.package_spec is not None:
        source = requirements.package_spec
    elif requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR:
        source = PackageSpec()
    else:
        raise UnsupportedMachineSpec("Machine execution requires explicit semantics and native dependencies")
    return MachineSpec(
        source=source,
        workdir=(
            requirements.working_directory
            if requirements.working_directory is not None
            else "" if requirements.docker_image else "/workspace"
        ),
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
    return machine if runtime.user is None else _UserMachine(machine, runtime.user)


def _output_tail(output: bytes) -> str:
    """The last ``SETUP_OUTPUT_CHARS`` characters of a command's output, for an error message."""
    text = output.decode(errors="replace").strip()
    if len(text) <= SETUP_OUTPUT_CHARS:
        return repr(text)
    return repr(f"...{text[-SETUP_OUTPUT_CHARS:]}")


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
        prepared = await asyncio.to_thread(
            prepare_machine_spec,
            requirements,
            factories[runtime.backend],
            _machine_spec(requirements, runtime),
            dict(os.environ),
        )
        machine = await _acquire_machine(runtime, prepared, factories, cleanup, owned)
        await _install_resources(machine, resources)
        for command in requirements.setup_commands:
            result = await machine.run(Command(("sh", "-c", command), timeout=runtime.startup_timeout, user="0"))
            if result.reason == ExitReason.TIMED_OUT:
                raise TimeoutError("Environment setup command timed out")
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Environment setup command failed: {result.reason}, exit={result.exit_code}; "
                    f"command: {command!r}; stdout: {_output_tail(result.stdout)}; stderr: {_output_tail(result.stderr)}"
                )
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
class _AttemptMachineFactory:
    """Create grading machines under the attempt's startup deadline, user, and cleanup."""

    runtime: MachineRuntimeSpec
    factories: Mapping[str, MachineFactory]
    cleanup: _Cleanup
    owned: AsyncExitStack

    @property
    def backend(self) -> Backend:
        return self.factories[self.runtime.backend].backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factories[self.runtime.backend].check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        async with asyncio.timeout(self.runtime.startup_timeout):
            machine = await _acquire_machine(self.runtime, spec, self.factories, self.cleanup, self.owned)
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
    cleanup = _Cleanup(cleanup_timeout)
    resources = task.resources.all + task.resources.worker
    try:
        async with AsyncExitStack() as owned:
            machine = await _prepare_machine(
                task.environment_requirements, runtime, resources, {runtime.backend: factory}, cleanup, owned
            )
            assert machine is not None
            yield machine
    except BaseException as error:
        if cleanup.errors:
            error.add_note(f"Machine cleanup failed: {[failure.operation for failure in cleanup.errors]}")
        raise
    if cleanup.errors:
        raise RuntimeError(f"Machine cleanup failed: {[error.operation for error in cleanup.errors]}")
