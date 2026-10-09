# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, resource installation, and bounded cleanup."""

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import (
    Backend,
    Command,
    ExitReason,
    HostImage,
    Machine,
    MachineFactory,
    MachineSpec,
    Result,
    ShellSimBuiltins,
)
from taskcompendium.models import EnvironmentRequirements, TaskResource
from taskcompendium.runtime.environment import resolve_env_vars
from taskcompendium.runtime.local import local_factory, local_runtime
from taskcompendium.runtime.resources import resource_bytes

from rolloutengine.cleanup import _Cleanup, _retain_task
from rolloutengine.spec import MachineRuntimeSpec


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
    if requirements.docker_image is not None:
        source = RegistryImage(requirements.docker_image)
    elif requirements.packages_lock is not None:
        source = HostImage()
    else:
        source = ShellSimBuiltins()
    return MachineSpec(
        source=source,
        workdir=(
            requirements.working_directory
            if requirements.working_directory is not None
            else "" if requirements.docker_image else "/workspace"
        ),
        env=resolve_env_vars(requirements.environment_variables, os.environ),
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
    requirements: EnvironmentRequirements,
) -> Machine:
    """Create a machine closed by ``owned``, retaining ownership if creation outlives cancellation."""
    machine_cleanup = cleanup
    if runtime.cleanup_timeout is not None:
        machine_cleanup = _Cleanup(runtime.cleanup_timeout, cleanup.errors)
    factory = factories[runtime.backend]
    if requirements.packages_lock is not None and requirements.docker_image is None:
        assert isinstance(factory, LocalMachineFactory)
        environment = await asyncio.to_thread(local_runtime, requirements.packages_lock)
        await asyncio.to_thread(environment.ensure_built)
        factory = await asyncio.to_thread(local_factory, environment, factory)
        spec = replace(spec, env={**environment.variables, **spec.env})
    creation = asyncio.create_task(factory.create(spec))
    try:
        machine = await asyncio.shield(creation)
    except asyncio.CancelledError:
        _retain_task(asyncio.create_task(_discard_machine(creation, machine_cleanup)))
        raise
    owned.push_async_callback(machine_cleanup.run, "machine_close", machine.close)
    return machine if runtime.user is None else _UserMachine(machine, runtime.user)


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
        machine = await _acquire_machine(
            runtime, _machine_spec(requirements, runtime), factories, cleanup, owned, requirements
        )
        await _install_resources(machine, resources)
        for command in requirements.setup_commands:
            result = await machine.run(Command(("sh", "-c", command), timeout=runtime.startup_timeout, user="0"))
            if result.reason == ExitReason.TIMED_OUT:
                raise TimeoutError("Environment setup command timed out")
            if result.exit_code != 0:
                raise RuntimeError(f"Environment setup command failed: {result.reason}, exit={result.exit_code}")
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
    environment: EnvironmentRequirements

    @property
    def backend(self) -> Backend:
        return self.factories[self.runtime.backend].backend

    async def create(self, spec: MachineSpec) -> Machine:
        async with asyncio.timeout(self.runtime.startup_timeout):
            machine = await _acquire_machine(
                self.runtime, spec, self.factories, self.cleanup, self.owned, self.environment
            )
        return _OwnedMachine(machine)
