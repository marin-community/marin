# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, public resource installation, and bounded cleanup."""

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor_config.env import resolve_env_vars
from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, Machine, MachineFactory, MachineSpec, ShellSimBuiltins
from taskcompendium.models import EnvironmentRequirements, TaskResource
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

    async def download(self, source: str, target: Path, *, max_bytes: int | None = None) -> None:
        await self.machine.download(source, target, max_bytes=max_bytes)

    async def close(self) -> None:
        await self.machine.close()


async def _install_resources(machine: Machine, resources: tuple[TaskResource, ...], *, root: str = "/") -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, resource in enumerate(resources):
            source = Path(directory) / str(index)
            source.write_bytes(resource_bytes(resource))
            source.chmod(0o644 if resource.mode is None else int(resource.mode, 8))
            if resource.mtime_ns is not None:
                os.utime(source, ns=(resource.mtime_ns, resource.mtime_ns))
            await machine.upload(source, f"{root.rstrip('/')}/{resource.path}")


def _machine_spec(requirements: EnvironmentRequirements, runtime: MachineRuntimeSpec) -> MachineSpec:
    return MachineSpec(
        source=RegistryImage(requirements.docker_image) if requirements.docker_image else ShellSimBuiltins(),
        workdir=(
            requirements.working_directory
            if requirements.working_directory is not None
            else "" if requirements.docker_image else "/workspace"
        ),
        env=resolve_env_vars(requirements.environment_variables),
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


async def _prepare_machine(
    requirements: EnvironmentRequirements,
    runtime: MachineRuntimeSpec | None,
    resources: tuple[TaskResource, ...],
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    owned: AsyncExitStack,
) -> Machine | None:
    """Prepare a machine and retain ownership if creation outlives cancellation."""
    if runtime is None:
        return None
    machine_cleanup = cleanup
    if runtime.cleanup_timeout is not None:
        machine_cleanup = _Cleanup(runtime.cleanup_timeout, cleanup.errors)
    async with asyncio.timeout(runtime.startup_timeout):
        creation = asyncio.create_task(factories[runtime.backend].create(_machine_spec(requirements, runtime)))
        try:
            machine = await asyncio.shield(creation)
        except asyncio.CancelledError:
            _retain_task(asyncio.create_task(_discard_machine(creation, machine_cleanup)))
            raise
        owned.push_async_callback(machine_cleanup.run, "machine_close", machine.close)
        if runtime.user is not None:
            machine = _UserMachine(machine, runtime.user)
        await _install_resources(machine, resources)
        for command in requirements.setup_commands:
            result = await machine.run(Command(("sh", "-c", command), timeout=runtime.startup_timeout, user="0"))
            if result.reason == ExitReason.TIMED_OUT:
                raise TimeoutError("Environment setup command timed out")
            if result.exit_code != 0:
                raise RuntimeError(f"Environment setup command failed: {result.reason}, exit={result.exit_code}")
        if runtime.user is not None:
            result = await machine.run(Command(("true",), timeout=runtime.startup_timeout))
            if result.reason == ExitReason.TIMED_OUT:
                raise TimeoutError("Execution user startup probe timed out")
            if result.exit_code != 0:
                raise RuntimeError(f"Execution user startup probe failed: {result.reason}, exit={result.exit_code}")
    return machine
