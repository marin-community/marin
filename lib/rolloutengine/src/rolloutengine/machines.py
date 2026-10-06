# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, setup, and cleanup for task execution."""

import asyncio
import os
from collections.abc import Iterable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from harbor_config.env import resolve_env_vars
from shellbox.image import DockerfileSource
from shellbox.image import RegistryImage as ShellboxRegistryImage
from shellbox.machine import Command, ExitReason, Machine, MachineFactory, MachineSpec, NetworkPolicy, ShellSimBuiltins
from taskcompendium.environment import (
    DockerBuild,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    HealthcheckSpec,
    RegistryImage,
)

from rolloutengine.cleanup import _Cleanup, _retain_task
from rolloutengine.contracts import TaskSetupError, TaskSetupTimeout


async def _install_files(machine: Machine, files: tuple[EnvironmentFile, ...]) -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, file in enumerate(files):
            source = Path(directory) / str(index)
            source.write_bytes(file.content)
            source.chmod(file.mode)
            if file.mtime_ns is not None:
                os.utime(source, ns=(file.mtime_ns, file.mtime_ns))
            await machine.upload(source, file.path)


def _machine_command(command: EnvironmentCommand) -> Command:
    return Command(
        argv=command.argv,
        cwd=command.cwd,
        env=resolve_env_vars(command.env),
        timeout=command.timeout,
        user=command.user,
    )


async def _run_setup_commands(
    machine: Machine,
    commands: Iterable[EnvironmentCommand],
    failure_prefix: str,
    stage: str | None,
) -> None:
    for command in commands:
        result = await machine.run(_machine_command(command))
        if result.reason == ExitReason.TIMED_OUT:
            raise TaskSetupTimeout(f"{failure_prefix} timed out", stage=stage, command=command.argv)
        if result.exit_code != 0:
            raise TaskSetupError(
                f"{failure_prefix} failed: {result.reason}, exit={result.exit_code}",
                stage=stage,
                command=command.argv,
                exit_code=result.exit_code,
            )


async def _wait_for_healthcheck(machine: Machine, healthcheck: HealthcheckSpec, stage: str | None) -> None:
    loop = asyncio.get_running_loop()
    grace_end = loop.time() + healthcheck.start_period
    failures = 0
    while True:
        in_grace = loop.time() < grace_end
        result = await machine.run(_machine_command(healthcheck.command))
        if result.reason == ExitReason.TIMED_OUT:
            raise TaskSetupTimeout("Environment healthcheck timed out", stage=stage, command=healthcheck.command.argv)
        if result.exit_code == 0:
            return
        if not in_grace:
            failures += 1
            if failures >= healthcheck.retries:
                raise TaskSetupError(
                    f"Environment healthcheck failed after {failures} attempts",
                    stage=stage,
                    command=healthcheck.command.argv,
                    exit_code=result.exit_code,
                )
        await asyncio.sleep(healthcheck.start_interval if in_grace else healthcheck.interval)


async def _create_machine(environment: EnvironmentSpec, factories: Mapping[EnvironmentKind, MachineFactory]) -> Machine:
    async with AsyncExitStack() as resources:
        if environment.kind == EnvironmentKind.SHELLSIM:
            source = ShellSimBuiltins()
        elif isinstance(environment.image, RegistryImage):
            source = ShellboxRegistryImage(environment.image.reference)
        else:
            assert isinstance(environment.image, DockerBuild)
            directory = Path(resources.enter_context(TemporaryDirectory(prefix="rollout-build-")))
            for file in environment.image.files:
                path = directory / file.path.removeprefix("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.content)
                path.chmod(file.mode)
            source = DockerfileSource(directory, directory / environment.image.dockerfile.removeprefix("/"))
        return await factories[environment.kind].create(
            MachineSpec(
                source=source,
                workdir=environment.workdir,
                env=resolve_env_vars(environment.env),
                network=NetworkPolicy.ALLOW if environment.network else NetworkPolicy.DENY,
                memory_mb=environment.memory_mb,
                cpus=environment.cpus,
                storage_mb=environment.storage_mb,
                gpus=environment.gpus,
                startup_timeout=environment.startup_timeout,
            )
        )


async def _discard_machine(creation: asyncio.Task[Machine], cleanup_timeout: float) -> None:
    machine = await creation
    cleanup = _Cleanup(cleanup_timeout)
    await cleanup.run("late_machine_close", machine.close)


@asynccontextmanager
async def _task_machine(
    environment: EnvironmentSpec, factories: Mapping[EnvironmentKind, MachineFactory], cleanup: _Cleanup
):
    """Prepare a machine. Close a machine created after cancellation in the background."""
    if environment.kind == EnvironmentKind.NULL:
        yield None
        return
    async with AsyncExitStack() as resources:
        async with asyncio.timeout(environment.startup_timeout):
            creation = asyncio.create_task(_create_machine(environment, factories))
            try:
                machine = await asyncio.shield(creation)
            except asyncio.CancelledError:
                _retain_task(asyncio.create_task(_discard_machine(creation, cleanup.timeout)))
                raise
            resources.push_async_callback(cleanup.run, "machine_close", machine.close)
            await _install_files(machine, environment.files)
            await _run_setup_commands(machine, environment.setup, "Environment setup command", None)
            if environment.healthcheck is not None:
                await _wait_for_healthcheck(machine, environment.healthcheck, None)
        yield machine
