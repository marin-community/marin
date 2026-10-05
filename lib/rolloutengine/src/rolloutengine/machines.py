# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Machine creation, setup, and cleanup for task execution."""

import asyncio
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

from rolloutengine.contracts import TaskSetupError, TaskSetupTimeout


async def _install_files(machine: Machine, files: tuple[EnvironmentFile, ...]) -> None:
    with TemporaryDirectory(prefix="rollout-files-") as directory:
        for index, file in enumerate(files):
            source = Path(directory) / str(index)
            source.write_bytes(file.content)
            source.chmod(file.mode)
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


@asynccontextmanager
async def _task_machine(environment: EnvironmentSpec, factories: Mapping[EnvironmentKind, MachineFactory]):
    """Yield a prepared machine, or none for a null environment, and release it after use."""
    if environment.kind == EnvironmentKind.NULL:
        yield None
        return
    async with AsyncExitStack() as resources:
        if environment.kind == EnvironmentKind.SHELLSIM:
            source = ShellSimBuiltins()
        elif isinstance(environment.image, RegistryImage):
            source = ShellboxRegistryImage(environment.image.reference)
        else:
            assert isinstance(environment.image, DockerBuild)
            directory = Path(resources.enter_context(TemporaryDirectory(prefix="rollout-build-")))
            for file in environment.image.files:
                path = directory / file.path.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file.content)
                path.chmod(file.mode)
            source = DockerfileSource(directory, directory / environment.image.dockerfile.lstrip("/"))
        async with asyncio.timeout(environment.startup_timeout):
            machine = await factories[environment.kind].create(
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
            resources.push_async_callback(_close_machine, machine)
            await _install_files(machine, environment.files)
            await _run_setup_commands(machine, environment.setup, "Environment setup command", None)
            if environment.healthcheck is not None:
                await _wait_for_healthcheck(machine, environment.healthcheck, None)
        yield machine


async def _close_machine(machine: Machine) -> None:
    # A total-attempt deadline can expire during cleanup after a startup timeout.
    cleanup = asyncio.create_task(machine.close())
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise
