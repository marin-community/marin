"""Shellbox backends shared by task workers and deployment preflight checks."""

import asyncio
import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from daytona import AsyncDaytona, DaytonaConfig
from omegaconf import DictConfig
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import SYSTEM_DIRECTORIES, LocalMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, ExecutionRequirements, Machine, MachineFactory, MachineSpec, Result

HOST_PYTHON_ENVIRONMENT = "host"
UV_ARCHIVE_DIRECTORY = "archive-v0"


def task_machine_factories(task_sessions: DictConfig) -> dict[str, MachineFactory]:
    """Build the configured task backends, probing host capabilities immediately."""
    factories: dict[str, MachineFactory] = {"shellsim": ShellSimMachineFactory()}
    local = task_sessions.get("local")
    if local is not None:
        factories[Backend.LOCAL] = local_machine_factory(local)
    daytona = task_sessions.get("daytona")
    if daytona is not None:
        factories[Backend.DAYTONA] = DaytonaMachineFactory(
            client_factory=partial(AsyncDaytona, DaytonaConfig(api_key=os.environ[daytona.api_key_env])),
            ttl_minutes=int(daytona.ttl_minutes),
        )
    if shutil.which("docker") is not None:
        factories[Backend.DOCKER] = DockerMachineFactory()
    return factories


def host_python_mounts() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """The read-only directories and the ``bin`` directory of the Python environment that runs this process.

    The directories are the environment, the source directories of its editable installs, the uv archive that
    its files link to (the task bootstrap runs ``uv sync --link-mode symlink``), and the base interpreter. The editable sources of SkyRL's own environment are Marin's ``lib`` packages outside the
    environment, for example ``verifyit``, which the script graders import. Every sandbox sees the system
    directories already, and mounts inside them fail; a directory inside another mount is not mounted again.
    """
    environment = Path(sys.prefix)
    candidates = (environment, *editable_sources(environment), *linked_archives(environment), Path(sys.base_prefix))
    usable = [
        path
        for path in dict.fromkeys(candidates)
        if path.is_dir() and not any(path.is_relative_to(system) for system in SYSTEM_DIRECTORIES)
    ]
    mounts = tuple(path for path in usable if not any(path != other and path.is_relative_to(other) for other in usable))
    return mounts, (environment / "bin",)


def linked_archives(environment: Path) -> tuple[Path, ...]:
    """The uv ``archive-v0`` directories that the files of ``environment`` link to, so that the links resolve in a
    sandbox. uv links every file of a package, so the first link of each site-packages directory is enough."""
    archives = []
    for site_packages in environment.glob("lib/python*/site-packages"):
        link = next((path for path in site_packages.rglob("*") if path.is_symlink()), None)
        if link is None:
            continue
        archive = next((parent for parent in link.resolve().parents if parent.name == UV_ARCHIVE_DIRECTORY), None)
        if archive is not None:
            archives.append(archive)
    return tuple(archives)


def editable_sources(environment: Path) -> tuple[Path, ...]:
    """The absolute directories that the ``.pth`` files of ``environment`` put on ``sys.path``: its editable installs,
    and the site-packages of a layered environment (``uv run --with``) with the ``.pth`` files of their own."""
    directories: list[Path] = []
    pending = sorted(environment.glob("lib/python*/site-packages"))
    seen: set[Path] = set()
    while pending:
        site_packages = pending.pop()
        if site_packages in seen:
            continue
        seen.add(site_packages)
        for pth in sorted(site_packages.glob("*.pth")):
            for line in pth.read_text().splitlines():
                path = Path(line.strip())
                if not path.is_absolute() or not path.is_dir():
                    continue
                directories.append(path)
                if path.name == "site-packages":
                    pending.append(path)
    return tuple(directories)


def local_machine_factory(settings: Mapping[str, Any]) -> MachineFactory:
    """A bubblewrap factory from ``environment.task_sessions.local``."""
    unknown = set(settings) - {"read_only", "bin_dirs", "python_environment", "max_machines"}
    if unknown:
        raise ValueError(f"Unknown environment.task_sessions.local keys: {', '.join(sorted(unknown))}")
    read_only = tuple(Path(path) for path in settings.get("read_only") or ())
    bin_dirs = tuple(Path(path) for path in settings.get("bin_dirs") or ())
    python_environment = settings.get("python_environment")
    if python_environment == HOST_PYTHON_ENVIRONMENT:
        mounts, environment_bin = host_python_mounts()
        read_only, bin_dirs = (*read_only, *mounts), (*bin_dirs, *environment_bin)
    elif python_environment is not None:
        raise ValueError(f"environment.task_sessions.local.python_environment must be {HOST_PYTHON_ENVIRONMENT!r}")
    factory = LocalMachineFactory(read_only=tuple(dict.fromkeys(read_only)), bin_dirs=tuple(dict.fromkeys(bin_dirs)))
    max_machines = settings.get("max_machines")
    return factory if max_machines is None else BoundedMachineFactory(factory, asyncio.Semaphore(int(max_machines)))


@dataclass
class BoundedMachineFactory:
    """A machine factory with at most as many open machines as ``slots`` permits.

    A script grader that measures time can fail a correct answer while it waits for a CPU.
    """

    factory: MachineFactory
    slots: asyncio.Semaphore

    @property
    def backend(self) -> Backend:
        return self.factory.backend

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        await self.slots.acquire()
        try:
            return _ReleasingMachine(await self.factory.create(spec), self.slots)
        except BaseException:
            self.slots.release()
            raise


@dataclass
class _ReleasingMachine:
    machine: Machine
    slots: asyncio.Semaphore

    async def run(self, command: Command) -> Result:
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        try:
            await self.machine.close()
        finally:
            self.slots.release()
