# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initialize workdirs, task aliases, and requested tools inside a local guest."""

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shellbox.backends.local import sandbox
from shellbox.backends.local.runtime import LocalMachine
from shellbox.machine import (
    Command,
    HostImage,
    Mount,
    UnsupportedMachineSpec,
)


async def start_host_machine(
    machine: LocalMachine, overlaid_workdir: PurePosixPath | None, tools: tuple[str, ...]
) -> None:
    """Make a workdir that lies in the machine's layer, then check that each requested tool is executable."""
    if overlaid_workdir is not None:
        made = await machine._control('exec mkdir -p -- "$1"', overlaid_workdir)
        if made.exit_code != 0:
            raise RuntimeError(f"Cannot create workdir {overlaid_workdir}: {made.stderr.decode(errors='replace')}")
    if not tools:
        return
    result = await machine.run(
        Command(
            (
                "/bin/sh",
                "-c",
                'for tool do path=$(command -v "$tool") && test -x "$path" || exit; done',
                "sh",
                *tools,
            )
        )
    )
    if result.exit_code != 0:
        raise UnsupportedMachineSpec("Requested tools must be executable in the sandbox PATH")


async def start_image_machine(machine: LocalMachine, aliases: str | None) -> None:
    if aliases is not None:
        hosts = await machine._control("mkdir -p /etc && cat > /etc/hosts", PurePosixPath("/"), stdin=aliases.encode())
        if hosts.exit_code != 0:
            raise RuntimeError(f"Cannot prepare task aliases: {hosts.stderr.decode(errors='replace')}")
    result = await machine._control('mkdir -p -- "$1"', sandbox._absolute_path(machine.spec.workdir))
    if result.exit_code != 0:
        raise RuntimeError(f"Cannot create image workdir: {result.stderr.decode(errors='replace')}")


def host_mounts(host: HostImage, read_only: tuple[Path, ...], system_mounts: tuple[Mount, ...]) -> tuple[Mount, ...]:
    # A read-only directory is a mount at its own path; the factory's and the image's may name the same one.
    directories = dict.fromkeys((*read_only, *(directory.absolute() for directory in host.read_only)))
    mounts = tuple(
        dict.fromkeys(
            (
                *system_mounts,
                *(Mount(directory, PurePosixPath(directory)) for directory in directories),
                *(Mount(mount.host.absolute(), mount.guest) for mount in host.mounts),
            )
        )
    )
    sandbox._check_mounts(mounts)
    return mounts


def host_environment(
    host: HostImage, bin_dirs: tuple[Path, ...], system_bin_dirs: tuple[Path, ...], hash_seed: str | None
) -> dict[str, str]:
    environment = {
        "PATH": os.pathsep.join(
            [
                *(str(directory.absolute()) for directory in host.bin_dirs),
                *map(str, bin_dirs),
                sandbox.DEFAULT_PATH,
                *map(str, system_bin_dirs),
            ]
        ),
        "HOME": str(sandbox.HOME),
        "LANG": "C.UTF-8",
    }
    if hash_seed is not None:
        environment["PYTHONHASHSEED"] = hash_seed
    return environment


@dataclass(frozen=True)
class HostLayout:
    mounts: tuple[Mount, ...]
    workdir: PurePosixPath
    overlaid_workdir: PurePosixPath | None


def host_layout(
    host: HostImage,
    read_only: tuple[Path, ...],
    system_mounts: tuple[Mount, ...],
    workdir: str,
    system_directories: sandbox.SystemDirectories,
) -> HostLayout:
    mounts = host_mounts(host, read_only, system_mounts)
    path = sandbox._absolute_path(workdir or "/")
    overlaid = (
        path
        if system_directories is sandbox.SystemDirectories.COPY_ON_WRITE
        and sandbox._within(path, sandbox.OVERLAID_DIRECTORIES)
        else None
    )
    if (sandbox._within(path, sandbox.SYSTEM_DIRECTORIES) and overlaid is None) or any(
        path.is_relative_to(mount.guest) for mount in mounts
    ):
        raise UnsupportedMachineSpec(f"The local backend's workdir {workdir} is a read-only host path")
    return HostLayout(mounts, path, overlaid)
