# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe bubblewrap startup and command entry on the current host."""

import asyncio
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from shellbox.backends.local import sandbox
from shellbox.backends.local.runtime import LocalMachine
from shellbox.machine import (
    HostImage,
    MachineSpec,
    Mount,
)


def _sandbox_error(
    bwrap: Path, system_directories: sandbox.SystemDirectories, mounts: tuple[Mount, ...], environment: dict[str, str]
) -> str | None:
    """Return why namespace creation or command entry fails, or None when both work."""

    async def probe() -> str | None:
        root, layers = sandbox._machine_directories(None, system_directories=system_directories)
        machine = LocalMachine(
            MachineSpec(HostImage(), workdir="/"),
            bwrap=bwrap,
            root=root,
            layers=layers,
            mounts=mounts,
            environment=environment,
        )
        try:
            try:
                async with asyncio.timeout(sandbox.PROBE_TIMEOUT):
                    await machine._start_keeper()
                    result = await machine._control("exit 0", PurePosixPath("/"))
            finally:
                await machine.close()
        except (OSError, RuntimeError, TimeoutError) as error:
            if machine._keeper is not None:
                assert machine._keeper.stderr is not None
                stderr = (await machine._keeper.stderr.read()).decode(errors="replace").strip()
                if stderr:
                    return stderr
            return str(error) or type(error).__name__
        if result.exit_code == 0:
            return None
        return result.stderr.decode(errors="replace").strip() or f"exit status {result.exit_code}"

    # Factory construction is synchronous and may run inside an existing event loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, probe()).result()


def _working_bwrap(
    candidates: Iterable[Path],
    system_directories: sandbox.SystemDirectories,
    mounts: tuple[Mount, ...],
    environment: dict[str, str],
) -> Path:
    errors = []
    for candidate in candidates:
        error = _sandbox_error(candidate, system_directories, mounts, environment)
        if error is None:
            return candidate
        errors.append(f"{candidate}: {error}")
    overlay = (
        "Copy-on-write system directories also need overlayfs and a temporary directory that is not on overlayfs. "
        if system_directories == sandbox.SystemDirectories.COPY_ON_WRITE
        else ""
    )
    raise sandbox.SandboxUnavailable(
        "No bwrap can sandbox commands on this host. A root process needs CAP_SYS_ADMIN, CAP_NET_ADMIN, no "
        "AppArmor confinement and a seccomp filter that allows pivot_root; another user needs unprivileged "
        "user namespaces. " + overlay + "; ".join(errors)
    )
