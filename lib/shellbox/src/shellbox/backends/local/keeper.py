# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start and pin the sandbox init before releasing its startup barrier."""

import asyncio
import fcntl
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from shellbox.backends.local import sandbox
from shellbox.backends.local.cgroup import MachineCgroup
from shellbox.backends.local.images import PreparedLocalImage

NAMESPACE_NAMES = ("user", "net", "ipc", "uts", "mnt", "pid", "cgroup")


@dataclass(frozen=True)
class GuestIdentity:
    pid: int
    namespace_inodes: dict[str, int]


class KeeperHost(Protocol):
    root: Path
    image: PreparedLocalImage | None
    _environment: dict[str, str]
    _keeper_cgroup: MachineCgroup | None
    _keeper: asyncio.subprocess.Process | None
    _keeper_pidfd: int | None
    _init_pidfd: int | None
    _root_fd: int | None
    _owned_namespaces: tuple[int, ...]
    _namespaces: tuple[int, ...]
    _network_namespaces: tuple[int, ...]


def _pin_init(pid: int, namespace: int) -> int:
    """Return an owned pidfd for the verified guest init."""
    directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    descriptor = None
    try:
        descriptor = os.pidfd_open(pid)
        process_namespace = os.open("ns/pid", os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory)
        try:
            if os.fstat(process_namespace).st_ino != namespace:
                raise sandbox.SandboxUnavailable("Bubblewrap init namespace identity changed")
        finally:
            os.close(process_namespace)
        return descriptor
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(directory)


async def _start_launcher(
    state: KeeperHost, arguments: list[str], prefix: tuple[int, ...], block_read: int, information: BinaryIO
) -> None:
    arguments += [
        "--block-fd",
        str(block_read),
        "--info-fd",
        str(information.fileno()),
        "--clearenv",
        "--setenv",
        "PATH",
        state._environment["PATH"],
        "/bin/sh",
        "-c",
        sandbox.KEEPER_PAYLOAD,
    ]
    config = {
        "namespaces": prefix,
        "program": arguments,
        "cgroup": None if state._keeper_cgroup is None else str(state._keeper_cgroup.path / "cgroup.procs"),
    }
    with tempfile.TemporaryFile() as configuration:
        configuration.write(json.dumps(config).encode())
        configuration.seek(0)
        state._keeper = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            sandbox.LAUNCH_SOURCE,
            str(configuration.fileno()),
            pass_fds=(configuration.fileno(), information.fileno(), block_read, *prefix),
            env=sandbox.CONTROL_ENVIRONMENT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if state._keeper.returncode is not None:
            raise ProcessLookupError("Local keeper exited before startup completed")
        state._keeper_pidfd = os.pidfd_open(state._keeper.pid)


async def _guest_identity(state: KeeperHost, information: BinaryIO) -> GuestIdentity:
    assert state._keeper is not None
    async with asyncio.timeout(sandbox.PROBE_TIMEOUT):
        while True:
            # The inherited file description shares bwrap's write
            # offset; seeking here can corrupt its multi-write JSON.
            payload = os.pread(information.fileno(), os.fstat(information.fileno()).st_size, 0)
            if payload:
                try:
                    status = json.loads(payload)
                    pid = status["child-pid"]
                    if state._init_pidfd is None:
                        state._init_pidfd = _pin_init(pid, status["pid-namespace"])
                    # info-fd is written before bwrap finishes mounting
                    # and pivoting. Pinning root earlier would capture
                    # the host root rather than the guest root.
                    expected = state.root.stat()
                    actual = os.stat(f"/proc/{pid}/root")
                    expected_uid = 0 if state.image is not None else os.geteuid()
                    mapped = Path(f"/proc/{pid}/uid_map").read_text().splitlines()
                    final_user = any(int(row.split()[0]) == expected_uid for row in mapped)
                    if final_user and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
                        return GuestIdentity(
                            pid,
                            {
                                name: status[f"{name}-namespace"]
                                for name in NAMESPACE_NAMES
                                if f"{name}-namespace" in status
                            },
                        )
                except json.JSONDecodeError:
                    pass  # bwrap may still be writing its single status document.
            if state._keeper.returncode is not None:
                assert state._keeper.stderr is not None
                error = (await state._keeper.stderr.read()).decode(errors="replace")
                raise sandbox.SandboxUnavailable(f"Persistent bubblewrap keeper failed: {error.strip()}")
            await asyncio.sleep(0.01)


def _pin_namespaces(state: KeeperHost, identity: GuestIdentity, prefix: tuple[int, ...]) -> None:
    # Pinned handles retain namespace identity even after the kernel reuses the PID.
    with ExitStack() as resources:
        descriptors = []
        for namespace in NAMESPACE_NAMES:
            descriptor = os.open(f"/proc/{identity.pid}/ns/{namespace}", os.O_RDONLY | os.O_CLOEXEC)
            resources.callback(os.close, descriptor)
            descriptors.append(descriptor)
            expected_inode = identity.namespace_inodes.get(namespace)
            if expected_inode is not None and os.fstat(descriptor).st_ino != expected_inode:
                raise sandbox.SandboxUnavailable("Bubblewrap startup namespace identity changed")
        state._root_fd = os.open(f"/proc/{identity.pid}/root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        owner = fcntl.ioctl(descriptors[NAMESPACE_NAMES.index("mnt")], sandbox.NS_GET_USERNS)
        resources.callback(os.close, owner)
        os.set_inheritable(owner, False)
        state._owned_namespaces = (owner, *descriptors)
        # Join mount/network owners before bwrap's final restricted user namespace.
        state._namespaces = (*prefix, owner, *descriptors[1:], descriptors[0])
        state._network_namespaces = (*prefix, owner, descriptors[1])
        resources.pop_all()  # Ownership has moved into the runtime, including partial startup cleanup.


async def allocate_keeper(
    state: KeeperHost, arguments: list[str], peer: "KeeperHost | None", block_read: int, block_write: int
) -> None:
    prefix = () if peer is None else peer._network_namespaces
    with tempfile.TemporaryFile() as information:
        await _start_launcher(state, arguments, prefix, block_read, information)
        identity = await _guest_identity(state, information)
        _pin_namespaces(state, identity, prefix)
        os.write(block_write, b"1")
        assert state._keeper is not None
        assert state._keeper.stdin is not None
        state._keeper.stdin.write(b"shellbox-ready\n")
        await state._keeper.stdin.drain()
        state._keeper.stdin.close()
