# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start and pin the sandbox init before releasing its startup barrier."""

import asyncio
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from shellbox.backends.local import sandbox
from shellbox.backends.local.cgroup import MachineCgroup
from shellbox.backends.local.images import PreparedLocalImage


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
    """Pin bwrap's init, verifying its identity through an anchored proc directory."""
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


async def allocate_keeper(
    state: KeeperHost, arguments: list[str], peer: "KeeperHost | None", block_read: int, block_write: int
) -> None:
    with tempfile.TemporaryFile() as information:
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
        prefix = () if peer is None else peer._network_namespaces
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
            state._keeper_pidfd = os.pidfd_open(state._keeper.pid)
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
                            break
                    except json.JSONDecodeError:
                        pass  # bwrap may still be writing its single status document.
                if state._keeper.returncode is not None:
                    assert state._keeper.stderr is not None
                    error = (await state._keeper.stderr.read()).decode(errors="replace")
                    raise sandbox.SandboxUnavailable(f"Persistent bubblewrap keeper failed: {error.strip()}")
                await asyncio.sleep(0.01)
        # The keeper cannot be reused after exit: the opened descriptors pin
        # this namespace and root even if the kernel later reuses its PID.
        descriptors = []
        try:
            for namespace in ("user", "net", "ipc", "uts", "mnt", "pid", "cgroup"):
                descriptors.append(os.open(f"/proc/{pid}/ns/{namespace}", os.O_RDONLY | os.O_CLOEXEC))
                expected_inode = json.loads(payload).get(f"{namespace}-namespace")
                if expected_inode is not None and os.fstat(descriptors[-1]).st_ino != expected_inode:
                    raise sandbox.SandboxUnavailable("Bubblewrap startup namespace identity changed")
            state._root_fd = os.open(f"/proc/{pid}/root", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            owner = fcntl.ioctl(descriptors[4], sandbox.NS_GET_USERNS)
            os.set_inheritable(owner, False)
        except BaseException:
            for descriptor in descriptors:
                os.close(descriptor)
            raise
        state._owned_namespaces = (owner, *descriptors)
        # bwrap drops capabilities by entering a final child user namespace.
        # Join mount/network owners first, then that restricted namespace.
        state._namespaces = (*prefix, owner, *descriptors[1:], descriptors[0])
        state._network_namespaces = (*prefix, owner, descriptors[1])
        os.write(block_write, b"1")
        assert state._keeper.stdin is not None
        state._keeper.stdin.write(b"shellbox-ready\n")
        await state._keeper.stdin.drain()
        state._keeper.stdin.close()
