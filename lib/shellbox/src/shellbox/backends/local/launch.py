# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start bubblewrap, relay commands to its keeper, or launch inside a sandbox.

Usage: ``python -I -S -c <source> <configuration FD>`` for the privileged path,
``--broker <socket>`` inside an unprivileged keeper, or ``--relay <configuration FD>``
outside it. Configuration stays on a private descriptor that closes before guest execution.
"""

import ctypes
import errno
import json
import os
import resource
import select
import signal
import socket
import struct
import sys
import time
import traceback
from array import array

COMMAND_NICENESS = 10
"""How far below the host process a command's CPU priority sits, so busy commands cannot starve the host."""
PR_CAPBSET_DROP = 24
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522
SETUP_ERROR_LIMIT_BYTES = 512
"""At most POSIX's minimum PIPE_BUF, so reporting cannot block before the host reads."""
BROKER_FD_COUNT = 5
BROKER_CONNECT_TIMEOUT = 30
BROKER_CONNECT_RETRY_INTERVAL = 0.01


def execute_in_namespaces(configuration_fd: int) -> None:
    """Join pinned namespaces and exec the configured program or sandboxed command."""
    with os.fdopen(configuration_fd) as configuration:
        config = json.load(configuration)
    status = config.get("status")
    if status is not None:
        os.set_inheritable(status, False)
    try:
        if config.get("cgroup") is not None:
            # Joining a cgroup namespace changes visibility, not membership.
            # Move the trusted launcher before it can fork any guest process.
            with open(config["cgroup"], "w") as membership:
                membership.write(str(os.getpid()))
        if config.get("program_user") is not None:
            # A peer's user namespace may disallow setgroups, so clear groups
            # before entering it while retaining the privilege needed for setns.
            os.setgroups([])
        entered = {
            os.stat(f"/proc/self/ns/{name}").st_ino for name in ("user", "net", "ipc", "uts", "mnt", "pid", "cgroup")
        }
        for descriptor in config["namespaces"]:
            # Entering the current user namespace is forbidden even when a later
            # namespace belongs to it. Group members can share this first entry.
            inode = os.fstat(descriptor).st_ino
            if inode not in entered:
                try:
                    os.setns(descriptor, 0)
                except OSError as error:
                    raise OSError(
                        error.errno, f"Cannot enter namespace {os.readlink(f'/proc/self/fd/{descriptor}')}"
                    ) from error
                entered.add(inode)
        if config.get("program") is not None:
            # Creating another member needs bwrap's own namespace setup. Its
            # descriptors are consumed and closed by bwrap, before the keeper runs.
            if config.get("program_user") is not None:
                uid, gid = config["program_user"]
                os.setgid(gid)
                os.setuid(uid)
            os.execv(config["program"][0], config["program"])
        os.fchdir(config["root"])
        os.chroot(".")
        os.chdir("/")
        for descriptor in (*config["namespaces"], config["root"]):
            os.close(descriptor)
        for limit, value in config["limits"]:
            soft, _ = resource.getrlimit(limit)
            value = value if soft == resource.RLIM_INFINITY else min(value, soft)
            resource.setrlimit(limit, (value, value))
        os.nice(COMMAND_NICENESS)
        # setns(PID) affects only children. This supervisor stays outside the guest
        # PID namespace, so neither its pre-exec state nor control FDs are visible.
        child = os.fork()
        if child == 0:
            drop_privileges(config["account"])
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)
            signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
            os.execvpe(
                "/bin/sh",
                ("/bin/sh", "-c", config["script"], "shellbox-local", config["cwd"], *config["argv"]),
                config["environment"],
            )
    except Exception as error:
        if status is not None:
            os.write(status, f"{type(error).__name__}: {error}".encode()[:SETUP_ERROR_LIMIT_BYTES])
        raise
    if status is not None:
        os.close(status)
    _, status = os.waitpid(child, 0)
    code = os.waitstatus_to_exitcode(status)
    sys.exit(code if code >= 0 else 128 - code)


def drop_privileges(account: list[int] | None) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    # The bounding set must be dropped while the namespace still grants
    # CAP_SETPCAP. no_new_privs also prevents setuid and file-capability gains.
    for capability in range(64):
        if libc.prctl(PR_CAPBSET_DROP, capability, 0, 0, 0) != 0 and ctypes.get_errno() != errno.EINVAL:
            raise OSError(ctypes.get_errno(), "Cannot drop capability bounding set")
    if libc.prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot clear ambient capabilities")
    if account is not None:
        os.setgroups([])
        os.setgid(account[1])
        os.setuid(account[0])
    header = (ctypes.c_uint32 * 2)(LINUX_CAPABILITY_VERSION_3, 0)
    data = (ctypes.c_uint32 * 6)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "Cannot clear process capabilities")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set no_new_privs")


def execute_via_broker(configuration_fd: int) -> None:
    """Relay a command to the keeper without entering its user namespace."""
    with os.fdopen(os.dup(configuration_fd)) as configuration:
        config = json.load(configuration)
    os.lseek(configuration_fd, 0, os.SEEK_SET)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        # The first command can start as soon as bwrap releases its block FD,
        # before the keeper has bound its broker socket.
        deadline = time.monotonic() + BROKER_CONNECT_TIMEOUT
        while True:
            try:
                connection.connect(config["broker"])
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(BROKER_CONNECT_RETRY_INTERVAL)
        peer_pid, _, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if peer_pid != config["broker_pid"]:
            raise RuntimeError("Bubblewrap command broker identity changed")
        descriptors = [configuration_fd, 0, 1, 2]
        if config.get("status") is not None:
            descriptors.append(config["status"])
        connection.sendmsg([b"L"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", descriptors))])
        data, ancillary, _, _ = connection.recvmsg(1, socket.CMSG_SPACE(array("i").itemsize))
        if data != b"P":
            raise RuntimeError("Bubblewrap command broker did not start the launcher")
        pidfds = _received_fds(ancillary)
        if len(pidfds) != 1:
            raise RuntimeError("Bubblewrap command broker did not return a pidfd")
        with os.fdopen(pidfds[0], "rb", closefd=True) as pidfd:
            with open(f"/proc/self/fdinfo/{pidfd.fileno()}") as information:
                pid = int(next(line.split()[1] for line in information if line.startswith("Pid:")))
            if config.get("cgroup") is not None:
                with open(config["cgroup"], "w") as membership:
                    membership.write(str(pid))
            connection.send(b"G")
            result = connection.recv(5)
        if len(result) != 5 or result[:1] != b"E":
            raise RuntimeError("Bubblewrap command broker lost the launcher")
        sys.exit(struct.unpack("!i", result[1:])[0])


def _received_fds(ancillary: list[tuple[int, int, bytes]]) -> list[int]:
    descriptors = array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[: len(data) - len(data) % descriptors.itemsize])
    return descriptors.tolist()


def _broker_command(config: dict, descriptors: list[int], barrier: int) -> None:
    status = descriptors[4] if len(descriptors) == BROKER_FD_COUNT else None
    try:
        os.setpgid(0, 0)
        if os.read(barrier, 1) != b"G":
            os._exit(127)
        for target, descriptor in enumerate(descriptors[1:4]):
            os.dup2(descriptor, target)
        if status is not None:
            os.dup2(status, 3)
            status = 3
            os.set_inheritable(3, False)
        for descriptor in (int(name) for name in os.listdir("/proc/self/fd") if name.isdigit()):
            if descriptor > 3:
                try:
                    os.close(descriptor)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
        for limit, value in config["limits"]:
            soft, _ = resource.getrlimit(limit)
            value = value if soft == resource.RLIM_INFINITY else min(value, soft)
            resource.setrlimit(limit, (value, value))
        os.nice(COMMAND_NICENESS)
        # bwrap already dropped every capability before starting the broker.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot set no_new_privs")
        os.execvpe(
            "/bin/sh",
            ("/bin/sh", "-c", config["script"], "shellbox-local", config["cwd"], *config["argv"]),
            config["environment"],
        )
    except BaseException as error:
        if status is not None:
            os.write(status, f"{type(error).__name__}: {error}".encode()[:SETUP_ERROR_LIMIT_BYTES])
        os._exit(127)


def _supervise_broker_command(connection: socket.socket, descriptors: list[int]) -> None:
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    with os.fdopen(descriptors[0]) as configuration:
        config = json.load(configuration)
    barrier_read, barrier_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(barrier_write)
        _broker_command(config, descriptors, barrier_read)
        os._exit(127)
    os.close(barrier_read)
    for descriptor in descriptors[1:]:
        os.close(descriptor)
    reaped = False
    try:
        with os.fdopen(os.pidfd_open(child), "rb") as pidfd:
            connection.sendmsg([b"P"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [pidfd.fileno()]))])
            if connection.recv(1) != b"G":
                return
        os.write(barrier_write, b"G")
        while True:
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid:
                reaped = True
                code = os.waitstatus_to_exitcode(status)
                connection.send(b"E" + struct.pack("!i", code if code >= 0 else 128 - code))
                return
            if select.select([connection], [], [], 0.05)[0] and not connection.recv(1):
                return
    finally:
        os.close(barrier_write)
        if not reaped:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.waitpid(child, 0)


def _reap_broker_children(_signal: int, _frame: object) -> None:
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def run_broker(path: str) -> None:
    """Accept trusted host launchers from inside the keeper's namespaces."""
    signal.signal(signal.SIGCHLD, _reap_broker_children)
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
        listener.bind(path)
        listener.listen()
        if sys.stdin.readline() != "shellbox-ready\n":
            return
        while True:
            connection, _ = listener.accept()
            with connection:
                data, ancillary, _, _ = connection.recvmsg(1, socket.CMSG_SPACE(BROKER_FD_COUNT * array("i").itemsize))
                descriptors = _received_fds(ancillary)
                if data != b"L" or len(descriptors) not in (4, BROKER_FD_COUNT):
                    for descriptor in descriptors:
                        os.close(descriptor)
                    continue
                child = os.fork()
                if child == 0:
                    listener.close()
                    try:
                        _supervise_broker_command(connection, descriptors)
                    except BaseException:
                        traceback.print_exc()
                    finally:
                        os._exit(0)
                for descriptor in descriptors:
                    os.close(descriptor)


if __name__ == "__main__":
    if sys.argv[1] == "--broker":
        run_broker(sys.argv[2])
    elif sys.argv[1] == "--relay":
        execute_via_broker(int(sys.argv[2]))
    else:
        execute_in_namespaces(int(sys.argv[1]))
