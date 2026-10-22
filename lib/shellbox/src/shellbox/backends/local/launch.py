# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start one command inside a machine's sandbox, end its process group, and report how it ended.

Usage: ``python -I -S -c <source> <settings JSON> <program> <args>...``. The settings name descriptors the
launcher inherits: ``sandbox``, a pidfd of the sandbox's first process; ``environment``, the command's
environment as NUL-terminated names and values; and ``status``, a pipe for the report. They also hold the
resource ``limits`` and the ``account`` (UID and GID) to switch to, or null.

The launcher applies the limits and its priority, enters the sandbox's namespaces and root, and forks the
command, which starts a session, switches account, drops every capability, sets ``no_new_privs`` and executes
``program`` with exactly the given environment. Loader variables such as ``LD_PRELOAD`` therefore take effect
only in the sandboxed program. The launcher stays outside the sandbox's PID namespace, where commands cannot
signal it. When the command's first process ends, the launcher kills the rest of its process group; SIGTERM
makes it kill the whole group at once.

The status pipe receives ``S`` when the command starts, then ``X<code>`` when it exits or ``K<signal>`` when a
signal ends it, each line ending in a newline. A launcher or command setup failure writes ``E<message>``.
"""

import ctypes
import json
import os
import resource
import select
import signal
import sys

COMMAND_NICENESS = 10
"""How far below the host process a command's CPU priority sits, so busy commands cannot starve the host."""
# The kernel's flag for each namespace in /proc/<pid>/ns. Some Linux builds of Python lack os.CLONE_NEWCGROUP.
NAMESPACES = (
    ("user", 0x10000000),
    ("mnt", 0x00020000),
    ("cgroup", 0x02000000),
    ("ipc", 0x08000000),
    ("uts", 0x04000000),
    ("net", 0x40000000),
    ("pid", 0x20000000),
)
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapabilityHeader(ctypes.Structure):
    _fields_ = (("version", ctypes.c_uint32), ("pid", ctypes.c_int))


class _CapabilitySet(ctypes.Structure):
    _fields_ = (("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32))


def _environment(fd: int) -> dict[bytes, bytes]:
    with os.fdopen(fd, "rb") as source:
        fields = source.read().split(b"\0")[:-1]
    return dict(zip(fields[::2], fields[1::2], strict=True))


def _enter(sandbox: int) -> None:
    """Join the namespaces in which the sandbox's first process differs from this one, and its root directory."""
    with open(f"/proc/self/fdinfo/{sandbox}") as fdinfo:
        pid = next(int(line.split()[1]) for line in fdinfo if line.startswith("Pid:"))
    flags = 0
    for name, flag in NAMESPACES:
        if os.stat(f"/proc/self/ns/{name}").st_ino != os.stat(f"/proc/{pid}/ns/{name}").st_ino:
            flags |= flag
    root = os.open(f"/proc/{pid}/root", os.O_RDONLY | os.O_DIRECTORY)
    # The PID still names the sandbox's process if that process had not exited once the root was open.
    poller = select.poll()
    poller.register(sandbox, select.POLLIN)
    if poller.poll(0):
        raise ProcessLookupError("The machine's sandbox has ended")
    os.setns(sandbox, flags)
    os.close(sandbox)
    os.fchdir(root)
    os.chroot(".")
    os.close(root)


def _drop_privileges(libc: ctypes.CDLL, last_capability: int, account: list[int] | None) -> None:
    """Switch to ``account`` and give up every capability, now and across exec, as bwrap's ``--cap-drop ALL`` does."""

    def prctl(*arguments: int) -> None:
        if libc.prctl(*(ctypes.c_ulong(argument) for argument in arguments)) != 0:
            raise OSError(ctypes.get_errno(), f"prctl{arguments}")

    # Dropping from the bounding set needs CAP_SETPCAP, which the account switch clears.
    for capability in range(last_capability + 1):
        prctl(PR_CAPBSET_DROP, capability, 0, 0, 0)
    prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0, 0, 0)
    if account is not None:
        uid, gid = account
        os.setgroups([])
        os.setresgid(gid, gid, gid)
        os.setresuid(uid, uid, uid)
    header = _CapabilityHeader(LINUX_CAPABILITY_VERSION_3, 0)
    empty = (_CapabilitySet * 2)()
    if libc.capset(ctypes.byref(header), empty) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)


def _command(
    status: int, libc: ctypes.CDLL, last_capability: int, account: list[int] | None, environment: dict[bytes, bytes]
) -> None:
    """The forked command: never returns."""
    try:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.setsid()
        _drop_privileges(libc, last_capability, account)
        os.write(status, b"S")
        os.execve(sys.argv[2], sys.argv[2:], environment)
    except BaseException as error:
        os.write(status, f"E{error!r}\n".encode())
    finally:
        os._exit(127)


def main() -> None:
    settings = json.loads(sys.argv[1])
    status = settings["status"]
    os.set_inheritable(status, False)
    try:
        environment = _environment(settings["environment"])
        libc = ctypes.CDLL(None, use_errno=True)
        with open("/proc/sys/kernel/cap_last_cap") as source:
            last_capability = int(source.read())
        for limit, value in settings["limits"]:
            soft, _ = resource.getrlimit(limit)
            value = value if soft == resource.RLIM_INFINITY else min(value, soft)
            resource.setrlimit(limit, (value, value))
        os.nice(COMMAND_NICENESS)
        # Python ignores these signals at startup, and exec would pass the ignored dispositions on to the command.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        _enter(settings["sandbox"])
    except Exception as error:
        os.write(status, f"E{error!r}\n".encode())
        sys.exit(1)

    command = 0

    def stop(signum: int, frame: object) -> None:
        if command > 0:
            # Before the command starts its session, its group does not exist yet; kill the process itself too.
            for kill in (os.killpg, os.kill):
                try:
                    kill(command, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGTERM, stop)
    try:
        command = os.fork()
    except OSError as error:
        os.write(status, f"E{error!r}\n".encode())
        sys.exit(1)
    if command == 0:
        _command(status, libc, last_capability, settings["account"], environment)
    # Wait without reaping, so the command's PID, and with it its process group's, cannot be reused yet.
    os.waitid(os.P_PID, command, os.WEXITED | os.WNOWAIT)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    stop(signal.SIGTERM, None)
    _, code = os.waitpid(command, 0)
    code = os.waitstatus_to_exitcode(code)
    os.write(status, (f"K{-code}\n" if code < 0 else f"X{code}\n").encode())


if __name__ == "__main__":
    main()
