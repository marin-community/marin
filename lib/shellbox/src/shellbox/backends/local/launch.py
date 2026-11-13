# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply a command's resource limits and scheduling priority, then exec the sandboxed program.

Usage: ``python -I -S -c <source> <configuration FD>``. Configuration stays on a private descriptor
that closes before guest execution.
"""

import ctypes
import errno
import json
import os
import resource
import signal
import struct
import sys
import tempfile

COMMAND_NICENESS = 10
"""How far below the host process a command's CPU priority sits, so busy commands cannot starve the host."""
PR_CAPBSET_DROP = 24
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522
SETUP_ERROR_LIMIT_BYTES = 512
"""At most POSIX's minimum PIPE_BUF, so reporting cannot block before the host reads."""

PTRACE_SYSCALL = 24
PTRACE_DETACH = 17
PTRACE_GETEVENTMSG = 0x4201
PTRACE_SEIZE = 0x4206
PTRACE_INTERRUPT = 0x4207
PTRACE_GET_SYSCALL_INFO = 0x420E
PTRACE_OPTIONS = 0x00000001 | 0x00000002 | 0x00000004 | 0x00000008 | 0x00000010 | 0x00100000
WAIT_ALL_THREADS = 0x40000000
SYSCALL_STOP = signal.SIGTRAP | 0x80
SYSCALL_ENTRY = 1
SYSCALL_EXIT = 2
FORK_EVENTS = (1, 2, 3)
EXEC_EVENT = 4
EVENT_STOP = 128
ALLOCATION_REPORT = struct.Struct("=qi")
# Linux UAPI syscall numbers, keyed by ptrace's AUDIT_ARCH value. Path-based
# allocation calls can only write the task filesystem: other mounts are read-only.
# FD-based calls additionally check the actual filesystem, excluding pipes,
# sockets and devices such as /dev/full from quota classification.
ALLOCATION_SYSCALLS = {
    0xC000003E: (
        {
            1: 0,
            16: 0,
            18: 0,
            20: 0,
            40: 0,
            74: 0,
            75: 0,
            77: 0,
            91: 0,
            93: 0,
            190: 0,
            199: 0,
            275: 2,
            276: 1,
            277: 0,
            285: 0,
            296: 0,
            326: 2,
            328: 0,
        },
        {
            2,
            76,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            90,
            92,
            94,
            133,
            188,
            189,
            197,
            198,
            257,
            258,
            259,
            260,
            263,
            264,
            265,
            266,
            268,
            316,
            437,
        },
    ),
    0xC00000B7: (
        {
            7: 0,
            16: 0,
            29: 0,
            46: 0,
            47: 0,
            52: 0,
            55: 0,
            64: 0,
            66: 0,
            68: 0,
            70: 0,
            71: 0,
            76: 2,
            77: 1,
            82: 0,
            83: 0,
            84: 0,
            285: 2,
            287: 0,
        },
        {5, 6, 14, 15, 33, 34, 35, 36, 37, 38, 45, 53, 54, 56, 276, 437},
    ),
}


def ptrace(request: int, pid: int, address: int = 0, data: int = 0) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    if libc.ptrace(request, pid, ctypes.c_void_p(address), ctypes.c_void_p(data)) == -1:
        raise OSError(ctypes.get_errno(), f"ptrace {request} failed for {pid}")


def observe_allocations(child: int, barrier: int, report: int, proc: int, devices: tuple[int, ...]) -> int:
    """Observe kernel syscall results, keeping the report FD outside guest code.

    GET_SYSCALL_INFO is architecture independent (Linux 5.3+). Descendants are
    traced until the foreground exits, then detached without killing services.
    The supervisor remains outside the guest PID namespace, as in untraced runs.
    """
    ptrace(PTRACE_SEIZE, child, data=PTRACE_OPTIONS)
    ptrace(PTRACE_INTERRUPT, child)
    os.write(barrier, b"1")
    os.close(barrier)
    traced = {child}
    syscalls: dict[int, tuple[int, int | None]] = {}
    foreground_status = None
    reported = False
    while traced:
        pid, status = os.waitpid(-1, WAIT_ALL_THREADS)
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            traced.discard(pid)
            syscalls.pop(pid, None)
            if pid == child:
                foreground_status = status
                for remaining in traced:
                    try:
                        ptrace(PTRACE_INTERRUPT, remaining)
                    except ProcessLookupError:
                        pass  # An exit notification is already pending.
            continue
        assert os.WIFSTOPPED(status)
        traced.add(pid)
        event = status >> 16
        stop = os.WSTOPSIG(status)
        if event in FORK_EVENTS or event == EXEC_EVENT:
            message = ctypes.c_ulong()
            ptrace(PTRACE_GETEVENTMSG, pid, data=ctypes.addressof(message))
            if event in FORK_EVENTS:
                traced.add(message.value)
            elif message.value != pid:
                traced.discard(message.value)
                syscalls.pop(message.value, None)
        if stop == SYSCALL_STOP:
            information = ctypes.create_string_buffer(88)
            ptrace(PTRACE_GET_SYSCALL_INFO, pid, len(information), ctypes.addressof(information))
            operation = information.raw[0]
            if operation == SYSCALL_ENTRY:
                architecture = struct.unpack_from("=I", information.raw, 4)[0]
                fd_calls, path_calls = ALLOCATION_SYSCALLS[architecture]
                number, *arguments = struct.unpack_from("=7Q", information.raw, 24)
                syscalls.pop(pid, None)
                if number in fd_calls:
                    syscalls[pid] = (number, arguments[fd_calls[number]])
                elif number in path_calls:
                    syscalls[pid] = (number, None)
            elif operation == SYSCALL_EXIT and information.raw[32]:
                result = struct.unpack_from("=q", information.raw, 24)[0]
                allocation = syscalls.get(pid)
                if result in (-errno.ENOSPC, -errno.EDQUOT) and allocation is not None and not reported:
                    number, descriptor = allocation
                    on_task_disk = descriptor is None
                    if descriptor is not None:
                        try:
                            on_task_disk = os.stat(f"{pid}/fd/{descriptor}", dir_fd=proc).st_dev in devices
                        except FileNotFoundError:
                            pass  # Another guest thread already closed this FD.
                    if on_task_disk:
                        os.write(report, ALLOCATION_REPORT.pack(number, -result))
                        reported = True
        delivered = 0 if stop == SYSCALL_STOP or event else stop
        if foreground_status is not None:
            ptrace(PTRACE_DETACH, pid, data=delivered)
            traced.discard(pid)
        elif event == EVENT_STOP and stop != signal.SIGTRAP:
            # Preserve a real job-control stop rather than suppressing it.
            ptrace(0x4208, pid)  # PTRACE_LISTEN
        else:
            ptrace(PTRACE_SYSCALL, pid, data=delivered)
    assert foreground_status is not None
    return foreground_status


def probe_allocations(path: str) -> None:
    """Require kernel observation to work before advertising disk enforcement."""
    with tempfile.TemporaryFile() as report:
        proc = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY)
        barrier_read, barrier_write = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(report.fileno())
            os.close(proc)
            os.close(barrier_write)
            if os.read(barrier_read, 1) != b"1":
                os._exit(2)
            os.close(barrier_read)
            try:
                with open(path, "wb", buffering=0) as output:
                    output.write(b"1")
            except OSError as error:
                os._exit(0 if error.errno in (errno.ENOSPC, errno.EDQUOT) else 2)
            os._exit(2)
        os.close(barrier_read)
        status = observe_allocations(
            child, barrier_write, report.fileno(), proc, (os.stat(os.path.dirname(path)).st_dev,)
        )
        os.close(proc)
        assert os.waitstatus_to_exitcode(status) == 0
        report.seek(0)
        syscall, allocation_errno = ALLOCATION_REPORT.unpack(report.read())
        assert syscall >= 0 and allocation_errno in (errno.ENOSPC, errno.EDQUOT)


def execute_in_namespaces(configuration_fd: int) -> None:
    """Join pinned namespaces and exec the configured program or sandboxed command."""
    with os.fdopen(configuration_fd) as configuration:
        config = json.load(configuration)
    status = config.get("status")
    if status is not None:
        os.set_inheritable(status, False)
    allocation_status = config.get("allocation_status")
    proc = os.open("/proc", os.O_RDONLY | os.O_DIRECTORY) if allocation_status is not None else None
    try:
        if config.get("cgroup") is not None:
            # Joining a cgroup namespace changes visibility, not membership.
            # Move the trusted launcher before it can fork any guest process.
            with open(config["cgroup"], "w") as membership:
                membership.write(str(os.getpid()))
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
        barrier_read, barrier_write = os.pipe() if allocation_status is not None else (-1, -1)
        child = os.fork()
        if child == 0:
            if allocation_status is not None:
                os.close(allocation_status)
                assert proc is not None
                os.close(proc)
                os.close(barrier_write)
                if os.read(barrier_read, 1) != b"1":
                    os._exit(1)
                os.close(barrier_read)
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
    if allocation_status is None:
        _, status = os.waitpid(child, 0)
    else:
        os.close(barrier_read)
        assert proc is not None
        status = observe_allocations(child, barrier_write, allocation_status, proc, tuple(config["allocation_devices"]))
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


if __name__ == "__main__":
    if sys.argv[1] == "--probe-allocations":
        probe_allocations(sys.argv[2])
    else:
        execute_in_namespaces(int(sys.argv[1]))
