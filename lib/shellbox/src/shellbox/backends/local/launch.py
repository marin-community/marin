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
import sys

COMMAND_NICENESS = 10
"""How far below the host process a command's CPU priority sits, so busy commands cannot starve the host."""
PR_CAPBSET_DROP = 24
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522
CAP_CHOWN = 0
CAP_DAC_OVERRIDE = 1
CAP_FOWNER = 3
IMAGE_FILESYSTEM_CAPABILITIES = (CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FOWNER)
SETUP_ERROR_LIMIT_BYTES = 512
"""At most POSIX's minimum PIPE_BUF, so reporting cannot block before the host reads."""


def execute_in_namespaces(configuration_fd: int) -> None:
    """Join pinned namespaces and exec the configured program or sandboxed command."""
    with os.fdopen(configuration_fd) as configuration:
        config = json.load(configuration)
    status = config.get("status")
    if status is not None:
        os.set_inheritable(status, False)
    disk_status = config.get("disk_status")
    disk = None
    namespace_directory = None
    try:
        namespace_directory = os.open("/proc/self/ns", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        outer_user_namespace = os.stat("user", dir_fd=namespace_directory).st_ino
        if disk_status is not None:
            disk = os.open(config["disk_root"], os.O_RDONLY | os.O_DIRECTORY)
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
        # The new /proc belongs to the guest PID namespace; this supervisor is
        # outside it until fork, so use the original process's pinned proc dir.
        if config["capabilities"] and os.stat("user", dir_fd=namespace_directory).st_ino == outer_user_namespace:
            raise ValueError("Image ownership requires an isolated guest user namespace")
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
            if disk is not None:
                os.close(disk)
                os.close(disk_status)
            drop_privileges(config["account"], config["capabilities"])
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
    finally:
        if namespace_directory is not None:
            os.close(namespace_directory)
    if status is not None:
        os.close(status)
    _, status = os.waitpid(child, 0)
    code = os.waitstatus_to_exitcode(status)
    if disk is not None:
        if code:
            filesystem = os.fstatvfs(disk)
            full = filesystem.f_favail == 0 or filesystem.f_bavail * filesystem.f_bsize < config["disk_failure_headroom"]
            os.write(disk_status, bytes((full,)))
        os.close(disk)
    sys.exit(code if code >= 0 else 128 - code)


def drop_privileges(account: list[int] | None, capabilities: list[int]) -> None:
    if capabilities not in ([], list(IMAGE_FILESYSTEM_CAPABILITIES)):
        raise ValueError("Only isolated image filesystem operations may retain capabilities")
    libc = ctypes.CDLL(None, use_errno=True)
    # Retain only image-local filesystem operations. Drop the other bounding bits
    # while CAP_SETPCAP remains available; no_new_privs prevents setuid/file gains.
    for capability in range(64):
        if capability in capabilities:
            continue
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
    for capability in capabilities:
        data[0] |= 1 << capability
        data[1] |= 1 << capability
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "Cannot clear process capabilities")
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot set no_new_privs")


if __name__ == "__main__":
    execute_in_namespaces(int(sys.argv[1]))
