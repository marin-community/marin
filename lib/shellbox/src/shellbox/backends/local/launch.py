# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Confine this process, switch to the command's user, and exec the command.

The local backend runs this source as ``python -I -S -c <source> <config JSON> <program> <args>...``,
so it imports only the standard library and never needs a file the command's user may lack access to.
"""

import ctypes
import json
import os
import resource
import signal
import stat
import struct
import sys

PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
# Linux gives the Landlock syscalls these numbers on every architecture.
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
# Access rights and scopes from linux/landlock.h.
LANDLOCK_EXECUTE = 1 << 0
LANDLOCK_WRITE_FILE = 1 << 1
LANDLOCK_READ_FILE = 1 << 2
LANDLOCK_READ_DIR = 1 << 3
LANDLOCK_TRUNCATE = 1 << 14
LANDLOCK_IOCTL_DEV = 1 << 15
LANDLOCK_NET_TCP = 0b11  # Bind and connect.
LANDLOCK_SCOPES = 0b11  # Abstract Unix sockets and signals.
# The rights a rule on a file, rather than a directory, may grant.
LANDLOCK_FILE_RIGHTS = (
    LANDLOCK_EXECUTE | LANDLOCK_WRITE_FILE | LANDLOCK_READ_FILE | LANDLOCK_TRUNCATE | LANDLOCK_IOCTL_DEV
)

libc = ctypes.CDLL(None, use_errno=True)


def syscall(number: int, *args: object) -> int:
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(a) if isinstance(a, int) else a for a in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def no_new_privs_available() -> bool:
    return libc.prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) >= 0


def landlock_abi() -> int:
    """The kernel's Landlock ABI version, or 0 when the kernel or a seccomp filter refuses Landlock."""
    try:
        return syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)
    except OSError:
        return 0


def restrict(handled_fs: int, handled_net: int, scoped: int, rules: list[tuple[str, int]]) -> None:
    """Deny this process every handled access except each rule's rights beneath its path."""
    # struct landlock_ruleset_attr; a kernel with an older ABI accepts the zeroed fields it does not know.
    attr = struct.pack("=QQQ", handled_fs, handled_net, scoped)
    ruleset = syscall(SYS_LANDLOCK_CREATE_RULESET, attr, len(attr), 0)
    for path, rights in rules:
        try:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except FileNotFoundError:
            continue  # A missing path needs no rule.
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            rights &= LANDLOCK_FILE_RIGHTS
        rule = struct.pack("=Qi", rights, descriptor)  # The packed struct landlock_path_beneath_attr.
        syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, rule, 0)
        os.close(descriptor)
    syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0)
    os.close(ruleset)


def main() -> None:
    config = json.loads(sys.argv[1])
    for limit, value in config["rlimits"]:
        soft, _ = resource.getrlimit(limit)
        value = value if soft == resource.RLIM_INFINITY else min(value, soft)
        resource.setrlimit(limit, (value, value))
    if config["no_new_privs"] and libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if config["landlock"] is not None:
        restrict(**config["landlock"])
    if config["account"] is not None:
        uid, gid = config["account"]
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    # Python ignores these signals at startup, and exec would pass the ignored dispositions on to the command.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    os.execv(sys.argv[2], sys.argv[2:])


if __name__ == "__main__":
    main()
