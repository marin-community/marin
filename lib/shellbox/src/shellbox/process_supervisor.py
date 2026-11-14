# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Linux command ownership, executed in the guest using only the standard library."""

from __future__ import annotations

import ctypes
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

PR_SET_CHILD_SUBREAPER = 36
CLEANUP_TIMEOUT = 8
CANCEL_CONFIRMATION_GRACE = 2
POLL_INTERVAL = 0.01
TIMEOUT_MARKER = "SHELLBOX_COMMAND_TIMEOUT"
FAILURE_MARKER = "SHELLBOX_CLEANUP_FAILED"
LOCK_FILE = "lock"
CANCEL_FILE = "cancel"
FAILED_FILE = "failed"
DONE_FILE = "done"
STARTED_FILE = "started"
PID_FILE = "pid"


def reap_children() -> None:
    """Kill and reap every child, including newly adopted orphan descendants."""
    deadline = time.monotonic() + CLEANUP_TIMEOUT
    parent_pid = os.getpid()
    while True:
        # A direct child's PID cannot be reused until we reap it. Killing each
        # generation makes its descendants our children, even across setsid().
        # The task/children entry requires CONFIG_CHECKPOINT_RESTORE; stat is
        # available on kernels without that optional feature. Process names can
        # contain spaces and parentheses, so split after the final closing one.
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue  # An unrelated process exited during enumeration.
            if int(stat.rsplit(")", 1)[1].split()[1]) != parent_pid:
                continue
            try:
                os.kill(int(stat_path.parent.name), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return  # ECHILD confirms no running or unreaped descendants remain.
        if time.monotonic() >= deadline:
            raise TimeoutError("Command descendants did not terminate within cleanup grace")
        time.sleep(POLL_INTERVAL)


def cancel(control: Path) -> None:
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (control / LOCK_FILE).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (control / CANCEL_FILE).touch()
        if (control / FAILED_FILE).exists():
            raise RuntimeError("Command teardown failed")
        if (control / DONE_FILE).exists():
            return
        if not (control / PID_FILE).exists():
            if (control / STARTED_FILE).exists():
                raise RuntimeError("Command supervisor identity is missing")
            return  # A queued launch observes cancel before creating children.
        os.kill(int((control / PID_FILE).read_text()), signal.SIGTERM)
    deadline = time.monotonic() + CLEANUP_TIMEOUT + CANCEL_CONFIRMATION_GRACE
    while not (control / DONE_FILE).exists():
        if (control / FAILED_FILE).exists() or time.monotonic() >= deadline:
            raise RuntimeError("Command teardown could not be confirmed")
        time.sleep(POLL_INTERVAL)


def run(control: Path, budget: str, argv: list[str]) -> int:
    control.mkdir(mode=0o700, parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot become a child subreaper")
    stop_code = None

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_code
        stop_code = 128 + signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with (control / LOCK_FILE).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (control / CANCEL_FILE).exists():
            return 143
        if (control / PID_FILE).exists() or (control / DONE_FILE).exists():
            raise RuntimeError("Command scope has already been started")
        (control / STARTED_FILE).touch()
        (control / PID_FILE).write_text(str(os.getpid()))
    code = 143
    timed_out = False
    try:
        if stop_code is None:
            deadline = None if budget == "none" else time.monotonic() + float(budget)
            child = subprocess.Popen(argv)
            while True:
                if stop_code is not None:
                    code = stop_code
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    print(TIMEOUT_MARKER, file=sys.stderr, flush=True)
                    timed_out = True
                    code = 124
                    break
                result = child.poll()
                if result is not None:
                    code = result if result >= 0 else 128 - result
                    break
                time.sleep(POLL_INTERVAL)
    finally:
        reap_children()
    with (control / LOCK_FILE).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (control / PID_FILE).unlink()
        (control / DONE_FILE).write_text("timeout" if timed_out else "complete")
    return code


def main() -> int:
    action, directory, *args = sys.argv[1:]
    control = Path(directory)
    try:
        if action == "cancel":
            cancel(control)
            return 0
        if action != "run":
            raise ValueError(f"Unknown supervisor action: {action}")
        pid = os.fork()
        if pid:
            _, status = os.waitpid(pid, 0)
            return os.waitstatus_to_exitcode(status)
        # A cancelled command transport may kill its process group. Keep the
        # subreaper alive so the independent cancel command can reach it.
        os.setsid()
        return run(control, args[0], args[1:])
    except Exception as error:
        control.mkdir(mode=0o700, parents=True, exist_ok=True)
        (control / FAILED_FILE).touch()
        print(FAILURE_MARKER, file=sys.stderr, flush=True)
        print(str(error), file=sys.stderr, flush=True)
        return 125


if __name__ == "__main__":
    sys.exit(main())
