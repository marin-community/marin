"""Linux subreaper for Pi, including tools orphaned by an exiting parent.

Executed inside the sandbox with the system Python; uses only the standard library.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

PR_SET_CHILD_SUBREAPER = 36
CLEANUP_TIMEOUT = 8
POLL_INTERVAL = 0.01


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
            raise TimeoutError("Pi descendants did not terminate within cleanup grace")
        time.sleep(POLL_INTERVAL)


def main() -> int:
    control = Path(sys.argv[1])
    budget = sys.argv[2]
    command = sys.argv[3]
    control.mkdir(parents=True, exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot become Pi child subreaper")
    stop_code: int | None = None

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_code
        stop_code = 128 + signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # A cancelled submission can arrive before the supervisor starts. Publish
    # the PID before the second check so cancellation cannot miss startup.
    cancelled = control / "cancel"
    if cancelled.exists():
        shutil.rmtree(control)
        return 143
    (control / "pid").write_text(f"{os.getpid()}\n")
    code = 143
    try:
        if not cancelled.exists() and stop_code is None:
            deadline = None if budget == "none" else time.monotonic() + float(budget)
            child = subprocess.Popen(["bash", "-o", "pipefail", "-c", command])
            while True:
                if stop_code is not None:
                    code = stop_code
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    print("HARBOR_PI_AGENT_TIMEOUT", file=sys.stderr, flush=True)
                    code = 124
                    break
                result = child.poll()
                if result is not None:
                    code = result if result >= 0 else 128 - result
                    break
                time.sleep(POLL_INTERVAL)
    finally:
        # Also reap background tools after ordinary completion. Closing the
        # pipeline does not prove that all its tools have exited.
        reap_children()
    shutil.rmtree(control)
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        # Retain the control directory on failure; verification must not proceed.
        print("HARBOR_PI_CLEANUP_FAILED", file=sys.stderr, flush=True)
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
