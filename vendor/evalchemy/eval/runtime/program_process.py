"""Own and reap the process groups used by canonical program checkers."""

import os
import resource
import signal
import subprocess
from multiprocessing.process import BaseProcess
from typing import Optional

_PROGRAM_SESSION_ID: Optional[int] = None
PROGRAM_MEMORY_BYTES = 30 * 1024**3


def start_program_session() -> None:
    """Own a candidate's descendants and cap each process's address/data space."""
    os.setsid()
    global _PROGRAM_SESSION_ID
    _PROGRAM_SESSION_ID = os.getpid()
    # Native checker safe_mode flags control unrelated execution behavior. Keep
    # the memory budget independent, including when those flags are disabled.
    for kind in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
        soft, hard = resource.getrlimit(kind)
        soft = PROGRAM_MEMORY_BYTES if soft == resource.RLIM_INFINITY else min(soft, PROGRAM_MEMORY_BYTES)
        hard = PROGRAM_MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(hard, PROGRAM_MEMORY_BYTES)
        resource.setrlimit(kind, (soft, hard))


def program_session_active() -> bool:
    """Whether a canonical checker owns the current session's cleanup."""
    return _PROGRAM_SESSION_ID is not None and os.getsid(0) == _PROGRAM_SESSION_ID


def kill_program_group(pid: int) -> None:
    """Kill an owned session leader's group, including surviving descendants."""
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_program_subprocess(process: subprocess.Popen) -> None:
    """Reap a command while leaving its native owner responsible for descendants."""
    if program_session_active():
        if process.poll() is None:
            process.kill()
    else:
        kill_program_group(process.pid)
    process.wait()


def run_program_process(process: BaseProcess, timeout: float) -> None:
    """Keep the canonical deadline and reap descendants on every return path."""
    process.start()
    try:
        process.join(timeout=timeout)
    finally:
        try:
            # Stop the leader first so it cannot create a session after cleanup
            # raced with its startup. Its first action establishes that session.
            if process.is_alive():
                process.kill()
            kill_program_group(process.pid)
        finally:
            process.join()
