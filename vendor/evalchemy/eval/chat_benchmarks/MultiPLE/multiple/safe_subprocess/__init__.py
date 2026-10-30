import os
import fcntl
import time
import subprocess
from eval.runtime.program_process import program_session_active, stop_program_subprocess
from typing import List

MAX_BYTES_PER_READ = 1024
SLEEP_BETWEEN_READS = 0.1


class Result:
    timeout: int
    exit_code: int
    stdout: str
    stderr: str

    def __init__(self, timeout, exit_code, stdout, stderr):
        self.timeout = timeout
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def set_nonblocking(reader):
    fd = reader.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)


def run(
    args: List[str], timeout_seconds: int = 15, max_output_size: int = 2048, env=None, cwd: str | None = None
) -> Result:
    """
    Runs the given program with arguments. After the timeout elapses, kills the process
    and all other processes in the process group. Captures at most max_output_size bytes
    of stdout and stderr each, and discards any output beyond that.
    """
    p = subprocess.Popen(
        args,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=not program_session_active(),
        bufsize=MAX_BYTES_PER_READ,
        cwd=cwd,
    )
    set_nonblocking(p.stdout)
    set_nonblocking(p.stderr)

    # We sleep for 0.1 seconds in each iteration.
    max_iterations = timeout_seconds * 10
    stdout_saved_bytes = []
    stderr_saved_bytes = []
    stdout_bytes_read = 0
    stderr_bytes_read = 0

    try:
        for _ in range(max_iterations):
            this_stdout_read = p.stdout.read(MAX_BYTES_PER_READ)
            this_stderr_read = p.stderr.read(MAX_BYTES_PER_READ)
            # this_stdout_read and this_stderr_read may be None if stdout or stderr
            # are closed. Without these checks, test_close_output fails.
            if this_stdout_read is not None and stdout_bytes_read < max_output_size:
                stdout_saved_bytes.append(this_stdout_read)
                stdout_bytes_read += len(this_stdout_read)
            if this_stderr_read is not None and stderr_bytes_read < max_output_size:
                stderr_saved_bytes.append(this_stderr_read)
                stderr_bytes_read += len(this_stderr_read)
            exit_code = p.poll()
            if exit_code is not None:
                break
            time.sleep(SLEEP_BETWEEN_READS)
    finally:
        # Also run when the outer canonical checker interrupts this helper.
        stop_program_subprocess(p)

    timeout = exit_code is None
    exit_code = exit_code if exit_code is not None else -1
    stdout = b"".join(stdout_saved_bytes).decode("utf-8", errors="ignore")
    stderr = b"".join(stderr_saved_bytes).decode("utf-8", errors="ignore")
    return Result(timeout=timeout, exit_code=exit_code, stdout=stdout, stderr=stderr)
