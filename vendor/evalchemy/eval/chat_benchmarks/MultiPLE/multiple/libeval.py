import subprocess
from typing import List
from eval.runtime.program_process import program_session_active, stop_program_subprocess


def run_without_exn(args: List[str]):
    """
    Runs the given program with a five second timeout. Does not throw an exception
    no matter what happens. The output is a dictionary of the format that we expect
    for our evaluation scripts. The "status" field is "OK" when the exit code is
    zero. If that isn't enough, you may want to tweak the status based on the
    captured stderr and stdout.
    """
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=not program_session_active())
    try:
        stdout, stderr = p.communicate(timeout=5)
        exit_code = p.returncode
        status = "OK" if exit_code == 0 else "Exception"
    except subprocess.TimeoutExpired as exc:
        stop_program_subprocess(p)
        stdout, stderr = exc.output, exc.stderr
        exit_code = -1
        status = "Timeout"
    finally:
        stop_program_subprocess(p)

    if stdout is None:
        stdout = b""
    if stderr is None:
        stderr = b""
    return {
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout.decode("utf-8", errors="ignore"),
        "stderr": stderr.decode("utf-8", errors="ignore"),
    }
