from pathlib import Path
from .safe_subprocess import run
import subprocess


def eval_script(path: Path):
    # since lean is a theorem prover first and not a programming environment,
    # the return code is always 1. idk.
    try:
        output = subprocess.run(["lean", str(path)], capture_output=True, timeout=5)
        outmessage = str(output)

        if "error: tactic 'rfl' failed" in outmessage:  # :skull:
            status = "AssertionError"
        elif outmessage == "":
            status = "OK"
        else:
            status = "SyntaxError"
        returncode = output.returncode

    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        output = exc
        returncode = -1
    return {
        "status": status,
        "exit_code": returncode,
        "stdout": "" if output.stdout is None else output.stdout.decode("utf-8"),
        "stderr": "" if output.stderr is None else output.stderr.decode("utf-8"),
    }
