import argparse
from sys import exit
import subprocess
from pathlib import Path
from .generic_eval import main as gmain


def eval_script(path: Path):
    try:
        # Assumes exit-code 0 is all okay
        output = subprocess.run(["elixir", str(path)], capture_output=True, timeout=5)

        if output.returncode == 0:
            status = "OK"
        else:
            outmessage = str(output)
            if "Assertion with == failed" in outmessage:
                status = "AssertionError"
            elif "SyntaxError" in outmessage:
                status = "SyntaxError"
            else:
                status = "Exception"
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


if __name__ == "__main__":
    gmain(eval_script, "Elixir", ".exs")
