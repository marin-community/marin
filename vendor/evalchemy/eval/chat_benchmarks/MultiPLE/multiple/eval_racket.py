"""
Evaluates a generated Racket program (.rkt).
"""

import os
from pathlib import Path
from .safe_subprocess import run
from .libeval import run_without_exn


def eval_script(path: Path):
    result = run(["racket", str(path)])

    if "standard-module-name-resolver: collection not found\n  for module path: rackunit" in result.stderr:
        print(f"Failed to run evaluation for {path}: rackunit is not installed")
        return None

    # rackunit produces exit code 0 even if tests fail.
    if len(result.stderr) > 0 or result.exit_code != 0:
        if "read-syntax" in result.stderr:
            status = "SyntaxError"
        else:
            status = "Exception"
    else:
        status = "OK"

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
