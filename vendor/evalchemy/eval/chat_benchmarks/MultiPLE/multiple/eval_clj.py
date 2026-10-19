"""
Evaluates a generated Clojure program (.clj).
"""

import os
from pathlib import Path
from .safe_subprocess import run
from .libeval import run_without_exn


def eval_script(path: Path):
    result = run(["clojure", "-J-Dclojure.main.report=stderr", "-M", str(path)])

    if result.timeout:
        status = "Timeout"
    elif result.exit_code != 0:
        status = "Exception"
    elif "\n0 failures, 0 errors.\n" in result.stdout:
        status = "OK"
    else:  # test failure
        status = "Exception"

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


if __name__ == "__main__":
    main()
