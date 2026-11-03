from pathlib import Path
from .safe_subprocess import run


def eval_script(path: Path):
    r = run(["luau-analyze", str(path)])
    if r.timeout:
        status = "Timeout"
    elif r.exit_code == 0:
        r = run(["luau", str(path)])
        if r.timeout:
            status = "Timeout"
        elif r.exit_code == 0:
            status = "OK"
        else:
            status = "Exception"
    elif "SyntaxError" in r.stderr:
        status = "SyntaxError"
    else:
        status = "TypeError"
    return {
        "status": status,
        "exit_code": r.exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
