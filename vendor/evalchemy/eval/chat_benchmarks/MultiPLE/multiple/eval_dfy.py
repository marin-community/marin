from pathlib import Path
from .safe_subprocess import run
from .generic_eval import main


# 0 – success
# 1 – invalid command-line arguments
# 2 – syntax, parse, or name or type resolution errors
# 3 – compilation errors
# 4 – verification errors


def eval_script(path: Path):
    r = run(["dafny", "run", str(path)])
    if r.timeout:
        status = "Timeout"
    elif r.exit_code == 0:
        status = "OK"
    elif r.exit_code == 2:
        status = "SyntaxError"
    elif r.exit_code == 3:
        status = "CompilationError"
    elif r.exit_code == 4:
        status = "VerificationError"
    else:
        status = "Exception"
    return {
        "status": status,
        "exit_code": r.exit_code,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
