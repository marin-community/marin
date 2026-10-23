import os
import subprocess
from pathlib import Path


def eval_script(path: Path):
    try:
        # Assumes exit-code 0 is all okay
        # Run R on the file, capturing stderr
        output = subprocess.run(["Rscript", str(path)], capture_output=True, timeout=5)
        if output.returncode == 0:
            status = "OK"
        else:
            outmessage = str(output)
            if "unexpected" in outmessage:
                status = "SyntaxError"
            elif "err=b''" in outmessage:
                status = "AssertionError"
            else:
                status = "Exception"
        returncode = output.returncode
    except subprocess.TimeoutExpired as exc:
        status = "Timeout"
        output = exc
        returncode = -1
    except subprocess.CalledProcessError as exc:
        status = "Exception"
        returncode = exc.returncode
        output = exc
    return {"status": status, "exit_code": returncode, "stdout": output.stdout, "stderr": output.stderr}
