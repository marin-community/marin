import os
import subprocess
from pathlib import Path
from .safe_subprocess import run
import sys
import re

ENABLE_SYNTAX_CHECK = False


def eval_script(path: Path):
    result = run(["rdmd", "-unittest", str(path)], timeout_seconds=15)
    if "might not be correctly installed" in result.stderr:
        raise Exception("D is not correctly installed")

    if result.timeout:
        status = "Timeout"
    elif result.exit_code == 0:
        status = "OK"
    elif "Error:" in result.stderr:
        status = "SyntaxError"
    else:
        status = "Exception"

    return {
        "status": status,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


DIR = "d-keep-code_davinci_001_temp_0.2"
