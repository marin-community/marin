# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native-tool execution with retained commands and output artifacts."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
)
COMMAND_TIMEOUT = 300


def execute(argv: list[str], work: Path, output: str) -> Path:
    """Execute a native data operation and retain its command, status, and streams."""
    destination = work / output
    error = work / (output + ".stderr")
    environment = dict(os.environ)
    environment.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    start = time.monotonic()
    status = {"argv": argv, "exit_code": None, "stdout": destination.name, "stderr": error.name}
    try:
        with destination.open("wb") as stdout, error.open("wb") as stderr:
            with subprocess.Popen(
                argv, cwd=work, env=environment, stdout=stdout, stderr=stderr, start_new_session=True
            ) as process:
                try:
                    status["exit_code"] = process.wait(timeout=COMMAND_TIMEOUT)
                finally:
                    # Workflow engines can own grandchildren; terminate the whole command group.
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if status["exit_code"] != 0:
            raise subprocess.CalledProcessError(status["exit_code"], argv)
    except (OSError, subprocess.SubprocessError) as failure:
        status["error"] = str(failure)
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        with (work / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps(status) + "\n")
    return destination
