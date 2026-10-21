# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a task program: a Python module that writes one TaskSpec, in a child process of the caller's environment.

The program runs as ``python3 program.py <task id> <output path>`` with the caller's interpreter, so it can import
``taskcompendium`` and ``verifyit``. Its environment is ``PATH`` alone, so the caller's variables and credentials
stay in the caller.
"""

import asyncio
import contextlib
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from taskcompendium.models import TaskSpec

PROGRAM_FILE = "program.py"
TASK_FILE = "task.json"
PROGRAM_ENV = {"PATH": os.defpath}
"""The program's entire environment, ``PATH`` alone, so the driver's variables and credentials stay in the driver."""


# NOTE: The caller holds stdout and stderr in memory, so a program printing without limit can exhaust memory.
# Truncate them or stream them to files if that happens.
@dataclass(frozen=True)
class ProgramRun:
    """A run of a task program.

    ``exit_code`` is None after a timeout. ``failure`` is None exactly when ``task`` is set.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    task: TaskSpec | None
    failure: str | None


async def run_program(source: str, task_id: str, timeout: float) -> ProgramRun:
    """Run ``source`` as ``python3 program.py <task_id> <output path>`` and parse the TaskSpec it writes.

    The program starts in a new session. On timeout the driver kills the whole process group, taking
    the program's own children down with it.
    """
    with tempfile.TemporaryDirectory(prefix="task-program-") as workdir:
        program, output = Path(workdir) / PROGRAM_FILE, Path(workdir) / TASK_FILE
        program.write_text(source)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(program),
            task_id,
            str(output),
            cwd=workdir,
            env=PROGRAM_ENV,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError:
            # The group may have exited between the timeout and the kill; that is still a timeout.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return ProgramRun(None, "", "", None, f"the program did not finish within {timeout}s")
        code = process.returncode
        out, err = stdout.decode(errors="replace"), stderr.decode(errors="replace")
        if code != 0:
            return ProgramRun(code, out, err, None, f"the program exited with status {code}")
        if not output.is_file():
            return ProgramRun(code, out, err, None, "the program wrote no file at the output path")
        try:
            task = TaskSpec.model_validate_json(output.read_bytes())
        except ValidationError as error:
            return ProgramRun(code, out, err, None, f"the output is not a TaskSpec: {error}")
        return ProgramRun(code, out, err, task, None)
