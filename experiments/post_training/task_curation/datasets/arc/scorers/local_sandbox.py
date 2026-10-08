# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run NVARC's sandbox requests as subprocesses of the grader.

``nvarc.grade_inductive_arc`` executes the submitted transform through a NeMo Skills sandbox
server (``sandbox.SandboxClient``). The grader machine is already a sandbox, so ``LocalSandbox`` takes
the client's place and runs the program with ``python -c``, answering with the server's response for
``language="python"``: ``completed`` whatever the exit status, ``timeout`` with empty stdout when the
deadline passes, and each stream cut to ``max_output_characters`` with an ``<output cut>`` marker.
"""

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

OUTPUT_CUT = "<output cut>"


def _cut(text: str, limit: int) -> str:
    return text[:limit] + OUTPUT_CUT if len(text) > limit else text


@dataclass(frozen=True)
class LocalSandbox:
    """Drop-in for ``SandboxClient`` that runs each program in a fresh interpreter.

    ``user`` is the uid and gid the program runs as. A root grader should pass an unprivileged id such as
    65534 (``nobody``) and make ``/tests`` unreadable to it, so the program cannot read the expected
    output; ``None`` runs the program as the grader's own user.
    """

    user: int | None

    def execute(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: float,
        session_id: str | None = None,
        max_output_characters: int = 1000,
    ) -> dict[str, Any]:
        if language != "python" or session_id is not None:
            raise ValueError(f"LocalSandbox runs stateless python only, not {language!r} session {session_id!r}")
        identity = {} if self.user is None else {"user": self.user, "group": self.user, "extra_groups": []}
        process = subprocess.Popen(
            (sys.executable, "-c", code),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tempfile.gettempdir(),
            start_new_session=True,
            **identity,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return {
                "process_status": "timeout",
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds\n",
            }
        result: dict[str, Any] = {
            "process_status": "completed",
            "stdout": _cut(stdout, max_output_characters),
            "stderr": _cut(stderr, max_output_characters),
        }
        if OUTPUT_CUT in result["stdout"] or OUTPUT_CUT in result["stderr"]:
            result["output_truncated"] = True
        return result
