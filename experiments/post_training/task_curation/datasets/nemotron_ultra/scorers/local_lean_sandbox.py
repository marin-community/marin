# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile Lean proofs as subprocesses of the grader, answering like the NeMo Skills sandbox.

``skyrl_gym.envs.nemotron_ultra.lean.verify_lean_attempt`` compiles a proof through a NeMo Skills
sandbox server (``sandbox.SandboxClient``). The grader machine is already a sandbox with Lean and
Mathlib installed by ``lean.Dockerfile``, so ``LocalLeanSandbox`` takes the client's place: it writes
the proof to a ``.lean`` file in the Mathlib project and runs ``lake env lean`` on it, answering with
the server's response for ``language="lean4"``: ``completed`` on exit status 0, ``failed`` otherwise,
``timeout`` when the deadline passes, and each stream cut to ``max_output_characters`` with an
``<output cut>`` marker. The file lives under the project path so ``lean_feedback.parse_error`` finds
the diagnostics' ``/lean4/my_project/...`` prefix.
"""

import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEAN_PROJECT_DIR = Path("/lean4/my_project")
"""The Mathlib project ``lean.Dockerfile`` builds; its ``scratch`` directory is world-writable."""
ELAN_HOME = Path("/opt/elan")
LAKE = ELAN_HOME / "bin" / "lake"
SCRATCH_DIR_NAME = "scratch"
OUTPUT_CUT = "<output cut>"


def _cut(text: str, limit: int) -> str:
    return text[:limit] + OUTPUT_CUT if len(text) > limit else text


@dataclass(frozen=True)
class LocalLeanSandbox:
    """Drop-in for ``SandboxClient`` that compiles each proof with the image's Lean toolchain.

    ``user`` is the uid and gid the compiler runs as; ``None`` runs it as the grader's own user. The
    proof file is readable by everyone, so an unprivileged id such as 65534 (``nobody``) can compile it.
    """

    user: int | None
    project_dir: Path = LEAN_PROJECT_DIR
    lake: Path = LAKE
    elan_home: Path = ELAN_HOME

    def execute(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: float,
        session_id: str | None = None,
        max_output_characters: int = 1000,
    ) -> dict[str, Any]:
        if language != "lean4" or session_id is not None:
            raise ValueError(f"LocalLeanSandbox compiles stateless lean4 only, not {language!r} session {session_id!r}")
        identity = {} if self.user is None else {"user": self.user, "group": self.user, "extra_groups": []}
        environment = {
            "PATH": f"{self.lake.parent}:{os.environ['PATH']}",
            "ELAN_HOME": str(self.elan_home),
        }
        with tempfile.NamedTemporaryFile("w", dir=self.project_dir / SCRATCH_DIR_NAME, suffix=".lean") as proof:
            proof.write(code)
            proof.flush()
            os.chmod(proof.name, 0o644)
            process = subprocess.Popen(
                (str(self.lake), "env", "--dir", str(self.project_dir), "lean", proof.name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_dir,
                env=environment,
                start_new_session=True,
                **identity,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                return _response(
                    "timeout",
                    stdout,
                    stderr + f"Execution timed out after {timeout_seconds} seconds\n",
                    max_output_characters,
                )
        return _response("completed" if process.returncode == 0 else "failed", stdout, stderr, max_output_characters)


def _response(process_status: str, stdout: str, stderr: str, max_output_characters: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "process_status": process_status,
        "stdout": _cut(stdout, max_output_characters),
        "stderr": _cut(stderr, max_output_characters),
    }
    if OUTPUT_CUT in result["stdout"] or OUTPUT_CUT in result["stderr"]:
        result["output_truncated"] = True
    return result
