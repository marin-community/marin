# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile Lean proofs in the grader's own machine, answering like the NeMo Skills sandbox.

``LocalLeanSandbox`` stands in for ``sandbox.SandboxClient`` when the grader runs in the image
``lean.Dockerfile`` builds. For ``language="lean4"`` it returns the server's response fields:
``process_status`` of ``completed`` on exit status 0, ``failed`` otherwise and ``timeout`` past the
deadline; ``stdout`` and ``stderr`` cut to ``max_output_characters`` with an ``<output cut>`` marker;
and ``output_truncated`` when either stream was cut. Diagnostics name the proof file under
``LEAN_PROJECT_DIR``, the path ``lean_feedback.parse_error`` expects.
"""

import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

LEAN_PROJECT_DIR = Path("/lean4/my_project")
"""The Mathlib project ``lean.Dockerfile`` builds; its ``scratch`` directory is world-writable."""
ELAN_HOME = Path("/opt/elan")
LAKE = ELAN_HOME / "bin" / "lake"
SCRATCH_DIR_NAME = "scratch"
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
"""Where ``lake env`` finds git; the toolchain proxies come from ``lake``'s own directory."""
OUTPUT_CUT = "<output cut>"

type SandboxResponse = dict[str, str | bool]


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
    ) -> SandboxResponse:
        if language != "lean4" or session_id is not None:
            raise ValueError(f"LocalLeanSandbox compiles stateless lean4 only, not {language!r} session {session_id!r}")
        identity = {} if self.user is None else {"user": self.user, "group": self.user, "extra_groups": []}
        environment = {"PATH": f"{self.lake.parent}:{SYSTEM_PATH}", "ELAN_HOME": str(self.elan_home)}
        scratch = self.project_dir / SCRATCH_DIR_NAME
        # The compiler's streams go to files so a proof that prints without limit cannot exhaust the
        # grader's memory; only the first ``max_output_characters`` plus one are read back.
        with (
            tempfile.NamedTemporaryFile("w", dir=scratch, suffix=".lean") as proof,
            tempfile.TemporaryFile("w+", dir=scratch) as stdout,
            tempfile.TemporaryFile("w+", dir=scratch) as stderr,
        ):
            proof.write(code)
            proof.flush()
            os.chmod(proof.name, 0o644)
            process = subprocess.Popen(
                (str(self.lake), "env", "--dir", str(self.project_dir), "lean", proof.name),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                cwd=self.project_dir,
                env=environment,
                start_new_session=True,
                **identity,
            )
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = "timeout"
            else:
                status = "completed" if process.returncode == 0 else "failed"
            out, err = (_read_head(stream, max_output_characters) for stream in (stdout, stderr))
        if status == "timeout":
            err += f"Execution timed out after {timeout_seconds} seconds\n"
        result: SandboxResponse = {
            "process_status": status,
            "stdout": _cut(out, max_output_characters),
            "stderr": _cut(err, max_output_characters),
        }
        if OUTPUT_CUT in result["stdout"] or OUTPUT_CUT in result["stderr"]:
            result["output_truncated"] = True
        return result


def _read_head(stream, limit: int) -> str:
    stream.seek(0)
    return stream.read(limit + 1)
