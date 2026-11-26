"""Task-local Lean compiler process. The task image supplies Lake and its project."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

COMPILER_OUTPUT_LIMIT_BYTES = 65536


def main() -> None:
    data = json.load(sys.stdin)
    with TemporaryDirectory(prefix="proof-", dir=data["project"]) as directory:
        proof = Path(directory) / "Proof.lean"
        proof.write_text(data["proof"])
        with TemporaryFile() as stdout, TemporaryFile() as stderr:
            process = subprocess.Popen(
                ["lake", "env", "lean", str(proof)],
                cwd=data["project"],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            reason = "exited"
            try:
                process.wait(timeout=data["timeout"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                reason = "timed_out"
            stdout.seek(0)
            stderr.seek(0)
            output = stdout.read(COMPILER_OUTPUT_LIMIT_BYTES + 1)
            errors = stderr.read(COMPILER_OUTPUT_LIMIT_BYTES + 1)
            print(
                json.dumps(
                    {
                        "reason": reason,
                        "exit_code": process.returncode if reason == "exited" else None,
                        "stdout": output[:COMPILER_OUTPUT_LIMIT_BYTES].decode(errors="replace"),
                        "stderr": errors[:COMPILER_OUTPUT_LIMIT_BYTES].decode(errors="replace"),
                        "stdout_truncated": len(output) > COMPILER_OUTPUT_LIMIT_BYTES,
                        "stderr_truncated": len(errors) > COMPILER_OUTPUT_LIMIT_BYTES,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
