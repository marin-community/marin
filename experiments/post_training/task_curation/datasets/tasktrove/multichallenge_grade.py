# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a source's RewardKit judge suite and report its reward as a verifyit script verdict.

verifyit's script mode runs this file with the task's tests directory in ``VERIFYIT_TESTS_DIR``.
The source's ``test.sh`` copies the answer to ``/app/response.txt``, runs RewardKit with the
source's judge, and writes ``/logs/verifier/reward.json``. Provider errors can quote request
headers or the candidate text, so the verdict reports only error types; the source's own logs stay
in the grading machine.
"""

import json
import math
import os
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from verifyit.execution.command import run_command

SOURCE_TIMEOUT = 600.0
SOURCE_JUDGE = "together_ai/Qwen/Qwen3.5-9B"
REWARDKIT_VERSION = "0.1.4"
VERDICT_FILENAME = "source-verdict.json"
SOURCE_PYTHON_FILES = ("deterministic_gate", "sitecustomize.py", "verifier.py")
SOURCE_JSON_FILES = (
    "verifier_data.json",
    "criterion_partition.json",
    "deterministic_criteria.json",
    "semantic_criteria.json",
)


def malformed_source(tests: Path) -> dict | None:
    """An ``invalid_task`` verdict for a source file that does not parse, found without running code or a judge."""
    for name in (*SOURCE_PYTHON_FILES, *SOURCE_JSON_FILES, "judge.toml"):
        path = tests / name
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
            if name in SOURCE_PYTHON_FILES:
                compile(content, name, "exec")
            elif name in SOURCE_JSON_FILES:
                json.loads(content)
            else:
                tomllib.loads(content.decode())
        except (SyntaxError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
            # Exception text can quote hidden source files or references; report only the file and error type.
            return {
                "status": "invalid_task",
                "reward": 0.0,
                "detail": {"error": "malformed_source_suite", "file": name, "error_type": type(error).__name__},
            }
    return None


def run_source(tests: Path, logs: Path, timeout: float) -> dict:
    """Score with the source's own reward; a failed run is an infrastructure error, never a zero reward."""
    invalid = malformed_source(tests)
    if invalid is not None:
        return invalid
    # The source scripts use absolute /tests, /app and /logs/verifier paths. The reward file is read
    # only after the script succeeds.
    completed = run_command(
        ["bash", str(tests / "test.sh")],
        cwd=Path.cwd(),
        env={**os.environ, "REWARDKIT_JUDGE": SOURCE_JUDGE},
        timeout=timeout,
    )
    if completed.timed_out or completed.returncode != 0:
        return {
            "status": "infra_error",
            "reward": 0.0,
            "detail": {
                "error": "source_producer_failed",
                "exit_code": completed.returncode,
                "timed_out": completed.timed_out,
            },
        }
    reward = json.loads((logs / "reward.json").read_text())
    if not isinstance(reward, dict) or set(reward) != {"reward"}:
        raise ValueError("Unexpected source reward shape")
    value = reward["reward"]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid source reward")
    return {"status": "scored", "reward": value, "detail": {"source": "harbor-rewardkit"}}


def main() -> None:
    try:
        if sys.version_info[:2] != (3, 12) or version("harbor-rewardkit") != REWARDKIT_VERSION:
            raise RuntimeError("source_runtime_mismatch")
        verdict = run_source(Path(os.environ["VERIFYIT_TESTS_DIR"]), Path("/logs/verifier"), SOURCE_TIMEOUT)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"error": type(error).__name__}}
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / VERDICT_FILENAME).write_text(json.dumps(verdict))


if __name__ == "__main__":
    main()
