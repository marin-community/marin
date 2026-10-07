# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private bridge to a source's RewardKit runner, without exporting provider diagnostics."""

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
VERDICT_FILENAME = "source-verdict.json"
SOURCE_PYTHON_FILES = ("deterministic_gate", "sitecustomize.py", "verifier.py")
SOURCE_JSON_FILES = (
    "verifier_data.json",
    "criterion_partition.json",
    "deterministic_criteria.json",
    "semantic_criteria.json",
)


def malformed_source(tests: Path) -> dict | None:
    """Identify definite source syntax defects without executing code or a judge."""
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
            # Exception text can quote private source or references. Export only
            # the known filename and type; I/O and runtime failures remain infra.
            return {
                "status": "invalid_task",
                "reward": 0.0,
                "detail": {"error": "malformed_source_suite", "file": name, "error_type": type(error).__name__},
            }
    return None


def run_source(tests: Path, logs: Path, timeout: float) -> dict:
    """Keep source reward semantics; failed producers never become negative controls."""
    invalid = malformed_source(tests)
    if invalid is not None:
        return invalid
    # Original scripts retain absolute /tests, /app and /logs/verifier paths.
    # The output file is read only after successful producer completion.
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
        if sys.version_info[:2] != (3, 12) or version("harbor-rewardkit") != "0.1.4":
            raise RuntimeError("source_runtime_mismatch")
        verdict = run_source(Path(os.environ["VERIFYIT_TESTS_DIR"]), Path("/logs/verifier"), SOURCE_TIMEOUT)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        # Provider failures can include request headers or candidate text. Keep
        # detailed source logs inside the private job; export only the error type.
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"error": type(error).__name__}}
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / VERDICT_FILENAME).write_text(json.dumps(verdict))


if __name__ == "__main__":
    main()
