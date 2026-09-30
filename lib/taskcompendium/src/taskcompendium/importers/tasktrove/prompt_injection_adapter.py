#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept the private checker mount paths and emit a script result record."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _result(verifier: Path, status: str, reward: float | None = None, error: str | None = None) -> None:
    payload: dict[str, str | float] = {"status": status}
    if reward is not None:
        payload["reward"] = reward
    if error is not None:
        payload["error"] = error
    (verifier / "result.json").write_text(json.dumps(payload) + "\n")


def main(tests: Path, workspace: Path, verifier: Path, timeout: float) -> None:
    submission = json.loads((verifier / "submission.json").read_text())
    answer = submission["answer"]
    if not isinstance(answer, str):
        _result(verifier, "invalid_task", error="Direct text submission is missing")
        return

    # The source checker reads an answer file. Populate it only in the
    # verifier's disposable workspace copy, never in the agent environment.
    (workspace / "answer.txt").write_text(answer)
    logs = verifier / "source_logs"
    logs.mkdir()
    environment = {
        **os.environ,
        "TASKTROVE_TESTS_DIR": str(tests),
        "TASKTROVE_WORKSPACE": str(workspace),
        "TASKTROVE_LOGS_DIR": str(logs),
    }
    try:
        completed = subprocess.run(
            [sys.executable, str(tests / "source_checker.py")],
            cwd=workspace,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _result(verifier, "scored", 0.0, "timeout")
        return
    try:
        payload = json.loads((logs / "reward.json").read_text())
        reward = payload["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("Source checker reward is not numeric")
        reward = float(reward)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _result(verifier, "infra_error", error=f"Source checker exited {completed.returncode} without a reward: {error}")
        return
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        _result(verifier, "scored", 0.0, "reward_out_of_range")
        return
    _result(verifier, "scored", reward)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]))
