#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt a private TaskTrove calendar script to TaskCompendium's result protocol."""

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_LOG_BYTES = 64 * 1024
READ_CHUNK_BYTES = 8 * 1024


def _result(verifier: Path, status: str, reward: float | None = None, error: str | None = None) -> None:
    payload: dict[str, str | float] = {"status": status}
    if reward is not None:
        payload["reward"] = reward
    if error is not None:
        payload["error"] = error
    (verifier / "result.json").write_text(json.dumps(payload) + "\n")


def _run_checker(
    command: list[str], *, cwd: Path, environment: dict[str, str], timeout: float, logs: Path
) -> int | None:
    """Run the checker while retaining bounded private stdout and stderr logs."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    streams = {process.stdout: logs / "stdout.log", process.stderr: logs / "stderr.log"}
    totals = {stream: 0 for stream in streams}
    started = time.monotonic()

    with selectors.DefaultSelector() as selector:
        for stream, destination in streams.items():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, (destination.open("wb"), stream))
        timed_out = False
        while selector.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for key, _ in selector.select(max(0.0, min(remaining, 0.1))):
                output, stream = key.data
                try:
                    chunk = os.read(stream.fileno(), READ_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    output.close()
                    continue
                retained = min(len(chunk), MAX_LOG_BYTES - totals[stream])
                if retained > 0:
                    output.write(chunk[:retained])
                    totals[stream] += retained
        if timed_out:
            return None
    try:
        return process.wait(timeout=max(0.0, timeout - (time.monotonic() - started)))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return None


def main(tests: Path, workspace: Path, verifier: Path, timeout: float) -> None:
    submission = json.loads((verifier / "submission.json").read_text())
    answer = submission["answer"]
    if not isinstance(answer, str):
        _result(verifier, "invalid_task", error="Direct text submission is missing")
        return

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
        return_code = _run_checker(
            [sys.executable, str(tests / "source_checker.py")],
            cwd=workspace,
            environment=environment,
            timeout=timeout,
            logs=logs,
        )
    except OSError as error:
        _result(verifier, "infra_error", error=f"Could not start source checker: {error}")
        return
    if return_code is None:
        _result(verifier, "infra_error", error="Source checker timed out")
        return
    if return_code != 0:
        _result(verifier, "infra_error", error=f"Source checker exited with status {return_code}")
        return
    try:
        payload = json.loads((logs / "reward.json").read_text())
        reward = payload["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("Source checker reward is not numeric")
        reward = float(reward)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _result(verifier, "infra_error", error=f"Source checker did not write a valid reward: {error}")
        return
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        _result(verifier, "infra_error", error="Source checker reward is outside [0, 1]")
        return
    _result(verifier, "scored", reward)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]))
