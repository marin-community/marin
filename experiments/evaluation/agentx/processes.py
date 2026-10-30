# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run isolated CLIs with retained logs and process-group cleanup."""

import os
import signal
import subprocess
from pathlib import Path


def logged_process(command: list[str], environment: dict[str, str], log: Path) -> subprocess.Popen:
    """Keep file artifacts and native Iris logs, including interrupted attempts."""
    return subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", '"$@" 2>&1 | tee "$AGENTX_LOG_FILE"', "agentx-log", *command],
        env=environment | {"AGENTX_LOG_FILE": str(log)},
        start_new_session=True,
    )


def stop_group(process: subprocess.Popen) -> None:
    """Reap the command and stop its descendants before releasing the allocation."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_logged(command: list[str], environment: dict[str, str], log: Path, timeout: int) -> None:
    with logged_process(command, environment, log) as process:
        try:
            result = process.wait(timeout=timeout)
            if result:
                raise subprocess.CalledProcessError(result, command)
        finally:
            stop_group(process)
