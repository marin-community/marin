# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a Multi-LCB reply with the vendored harness: 1 when the extracted program passes every test.

As upstream does, the reply's reasoning is truncated before the first fenced block is extracted, and a run
that times out is evaluated again, up to ``EVALUATION_ATTEMPTS`` in all. ``/tests/config.json`` holds the
language and a deadline across all attempts; past it the reply scores 0. The harness can print, so stdout
points at stderr while it runs and carries only the reward. After each program the harness runs
``pkill -f <source file>``, which the worker image lacks, so the script puts :data:`PKILL` first on ``PATH``.
"""

import json
import os
import signal
import sys
import tempfile
import zlib
from pathlib import Path

sys.path.insert(0, "/tests")

from lcb_runner.evaluation.testing_plang import ExecutionErrorMeta, Status, eval_plang_code
from lcb_runner.utils.extraction_utils import extract_code, truncate_reasoning

ANSWER_PATH = Path("/app/answer.txt")
CONFIG_PATH = Path("/tests/config.json")
TESTS_PATH = Path("/tests/tests.json.zlib")
TEST_TIMEOUT = 6
"""Upstream's default per-test timeout, in seconds."""
EVALUATION_ATTEMPTS = 3
"""Upstream's default ``eval_restarts``."""
RETRIED_ERRORS = (Status.TimeoutExpired, Status.BuildTimeOut)
PKILL = """#!/usr/bin/env python3
# pkill -f PATTERN: send SIGTERM to every other process whose command line matches PATTERN.
import os, re, signal, sys

pattern = re.compile(sys.argv[-1].encode())
for entry in os.listdir("/proc"):
    if not entry.isdigit() or int(entry) == os.getpid():
        continue
    try:
        with open(f"/proc/{entry}/cmdline", "rb") as cmdline:
            if pattern.search(cmdline.read().replace(b"\\0", b" ")):
                os.kill(int(entry), signal.SIGTERM)
    except OSError:
        pass  # The process exited or is not ours to signal.
"""


class DeadlineExceeded(BaseException):
    """Raised by the alarm. A ``BaseException`` passes the harness's ``except Exception`` around each run."""


def _raise_deadline(signum: int, frame: object) -> None:
    raise DeadlineExceeded


def install_pkill() -> None:
    directory = Path(tempfile.mkdtemp())
    (directory / "pkill").write_text(PKILL)
    (directory / "pkill").chmod(0o755)
    os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ['PATH']}"


def passes(code: str, tests: dict[str, list[str]], language: str) -> bool:
    for _ in range(EVALUATION_ATTEMPTS):
        scores, metadata = eval_plang_code(code, tests["inputs"], tests["outputs"], plang=language, timeout=TEST_TIMEOUT)
        if not (isinstance(metadata, ExecutionErrorMeta) and metadata.error in RETRIED_ERRORS):
            break
    print(repr(metadata))
    return all(score.value > 0 for score in scores)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    tests = json.loads(zlib.decompress(TESTS_PATH.read_bytes()))
    code = extract_code(truncate_reasoning(ANSWER_PATH.read_text()))
    install_pkill()
    stdout = os.dup(1)
    os.dup2(2, 1)
    signal.signal(signal.SIGALRM, _raise_deadline)
    signal.alarm(config["deadline"])
    try:
        score = 1.0 if passes(code, tests, config["language"]) else 0.0
    except DeadlineExceeded:
        print(f"Deadline of {config['deadline']} seconds exceeded")
        score = 0.0
    signal.alarm(0)
    sys.stdout.flush()
    os.dup2(stdout, 1)
    print(score)


if __name__ == "__main__":
    main()
