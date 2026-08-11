# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARD_KILL_MESSAGE = "pytest-timeout signal handler did not stop the test; hard-killing process"


def _run_hanging_test(tmp_path: Path, test_body: str) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_hang.py"
    test_file.write_text(test_body)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "conftest",
            "-c",
            str(REPO_ROOT / "pyproject.toml"),
            "-o",
            "addopts=",
            "--timeout=0.1",
            "-q",
            str(test_file),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="pytest-timeout uses signals only on POSIX")
def test_timeout_signal_failure_stops_hanging_test(tmp_path: Path):
    result = _run_hanging_test(
        tmp_path,
        """
import threading


def test_hangs():
    threading.Event().wait()
""",
    )

    assert result.returncode != 0
    assert "Timeout (>0.1s) from pytest-timeout" in result.stdout
    assert HARD_KILL_MESSAGE not in result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="pytest-timeout uses signals only on POSIX")
def test_timeout_hard_kills_test_that_swallows_signal_failure(tmp_path: Path):
    result = _run_hanging_test(
        tmp_path,
        """
import threading


def test_swallows_timeout():
    while True:
        try:
            threading.Event().wait()
        except BaseException:
            pass
""",
    )

    assert result.returncode != 0
    assert HARD_KILL_MESSAGE in result.stderr
