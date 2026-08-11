# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_hanging_test(tmp_path: Path, signal_seen: Path) -> subprocess.CompletedProcess[str]:
    test_file = tmp_path / "test_hang.py"
    test_file.write_text(
        f"""
import threading
from pathlib import Path


def test_swallows_timeout():
    while True:
        try:
            threading.Event().wait()
        except BaseException:
            Path({str(signal_seen)!r}).touch()
"""
    )

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
        timeout=15,
        check=False,
    )


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="pytest-timeout uses signals only on POSIX")
def test_timeout_hard_kills_test_that_swallows_signal_failure(tmp_path: Path):
    signal_seen = tmp_path / "signal_seen"
    result = _run_hanging_test(tmp_path, signal_seen)

    assert signal_seen.exists()
    assert result.returncode != 0
