# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Real worker deadlines and descendant cleanup, including abnormal exits."""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest
from tasktrove_verify.execution.command import run_command
from tasktrove_verify.execution.worker import call_bounded
from tasktrove_verify.grade import InvalidTask

from .bounded_grading import grading_with_child, grading_with_large_result

pytestmark = pytest.mark.usefixtures("importable_grading_modules")


@pytest.fixture
def receipt(tmp_path):
    path = tmp_path / "child.pid"
    try:
        yield path
    finally:
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_bounded_grading_drains_large_result_without_exposing_worker_output(capfd):
    result = call_bounded(grading_with_large_result, "private grading input", timeout=5)

    assert result == b"x" * (1024 * 1024)
    assert capfd.readouterr() == ("", "")


def assert_child_stopped(receipt):
    assert receipt.exists()
    pid = int(receipt.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not state or state.startswith("Z"):
            break
        threading.Event().wait(0.01)
    assert not state or state.startswith("Z"), f"grading descendant {pid} survived"


@pytest.mark.parametrize("behavior", ["success", "invalid", "crash", "unserializable", "timeout", "interrupt"])
def test_bounded_grading_reaps_descendants_on_every_exit(receipt, behavior):
    interrupter = None
    cancelled = threading.Event()
    if behavior == "interrupt":

        def interrupt():
            deadline = time.monotonic() + 5
            while not receipt.exists() and not cancelled.is_set() and time.monotonic() < deadline:
                cancelled.wait(0.01)
            if receipt.exists() and not cancelled.is_set():
                os.kill(os.getpid(), signal.SIGINT)

        interrupter = threading.Thread(target=interrupt)
        interrupter.start()
    try:
        if behavior == "success":
            assert call_bounded(grading_with_child, str(receipt), behavior, timeout=5) == {"score": 0.5}
        else:
            expected = {
                "invalid": InvalidTask,
                "crash": RuntimeError,
                "unserializable": RuntimeError,
                "timeout": TimeoutError,
                "interrupt": KeyboardInterrupt,
            }[behavior]
            with pytest.raises(expected):
                call_bounded(grading_with_child, str(receipt), behavior, timeout=1 if behavior == "timeout" else 5)
    finally:
        cancelled.set()
        if interrupter:
            interrupter.join(timeout=6)
    assert_child_stopped(receipt)


def test_command_exited_parent_does_not_leave_descendant_holding_output(receipt):
    program = (
        "import subprocess,sys;from pathlib import Path;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "Path(sys.argv[1]).write_text(str(p.pid))"
    )
    result = run_command([sys.executable, "-c", program, str(receipt)], receipt.parent, 0.5)
    assert result.timed_out
    assert_child_stopped(receipt)
