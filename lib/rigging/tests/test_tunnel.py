# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import signal
import subprocess
from typing import cast

import pytest
import rigging.tunnel as tunnel


class _ProcessStub:
    pid = 123

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("process", timeout)
        return self.returncode

    def send_signal(self, _signal: int) -> None:
        raise AssertionError("process-group signaling should not fall back to the leader")


def test_terminate_process_group_waits_for_graceful_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _ProcessStub()
    signals: list[int] = []

    monkeypatch.setattr(tunnel, "_process_group", lambda _process: 456)

    def signal_group(_group_id: int, signum: int) -> None:
        signals.append(signum)
        process.returncode = -signum

    monkeypatch.setattr(tunnel.os, "killpg", signal_group)

    tunnel.terminate_process_group(cast(subprocess.Popen, process), grace_period=1)

    assert signals == [signal.SIGTERM]
    assert process.wait_timeouts == [1]


def test_terminate_process_group_escalates_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    process = _ProcessStub()
    signals: list[int] = []

    monkeypatch.setattr(tunnel, "_process_group", lambda _process: 456)

    def signal_group(_group_id: int, signum: int) -> None:
        signals.append(signum)
        if signum == signal.SIGKILL:
            process.returncode = -signum

    monkeypatch.setattr(tunnel.os, "killpg", signal_group)

    tunnel.terminate_process_group(cast(subprocess.Popen, process), grace_period=0.01)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.wait_timeouts == [0.01, None]
