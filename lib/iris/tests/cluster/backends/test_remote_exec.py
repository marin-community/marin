# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry behavior of run_streaming_with_retry.

These exercise realistic transport failure modes (transient gcloud crash vs a
genuine command failure) through the public retry helper.
"""

import pytest
from iris.cluster.backends import remote_exec
from iris.cluster.backends.remote_exec import run_streaming_with_retry


class _FakeProc:
    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stdout = iter(lines)
        self.returncode = returncode
        self.args = "gcloud compute tpus tpu-vm ssh"

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


class _ScriptedRemoteExec:
    """Replays a scripted list of (output_lines, returncode) across run_streaming calls."""

    def __init__(self, script: list[tuple[list[str], int]]) -> None:
        self._script = script
        self.calls = 0

    def run(self, command, timeout=None):
        raise NotImplementedError

    def run_streaming(self, command: str) -> _FakeProc:
        lines, returncode = self._script[self.calls]
        self.calls += 1
        return _FakeProc(lines, returncode)

    @property
    def address(self) -> str:
        return "fake-host"

    @property
    def zone(self) -> str:
        return ""


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(remote_exec.time, "sleep", lambda _seconds: None)


def test_retries_transient_gcloud_database_locked_crash():
    # gcloud's local credential SQLite DB can fail its lock under concurrent
    # invocations and abort (exit 1) before the remote command runs. The host is
    # healthy, so the bootstrap should retry rather than report a failure.
    conn = _ScriptedRemoteExec(
        [
            (["[iris-init] Starting", "ERROR: gcloud crashed (OperationalError): database is locked"], 1),
            (["[iris-init] Worker is healthy", "[iris-init] Bootstrap complete"], 0),
        ]
    )

    result = run_streaming_with_retry(conn, "bash -c bootstrap")

    assert result.returncode == 0
    assert conn.calls == 2


def test_does_not_retry_ordinary_command_failure():
    # A genuine non-zero exit with no transient signature must surface
    # immediately on the first attempt, not be masked by retries.
    conn = _ScriptedRemoteExec([(["bootstrap step failed", "fatal: real error"], 1)])

    result = run_streaming_with_retry(conn, "bash -c bootstrap")

    assert result.returncode == 1
    assert conn.calls == 1
