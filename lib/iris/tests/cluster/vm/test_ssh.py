# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SSH utilities and connection implementations."""

import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from iris.cluster.vm.ssh import (
    FakePopen,
    InMemorySshConnection,
    check_health,
    connection_available,
    run_streaming_with_retry,
    shutdown_worker,
    wait_for_connection,
)


def test_in_memory_connection_returns_success_for_echo():
    """InMemorySshConnection returns success for echo commands."""
    conn = InMemorySshConnection(_address="10.0.0.1")
    result = conn.run("echo ok")
    assert result.returncode == 0
    assert result.stdout == "ok\n"


def test_fake_popen_yields_bootstrap_output():
    """FakePopen yields simulated bootstrap output lines."""
    popen = FakePopen()
    lines = list(popen.stdout)
    assert len(lines) == 7
    assert "[iris-init] Starting Iris worker bootstrap\n" in lines
    assert "[iris-init] Bootstrap complete\n" in lines


def test_connection_available_returns_true_on_success():
    """connection_available returns True when command succeeds."""
    conn = InMemorySshConnection(_address="10.0.0.1")
    assert connection_available(conn) is True


def test_connection_available_returns_false_on_failure():
    """connection_available returns False when command fails."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=1)
    assert connection_available(conn) is False


def test_connection_available_returns_false_on_timeout():
    """connection_available returns False on timeout."""
    conn = MagicMock()
    conn.run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=30)
    assert connection_available(conn) is False


def test_connection_available_returns_false_on_os_error():
    """connection_available returns False on OSError."""
    conn = MagicMock()
    conn.run.side_effect = OSError("Connection refused")
    assert connection_available(conn) is False


@patch("iris.cluster.vm.ssh.time.sleep")
@patch("iris.cluster.vm.ssh.connection_available")
def test_wait_for_connection_returns_true_immediately(mock_conn_avail, _mock_sleep):
    """wait_for_connection returns True if connection available immediately."""
    mock_conn_avail.return_value = True
    conn = MagicMock()
    assert wait_for_connection(conn, timeout_seconds=60, poll_interval=5) is True
    mock_conn_avail.assert_called_once()


@patch("iris.cluster.vm.ssh.time.sleep")
@patch("iris.cluster.vm.ssh.connection_available")
@patch("iris.cluster.vm.ssh.time.time")
def test_wait_for_connection_returns_false_on_timeout(mock_time, mock_conn_avail, _mock_sleep):
    """wait_for_connection returns False when timeout expires."""
    mock_conn_avail.return_value = False
    mock_time.side_effect = [0, 5, 11]  # Simulates time passing past 10s timeout
    conn = MagicMock()
    assert wait_for_connection(conn, timeout_seconds=10, poll_interval=5) is False


@patch("iris.cluster.vm.ssh.time.sleep")
@patch("iris.cluster.vm.ssh.connection_available")
@patch("iris.cluster.vm.ssh.time.time")
def test_wait_for_connection_respects_stop_event(mock_time, mock_conn_avail, _mock_sleep):
    """wait_for_connection returns False when stop_event is set."""
    mock_conn_avail.return_value = False
    mock_time.return_value = 0
    conn = MagicMock()
    stop_event = threading.Event()
    stop_event.set()
    assert wait_for_connection(conn, timeout_seconds=60, poll_interval=5, stop_event=stop_event) is False


def test_check_health_returns_true_on_success():
    """check_health returns True when curl succeeds."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=0)
    assert check_health(conn, port=10001) is True
    # Verify curl command was issued
    call_args = conn.run.call_args[0][0]
    assert "curl" in call_args
    assert "10001" in call_args


def test_check_health_returns_false_on_failure():
    """check_health returns False when curl fails."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=1)
    assert check_health(conn, port=10001) is False


def test_check_health_returns_false_on_exception():
    """check_health returns False on exception (safe helper)."""
    conn = MagicMock()
    conn.run.side_effect = Exception("Network error")
    assert check_health(conn, port=10001) is False


def test_shutdown_worker_graceful():
    """shutdown_worker with graceful=True runs docker stop."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=0)
    assert shutdown_worker(conn, graceful=True) is True
    call_args = conn.run.call_args[0][0]
    assert "docker stop iris-worker" in call_args


def test_shutdown_worker_forceful():
    """shutdown_worker with graceful=False runs docker kill."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=0)
    assert shutdown_worker(conn, graceful=False) is True
    call_args = conn.run.call_args[0][0]
    assert "docker kill iris-worker" in call_args


def test_shutdown_worker_returns_false_on_exception():
    """shutdown_worker returns False on exception (safe helper)."""
    conn = MagicMock()
    conn.run.side_effect = Exception("Docker error")
    assert shutdown_worker(conn, graceful=True) is False


def test_run_streaming_with_retry_success_first_attempt():
    """run_streaming_with_retry succeeds on first attempt."""
    conn = InMemorySshConnection(_address="10.0.0.1")
    lines_received = []
    result = run_streaming_with_retry(conn, "bootstrap script", max_retries=3, on_line=lines_received.append)
    assert result.returncode == 0
    assert any("iris-init" in line for line in lines_received)


@patch("iris.cluster.vm.ssh.time.sleep")
def test_run_streaming_with_retry_retries_on_connection_error(_mock_sleep):
    """run_streaming_with_retry retries on connection error with backoff."""
    call_count = [0]

    def run_streaming_side_effect(_command: str):
        call_count[0] += 1
        if call_count[0] < 3:
            raise OSError("Connection refused")
        return FakePopen()

    conn = MagicMock()
    conn.run_streaming.side_effect = run_streaming_side_effect

    result = run_streaming_with_retry(conn, "bootstrap script", max_retries=3)

    assert result.returncode == 0
    assert call_count[0] == 3


@patch("iris.cluster.vm.ssh.time.sleep")
def test_run_streaming_with_retry_raises_after_max_retries(_mock_sleep):
    """run_streaming_with_retry raises RuntimeError after max retries."""
    conn = MagicMock()
    conn.run_streaming.side_effect = OSError("Connection refused")

    with pytest.raises(RuntimeError, match="Command failed after 3 attempts"):
        run_streaming_with_retry(conn, "bootstrap script", max_retries=3)


def test_run_streaming_with_retry_calls_on_line_callback():
    """run_streaming_with_retry calls on_line callback for each line."""
    conn = InMemorySshConnection(_address="10.0.0.1")
    lines_received = []
    run_streaming_with_retry(conn, "bootstrap script", on_line=lines_received.append)
    assert len(lines_received) == 7  # FakePopen yields 7 lines
