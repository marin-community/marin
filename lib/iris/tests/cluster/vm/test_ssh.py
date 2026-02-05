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

from iris.cluster.platform.controller_vm import check_health
from iris.cluster.platform.ssh import (
    connection_available,
    run_streaming_with_retry,
    wait_for_connection,
)
from iris.time_utils import Duration


def make_fake_popen(lines: list[str] | None = None):
    """Create a mock Popen-like object for streaming tests."""
    if lines is None:
        lines = ["[iris-init] Bootstrap starting", "[iris-init] Bootstrap complete"]
    mock = MagicMock()
    mock.stdout = iter(line + "\n" for line in lines)
    mock.returncode = 0
    mock.wait.return_value = 0
    mock.args = []
    return mock


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


@patch("iris.cluster.platform.ssh.time.sleep")
@patch("iris.cluster.platform.ssh.connection_available")
def test_wait_for_connection_returns_true_immediately(mock_conn_avail, _mock_sleep):
    """wait_for_connection returns True if connection available immediately."""
    mock_conn_avail.return_value = True
    conn = MagicMock()
    # Test behavior: function should return True when connection is available
    result = wait_for_connection(conn, timeout=Duration.from_seconds(60), poll_interval=Duration.from_seconds(5))
    assert result is True


@patch("iris.cluster.platform.ssh.time.sleep")
@patch("iris.cluster.platform.ssh.connection_available")
def test_wait_for_connection_returns_false_on_timeout(mock_conn_avail, _mock_sleep):
    """wait_for_connection returns False when timeout expires."""
    mock_conn_avail.return_value = False
    conn = MagicMock()
    # Use a very short timeout so the real monotonic deadline expires quickly
    assert wait_for_connection(conn, timeout=Duration.from_ms(50), poll_interval=Duration.from_ms(10)) is False


@patch("iris.cluster.platform.ssh.time.sleep")
@patch("iris.cluster.platform.ssh.connection_available")
def test_wait_for_connection_respects_stop_event(mock_conn_avail, _mock_sleep):
    """wait_for_connection returns False when stop_event is set."""
    mock_conn_avail.return_value = False
    conn = MagicMock()
    stop_event = threading.Event()
    stop_event.set()
    assert (
        wait_for_connection(
            conn, timeout=Duration.from_seconds(60), poll_interval=Duration.from_seconds(5), stop_event=stop_event
        )
        is False
    )


def test_check_health_returns_healthy_on_success():
    """check_health returns healthy result when curl succeeds."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=0, stdout="OK")
    result = check_health(conn, port=10001)
    assert result.healthy is True


def test_check_health_returns_unhealthy_on_failure():
    """check_health returns unhealthy result when curl fails."""
    conn = MagicMock()
    conn.run.return_value = MagicMock(returncode=1, stderr="Connection refused", stdout="")
    result = check_health(conn, port=10001)
    assert result.healthy is False
    assert "exit code 1" in result.curl_error or "Connection refused" in result.curl_error


def test_check_health_returns_unhealthy_on_exception():
    """check_health returns unhealthy result on exception."""
    conn = MagicMock()
    conn.run.side_effect = Exception("Network error")
    result = check_health(conn, port=10001)
    assert result.healthy is False
    assert "Network error" in result.curl_error


def test_run_streaming_with_retry_success_first_attempt():
    """run_streaming_with_retry succeeds on first attempt."""
    conn = MagicMock()
    conn.run_streaming.return_value = make_fake_popen()
    lines_received: list[str] = []
    result = run_streaming_with_retry(conn, "bootstrap script", max_retries=3, on_line=lines_received.append)
    assert result.returncode == 0
    assert len(lines_received) > 0


@patch("iris.cluster.platform.ssh.time.sleep")
def test_run_streaming_with_retry_retries_on_connection_error(_mock_sleep):
    """run_streaming_with_retry eventually succeeds after connection errors."""
    # Simulate initial failures followed by success
    conn = MagicMock()
    conn.run_streaming.side_effect = [
        OSError("Connection refused"),
        OSError("Connection refused"),
        make_fake_popen(),
    ]

    # Test behavior: function should eventually succeed despite initial failures
    result = run_streaming_with_retry(conn, "bootstrap script", max_retries=3)

    # Verify successful outcome
    assert result.returncode == 0


@pytest.mark.slow  # Flaky in CI: background thread holds logging lock (gh#2551)
@patch("iris.cluster.platform.ssh.time.sleep")
def test_run_streaming_with_retry_raises_after_max_retries(_mock_sleep):
    """run_streaming_with_retry raises RuntimeError after max retries."""
    conn = MagicMock()
    conn.run_streaming.side_effect = OSError("Connection refused")

    with pytest.raises(RuntimeError, match="Command failed after 3 attempts"):
        run_streaming_with_retry(conn, "bootstrap script", max_retries=3)


def test_run_streaming_with_retry_calls_on_line_callback():
    """run_streaming_with_retry calls on_line callback for each line."""
    expected_lines = ["line one", "line two", "line three"]
    conn = MagicMock()
    conn.run_streaming.return_value = make_fake_popen(expected_lines)
    lines_received: list[str] = []
    run_streaming_with_retry(conn, "bootstrap script", on_line=lines_received.append)
    assert lines_received == expected_lines


# ============================================================================
# Regression tests for Duration/float type safety
# ============================================================================


class FakeSshConnection:
    """Minimal SshConnection implementation for testing timeout handling.

    Records calls so tests can verify Duration values are properly converted
    to float/int before reaching subprocess APIs.
    """

    def __init__(self, run_result: subprocess.CompletedProcess | None = None):
        self._run_result = run_result or subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
        self.last_timeout: Duration | None = None

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
        self.last_timeout = timeout
        return self._run_result

    def run_streaming(self, command: str) -> MagicMock:
        return make_fake_popen()

    @property
    def address(self) -> str:
        return "fake-host"

    @property
    def zone(self) -> str:
        return ""


def test_connection_available_accepts_duration_timeout():
    """connection_available works with Duration timeout (regression for TypeError)."""
    conn = FakeSshConnection()
    assert connection_available(conn, timeout=Duration.from_seconds(5)) is True
    assert conn.last_timeout == Duration.from_seconds(5)


def test_check_health_passes_duration_to_run():
    """check_health passes Duration to conn.run without TypeError."""
    conn = FakeSshConnection()
    result = check_health(conn, port=10001)
    assert result.healthy is True
    assert conn.last_timeout == Duration.from_seconds(10)


def test_direct_ssh_connection_accepts_duration_connect_timeout():
    """DirectSshConnection accepts Duration for connect_timeout field."""
    from iris.cluster.platform.ssh import DirectSshConnection

    conn = DirectSshConnection(
        host="10.0.0.1",
        connect_timeout=Duration.from_seconds(45),
    )
    cmd = conn._build_cmd("echo hello")
    # The SSH ConnectTimeout option should be the integer seconds value
    assert "ConnectTimeout=45" in " ".join(cmd)


def test_ssh_config_duration_flows_to_direct_ssh_connection():
    """SshConfig.connect_timeout (Duration) flows correctly to DirectSshConnection."""
    from iris.cluster.platform.worker_vm import SshConfig
    from iris.cluster.platform.ssh import DirectSshConnection

    ssh_config = SshConfig(connect_timeout=Duration.from_seconds(20))
    conn = DirectSshConnection(
        host="10.0.0.1",
        connect_timeout=ssh_config.connect_timeout,
    )
    cmd = conn._build_cmd("echo hello")
    assert "ConnectTimeout=20" in " ".join(cmd)
