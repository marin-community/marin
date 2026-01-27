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

"""SSH utilities and connection implementations for VM management.

This module provides:
- SshConnection protocol for transport abstraction
- GcloudSshConnection for TPU VM SSH via gcloud
- GceSshConnection for standard GCE VMs via gcloud compute ssh
- DirectSshConnection for raw SSH connections
- InMemorySshConnection for dry-run and testing
- Utility functions for connection testing, health checks, and streaming commands
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from iris.time_utils import ExponentialBackoff

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol
# ============================================================================


@runtime_checkable
class SshConnection(Protocol):
    """Execute commands on a remote host via SSH. Carries location metadata.

    Implementations must provide run() for synchronous execution and
    run_streaming() for commands with streaming output.
    """

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run command synchronously and wait for completion."""
        ...

    def run_streaming(self, command: str) -> Any:
        """Run command with streaming stdout.

        Returns subprocess.Popen in production, FakePopen in tests.
        """
        ...

    @property
    def address(self) -> str:
        """Target address (IP, hostname, or VM identifier)."""
        ...

    @property
    def zone(self) -> str:
        """Zone/location, or empty string if not applicable."""
        ...


# ============================================================================
# Production Implementations
# ============================================================================


@dataclass
class GcloudSshConnection:
    """SSH via gcloud compute tpus tpu-vm ssh.

    Used for connecting to TPU VMs via the gcloud CLI. The gcloud tool
    handles SSH key management and tunneling.
    """

    project_id: str
    _zone: str
    vm_id: str
    worker_index: int = 0
    _address: str = ""

    @property
    def address(self) -> str:
        return self._address or self.vm_id

    @property
    def zone(self) -> str:
        return self._zone

    def _build_cmd(self, command: str) -> list[str]:
        return [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            self.vm_id,
            f"--zone={self._zone}",
            f"--project={self.project_id}",
            f"--worker={self.worker_index}",
            "--command",
            command,
        ]

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout)

    def run_streaming(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            self._build_cmd(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


@dataclass
class GceSshConnection:
    """SSH via gcloud compute ssh for standard GCE VMs (not TPU VMs).

    Used for connecting to regular GCE instances like the controller VM.
    Unlike GcloudSshConnection which uses `gcloud compute tpus tpu-vm ssh`,
    this uses `gcloud compute ssh` for standard compute instances.
    """

    project_id: str
    zone: str
    vm_name: str

    @property
    def address(self) -> str:
        return self.vm_name

    def _build_cmd(self, command: str) -> list[str]:
        return [
            "gcloud",
            "compute",
            "ssh",
            self.vm_name,
            f"--zone={self.zone}",
            f"--project={self.project_id}",
            "--command",
            command,
        ]

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout)

    def run_streaming(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            self._build_cmd(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


@dataclass
class DirectSshConnection:
    """SSH via raw ssh command.

    Used for direct SSH connections to VMs when not using gcloud.
    Configures SSH for non-interactive use with known host checking disabled.
    """

    host: str
    user: str = "root"
    port: int = 22
    key_file: str | None = None
    connect_timeout: int = 30

    @property
    def address(self) -> str:
        return self.host

    @property
    def zone(self) -> str:
        return ""

    def _build_cmd(self, command: str) -> list[str]:
        cmd = ["ssh"]
        if self.key_file:
            cmd.extend(["-i", self.key_file])
        cmd.extend(
            [
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                f"ConnectTimeout={self.connect_timeout}",
                "-o",
                "BatchMode=yes",
                "-p",
                str(self.port),
                f"{self.user}@{self.host}",
                command,
            ]
        )
        return cmd

    def run(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout + 5)

    def run_streaming(self, command: str) -> subprocess.Popen:
        return subprocess.Popen(
            self._build_cmd(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


# ============================================================================
# SSH Utilities
# ============================================================================


def connection_available(conn: SshConnection, timeout: int = 30) -> bool:
    """Check if remote connection works by running a simple echo command."""
    try:
        result = conn.run("echo ok", timeout=timeout)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def wait_for_connection(
    conn: SshConnection,
    timeout_seconds: int,
    poll_interval: int = 5,
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for remote connection to become available.

    Polls the connection at the specified interval until it succeeds
    or the timeout expires.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return False
        if connection_available(conn):
            return True
        time.sleep(poll_interval)
    return False


@dataclass
class HealthCheckResult:
    """Result of a health check with diagnostic info."""

    healthy: bool
    curl_output: str = ""
    curl_error: str = ""
    container_status: str = ""
    container_logs: str = ""

    def __bool__(self) -> bool:
        """Allow using result directly as a boolean."""
        return self.healthy

    def summary(self) -> str:
        """Return a one-line summary of the health check result."""
        if self.healthy:
            return "healthy"
        parts = []
        if self.container_status:
            parts.append(f"container={self.container_status}")
        if self.curl_error:
            parts.append(f"curl_error={self.curl_error[:50]}")
        return ", ".join(parts) if parts else "unknown failure"


def check_health(
    conn: SshConnection,
    port: int = 10001,
    container_name: str = "iris-worker",
) -> HealthCheckResult:
    """Check if worker/controller is healthy via health endpoint.

    Returns HealthCheckResult with diagnostic info. The result can be used
    directly as a boolean (via __bool__), or inspected for details on failure.

    Args:
        conn: SSH connection to the host
        port: Port to check health on
        container_name: Container name for gathering diagnostics on failure
    """
    result = HealthCheckResult(healthy=False)

    # Try curl health check
    try:
        curl_result = conn.run(f"curl -sf http://localhost:{port}/health", timeout=10)
        if curl_result.returncode == 0:
            result.healthy = True
            result.curl_output = curl_result.stdout.strip()
            return result
        result.curl_error = curl_result.stderr.strip() or f"exit code {curl_result.returncode}"
    except Exception as e:
        result.curl_error = str(e)

    # Health check failed, gather diagnostics
    try:
        status_result = conn.run(
            f"sudo docker inspect --format='{{{{.State.Status}}}}' {container_name} 2>/dev/null || echo 'not_found'",
            timeout=10,
        )
        result.container_status = status_result.stdout.strip()
    except Exception as e:
        result.container_status = f"error: {e}"

    # If container is not running, get recent logs
    if result.container_status in ("restarting", "exited", "dead"):
        try:
            logs_result = conn.run(f"sudo docker logs {container_name} --tail 20 2>&1", timeout=15)
            result.container_logs = logs_result.stdout.strip()
        except Exception as e:
            result.container_logs = f"error fetching logs: {e}"

    return result


def shutdown_worker(conn: SshConnection, graceful: bool = True) -> bool:
    """Shutdown worker container via docker stop/kill.

    Returns False on any failure - this is a safe helper that never raises.
    """
    cmd = "docker stop iris-worker" if graceful else "docker kill iris-worker"
    try:
        conn.run(cmd, timeout=30)
        return True
    except Exception as e:
        logger.debug("Shutdown worker failed for %s: %s", conn.address, e)
        return False


SSH_MAX_RETRIES = 3


def run_streaming_with_retry(
    conn: SshConnection,
    command: str,
    max_retries: int = SSH_MAX_RETRIES,
    overall_timeout: int = 600,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run command with streaming output and exponential backoff retry on failures.

    Retries on connection errors with exponential backoff. Lines of output
    are passed to the on_line callback as they arrive.
    """
    backoff = ExponentialBackoff(initial=5.0, maximum=30.0, factor=2.0)
    last_error: Exception | None = None

    for attempt in range(max_retries):
        proc: Any = None
        try:
            logger.info("SSH: Running command (attempt %d/%d)", attempt + 1, max_retries)
            proc = conn.run_streaming(command)

            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    if on_line:
                        on_line(line)

            proc.wait(timeout=overall_timeout)
            return subprocess.CompletedProcess(proc.args, proc.returncode or 0, "", "")

        except subprocess.TimeoutExpired as e:
            last_error = e
            logger.warning("SSH: Command timeout on attempt %d", attempt + 1)
            if proc is not None:
                proc.kill()
                proc.wait()
        except OSError as e:
            last_error = e
            logger.warning("SSH: Connection error on attempt %d: %s", attempt + 1, e)

        if attempt < max_retries - 1:
            wait_time = backoff.next_interval()
            logger.info("SSH: Retrying in %.1fs...", wait_time)
            time.sleep(wait_time)

    raise RuntimeError(f"Command failed after {max_retries} attempts: {last_error}")
