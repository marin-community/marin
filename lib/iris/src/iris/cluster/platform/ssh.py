# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SSH utilities and connection implementations for VM management.

This module provides:
- SshConnection protocol for transport abstraction
- GcloudSshConnection for TPU VM SSH via gcloud
- GceSshConnection for standard GCE VMs via gcloud compute ssh
- DirectSshConnection for raw SSH connections
- Utility functions for connection testing and streaming commands
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Protocol

from iris.time_utils import Duration, ExponentialBackoff

logger = logging.getLogger(__name__)


# ============================================================================
# Protocol
# ============================================================================


class SshConnection(Protocol):
    """Execute commands on a remote host via SSH. Carries location metadata.

    Implementations must provide run() for synchronous execution and
    run_streaming() for commands with streaming output.
    """

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
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
            "--quiet",
            "--command",
            command,
        ]

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout.to_seconds())

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
            "--quiet",
            "--command",
            command,
        ]

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout.to_seconds())

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
    connect_timeout: Duration = dataclasses.field(default_factory=lambda: Duration.from_seconds(30))

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
                f"ConnectTimeout={int(self.connect_timeout.to_seconds())}",
                "-o",
                "BatchMode=yes",
                "-p",
                str(self.port),
                f"{self.user}@{self.host}",
                command,
            ]
        )
        return cmd

    def run(self, command: str, timeout: Duration = Duration.from_seconds(30)) -> subprocess.CompletedProcess:
        return subprocess.run(self._build_cmd(command), capture_output=True, text=True, timeout=timeout.to_seconds() + 5)

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


def connection_available(conn: SshConnection, timeout: Duration = Duration.from_seconds(30)) -> bool:
    """Check if remote connection works by running a simple echo command.

    Returns True if connection succeeds, False otherwise. Logs errors prominently
    to help diagnose SSH issues like platform incompatibility or network problems.
    """
    try:
        result = conn.run("echo ok", timeout=timeout)
        if result.returncode == 0:
            return True
        # Capture stderr for error diagnosis - this is the most useful info
        error_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
        logger.warning("SSH connection check failed to %s: %s", conn.address, error_msg)
        return False
    except subprocess.TimeoutExpired:
        logger.warning("SSH connection check timed out to %s after %s", conn.address, timeout)
        return False
    except OSError as e:
        logger.warning("SSH connection check failed to %s: %s", conn.address, e)
        return False


def wait_for_connection(
    conn: SshConnection,
    timeout: Duration,
    poll_interval: Duration,
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for remote connection to become available.

    Polls the connection at the specified interval until it succeeds
    or the timeout expires. Logs prominently on first failure and
    periodically during the wait to help diagnose connection issues.
    """
    timeout_seconds = timeout.to_seconds()
    deadline = time.time() + timeout_seconds
    start_time = deadline - timeout_seconds
    attempt = 0
    first_failure_logged = False

    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return False
        attempt += 1

        if connection_available(conn):
            if first_failure_logged:
                elapsed = int(time.time() - start_time)
                logger.info("SSH: Connection established to %s after %ds (%d attempts)", conn.address, elapsed, attempt)
            return True

        # Log first failure prominently (connection_available already logged the error details)
        if not first_failure_logged:
            logger.warning(
                "SSH: First connection attempt failed to %s (will retry for %ds)", conn.address, timeout_seconds
            )
            first_failure_logged = True
        elif attempt % 6 == 0:  # Every 30 seconds at 5-second intervals
            elapsed = int(time.time() - start_time)
            remaining = timeout_seconds - elapsed
            logger.info("SSH: Still waiting for %s (%ds elapsed, %ds remaining)", conn.address, elapsed, remaining)

        time.sleep(poll_interval.to_seconds())

    logger.error("SSH: Connection timeout after %ds to %s (%d attempts)", timeout_seconds, conn.address, attempt)
    return False


SSH_MAX_RETRIES = 3


SSH_RETRYABLE_EXIT_CODES = {255}  # SSH connection failures


def run_streaming_with_retry(
    conn: SshConnection,
    command: str,
    max_retries: int = SSH_MAX_RETRIES,
    overall_timeout: int = 600,
    on_line: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run command with streaming output and exponential backoff retry on failures.

    Retries on connection errors with exponential backoff. Lines of output
    are passed to the on_line callback as they arrive. Also retries on
    SSH-specific exit codes (255 = connection refused/failed).
    """
    backoff = ExponentialBackoff(initial=5.0, maximum=30.0, factor=2.0)
    last_error: Exception | str | None = None

    for attempt in range(max_retries):
        proc: Any = None
        try:
            logger.info("SSH: Running command (attempt %d/%d)", attempt + 1, max_retries)
            proc = conn.run_streaming(command)

            output_lines: list[str] = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    output_lines.append(line)
                    if on_line:
                        on_line(line)

            proc.wait(timeout=overall_timeout)
            returncode = proc.returncode or 0

            # Retry on SSH connection failures (exit code 255)
            if returncode in SSH_RETRYABLE_EXIT_CODES:
                last_error = f"SSH exit code {returncode}"
                logger.warning("SSH: Connection failed on attempt %d (exit code %d)", attempt + 1, returncode)
            else:
                return subprocess.CompletedProcess(proc.args, returncode, "\n".join(output_lines), "")

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
