# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared base for RemoteWorkerHandle implementations backed by remote execution.

GCP and Manual platform handles share identical logic for run_command(),
bootstrap(), wait_for_connection(), and reboot(). This base extracts that
shared implementation so the concrete handles only need to add
platform-specific operations (status queries, terminate, labels, etc.).

Local handles use subprocess rather than SSH, so they do not use this base.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field

from rigging.timing import Duration

from iris.cluster.providers.remote_exec import (
    RemoteExec,
    run_streaming_with_retry,
    wait_for_connection,
)
from iris.cluster.providers.types import CommandResult

logger = logging.getLogger(__name__)


@dataclass
class RemoteExecWorkerBase:
    """Shared implementation for RemoteWorkerHandle backed by a RemoteExec connection.

    Provides the five methods that are identical across GCP and Manual handles:
    run_command, bootstrap, bootstrap_log, wait_for_connection, and reboot.

    Subclasses must implement status() and any platform-specific operations
    (terminate, set_labels, set_metadata).
    """

    _remote_exec: RemoteExec
    _vm_id: str
    _internal_address: str
    _external_address: str | None = None
    _bootstrap_log_lines: list[str] = field(default_factory=list)

    @property
    def worker_id(self) -> str:
        return self._vm_id

    @property
    def vm_id(self) -> str:
        return self._vm_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return self._external_address

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        return wait_for_connection(self._remote_exec, timeout, poll_interval)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        if on_line:
            result = run_streaming_with_retry(self._remote_exec, command, max_retries=1, on_line=on_line)
            return CommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        ssh_timeout = timeout or Duration.from_seconds(30)
        result = self._remote_exec.run(command, timeout=ssh_timeout)
        return CommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def bootstrap(self, script: str) -> None:
        self._bootstrap_log_lines.clear()

        def on_line(line: str) -> None:
            self._bootstrap_log_lines.append(line)
            logger.info("[%s] %s", self._vm_id, line)

        result = run_streaming_with_retry(
            self._remote_exec,
            f"bash -c {shlex.quote(script)}",
            on_line=on_line,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Bootstrap failed on {self._vm_id}: exit code {result.returncode}")

    @property
    def bootstrap_log(self) -> str:
        return "\n".join(self._bootstrap_log_lines)

    def reboot(self) -> None:
        self._remote_exec.run("sudo reboot", timeout=Duration.from_seconds(10))

    def restart_worker(self, bootstrap_script: str) -> None:
        """Restart the worker with a fresh bootstrap script.

        Re-runs the full bootstrap which pulls the latest image, stops the
        old container, and starts a new one. The new worker process discovers
        and adopts existing task containers via Docker labels.
        """
        logger.info("Restarting worker on %s via bootstrap", self._vm_id)
        self.bootstrap(bootstrap_script)
