# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local handle implementations for in-process testing.

Provides LocalSliceHandle and worker handle classes used by InMemoryGcpService
in LOCAL mode. Local testing uses GcpWorkerProvider + InMemoryGcpService(mode=ServiceMode.LOCAL).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field

from rigging.timing import Duration, Timestamp

from iris.cluster.providers.types import (
    CloudSliceState,
    CloudWorkerState,
    CommandResult,
    Labels,
    SliceStatus,
    WorkerStatus,
)
from iris.cluster.worker.worker import Worker

logger = logging.getLogger(__name__)


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class _LocalWorkerHandle:
    """Handle to a local in-process worker.

    run_command() executes commands locally via subprocess.
    wait_for_connection() returns True immediately (local process).
    """

    _vm_id: str
    _internal_address: str
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
        return None

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        return True

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        timeout_secs = timeout.to_seconds() if timeout else 30.0
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        if on_line:
            for line in result.stdout.splitlines():
                on_line(line)
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def bootstrap(self, script: str) -> None:
        self._bootstrap_log_lines.clear()
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
        )
        self._bootstrap_log_lines.extend(result.stdout.splitlines())
        if result.returncode != 0:
            self._bootstrap_log_lines.extend(result.stderr.splitlines())
            raise RuntimeError(f"Bootstrap failed on {self._vm_id}: exit code {result.returncode}\n{result.stderr}")

    @property
    def bootstrap_log(self) -> str:
        return "\n".join(self._bootstrap_log_lines)

    def reboot(self) -> None:
        logger.info("Reboot requested for local VM %s (no-op)", self._vm_id)

    def restart_worker(self, bootstrap_script: str) -> None:
        logger.info("Worker restart requested for local VM %s (no-op)", self._vm_id)


@dataclass
class LocalSliceHandle:
    """Handle to a local in-process slice.

    list_vms() returns _LocalWorkerHandle instances for each "worker" in the slice.
    terminate() marks the slice as terminated and stops any real Worker instances.
    """

    _slice_id: str
    _vm_ids: list[str]
    _addresses: list[str]
    _labels: dict[str, str]
    _created_at: Timestamp
    _label_prefix: str
    _workers: list[Worker] = field(default_factory=list)
    _terminated: bool = False

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return "local"

    @property
    def scale_group(self) -> str:
        return self._labels.get(Labels(self._label_prefix).iris_scale_group, "")

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def created_at(self) -> Timestamp:
        return self._created_at

    def describe(self) -> SliceStatus:
        if self._terminated:
            return SliceStatus(state=CloudSliceState.DELETING, worker_count=0)
        workers = [
            _LocalWorkerHandle(_vm_id=vm_id, _internal_address=addr)
            for vm_id, addr in zip(self._vm_ids, self._addresses, strict=True)
        ]
        return SliceStatus(state=CloudSliceState.READY, worker_count=len(self._vm_ids), workers=workers)

    def terminate(self, *, wait: bool = False) -> None:
        if self._terminated:
            return
        self._terminated = True
        for worker in self._workers:
            worker.stop()
