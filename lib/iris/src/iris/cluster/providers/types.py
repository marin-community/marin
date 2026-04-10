# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared types, handle protocols, status enums, and exceptions for infrastructure providers.

Provider implementations and consumers depend on this single, self-contained
module without pulling in the full provider protocols or any concrete implementation.
"""

from __future__ import annotations

import datetime
import logging
import os
import socket
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from rigging.timing import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def generate_slice_suffix() -> str:
    """Generate a unique suffix for slice IDs: ``YYYYMMDD-HHMM-<uuid8>``."""
    now = datetime.datetime.now(datetime.timezone.utc)
    short_uuid = uuid.uuid4().hex[:8]
    return f"{now.strftime('%Y%m%d-%H%M')}-{short_uuid}"


class Labels:
    """Label keys for Iris-managed cloud resources.

    All keys follow ``iris-{prefix}-<suffix>`` so resources are self-documenting
    and namespaced per cluster.
    """

    def __init__(self, prefix: str):
        self.iris_managed = f"iris-{prefix}-managed"
        self.iris_scale_group = f"iris-{prefix}-scale-group"
        self.iris_controller = f"iris-{prefix}-controller"
        self.iris_controller_address = f"iris-{prefix}-controller-address"
        self.iris_slice_id = f"iris-{prefix}-slice-id"


def find_free_port(start: int = -1) -> int:
    """Find an available port.

    Args:
        start: Starting port for sequential scan. Default of -1 lets the kernel
            pick a random ephemeral port, which avoids collisions when multiple
            processes search for ports concurrently (e.g. pytest-xdist).
    """
    if start == -1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    for port in range(start, start + 1000):
        lock = Path(f"/tmp/iris/port_{port}")
        try:
            os.kill(int(lock.read_text()), 0)
            continue  # port locked by a live process
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                lock.parent.mkdir(parents=True, exist_ok=True)
                lock.write_text(str(os.getpid()))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 1000}")


def wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a TCP port to become connectable.

    Returns True if port is ready, False on timeout.
    """
    import time

    dl = Deadline.from_seconds(timeout)
    while not dl.expired():
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InfraError(Exception):
    """Base for infrastructure operation failures."""


class QuotaExhaustedError(InfraError):
    """No capacity in the requested zone. Try another zone or wait."""


class ResourceNotFoundError(InfraError):
    """The requested resource type/variant doesn't exist."""


class InfraUnavailableError(InfraError):
    """Transient infrastructure failure. Retry with backoff."""


# ---------------------------------------------------------------------------
# Status types
# ---------------------------------------------------------------------------


class CloudSliceState(StrEnum):
    """Cloud-level slice states. Provider implementations map to these."""

    CREATING = "CREATING"
    BOOTSTRAPPING = "BOOTSTRAPPING"
    READY = "READY"
    FAILED = "FAILED"
    REPAIRING = "REPAIRING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"


class CloudWorkerState(StrEnum):
    """Cloud-level worker states."""

    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SliceStatus:
    """Cloud-level slice status, including worker handles from the same query."""

    state: CloudSliceState
    worker_count: int
    workers: list[RemoteWorkerHandle] = field(default_factory=list)
    error_message: str = ""


@dataclass
class WorkerStatus:
    """Cloud-level worker status."""

    state: CloudWorkerState


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Handle protocols
# ---------------------------------------------------------------------------


class RemoteWorkerHandle(Protocol):
    """Handle to a single worker within a slice.

    Represents a remote worker process: a TPU VM on GCP, a Pod on CoreWeave,
    a thread in LOCAL mode. Provides infrastructure-level operations.

    No terminate -- slices are the atomic unit. Individual slice members
    cannot be terminated independently.

    Thread safety: implementations must be safe for concurrent run_command() calls.
    """

    @property
    def worker_id(self) -> str: ...

    @property
    def vm_id(self) -> str: ...

    @property
    def internal_address(self) -> str:
        """Internal/private IP address for intra-cluster communication."""
        ...

    @property
    def external_address(self) -> str | None:
        """External/public IP address, if available."""
        ...

    @property
    def bootstrap_log(self) -> str:
        """Most recent bootstrap output captured for this worker."""
        ...

    def status(self) -> WorkerStatus:
        """Cloud-level worker status."""
        ...

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Run a command on the worker. Optionally stream output lines."""
        ...

    def reboot(self) -> None:
        """Reboot the worker."""
        ...

    def restart_worker(self, bootstrap_script: str) -> None:
        """Restart the Iris worker process with a fresh bootstrap script.

        Runs the bootstrap script which pulls the latest image, stops the
        old container, and starts a new one. The new worker process discovers
        and adopts existing task containers via Docker labels.

        Args:
            bootstrap_script: Full bootstrap script to execute on the worker VM.
        """
        ...


class StandaloneWorkerHandle(RemoteWorkerHandle, Protocol):
    """Handle to a standalone worker (e.g., controller). Can be terminated and labeled.

    Returned by create_vm(). Extends RemoteWorkerHandle with operations that
    only make sense for independently-managed workers.
    """

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        """Wait until the worker is reachable via SSH/network.

        Returns True if connection succeeded within timeout, False otherwise.
        """
        ...

    def bootstrap(self, script: str) -> None:
        """Run the bootstrap script on the worker."""
        ...

    def terminate(self, *, wait: bool = False) -> None:
        """Destroy the worker."""
        ...

    def set_labels(self, labels: dict[str, str]) -> None:
        """Set labels on the worker (for discovery via list_vms).

        GCE label values: lowercase alphanumeric + hyphens, max 63 chars.
        """
        ...

    def set_metadata(self, metadata: dict[str, str]) -> None:
        """Set arbitrary key-value metadata on the worker.

        Unlike labels, metadata values have no character restrictions.
        On GCP, accessible via the metadata server from within the VM.
        """
        ...


class SliceHandle(Protocol):
    """Handle to an allocated slice of connected workers.

    A slice is the atomic scaling unit. For TPUs, it's a complete pod.
    For GPUs, it could be a set of IB-connected nodes.
    """

    @property
    def slice_id(self) -> str:
        """Unique identifier (e.g., 'iris-tpu_v5e_16-1738000000000')."""
        ...

    @property
    def zone(self) -> str:
        """Zone where this slice is allocated."""
        ...

    @property
    def scale_group(self) -> str:
        """Name of the scale group this slice belongs to."""
        ...

    @property
    def labels(self) -> dict[str, str]:
        """Labels/tags set on this slice at creation time."""
        ...

    @property
    def created_at(self) -> Timestamp:
        """When this slice was created."""
        ...

    def describe(self) -> SliceStatus:
        """Query cloud state, returning status and worker handles."""
        ...

    def terminate(self, *, wait: bool = False) -> None:
        """Destroy the slice and all its workers."""
        ...


# ---------------------------------------------------------------------------
# Default stop_all helper
# ---------------------------------------------------------------------------

TERMINATE_TIMEOUT_SECONDS = 60


def default_stop_all(
    list_all_slices: Callable[[], list[SliceHandle]],
    stop_controller: Callable[[], None],
    dry_run: bool = False,
) -> list[str]:
    """Discover all managed slices and the controller, terminate in parallel.

    Shared by GCP, Manual, and Local providers. Uses daemon threads with a hard
    timeout so timed-out threads don't block interpreter shutdown.

    Args:
        list_all_slices: Callable returning all managed slices.
        stop_controller: Callable that stops the controller.
        dry_run: When True, discover resources but don't terminate.
    """
    target_names: list[str] = ["controller"]
    all_slices = list_all_slices()
    for s in all_slices:
        logger.info("Found managed slice %s", s.slice_id)
        target_names.append(f"slice:{s.slice_id}")

    if dry_run:
        return target_names

    targets: list[tuple[str, Callable[[], None]]] = [
        ("controller", stop_controller),
    ]
    for s in all_slices:
        targets.append((f"slice:{s.slice_id}", s.terminate))

    logger.info("Terminating %d resource(s) in parallel", len(targets))

    errors: list[str] = []
    results: dict[str, Exception | None] = {}
    lock = threading.Lock()

    def _run(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            with lock:
                results[name] = exc
            return
        with lock:
            results[name] = None

    threads: dict[str, threading.Thread] = {}
    for name, fn in targets:
        t = threading.Thread(target=_run, args=(name, fn), daemon=True)
        t.start()
        threads[name] = t

    dl = Deadline.from_seconds(TERMINATE_TIMEOUT_SECONDS)
    for _name, t in threads.items():
        t.join(timeout=dl.remaining_seconds())

    for name, t in threads.items():
        if t.is_alive():
            logger.warning(
                "Termination of %s still running after %ds, giving up",
                name,
                TERMINATE_TIMEOUT_SECONDS,
            )
            errors.append(f"timeout:{name}")
        else:
            exc = results.get(name)
            if exc is not None:
                logger.exception("Failed to terminate %s", name, exc_info=exc)
                errors.append(name)

    if errors:
        logger.error("Errors when stopping cluster: %s", errors)

    return target_names
