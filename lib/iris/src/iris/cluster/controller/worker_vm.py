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

"""WorkerVm - per-VM lifecycle management with dedicated thread.

This module provides:
- WorkerVm: Unified per-VM lifecycle with bootstrap logic
- VmRegistry: Global tracking for all WorkerVm instances
- VmFactory: Protocol for creating WorkerVm instances
- TrackedVmFactory: VmFactory implementation that registers VMs
- Bootstrap script and utilities for worker initialization
- SSH configuration for remote execution
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol

from iris.cluster.platform.bootstrap import build_worker_bootstrap_script
from iris.cluster.platform.ssh import (
    SshConnection,
    run_streaming_with_retry,
    wait_for_connection,
)
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2, vm_pb2
from iris.rpc.proto_utils import vm_state_name
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================


class PoolExhaustedError(Exception):
    """Raised when no hosts are available in the pool."""


class BootstrapError(Exception):
    """Raised when bootstrap fails."""


class QuotaExceededError(Exception):
    """Raised when cloud provider quota is exceeded."""


# ============================================================================
# SSH Configuration
# ============================================================================


@dataclass
class SshConfig:
    """SSH configuration for manual slice workers."""

    user: str = "root"
    key_file: str | None = None
    port: int = 22
    connect_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(30))


# ============================================================================
# Constants
# ============================================================================

PARTIAL_SLICE_GRACE_MS = 5 * 60 * 1000  # 5 minutes


# ============================================================================
# WorkerVm - Unified per-VM lifecycle thread
# ============================================================================


class WorkerVm:
    """A VM with its own lifecycle management thread.

    Works for both GCP and manual VMs via the RemoteExecutor abstraction.
    Thread drives: BOOTING -> INITIALIZING -> READY (or FAILED).

    Bootstrap logic is inlined - no separate VmBootstrap class.
    If address is available at construction, checks if worker is already healthy
    before running bootstrap (useful for controller restart recovery).
    """

    def __init__(
        self,
        vm_id: str,
        slice_id: str,
        scale_group: str,
        zone: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        conn: SshConnection,
        labels: dict[str, str] | None = None,
        address: str | None = None,
        discovery_preamble: str = "",
    ):
        now = Timestamp.now()
        self.info = vm_pb2.VmInfo(
            vm_id=vm_id,
            slice_id=slice_id,
            scale_group=scale_group,
            state=vm_pb2.VM_STATE_BOOTING,
            address=address or "",
            zone=zone,
            created_at=now.to_proto(),
            state_changed_at=now.to_proto(),
            labels=labels or {},
        )
        self._conn = conn
        self._bootstrap_config = bootstrap_config
        self._timeouts = timeouts
        self._discovery_preamble = discovery_preamble

        # Inlined bootstrap state
        self._log_lines: list[str] = []
        self._phase: str = ""
        self._threads = ThreadContainer(f"vm-{vm_id}")
        self._managed_thread = self._threads.spawn(target=self._run, name=f"vm-{vm_id}")

    @property
    def _stop(self) -> threading.Event:
        return self._managed_thread.stop_event

    def stop(self, timeout: Duration = Duration.from_seconds(10.0)) -> None:
        """Signal VM thread to stop and wait for it to exit."""
        self._threads.stop(timeout=timeout)

    def _run(self, stop_event: threading.Event) -> None:
        """Main lifecycle: BOOTING -> INITIALIZING -> READY.

        Inlines bootstrap logic directly - no nested state machine.
        Worker ID is not set by WorkerVm - the worker generates its own ID
        from IRIS_VM_ADDRESS or host:port at startup.
        """
        boot_timeout = Duration.from_proto(self._timeouts.boot_timeout)
        poll_interval = Duration.from_proto(self._timeouts.ssh_poll_interval)
        vm_id = self.info.vm_id
        vm_address = self.info.address

        logger.info("VM %s: Starting lifecycle (state=BOOTING)", vm_id)

        try:
            # Always bootstrap VMs when controller adopts them to ensure latest image.
            # The controller re-bootstraps all VMs on startup to pull the latest worker image.
            # Wait for connection
            logger.info("VM %s: Waiting for connection (timeout=%.1fs)", vm_id, boot_timeout.to_seconds())
            if not wait_for_connection(self._conn, boot_timeout, poll_interval, self._stop):
                self.info.init_error = f"Boot timeout after {boot_timeout.to_seconds():.1f}s"
                logger.error("VM %s: Boot timeout after %.1fs", vm_id, boot_timeout.to_seconds())
                self._transition(vm_pb2.VM_STATE_FAILED)
                return

            logger.info("VM %s: Connection available", vm_id)
            self._transition(vm_pb2.VM_STATE_INITIALIZING)

            # Run bootstrap - pass vm_address for IRIS_VM_ADDRESS env var
            logger.info("VM %s: Starting bootstrap", vm_id)
            script = build_worker_bootstrap_script(self._bootstrap_config, vm_address, self._discovery_preamble)

            result = run_streaming_with_retry(
                self._conn,
                script,
                max_retries=3,
                overall_timeout=600,
                on_line=self._log,
            )

            if result.returncode != 0:
                error_msg = f"Exit code {result.returncode}"
                self._log(f"[iris-init] Bootstrap failed: {error_msg}")
                raise BootstrapError(error_msg)

            logger.info("VM %s: Bootstrap complete", vm_id)
            self._transition(vm_pb2.VM_STATE_READY)

        except BootstrapError as e:
            self.info.init_error = str(e)
            logger.error("VM %s: Bootstrap failed: %s", vm_id, e)
            self._dump_log_on_failure()
            self._transition(vm_pb2.VM_STATE_FAILED)

        except RuntimeError as e:
            # Raised by run_streaming_with_retry on max retries exceeded
            self.info.init_error = str(e)
            self._log(f"[iris-init] {e}")
            logger.error("VM %s: Bootstrap failed: %s", vm_id, e)
            self._dump_log_on_failure()
            self._transition(vm_pb2.VM_STATE_FAILED)

        except Exception as e:
            self.info.init_error = f"Unexpected error: {e}"
            logger.exception("VM %s: Unexpected error", vm_id)
            self._dump_log_on_failure()
            self._transition(vm_pb2.VM_STATE_FAILED)

    def _log(self, line: str, level: int = logging.DEBUG) -> None:
        """Append to bootstrap log and emit to logger.

        Lines starting with [iris-init] are logged at INFO level.
        Other lines use the specified level (default DEBUG).
        """
        self._log_lines.append(line)
        if line.startswith("[iris-init]"):
            logger.info("[iris-bootstrap %s] %s", self.info.vm_id, line)
            if "[iris-init] Phase:" in line:
                self._phase = line.split("Phase:")[-1].strip()
                self.info.init_phase = self._phase
        elif line:
            logger.log(level, "[iris-bootstrap %s] %s", self.info.vm_id, line)

    def _dump_log_on_failure(self, tail: int = 50) -> None:
        """Dump bootstrap log tail on failure as a formatted block."""
        vm_id = self.info.vm_id
        if not self._log_lines:
            logger.error("[iris-bootstrap %s] No bootstrap log available", vm_id)
            return

        lines = self._log_lines[-tail:] if len(self._log_lines) > tail else self._log_lines
        truncated = len(self._log_lines) > tail

        prefix = f"[iris-bootstrap {vm_id}]"
        formatted = "\n".join(f"{prefix} {line}" for line in lines)

        header = f"{prefix} === BOOTSTRAP FAILED ==="
        if truncated:
            header += f" (showing last {tail} of {len(self._log_lines)} lines)"

        logger.error("%s\n%s", header, formatted)

    def _check_worker_healthy(self) -> bool:
        """Check if worker container is already running and healthy."""
        port = self._bootstrap_config.worker_port or 10001
        try:
            result = self._conn.run(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
            return result.returncode == 0
        except Exception:
            return False

    def _transition(self, new_state: vm_pb2.VmState) -> None:
        """Update state with timestamp and log the transition."""
        old_state = self.info.state
        self.info.state = new_state
        self.info.state_changed_at.CopyFrom(Timestamp.now().to_proto())
        if self._phase:
            self.info.init_phase = self._phase
        logger.info("VM %s: %s -> %s", self.info.vm_id, vm_state_name(old_state), vm_state_name(new_state))

    def init_log(self, tail: int | None = None) -> str:
        """Return bootstrap log (empty if not yet initializing)."""
        lines = self._log_lines[-tail:] if tail else self._log_lines
        return "\n".join(lines)

    def check_health(self) -> bool:
        """Check if worker is healthy via health endpoint."""
        port = self._bootstrap_config.worker_port or 10001
        try:
            result = self._conn.run(f"curl -sf http://localhost:{port}/health", timeout=Duration.from_seconds(10))
            return result.returncode == 0
        except Exception:
            return False

    def shutdown(self, graceful: bool = True) -> bool:
        """Shutdown worker container."""
        cmd = "docker stop iris-worker" if graceful else "docker kill iris-worker"
        try:
            self._conn.run(cmd, timeout=Duration.from_seconds(30))
            return True
        except Exception:
            return False

    def reload(self) -> None:
        """Re-run bootstrap script to pull new image and restart container.

        This is faster than recreating the VM - it SSHs into the existing VM
        and re-runs the bootstrap sequence (pull image, stop old container,
        start new one, health check).

        Raises:
            RuntimeError: If bootstrap fails or health check times out
        """
        logger.info("VM %s: Reloading (re-running bootstrap)", self.info.vm_id)

        # Clear previous log and reset state
        self._log_lines = []
        self._phase = ""
        self._transition(vm_pb2.VM_STATE_INITIALIZING)

        # Build and run bootstrap script
        script = build_worker_bootstrap_script(
            self._bootstrap_config,
            self.info.address,
            self._discovery_preamble,
        )

        try:
            result = run_streaming_with_retry(
                self._conn,
                script,
                max_retries=3,
                overall_timeout=600,
                on_line=self._log,
            )

            if result.returncode != 0:
                error_msg = f"Exit code {result.returncode}"
                self._log(f"[iris-init] Reload failed: {error_msg}")
                raise RuntimeError(f"Reload failed: {error_msg}")

            logger.info("VM %s: Reload complete", self.info.vm_id)
            self._transition(vm_pb2.VM_STATE_READY)

        except Exception as e:
            self.info.init_error = str(e)
            self._dump_log_on_failure()
            self._transition(vm_pb2.VM_STATE_FAILED)
            raise RuntimeError(f"VM {self.info.vm_id} reload failed: {e}") from e

    @property
    def is_terminal(self) -> bool:
        """True if VM is in a terminal state."""
        return self.info.state in (
            vm_pb2.VM_STATE_READY,
            vm_pb2.VM_STATE_FAILED,
            vm_pb2.VM_STATE_TERMINATED,
            vm_pb2.VM_STATE_PREEMPTED,
        )


# ============================================================================
# VM Registry and Factory (merged from registry.py)
# ============================================================================


class VmRegistry:
    """Tracks all VMs across all scale groups for status reporting.

    Thread-safe registry that provides a global view of all managed VMs.
    Used by the autoscaler for status reporting and by VmFactory implementations
    to ensure all VMs are tracked.
    """

    def __init__(self) -> None:
        self._vms: dict[str, WorkerVm] = {}
        self._lock = threading.Lock()

    def register(self, vm: WorkerVm) -> None:
        """Register a VM for tracking.

        Called when a VM is created. If a VM with the same ID already exists,
        it will be replaced.
        """
        with self._lock:
            self._vms[vm.info.vm_id] = vm

    def unregister(self, vm_id: str) -> None:
        """Unregister a VM by ID.

        Called when a VM is terminated. Safe to call if the VM is not registered.
        """
        with self._lock:
            self._vms.pop(vm_id, None)

    def get_vm(self, vm_id: str) -> WorkerVm | None:
        """Get a specific VM by ID."""
        with self._lock:
            return self._vms.get(vm_id)

    def all_vms(self) -> list[WorkerVm]:
        """Return a snapshot of all tracked VMs."""
        with self._lock:
            return list(self._vms.values())

    def get_init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get initialization log for a VM.

        Returns empty string if VM is not found.
        """
        vm = self.get_vm(vm_id)
        return vm.init_log(tail) if vm else ""

    def vm_count(self) -> int:
        """Return the number of registered VMs."""
        with self._lock:
            return len(self._vms)


class VmFactory(Protocol):
    """Protocol for creating WorkerVm instances.

    VmManagers use a factory to create WorkerVm instances. This allows
    the registry to track all VMs regardless of which manager created them.
    """

    @property
    def registry(self) -> VmRegistry:
        """Access the underlying registry for cleanup operations."""
        ...

    def create_vm(
        self,
        vm_id: str,
        slice_id: str,
        scale_group: str,
        zone: str,
        conn: SshConnection,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        labels: dict[str, str],
        address: str | None = None,
        discovery_preamble: str = "",
    ) -> WorkerVm:
        """Create a new WorkerVm instance.

        Args:
            vm_id: Unique identifier for the VM
            slice_id: ID of the slice this VM belongs to
            scale_group: Name of the scale group
            zone: GCP zone or location
            conn: SshConnection for SSH commands
            bootstrap_config: Bootstrap configuration
            timeouts: Timeout configuration
            labels: Labels/tags for the VM
            address: Optional IP address if known at creation time
            discovery_preamble: Shell script that sets CONTROLLER_ADDRESS variable.
                GCP platforms inject metadata discovery, manual platforms use static address.

        Returns:
            A started WorkerVm instance
        """
        ...


class TrackedVmFactory:
    """VmFactory that registers VMs with a registry and starts them.

    This is the standard factory implementation used by VmManagers.
    It creates WorkerVm instances, registers them with the registry,
    and starts their lifecycle threads.
    """

    def __init__(self, registry: VmRegistry) -> None:
        self._registry = registry

    def create_vm(
        self,
        vm_id: str,
        slice_id: str,
        scale_group: str,
        zone: str,
        conn: SshConnection,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        labels: dict[str, str],
        address: str | None = None,
        discovery_preamble: str = "",
    ) -> WorkerVm:
        """Create a WorkerVm, register it, and start its lifecycle thread."""
        vm = WorkerVm(
            vm_id=vm_id,
            slice_id=slice_id,
            scale_group=scale_group,
            zone=zone,
            bootstrap_config=bootstrap_config,
            timeouts=timeouts,
            conn=conn,
            labels=labels,
            address=address,
            discovery_preamble=discovery_preamble,
        )
        self._registry.register(vm)
        return vm

    @property
    def registry(self) -> VmRegistry:
        """Access the underlying registry."""
        return self._registry
