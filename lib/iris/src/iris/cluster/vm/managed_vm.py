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

"""ManagedVm - per-VM lifecycle management with dedicated thread.

This module provides:
- ManagedVm: Unified per-VM lifecycle with bootstrap logic
- VmRegistry: Global tracking for all ManagedVm instances
- VmFactory: Protocol for creating ManagedVm instances
- TrackedVmFactory: VmFactory implementation that registers VMs
- Bootstrap script and utilities for worker initialization
- SSH configuration for remote execution
"""

from __future__ import annotations

import logging
import shlex
import threading
from dataclasses import dataclass, field
from typing import Protocol

from iris.cluster.vm.ssh import (
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
# Bootstrap Script
# ============================================================================

# Bootstrap script template for worker VMs.
#
# This script expects CONTROLLER_ADDRESS to be set as a shell variable before
# the main script runs. Platforms inject a discovery preamble that sets this:
# - GCP (TpuVmManager): Queries GCP instance metadata
# - Manual (ManualVmManager): Uses static address from config
BOOTSTRAP_SCRIPT = """
set -e

echo "[iris-init] Starting Iris worker bootstrap"

# Fetch TPU metadata from GCE and export as environment variables.
# GCP TPU VMs expose TPU info via instance metadata, not environment variables.
echo "[iris-init] Probing TPU metadata..."
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
METADATA_HEADER="Metadata-Flavor: Google"

# Derive TPU slice name from instance name by stripping worker suffix.
# Handles both -wN (original) and -w-N (GCP multi-host) formats.
# Example: "iris-v5litepod_16-abc123-w0" -> "iris-v5litepod_16-abc123"
# Example: "t1v-n-598bede5-w-0" -> "t1v-n-598bede5"
INSTANCE_NAME=$(curl -sf -H "$METADATA_HEADER" \
    "http://metadata.google.internal/computeMetadata/v1/instance/name" 2>/dev/null || echo "")
if [ -n "$INSTANCE_NAME" ]; then
    export TPU_NAME=$(echo "$INSTANCE_NAME" | sed 's/-w-*[0-9]*$//')
fi

# Fetch accelerator-type as TPU_TYPE (e.g., v5litepod-16)
export TPU_TYPE=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/accelerator-type" 2>/dev/null || echo "")

# Fetch agent-worker-number as TPU_WORKER_ID
export TPU_WORKER_ID=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/agent-worker-number" 2>/dev/null || echo "")

# Fetch worker hostnames for multi-host slices (JSON array of network endpoints)
TPU_HOSTNAMES_RAW=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/worker-network-endpoints" 2>/dev/null || echo "")
if [ -n "$TPU_HOSTNAMES_RAW" ]; then
    # Format is "unknown:unknown:ip1,unknown:unknown:ip2,..." — extract IPs (fields with dots)
    export TPU_WORKER_HOSTNAMES=$(echo "$TPU_HOSTNAMES_RAW" | \
        tr ',' '\\n' | while read -r entry; do echo "$entry" | tr ':' '\\n' | grep '[.]'; done | \\
        paste -sd ',' - 2>/dev/null || echo "")
fi

# Fetch chips per host bounds from tpu-env metadata (YAML-like key: value format)
TPU_ENV_RAW=$(curl -sf -H "$METADATA_HEADER" "$METADATA_URL/tpu-env" 2>/dev/null || echo "")
if [ -n "$TPU_ENV_RAW" ]; then
    TOPO_RAW=$(echo "$TPU_ENV_RAW" | grep "^CHIPS_PER_HOST_BOUNDS:" | sed "s/.*: *'\\(.*\\)'/\\1/" | tr -d "'")
    if [ -n "$TOPO_RAW" ]; then
        export TPU_CHIPS_PER_HOST_BOUNDS="$TOPO_RAW"
    fi
fi

if [ -n "$TPU_NAME" ]; then
    echo "[iris-init] TPU detected: name=$TPU_NAME type=$TPU_TYPE worker_id=$TPU_WORKER_ID"
else
    echo "[iris-init] No TPU metadata found (this may be a non-TPU VM)"
fi

echo "[iris-init] Phase: prerequisites"

# Install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[iris-init] Installing Docker..."
    curl -fsSL https://get.docker.com | sudo sh
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "[iris-init] Docker installed"
else
    echo "[iris-init] Docker already installed"
fi

# Ensure docker daemon is running
sudo systemctl start docker || true

# Create cache directory
sudo mkdir -p {cache_dir}

echo "[iris-init] Phase: docker_pull"
echo "[iris-init] Configuring docker authentication"
# Configure docker for common GCP Artifact Registry regions
sudo gcloud auth configure-docker \
  europe-west4-docker.pkg.dev,us-central1-docker.pkg.dev,us-docker.pkg.dev --quiet 2>/dev/null || true

echo "[iris-init] Pulling image: {docker_image}"
sudo docker pull {docker_image}

echo "[iris-init] Phase: worker_start"

# Force-remove existing worker (handles restart policy race)
sudo docker rm -f iris-worker 2>/dev/null || true

# Clean up ALL iris-managed task containers by label
echo "[iris-init] Cleaning up iris task containers"
IRIS_CONTAINERS=$(sudo docker ps -aq --filter "label=iris.managed=true" 2>/dev/null || true)
if [ -n "$IRIS_CONTAINERS" ]; then
    sudo docker rm -f $IRIS_CONTAINERS 2>/dev/null || true
fi

# Start worker container without restart policy first (fail fast during bootstrap)
# Note: CONTROLLER_ADDRESS is set by the discovery preamble prepended to this script
sudo docker run -d --name iris-worker \
    --network=host \
    -v {cache_dir}:{cache_dir} \
    -v /var/run/docker.sock:/var/run/docker.sock \
    {env_flags} \
    {docker_image} \
    python -m iris.cluster.worker.main serve \
        --host 0.0.0.0 --port {worker_port} \
        --controller-address "$CONTROLLER_ADDRESS"

echo "[iris-init] Worker container started"
echo "[iris-init] Phase: registration"
echo "[iris-init] Waiting for worker to register with controller..."

# Wait for worker to be healthy (poll health endpoint)
for i in $(seq 1 60); do
    # Check if container is still running
    if ! sudo docker ps -q -f name=iris-worker | grep -q .; then
        echo "[iris-init] ERROR: Worker container exited unexpectedly"
        echo "[iris-init] Container status:"
        sudo docker ps -a -f name=iris-worker --format "table {{{{.Status}}}}\t{{{{.State}}}}"
        echo "[iris-init] Container logs:"
        sudo docker logs iris-worker --tail 100
        exit 1
    fi

    if curl -sf http://localhost:{worker_port}/health > /dev/null 2>&1; then
        echo "[iris-init] Worker is healthy"
        # Now add restart policy for production
        sudo docker update --restart=unless-stopped iris-worker
        echo "[iris-init] Bootstrap complete"
        exit 0
    fi
    sleep 2
done

echo "[iris-init] ERROR: Worker failed to become healthy after 120s"
echo "[iris-init] Container status:"
sudo docker ps -a -f name=iris-worker --format "table {{{{.Status}}}}\t{{{{.State}}}}"
echo "[iris-init] Container logs:"
sudo docker logs iris-worker --tail 100
exit 1
"""


def _build_env_flags(config: config_pb2.BootstrapConfig, vm_address: str) -> str:
    """Generate docker -e flags with proper escaping.

    Note: IRIS_CONTROLLER_ADDRESS is set from the shell variable $CONTROLLER_ADDRESS
    which is populated by the discovery preamble.

    TPU environment variables (TPU_NAME, TPU_WORKER_ID, etc.) are passed through
    from the host if they exist. These are set by GCP on TPU VMs and are required
    for the worker to register with tpu-name and tpu-worker-id attributes needed
    for coscheduled job scheduling.
    """
    flags = []
    for k, v in config.env_vars.items():
        flags.append(f"-e {shlex.quote(k)}={shlex.quote(v)}")
    # Use shell variable set by discovery preamble, not Python string
    flags.append('-e IRIS_CONTROLLER_ADDRESS="$CONTROLLER_ADDRESS"')
    # Inject VM address so worker can include it in registration for autoscaler tracking
    if vm_address:
        flags.append(f"-e IRIS_VM_ADDRESS={shlex.quote(vm_address)}")

    # Pass through TPU environment variables from host if they exist.
    # These are fetched from GCE metadata by the bootstrap script preamble.
    # TPU_NAME = instance name with -w{{N}} suffix stripped (TPU slice name) for coscheduling group_by
    # TPU_TYPE = accelerator-type (e.g., v5litepod-16) for topology lookup
    tpu_env_vars = [
        "TPU_NAME",
        "TPU_TYPE",
        "TPU_WORKER_ID",
        "TPU_WORKER_HOSTNAMES",
        "TPU_CHIPS_PER_HOST_BOUNDS",
    ]
    for var in tpu_env_vars:
        # Use shell syntax to pass through only if set on host
        flags.append(f'-e {var}="${{{var}:-}}"')

    return " ".join(flags)


def _build_bootstrap_script(
    config: config_pb2.BootstrapConfig,
    vm_address: str,
    discovery_preamble: str = "",
) -> str:
    """Build the bootstrap script from config.

    Args:
        config: Bootstrap configuration
        vm_address: VM IP address for autoscaler tracking
        discovery_preamble: Shell script fragment that sets CONTROLLER_ADDRESS.
            This is prepended to the main bootstrap script and must define
            the CONTROLLER_ADDRESS variable for the worker to connect.
    """
    env_flags = _build_env_flags(config, vm_address)
    main_script = BOOTSTRAP_SCRIPT.format(
        cache_dir=config.cache_dir or "/var/cache/iris",
        docker_image=config.docker_image,
        worker_port=config.worker_port or 10001,
        env_flags=env_flags,
    )
    if discovery_preamble:
        return discovery_preamble + main_script
    return main_script


# ============================================================================
# SSH Configuration
# ============================================================================


@dataclass
class SshConfig:
    """SSH configuration for ManualVmManager."""

    user: str = "root"
    key_file: str | None = None
    port: int = 22
    connect_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(30))


# ============================================================================
# Constants
# ============================================================================

PARTIAL_SLICE_GRACE_MS = 5 * 60 * 1000  # 5 minutes


# ============================================================================
# ManagedVm - Unified per-VM lifecycle thread
# ============================================================================


class ManagedVm:
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

    def start(self) -> None:
        """No-op for backward compatibility; thread starts on construction."""

    def stop(self, timeout: float = 10.0) -> None:
        """Signal VM thread to stop and wait for it to exit."""
        self._threads.stop(timeout=timeout)

    def _run(self, stop_event: threading.Event) -> None:
        """Main lifecycle: BOOTING -> INITIALIZING -> READY.

        Inlines bootstrap logic directly - no nested state machine.
        Worker ID is not set by ManagedVm - the worker generates its own ID
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
            script = _build_bootstrap_script(self._bootstrap_config, vm_address, self._discovery_preamble)

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
        script = _build_bootstrap_script(
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
        self._vms: dict[str, ManagedVm] = {}
        self._lock = threading.Lock()

    def register(self, vm: ManagedVm) -> None:
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

    def get_vm(self, vm_id: str) -> ManagedVm | None:
        """Get a specific VM by ID."""
        with self._lock:
            return self._vms.get(vm_id)

    def all_vms(self) -> list[ManagedVm]:
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
    """Protocol for creating ManagedVm instances.

    VmManagers use a factory to create ManagedVm instances. This allows
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
    ) -> ManagedVm:
        """Create a new ManagedVm instance.

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
            A started ManagedVm instance
        """
        ...


class TrackedVmFactory:
    """VmFactory that registers VMs with a registry and starts them.

    This is the standard factory implementation used by VmManagers.
    It creates ManagedVm instances, registers them with the registry,
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
    ) -> ManagedVm:
        """Create a ManagedVm, register it, and start its lifecycle thread."""
        vm = ManagedVm(
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
        vm.start()
        return vm

    @property
    def registry(self) -> VmRegistry:
        """Access the underlying registry."""
        return self._registry
