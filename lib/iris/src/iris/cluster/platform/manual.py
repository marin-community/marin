# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""ManualPlatform implementation for pre-existing hosts.

Implements the full Platform interface for manually managed hosts. Hosts are
drawn from a configured pool and returned on slice termination. Remote execution
uses DirectSshRemoteExec (raw ssh, no gcloud).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field

from iris.cluster.platform._worker_base import RemoteExecWorkerBase
from iris.cluster.platform.base import (
    CloudSliceState,
    CloudWorkerState,
    PlatformError,
    SliceStatus,
    WorkerStatus,
)
from iris.cluster.platform.bootstrap import build_worker_bootstrap_script
from iris.cluster.platform.remote_exec import (
    DirectSshRemoteExec,
)
from iris.rpc import config_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


# ============================================================================
# Handle Implementations
# ============================================================================


@dataclass
class ManualWorkerHandle(RemoteExecWorkerBase):
    """Handle to a worker on a manual (pre-existing) host.

    Uses DirectSshRemoteExec for SSH. Thread-safe: each run_command() spawns
    a new SSH process.
    """

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)


@dataclass
class ManualStandaloneWorkerHandle(RemoteExecWorkerBase):
    """Handle to a standalone worker on a manual host (e.g., controller).

    Extends ManualWorkerHandle with terminate, set_labels, and set_metadata.
    Labels and metadata are tracked in-memory since manual hosts don't have
    a cloud metadata service.
    """

    _labels: dict[str, str] = field(default_factory=dict)
    _metadata: dict[str, str] = field(default_factory=dict)
    _on_terminate: Callable[[], None] | None = None

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self._metadata)

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def terminate(self) -> None:
        if self._on_terminate:
            self._on_terminate()

    def set_labels(self, labels: dict[str, str]) -> None:
        self._labels.update(labels)

    def set_metadata(self, metadata: dict[str, str]) -> None:
        self._metadata.update(metadata)


class ManualSliceHandle:
    """Handle to a slice of manual hosts.

    Hosts are pre-existing and not destroyed on terminate — they are returned
    to the pool instead. When bootstrap is requested, describe() composites
    the bootstrap state with the base state.
    """

    def __init__(
        self,
        *,
        _slice_id: str,
        _hosts: list[str],
        _labels: dict[str, str],
        _created_at: Timestamp,
        _label_prefix: str,
        _ssh_connections: list[DirectSshRemoteExec],
        _on_terminate: Callable[[list[str]], None] | None = None,
        _bootstrapping: bool = False,
    ):
        self._slice_id = _slice_id
        self._hosts = _hosts
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._ssh_connections = _ssh_connections
        self._on_terminate = _on_terminate
        self._terminated = False
        self._bootstrapping = _bootstrapping
        # Bootstrap state: None means bootstrap not yet completed.
        # Set by the platform's internal bootstrap thread.
        self._bootstrap_state: CloudSliceState | None = None
        self._bootstrap_lock = threading.Lock()

    @property
    def slice_id(self) -> str:
        return self._slice_id

    @property
    def zone(self) -> str:
        return "manual"

    @property
    def scale_group(self) -> str:
        return self._labels.get(f"{self._label_prefix}-scale-group", "")

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
            ManualWorkerHandle(
                _vm_id=f"{self._slice_id}-{host.replace('.', '-').replace(':', '-')}",
                _internal_address=host,
                _remote_exec=ssh,
            )
            for host, ssh in zip(self._hosts, self._ssh_connections, strict=True)
        ]

        # Composite state: if bootstrap was requested, reflect its progress
        if self._bootstrapping:
            with self._bootstrap_lock:
                bs = self._bootstrap_state
            if bs is None:
                state = CloudSliceState.BOOTSTRAPPING
            elif bs == CloudSliceState.READY:
                state = CloudSliceState.READY
            elif bs == CloudSliceState.FAILED:
                state = CloudSliceState.FAILED
            else:
                state = CloudSliceState.READY
        else:
            state = CloudSliceState.READY

        return SliceStatus(state=state, worker_count=len(self._hosts), workers=workers)

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self._on_terminate:
            self._on_terminate(list(self._hosts))
        logger.info("Terminated manual slice %s (%d hosts)", self._slice_id, len(self._hosts))


# ============================================================================
# ManualPlatform
# ============================================================================


class ManualPlatform:
    """Platform for pre-existing hosts managed without cloud provisioning.

    Hosts are drawn from a configured pool. On slice termination, hosts are
    returned to the pool for reuse. SSH uses DirectSshRemoteExec.
    """

    def __init__(
        self,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        hosts: list[str] | None = None,
    ):
        self._label_prefix = label_prefix
        self._ssh_config = ssh_config
        self._all_hosts = list(hosts or [])
        self._available_hosts: set[str] = set(self._all_hosts)
        self._allocated_hosts: set[str] = set()
        self._slices: dict[str, ManualSliceHandle] = {}
        self._vms: dict[str, ManualStandaloneWorkerHandle] = {}

    def create_vm(self, config: config_pb2.VmConfig) -> ManualStandaloneWorkerHandle:
        """Allocate a host from the pool for a standalone VM (e.g., controller)."""
        manual = config.manual
        host = manual.host

        # If specific host is requested, use it; otherwise pop from pool
        if host:
            if host in self._allocated_hosts:
                raise RuntimeError(f"Host {host} is already allocated")
            self._available_hosts.discard(host)
        elif self._available_hosts:
            host = self._available_hosts.pop()
        else:
            raise RuntimeError("No hosts available in manual platform pool")

        self._allocated_hosts.add(host)
        remote_exec = self._create_remote_exec(host, manual)

        def on_terminate() -> None:
            self._return_hosts([host])
            self._vms.pop(config.name, None)

        handle = ManualStandaloneWorkerHandle(
            _vm_id=config.name,
            _internal_address=host,
            _remote_exec=remote_exec,
            _labels=dict(config.labels),
            _metadata=dict(config.metadata),
            _on_terminate=on_terminate,
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        bootstrap_config: config_pb2.BootstrapConfig | None = None,
    ) -> ManualSliceHandle:
        """Allocate hosts from the pool for a slice.

        When bootstrap_config is provided, spawns a background thread that runs
        the bootstrap script on each worker. The handle's describe() composites
        bootstrap state with the base state.
        """
        manual = config.manual
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"

        # Use explicitly listed hosts if provided, otherwise draw from pool
        if manual.hosts:
            hosts = list(manual.hosts)
            already_allocated = self._allocated_hosts & set(hosts)
            if already_allocated:
                raise RuntimeError(f"Hosts already allocated: {sorted(already_allocated)}")
            for h in hosts:
                self._available_hosts.discard(h)
        else:
            needed = config.num_vms or 1
            if len(self._available_hosts) < needed:
                raise RuntimeError(f"Need {needed} hosts but only {len(self._available_hosts)} available")
            hosts = [self._available_hosts.pop() for _ in range(needed)]

        self._allocated_hosts.update(hosts)
        ssh_connections = [self._create_remote_exec(h, manual) for h in hosts]

        def on_terminate(terminated_hosts: list[str]) -> None:
            self._return_hosts(terminated_hosts)
            self._slices.pop(slice_id, None)

        handle = ManualSliceHandle(
            _slice_id=slice_id,
            _hosts=hosts,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _ssh_connections=ssh_connections,
            _on_terminate=on_terminate,
            _bootstrapping=bootstrap_config is not None,
        )
        self._slices[slice_id] = handle

        if bootstrap_config:

            def _bootstrap_worker():
                try:
                    self._run_bootstrap(handle, bootstrap_config)
                except Exception as e:
                    logger.error("Bootstrap failed for slice %s: %s", handle.slice_id, e)
                    with handle._bootstrap_lock:
                        handle._bootstrap_state = CloudSliceState.FAILED

            threading.Thread(
                target=_bootstrap_worker,
                name=f"bootstrap-{handle.slice_id}",
                daemon=True,
            ).start()

        return handle

    def _run_bootstrap(
        self,
        handle: ManualSliceHandle,
        bootstrap_config: config_pb2.BootstrapConfig,
    ) -> None:
        """Bootstrap all workers in the slice in parallel.

        Manual hosts are already reachable (no cloud provisioning wait), so we
        bootstrap all workers concurrently via wait_for_connection + bootstrap().
        """
        status = handle.describe()
        workers = status.workers
        logger.info("Bootstrapping %d workers for slice %s", len(workers), handle.slice_id)
        errors: list[tuple[str, Exception]] = []

        def _bootstrap_one(worker: RemoteExecWorkerBase) -> None:
            try:
                if not worker.internal_address:
                    raise PlatformError(f"Worker {worker.worker_id} in slice {handle.slice_id} has no internal address")
                if not worker.wait_for_connection(timeout=Duration.from_seconds(300)):
                    raise PlatformError(f"Worker {worker.worker_id} in slice {handle.slice_id} not reachable via SSH")
                script = build_worker_bootstrap_script(bootstrap_config, worker.internal_address)
                worker.bootstrap(script)
            except Exception as e:
                errors.append((worker.worker_id, e))

        threads: list[threading.Thread] = []
        for worker in workers:
            t = threading.Thread(
                target=_bootstrap_one,
                args=(worker,),
                name=f"bootstrap-{worker.worker_id}",
                daemon=True,
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        if errors:
            failed_ids = [wid for wid, _ in errors]
            raise PlatformError(
                f"Bootstrap failed for {len(errors)}/{len(workers)} workers in slice {handle.slice_id}: "
                f"{', '.join(failed_ids)}: {errors[0][1]}"
            )

        logger.info("Bootstrap completed for slice %s (%d workers)", handle.slice_id, len(workers))
        with handle._bootstrap_lock:
            handle._bootstrap_state = CloudSliceState.READY

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[ManualSliceHandle]:
        """List all manual slices, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        all manual slices report zone="manual".
        """
        results = list(self._slices.values())
        if labels:
            results = [s for s in results if all(s.labels.get(k) == v for k, v in labels.items())]
        return results

    def list_all_slices(self, labels: dict[str, str] | None = None) -> list[ManualSliceHandle]:
        return self.list_slices(zones=[], labels=labels)

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[ManualStandaloneWorkerHandle]:
        """List all manual standalone VMs, optionally filtered by labels.

        The zones parameter is accepted for interface compatibility but ignored —
        manual VMs have no zone concept.
        """
        results = list(self._vms.values())
        if labels:
            results = [v for v in results if all(v.labels.get(k) == v_ for k, v_ in labels.items())]
        return results

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        return nullcontext(address)

    def shutdown(self) -> None:
        pass

    def discover_controller(self, controller_config: config_pb2.ControllerVmConfig) -> str:
        """Return static controller address from config."""
        manual = controller_config.manual
        port = manual.port or 10000
        return f"{manual.host}:{port}"

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _create_remote_exec(
        self, host: str, manual_config: config_pb2.ManualVmConfig | config_pb2.ManualSliceConfig | None = None
    ) -> DirectSshRemoteExec:
        """Create a remote execution connection for the given host.

        Uses SSH config from the manual_config if provided (per-VM/slice overrides),
        falling back to the platform-level ssh_config.
        """
        user = "root"
        key_file: str | None = None
        connect_timeout = Duration.from_seconds(30)

        # Platform-level defaults
        if self._ssh_config:
            if self._ssh_config.user:
                user = self._ssh_config.user
            if self._ssh_config.key_file:
                key_file = self._ssh_config.key_file
            if self._ssh_config.HasField("connect_timeout"):
                connect_timeout = Duration.from_proto(self._ssh_config.connect_timeout)

        # Per-VM/slice overrides
        if manual_config is not None:
            ssh_user = getattr(manual_config, "ssh_user", "")
            ssh_key = getattr(manual_config, "ssh_key_file", "")
            if ssh_user:
                user = ssh_user
            if ssh_key:
                key_file = ssh_key

        return DirectSshRemoteExec(
            host=host,
            user=user,
            key_file=key_file,
            connect_timeout=connect_timeout,
        )

    def _return_hosts(self, hosts: list[str]) -> None:
        """Return hosts to the available pool."""
        for host in hosts:
            self._allocated_hosts.discard(host)
            if host in self._all_hosts:
                self._available_hosts.add(host)
                logger.debug("Host %s returned to pool", host)

    @property
    def available_host_count(self) -> int:
        return len(self._available_hosts)
