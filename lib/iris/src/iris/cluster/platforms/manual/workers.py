# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ManualWorkerProvider for pre-existing hosts.

Implements WorkerInfraProvider for manually managed hosts. Hosts are drawn from
a configured pool and returned on slice termination. Remote execution uses
DirectSshRemoteExec (raw ssh, no gcloud).
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from rigging.timing import Duration, Timestamp

from iris.cluster.config import (
    ManualSliceConfig,
    ManualVmConfig,
    SliceConfig,
    SshConfig,
    VmConfig,
    WorkerConfig,
)
from iris.cluster.platforms._worker_base import RemoteExecWorkerBase
from iris.cluster.platforms.gcp.worker_bootstrap import build_worker_bootstrap_script
from iris.cluster.platforms.remote_exec import DirectSshRemoteExec
from iris.cluster.platforms.types import (
    CloudSliceState,
    CloudWorkerState,
    InfraError,
    Labels,
    ListedSlice,
    SliceStatus,
    WorkerStatus,
    generate_slice_suffix,
)
from iris.cluster.worker.env_probe import construct_worker_id

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

    def terminate(self, *, wait: bool = False) -> None:
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
        _worker_port: int,
        _ssh_connections: list[DirectSshRemoteExec],
        _on_terminate: Callable[[list[str]], None] | None = None,
        _bootstrapping: bool = False,
    ):
        self._slice_id = _slice_id
        self._hosts = _hosts
        self._labels = _labels
        self._created_at = _created_at
        self._label_prefix = _label_prefix
        self._worker_port = _worker_port
        self._iris_labels = Labels(_label_prefix)
        self._ssh_connections = _ssh_connections
        self._on_terminate = _on_terminate
        self._terminated = False
        self._bootstrapping = _bootstrapping
        # Bootstrap state: None means bootstrap not yet completed.
        # Set by the provider's internal bootstrap thread.
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
        return self._labels.get(self._iris_labels.iris_scale_group, "")

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
                _vm_id=construct_worker_id(self._slice_id, i),
                _internal_address=host,
                _port=self._worker_port,
                _remote_exec=ssh,
            )
            for i, (host, ssh) in enumerate(zip(self._hosts, self._ssh_connections, strict=True))
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

    def terminate(self, *, wait: bool = False) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self._on_terminate:
            self._on_terminate(list(self._hosts))
        logger.info("Terminated manual slice %s (%d hosts)", self._slice_id, len(self._hosts))


# ============================================================================
# ManualWorkerProvider
# ============================================================================


class ManualWorkerProvider:
    """Worker infrastructure management for pre-existing hosts.

    Implements WorkerInfraProvider. Hosts are drawn from a configured pool.
    On slice termination, hosts are returned to the pool for reuse.
    SSH uses DirectSshRemoteExec.
    """

    def __init__(
        self,
        label_prefix: str,
        worker_port: int,
        ssh_config: SshConfig | None = None,
        hosts: list[str] | None = None,
    ):
        self._label_prefix = label_prefix
        self._worker_port = worker_port
        self._iris_labels = Labels(label_prefix)
        self._ssh_config = ssh_config
        self._all_hosts = list(hosts or [])
        self._available_hosts: set[str] = set(self._all_hosts)
        self._allocated_hosts: set[str] = set()
        self._slices: dict[str, ManualSliceHandle] = {}
        self._vms: dict[str, ManualStandaloneWorkerHandle] = {}

    @property
    def label_prefix(self) -> str:
        return self._label_prefix

    @property
    def iris_labels(self) -> Labels:
        return self._iris_labels

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        return image

    def create_vm(self, config: VmConfig) -> ManualStandaloneWorkerHandle:
        """Allocate a host from the pool for a standalone VM (e.g., controller)."""
        manual = config.manual
        host = manual.host

        if host:
            if host in self._allocated_hosts:
                raise RuntimeError(f"Host {host} is already allocated")
            self._available_hosts.discard(host)
        elif self._available_hosts:
            host = self._available_hosts.pop()
        else:
            raise RuntimeError("No hosts available in manual provider pool")

        self._allocated_hosts.add(host)
        remote_exec = self._create_remote_exec(host, manual)

        def on_terminate() -> None:
            self._return_hosts([host])
            self._vms.pop(config.name, None)

        handle = ManualStandaloneWorkerHandle(
            _vm_id=config.name,
            _internal_address=host,
            _port=self._worker_port,
            _remote_exec=remote_exec,
            _labels=dict(config.labels),
            _metadata=dict(config.metadata),
            _on_terminate=on_terminate,
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(
        self,
        config: SliceConfig,
        worker_config: WorkerConfig | None = None,
    ) -> ManualSliceHandle:
        """Allocate hosts from the pool for a slice.

        When worker_config is provided, spawns a background thread that runs
        the bootstrap script on each worker. The handle's describe() composites
        bootstrap state with the base state.
        """
        manual = config.manual
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"

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
            _worker_port=self._worker_port,
            _ssh_connections=ssh_connections,
            _on_terminate=on_terminate,
            _bootstrapping=worker_config is not None,
        )
        self._slices[slice_id] = handle

        if worker_config:

            def _bootstrap_worker() -> None:
                try:
                    _run_bootstrap(handle, worker_config)
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

    def list_all_slices(self) -> list[ListedSlice]:
        """List autoscaler-managed slices paired with cloud state.

        Excludes slices tagged iris_manual=true (operator-created via
        `iris cluster create-slice`). Manual slices have no real cloud
        lifecycle; non-terminated ones report READY.
        """
        all_managed = self.list_slices(zones=[], labels={self._iris_labels.iris_managed: "true"})
        manual_label = self._iris_labels.iris_manual
        return [
            ListedSlice(
                handle=s,
                state=CloudSliceState.DELETING if s._terminated else CloudSliceState.READY,
            )
            for s in all_managed
            if s.labels.get(manual_label) != "true"
        ]

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

    def shutdown(self) -> None:
        pass

    @property
    def available_host_count(self) -> int:
        return len(self._available_hosts)

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _create_remote_exec(
        self,
        host: str,
        manual_config: ManualVmConfig | ManualSliceConfig | None = None,
    ) -> DirectSshRemoteExec:
        """Create a remote execution connection for the given host.

        Uses SSH config from manual_config if provided (per-VM/slice overrides),
        falling back to the provider-level ssh_config.
        """
        user = "root"
        key_file: str | None = None
        connect_timeout = Duration.from_seconds(30)

        if self._ssh_config:
            if self._ssh_config.user:
                user = self._ssh_config.user
            if self._ssh_config.key_file:
                key_file = self._ssh_config.key_file
            if self._ssh_config.connect_timeout is not None:
                connect_timeout = self._ssh_config.connect_timeout

        if manual_config is not None:
            if manual_config.ssh_user:
                user = manual_config.ssh_user
            if manual_config.ssh_key_file:
                key_file = manual_config.ssh_key_file

        return DirectSshRemoteExec(
            host=host,
            user=user,
            key_file=key_file,
            connect_timeout=connect_timeout,
        )

    def _return_hosts(self, hosts: list[str]) -> None:
        for host in hosts:
            self._allocated_hosts.discard(host)
            if host in self._all_hosts:
                self._available_hosts.add(host)
                logger.debug("Host %s returned to pool", host)


# ============================================================================
# Bootstrap helpers
# ============================================================================


def _run_bootstrap(
    handle: ManualSliceHandle,
    worker_config: WorkerConfig,
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
                raise InfraError(f"Worker {worker.worker_id} in slice {handle.slice_id} has no internal address")
            if not worker.wait_for_connection(timeout=Duration.from_seconds(300)):
                raise InfraError(f"Worker {worker.worker_id} in slice {handle.slice_id} not reachable via SSH")
            per_worker_config = worker_config.model_copy(deep=True)
            per_worker_config.worker_id = worker.worker_id
            script = build_worker_bootstrap_script(per_worker_config)
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
        raise InfraError(
            f"Bootstrap failed for {len(errors)}/{len(workers)} workers in slice {handle.slice_id}: "
            f"{', '.join(failed_ids)}: {errors[0][1]}"
        )

    logger.info("Bootstrap completed for slice %s (%d workers)", handle.slice_id, len(workers))
    with handle._bootstrap_lock:
        handle._bootstrap_state = CloudSliceState.READY
