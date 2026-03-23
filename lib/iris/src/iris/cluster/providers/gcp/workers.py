# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GcpWorkerProvider — worker infrastructure management for GCP.

Implements the WorkerInfraProvider protocol: create/list slices and VMs
for the Autoscaler and ScalingGroup. Manages GCE instances (standalone VMs)
and TPU slices via GcpService.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable
import time
import urllib.error
import urllib.request

from iris.cluster.providers.gcp.bootstrap import (
    build_worker_bootstrap_script,
    rewrite_ghcr_to_ar_remote,
    zone_to_multi_region,
)
from iris.cluster.providers.gcp.handles import (
    CloudSliceState,
    GcpSliceHandle,
    GcpStandaloneWorkerHandle,
    GcpVmSliceHandle,
    _ACTIVE_VM_SLICE_STATES,
    _build_vm_slice_id,
)
from iris.cluster.providers.gcp.service import (
    CloudGcpService,
    GcpService,
    TpuCreateRequest,
    VmCreateRequest,
)
from iris.cluster.providers.types import (
    InfraError,
    Labels,
    SliceHandle,
    generate_slice_suffix,
)
from iris.cluster.service_mode import ServiceMode
from iris.cluster.worker.env_probe import construct_worker_id
from iris.cluster.providers.remote_exec import GceRemoteExec
from iris.rpc import config_pb2
from iris.time_utils import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


def _spawn_bootstrap_thread(
    handle: GcpSliceHandle | GcpVmSliceHandle,
    bootstrap_fn: Callable[[], None],
) -> None:
    """Launch a daemon thread that runs bootstrap_fn, marking the handle FAILED on error."""

    def _run():
        try:
            bootstrap_fn()
        except Exception as e:
            logger.error("Bootstrap failed for slice %s: %s", handle.slice_id, e)
            with handle._bootstrap_lock:
                handle._bootstrap_state = CloudSliceState.FAILED

    threading.Thread(target=_run, name=f"bootstrap-{handle.slice_id}", daemon=True).start()


DEFAULT_MACHINE_TYPE = "n2-standard-4"
DEFAULT_BOOT_DISK_SIZE_GB = 50
# pd-ssd provides ~6000 IOPS vs ~38 on pd-standard, critical for controller DB
DEFAULT_BOOT_DISK_TYPE = "pd-ssd"


def _validate_slice_config(config: config_pb2.SliceConfig) -> None:
    """Validate required fields on a SliceConfig before creating a GCP slice.

    Raises ValueError listing all missing fields so operators can fix config
    in one pass rather than discovering issues one-by-one.
    """
    missing: list[str] = []
    violations: list[str] = []
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if config.gcp.mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
        if not config.gcp.machine_type:
            missing.append("gcp.machine_type")
        if config.num_vms != 1:
            violations.append("GCP VM slice mode requires num_vms=1")
        if config.preemptible:
            violations.append("GCP VM slice mode does not support preemptible instances")
        if config.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
            violations.append("GCP VM slice mode requires accelerator_type=cpu")
        if config.accelerator_variant:
            violations.append("GCP VM slice mode does not support accelerator_variant")
    else:
        if not config.accelerator_variant:
            missing.append("accelerator_variant")
        if not config.gcp.runtime_version:
            missing.append("gcp.runtime_version")
    errors: list[str] = []
    if missing:
        errors.append(f"SliceConfig is missing required fields: {', '.join(missing)}")
    errors.extend(violations)
    if errors:
        raise ValueError("; ".join(errors))


def _validate_vm_config(config: config_pb2.VmConfig) -> None:
    """Validate required fields on a VmConfig before creating a GCE instance."""
    missing: list[str] = []
    if not config.name:
        missing.append("name")
    if not config.gcp.zone:
        missing.append("gcp.zone")
    if missing:
        raise ValueError(f"VmConfig is missing required fields: {', '.join(missing)}")


class GcpWorkerProvider:
    """Worker infrastructure management for Google Cloud Platform.

    Manages GCE instances (standalone VMs) and TPU slices via GcpService.
    Zones are stored from GcpPlatformConfig for list_all_slices(); per-slice
    zones come from SliceConfig.
    """

    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        gcp_service: GcpService | None = None,
    ):
        self._project_id = gcp_config.project_id
        self._label_prefix = label_prefix
        self._iris_labels = Labels(label_prefix)
        self._ssh_config = ssh_config
        self._zones = list(gcp_config.zones)
        self._gcp: GcpService = gcp_service or CloudGcpService(project_id=self._project_id)

    @property
    def gcp_service(self) -> GcpService:
        return self._gcp

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def label_prefix(self) -> str:
        return self._label_prefix

    @property
    def iris_labels(self) -> Labels:
        return self._iris_labels

    def resolve_image(self, image: str, zone: str | None = None) -> str:
        """Rewrite ``ghcr.io/`` images to the AR remote repo for *zone*'s continent.

        Non-GHCR images pass through unchanged.
        """
        if not image.startswith("ghcr.io/"):
            return image
        if not zone:
            raise ValueError("zone is required for GHCR→AR image rewriting on GCP")
        multi_region = zone_to_multi_region(zone)
        if not multi_region:
            return image
        return rewrite_ghcr_to_ar_remote(image, multi_region, self._project_id)

    def _best_effort_delete_tpu(self, slice_id: str, zone: str) -> None:
        """Try to delete a TPU VM that may have been partially created."""
        logger.info("Best-effort async cleanup of TPU %s in %s", slice_id, zone)
        try:
            self._gcp.tpu_delete(slice_id, zone)
        except InfraError as e:
            logger.warning("Cleanup of TPU %s failed: %s", slice_id, e)

    def _best_effort_delete_vm(self, vm_name: str, zone: str) -> None:
        """Try to delete a GCE VM that may have been partially created."""
        logger.info("Best-effort cleanup of VM %s in %s", vm_name, zone)
        try:
            self._gcp.vm_delete(vm_name, zone)
        except InfraError as e:
            logger.warning("Cleanup of VM %s failed: %s", vm_name, e)

    def create_vm(self, config: config_pb2.VmConfig) -> GcpStandaloneWorkerHandle:
        """Create a GCE instance. Returns a handle with SSH and label/metadata support."""
        _validate_vm_config(config)
        gcp = config.gcp
        zone = gcp.zone
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = gcp.boot_disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        request = VmCreateRequest(
            name=config.name,
            zone=zone,
            machine_type=machine_type,
            labels=dict(config.labels),
            metadata=dict(config.metadata),
            disk_size_gb=boot_disk_size,
            boot_disk_type=DEFAULT_BOOT_DISK_TYPE,
            image_family="debian-12",
            image_project="debian-cloud",
        )

        logger.info("Creating GCE instance: %s (zone=%s, type=%s)", config.name, zone, machine_type)
        try:
            vm_info = self._gcp.vm_create(request)
        except InfraError:
            self._best_effort_delete_vm(config.name, zone)
            raise

        remote_exec = GceRemoteExec(
            project_id=self._project_id,
            zone=zone,
            vm_name=config.name,
        )

        return GcpStandaloneWorkerHandle(
            _vm_id=construct_worker_id(config.name, 0),
            _internal_address=vm_info.internal_ip,
            _external_address=vm_info.external_ip,
            _gce_vm_name=config.name,
            _zone=zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp,
            _remote_exec=remote_exec,
        )

    def create_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a GCP-backed slice (TPU pod or single VM).

        In LOCAL mode, delegates to InMemoryGcpService to spawn local worker threads
        and returns a LocalSliceHandle.
        """
        if self._gcp.mode == ServiceMode.LOCAL:
            return self._create_local_slice(config, worker_config)
        _validate_slice_config(config)
        if config.gcp.mode == config_pb2.GcpSliceConfig.GCP_SLICE_MODE_VM:
            return self._create_vm_slice(config, worker_config)
        return self._create_tpu_slice(config, worker_config)

    def _create_local_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> SliceHandle:
        """Create a local slice via InMemoryGcpService(LOCAL)."""
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"
        return self._gcp.create_local_slice(slice_id, config, worker_config)

    def _create_tpu_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> GcpSliceHandle:
        """Create a TPU slice via GcpService.

        When worker_config is provided the bootstrap script is passed as TPU
        metadata (startup-script) so each worker VM self-bootstraps on first
        boot. Bootstrap progress is monitored via health endpoint polling.
        """
        gcp = config.gcp
        slice_id = f"{config.name_prefix}-{generate_slice_suffix()}"

        metadata: dict[str, str] = {}
        if worker_config:
            worker_config.docker_image = self.resolve_image(worker_config.docker_image, zone=gcp.zone)
            worker_config.slice_id = slice_id
            startup_script = build_worker_bootstrap_script(worker_config)
            metadata["startup-script"] = startup_script

        request = TpuCreateRequest(
            name=slice_id,
            zone=gcp.zone,
            accelerator_type=config.accelerator_variant,
            runtime_version=gcp.runtime_version,
            labels=dict(config.labels),
            metadata=metadata,
            preemptible=config.preemptible,
        )

        logger.info("Creating TPU slice: %s (type=%s, zone=%s)", slice_id, config.accelerator_variant, gcp.zone)
        try:
            self._gcp.tpu_create(request)
        except InfraError:
            self._best_effort_delete_tpu(slice_id, gcp.zone)
            raise

        handle = GcpSliceHandle(
            _slice_id=slice_id,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _labels=dict(config.labels),
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _accelerator_variant=config.accelerator_variant,
            _gcp_service=self._gcp,
            _ssh_config=self._ssh_config,
            _bootstrapping=worker_config is not None,
        )

        if worker_config:
            _spawn_bootstrap_thread(
                handle,
                lambda: _run_tpu_bootstrap(self._gcp, self._project_id, handle, worker_config),
            )

        return handle

    def _create_vm_slice(
        self,
        config: config_pb2.SliceConfig,
        worker_config: config_pb2.WorkerConfig | None = None,
    ) -> GcpVmSliceHandle:
        """Create a single GCE VM that behaves as a one-worker slice.

        When worker_config is provided the bootstrap script is passed as GCE
        startup-script metadata so the VM self-bootstraps on first boot.
        """
        gcp = config.gcp
        slice_id = _build_vm_slice_id(config.name_prefix, generate_slice_suffix())
        vm_name = slice_id
        machine_type = gcp.machine_type or DEFAULT_MACHINE_TYPE
        boot_disk_size = config.disk_size_gb or DEFAULT_BOOT_DISK_SIZE_GB

        labels = dict(config.labels)
        labels[self._iris_labels.iris_slice_id] = slice_id

        startup_script: str | None = None
        if worker_config:
            worker_config.docker_image = self.resolve_image(worker_config.docker_image, zone=gcp.zone)
            worker_config.worker_id = construct_worker_id(slice_id, 0)
            startup_script = build_worker_bootstrap_script(worker_config)

        request = VmCreateRequest(
            name=vm_name,
            zone=gcp.zone,
            machine_type=machine_type,
            labels=labels,
            disk_size_gb=boot_disk_size,
            image_family="debian-12",
            image_project="debian-cloud",
            startup_script=startup_script,
        )

        logger.info("Creating VM slice: %s (vm=%s, zone=%s, type=%s)", slice_id, vm_name, gcp.zone, machine_type)
        try:
            self._gcp.vm_create(request)
        except InfraError:
            self._best_effort_delete_vm(vm_name, gcp.zone)
            raise

        handle = GcpVmSliceHandle(
            _slice_id=slice_id,
            _vm_name=vm_name,
            _zone=gcp.zone,
            _project_id=self._project_id,
            _gcp_service=self._gcp,
            _labels=labels,
            _created_at=Timestamp.now(),
            _label_prefix=self._label_prefix,
            _ssh_config=self._ssh_config,
            _bootstrapping=worker_config is not None,
        )

        if worker_config:
            _spawn_bootstrap_thread(
                handle,
                lambda: _run_vm_slice_bootstrap(self._gcp, handle, worker_config),
            )

        return handle

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpSliceHandle | GcpVmSliceHandle]:
        """List TPU and VM slices across zones, optionally filtered by labels."""
        if self._gcp.mode == ServiceMode.LOCAL:
            return self._gcp.get_local_slices(labels)  # type: ignore[return-value]

        handles: list[GcpSliceHandle | GcpVmSliceHandle] = []

        tpu_infos = self._gcp.tpu_list(zones, labels)
        for tpu in tpu_infos:
            if tpu.state not in ("READY", "CREATING"):
                logger.info("Skipping TPU %s in state %s", tpu.name, tpu.state)
                continue
            handles.append(
                GcpSliceHandle(
                    _slice_id=tpu.name,
                    _zone=tpu.zone,
                    _project_id=self._project_id,
                    _labels=tpu.labels,
                    _created_at=tpu.created_at,
                    _label_prefix=self._label_prefix,
                    _accelerator_variant=tpu.accelerator_type,
                    _gcp_service=self._gcp,
                    _ssh_config=self._ssh_config,
                    _state=tpu.state,
                )
            )

        vm_infos = self._gcp.vm_list(zones, labels)
        for vm in vm_infos:
            if vm.status not in _ACTIVE_VM_SLICE_STATES:
                logger.info("Skipping VM instance %s in state %s", vm.name, vm.status)
                continue
            slice_id = vm.labels.get(self._iris_labels.iris_slice_id, "")
            if not slice_id:
                continue
            handles.append(
                GcpVmSliceHandle(
                    _slice_id=slice_id,
                    _vm_name=vm.name,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _labels=vm.labels,
                    _created_at=vm.created_at,
                    _label_prefix=self._label_prefix,
                    _ssh_config=self._ssh_config,
                )
            )

        return handles

    def list_all_slices(self) -> list[GcpSliceHandle | GcpVmSliceHandle]:
        """List all slices managed by this cluster.

        Uses project-wide queries (empty zones = all zones) via GcpService,
        filtered by iris-{prefix}-managed=true.
        """
        managed_labels = {self._iris_labels.iris_managed: "true"}

        if self._gcp.mode == ServiceMode.LOCAL:
            return self._gcp.get_local_slices(managed_labels)  # type: ignore[return-value]

        tpu_infos = self._gcp.tpu_list(zones=[], labels=managed_labels)
        vm_infos = self._gcp.vm_list(zones=[], labels=managed_labels)

        handles: list[GcpSliceHandle | GcpVmSliceHandle] = []

        for tpu in tpu_infos:
            if tpu.state not in ("READY", "CREATING"):
                continue
            handles.append(
                GcpSliceHandle(
                    _slice_id=tpu.name,
                    _zone=tpu.zone,
                    _project_id=self._project_id,
                    _labels=tpu.labels,
                    _created_at=tpu.created_at,
                    _label_prefix=self._label_prefix,
                    _accelerator_variant=tpu.accelerator_type,
                    _gcp_service=self._gcp,
                    _ssh_config=self._ssh_config,
                    _state=tpu.state,
                )
            )

        for vm in vm_infos:
            if vm.status not in _ACTIVE_VM_SLICE_STATES:
                continue
            slice_id = vm.labels.get(self._iris_labels.iris_slice_id, "")
            if not slice_id:
                continue
            handles.append(
                GcpVmSliceHandle(
                    _slice_id=slice_id,
                    _vm_name=vm.name,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _labels=vm.labels,
                    _created_at=vm.created_at,
                    _label_prefix=self._label_prefix,
                    _ssh_config=self._ssh_config,
                )
            )

        logger.info("list_all_slices: found %d managed slices", len(handles))
        return handles

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[GcpStandaloneWorkerHandle]:
        """List GCE instances across zones, optionally filtered by labels."""
        vm_infos = self._gcp.vm_list(zones, labels)

        handles: list[GcpStandaloneWorkerHandle] = []
        for vm in vm_infos:
            remote_exec = GceRemoteExec(
                project_id=self._project_id,
                zone=vm.zone,
                vm_name=vm.name,
            )
            handles.append(
                GcpStandaloneWorkerHandle(
                    _vm_id=construct_worker_id(vm.name, 0),
                    _internal_address=vm.internal_ip,
                    _external_address=vm.external_ip,
                    _gce_vm_name=vm.name,
                    _zone=vm.zone,
                    _project_id=self._project_id,
                    _gcp_service=self._gcp,
                    _remote_exec=remote_exec,
                )
            )
        return handles

    def shutdown(self) -> None:
        self._gcp.shutdown()


# ============================================================================
# Bootstrap monitoring (module-level functions)
# ============================================================================


def _run_tpu_bootstrap(
    gcp_service: GcpService,
    project_id: str,
    handle: GcpSliceHandle,
    worker_config: config_pb2.WorkerConfig,
    poll_interval: float = 10.0,
    cloud_ready_timeout: float = 600.0,
    bootstrap_timeout: float = 600.0,
) -> None:
    """Monitor TPU startup-script bootstrap via health endpoint polling.

    Phase 1: Wait for cloud READY with all worker IPs.
    Phase 2: Poll worker health endpoints until all respond healthy.
    On timeout: query Cloud Logging for [iris-init] entries for diagnostics.
    """
    deadline = Deadline.from_now(Duration.from_seconds(cloud_ready_timeout))
    while not deadline.expired():
        cloud_status = handle._describe_cloud()
        if cloud_status.state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
            raise InfraError(f"Slice {handle.slice_id} entered {cloud_status.state} while waiting for cloud READY")
        if cloud_status.state == CloudSliceState.READY:
            all_have_ips = all(w.internal_address for w in cloud_status.workers)
            if all_have_ips and len(cloud_status.workers) == cloud_status.worker_count:
                break
            logger.info(
                "Slice %s is READY but only %d/%d workers have IPs, waiting...",
                handle.slice_id,
                sum(1 for w in cloud_status.workers if w.internal_address),
                cloud_status.worker_count,
            )
        time.sleep(poll_interval)
    else:
        raise InfraError(f"Slice {handle.slice_id} did not reach cloud READY within {cloud_ready_timeout}s")

    workers = cloud_status.workers
    worker_addrs = [(w.worker_id, w.internal_address) for w in workers]
    healthy_workers: set[str] = set()
    health_deadline = Deadline.from_now(Duration.from_seconds(bootstrap_timeout))

    logger.info(
        "Polling health endpoints for %d workers in slice %s",
        len(worker_addrs),
        handle.slice_id,
    )

    while not health_deadline.expired():
        for worker_id, addr in worker_addrs:
            if worker_id in healthy_workers:
                continue
            try:
                resp = urllib.request.urlopen(
                    f"http://{addr}:{worker_config.port}/health",
                    timeout=5,
                )
                if resp.status == 200:
                    healthy_workers.add(worker_id)
                    logger.info("Worker %s is healthy", worker_id)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
                pass

        if len(healthy_workers) == len(worker_addrs):
            break
        time.sleep(poll_interval)
    else:
        _fetch_bootstrap_logs(project_id, handle)
        raise InfraError(
            f"TPU slice {handle.slice_id} bootstrap timed out: "
            f"{len(healthy_workers)}/{len(worker_addrs)} workers healthy"
        )

    logger.info("Bootstrap completed for TPU slice %s (%d workers)", handle.slice_id, len(workers))
    with handle._bootstrap_lock:
        handle._bootstrap_state = CloudSliceState.READY


def _fetch_bootstrap_logs(project_id: str, handle: GcpSliceHandle) -> None:
    """Fetch [iris-init] log entries from Cloud Logging for diagnostics."""
    log_filter = (
        f'resource.type="gce_instance" '
        f'textPayload:"[iris-init]" '
        f'labels."compute.googleapis.com/resource_name":"{handle._slice_id}"'
    )
    cmd = [
        "gcloud",
        "logging",
        "read",
        log_filter,
        f"--project={project_id}",
        "--freshness=30m",
        "--limit=200",
        "--format=value(textPayload)",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        logger.warning("Cloud Logging query timed out for %s", handle.slice_id)
        return

    if result.returncode == 0 and result.stdout.strip():
        logger.error("Bootstrap logs for %s:\n%s", handle.slice_id, result.stdout)
    else:
        logger.warning(
            "Could not fetch Cloud Logging for %s (rc=%d): %s",
            handle.slice_id,
            result.returncode,
            result.stderr.strip(),
        )


def _run_vm_slice_bootstrap(
    gcp_service: GcpService,
    handle: GcpVmSliceHandle,
    worker_config: config_pb2.WorkerConfig,
    poll_interval: float = 5.0,
    cloud_ready_timeout: float = 600.0,
) -> None:
    """Monitor GCE startup-script bootstrap via serial port output.

    The bootstrap script was baked into VM metadata at creation time, so the
    VM self-bootstraps on first boot. This method polls serial port output
    for [iris-init] log lines until the script emits ``Bootstrap complete``
    or the timeout expires.
    """
    deadline = Deadline.from_now(Duration.from_seconds(cloud_ready_timeout))
    poll_duration = Duration.from_seconds(poll_interval)

    while not deadline.expired():
        cloud_status = handle._describe_cloud()
        if cloud_status.state in (CloudSliceState.FAILED, CloudSliceState.DELETING):
            raise InfraError(f"VM slice {handle.slice_id} entered {cloud_status.state} while waiting for cloud READY")
        if cloud_status.state == CloudSliceState.READY and cloud_status.workers:
            if cloud_status.workers[0].internal_address:
                break
        time.sleep(poll_duration.to_seconds())
    else:
        raise InfraError(f"VM slice {handle.slice_id} did not reach cloud READY within {cloud_ready_timeout}s")

    serial_offset = 0
    bootstrap_complete = False
    bootstrap_failed = False

    while not deadline.expired():
        output = gcp_service.vm_get_serial_port_output(handle._vm_name, handle._zone, start=serial_offset)
        if output:
            for line in output.splitlines():
                if "[iris-init]" in line:
                    logger.info("[%s serial] %s", handle.slice_id, line.strip())
                if "Bootstrap complete" in line:
                    bootstrap_complete = True
                if "[iris-init] ERROR" in line:
                    bootstrap_failed = True

            serial_offset += len(output)

        if bootstrap_complete:
            break
        if bootstrap_failed:
            raise InfraError(f"Startup-script bootstrap failed for VM slice {handle.slice_id} (see serial output above)")

        time.sleep(poll_duration.to_seconds())
    else:
        raise InfraError(f"VM slice {handle.slice_id} startup-script did not complete within {cloud_ready_timeout}s")

    logger.info("Bootstrap completed for VM slice %s (via startup-script)", handle.slice_id)
    with handle._bootstrap_lock:
        handle._bootstrap_state = CloudSliceState.READY
