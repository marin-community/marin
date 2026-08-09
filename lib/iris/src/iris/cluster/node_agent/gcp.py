# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect physical node and accelerator telemetry on GCP workers."""

import threading
import time
from pathlib import Path

from rigging import telemetry
from rigging.auth import BearerTokenInjector, StaticTokenProvider
from rigging.telemetry.probes import nvidia

from iris.cluster.config import WorkerConfig
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, TELEMETRY_ENDPOINT_PATH
from iris.cluster.node_agent import SERVICE_NAME
from iris.cluster.node_agent.metrics import NodeMetrics, NodeTarget, publish_node_telemetry
from iris.cluster.types import AcceleratorType
from iris.cluster.worker.env_probe import (
    HardwareProbe,
    HostMetricsCollector,
    construct_worker_id,
    infer_worker_id,
    probe_hardware,
)
from iris.rpc.resource_client import ResourceRpcClient

DEFAULT_COLLECTION_INTERVAL = 30.0
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def _worker_telemetry_endpoint(config: WorkerConfig) -> str:
    address = config.controller_address
    if not address:
        raise ValueError("worker node telemetry requires controller_address")
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    interceptors: tuple[BearerTokenInjector, ...] = ()
    if config.auth_token:
        interceptors = (BearerTokenInjector(StaticTokenProvider(config.auth_token), "authorization"),)
    client = ResourceRpcClient(
        address,
        timeout_ms=10_000,
        interceptors=interceptors,
    )
    try:
        endpoints = client.resolve_endpoints(LOG_SERVER_ENDPOINT_NAME)
    finally:
        client.close()
    if not endpoints:
        raise ConnectionError(f"controller has no {LOG_SERVER_ENDPOINT_NAME!r} endpoint")
    return endpoints[0].address.rstrip("/") + TELEMETRY_ENDPOINT_PATH


def _configure(endpoint: str, *, node_name: str, node_uid: str, worker: str | None = None) -> None:
    attributes = {"node_name": node_name, "node_uid": node_uid, "role": str(telemetry.TelemetryRole.WORKER)}
    if worker:
        attributes["worker"] = worker
    telemetry.configure(
        endpoint=endpoint,
        service=SERVICE_NAME,
        attributes=attributes,
    )


def _boot_id() -> str:
    boot_id = _BOOT_ID_PATH.read_text().strip()
    if not boot_id:
        raise ValueError(f"host boot identity is empty: {_BOOT_ID_PATH}")
    return boot_id


def _local_metrics(collector: HostMetricsCollector, node_uid: str) -> NodeMetrics:
    snapshot = collector.collect()
    host_available = snapshot.memory_total_bytes > 0
    source_replica_uid = f"{node_uid}:{_boot_id()}"
    return NodeMetrics(
        cpu_pct=float(snapshot.host_cpu_percent),
        mem_used_bytes=snapshot.memory_used_bytes,
        mem_total_bytes=snapshot.memory_total_bytes,
        disk_used_bytes=snapshot.disk_used_bytes,
        disk_total_bytes=snapshot.disk_total_bytes,
        net_recv_bytes=snapshot.net_recv_bytes,
        net_sent_bytes=snapshot.net_sent_bytes,
        host_available=host_available,
        host_source_kind="procfs",
        host_source_replica_uid=source_replica_uid,
    )


def _publish_tpu_inventory(hardware: HardwareProbe) -> None:
    if not hardware.tpu_type:
        return
    attributes = {
        "device_kind": "tpu",
        "tpu_type": hardware.tpu_type,
        **telemetry.snapshot_attributes("gcp_metadata", telemetry.CURRENT_SNAPSHOT),
    }
    if hardware.tpu_name:
        attributes["tpu_name"] = hardware.tpu_name
    if hardware.tpu_worker_id:
        attributes["worker_index"] = hardware.tpu_worker_id
    telemetry.gauge("hardware_inventory").set(1.0, attributes=attributes)
    telemetry.gauge("hardware_source_available").set(
        1.0,
        attributes=telemetry.snapshot_attributes("gcp_metadata", telemetry.CURRENT_SNAPSHOT),
    )


def collect_once(
    collector: HostMetricsCollector,
    target: NodeTarget,
    hardware: HardwareProbe,
) -> None:
    """Publish one bounded local-host collection pass."""
    publish_node_telemetry(target, _local_metrics(collector, target.node_uid), time.time())
    _publish_tpu_inventory(hardware)
    telemetry.record_runtime_health()


def run(config_path: Path, stop: threading.Event) -> None:
    """Collect telemetry until the process receives a shutdown signal."""
    config = WorkerConfig.model_validate_json(config_path.read_text())
    hardware = probe_hardware()
    node_name = hardware.gce_instance_name or hardware.hostname
    node_uid = hardware.gce_instance_uid or hardware.gce_instance_name or hardware.hostname
    endpoint = _worker_telemetry_endpoint(config)
    worker = config.worker_id
    if not worker and config.slice_id:
        worker_index = int(hardware.tpu_worker_id) if hardware.tpu_worker_id else 0
        worker = construct_worker_id(config.slice_id, worker_index)
    if not worker:
        worker = infer_worker_id(hardware)
    _configure(endpoint, node_name=node_name, node_uid=node_uid, worker=worker)

    target = NodeTarget(
        name=node_name,
        node_uid=node_uid,
        internal_ip=hardware.ip_address,
        device_type="gpu" if config.accelerator_type == AcceleratorType.GPU or hardware.gpu_count else "cpu",
        device_variant=config.accelerator_variant or hardware.gpu_name,
    )
    collector = HostMetricsCollector(disk_path=config.cache_dir)
    nvidia_probe = nvidia.start() if target.device_type == "gpu" else None
    try:
        while not stop.is_set():
            collect_once(collector, target, hardware)
            stop.wait(DEFAULT_COLLECTION_INTERVAL)
    finally:
        if nvidia_probe is not None:
            nvidia_probe.shutdown(2.0)
        telemetry.shutdown(5.0)
