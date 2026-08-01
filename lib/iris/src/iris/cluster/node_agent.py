# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-owned physical node and accelerator telemetry."""

import logging
import signal
import threading
import time
from pathlib import Path

import click
from finelog.deploy.config import derive_endpoint_uri, load_finelog_config
from rigging import telemetry
from rigging.auth import BearerTokenInjector, StaticTokenProvider
from rigging.log_setup import configure_logging
from rigging.telemetry.probes import nvidia
from rigging.telemetry.probes.runner import BoundedCommandRunner

from iris.cluster.backends.k8s.node_metrics import (
    NodeMetrics,
    NodeStatsScraper,
    NodeTarget,
    publish_node_telemetry,
)
from iris.cluster.config import IrisClusterConfig, WorkerConfig, load_config
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME, resolve_endpoint_uri
from iris.cluster.platforms.k8s.service import CloudK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import AcceleratorType
from iris.cluster.worker.env_probe import (
    HardwareProbe,
    HostMetricsCollector,
    construct_worker_id,
    infer_worker_id,
    probe_hardware,
)
from iris.rpc import controller_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import EndpointServiceClientSync

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_INTERVAL = 30.0
NVIDIA_COLLECTION_INTERVAL = 10 * 60.0
K8S_API_TIMEOUT = 2.0
NODE_EXPORTER_ADDRESS = "127.0.0.1"
_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_K8S_GPU_MODEL_LABELS = ("nvidia.com/gpu.product", "gpu.nvidia.com/model")


def _telemetry_endpoint(cluster_config: IrisClusterConfig) -> str:
    spec = cluster_config.endpoints.get(LOG_SERVER_ENDPOINT_NAME)
    if cluster_config.finelog.config:
        if spec is not None:
            raise ValueError("cluster config cannot set both finelog.config and /system/log-server")
        finelog_config = load_finelog_config(cluster_config.finelog.config)
        uri, metadata = derive_endpoint_uri(finelog_config)
    elif spec is not None:
        uri, metadata = spec.uri, dict(spec.metadata)
    else:
        raise ValueError("node telemetry requires an external /system/log-server endpoint")
    return resolve_endpoint_uri(uri, metadata).rstrip("/") + "/v1/telemetry"


def _worker_telemetry_endpoint(config: WorkerConfig) -> str:
    address = config.controller_address
    if not address:
        raise ValueError("worker node telemetry requires controller_address")
    if not address.startswith(("http://", "https://")):
        address = f"http://{address}"
    interceptors: tuple[BearerTokenInjector, ...] = ()
    if config.auth_token:
        interceptors = (BearerTokenInjector(StaticTokenProvider(config.auth_token), "authorization"),)
    client = EndpointServiceClientSync(
        address=address,
        timeout_ms=10_000,
        interceptors=interceptors,
        accept_compression=IRIS_RPC_COMPRESSIONS,
        send_compression=None,
    )
    try:
        response = client.list_endpoints(
            controller_pb2.Controller.ListEndpointsRequest(prefix=LOG_SERVER_ENDPOINT_NAME, exact=True)
        )
    finally:
        client.close()
    if not response.endpoints:
        raise ConnectionError(f"controller has no {LOG_SERVER_ENDPOINT_NAME!r} endpoint")
    return response.endpoints[0].address.rstrip("/") + "/v1/telemetry"


def _configure(endpoint: str, *, node_name: str, node_uid: str, worker: str | None = None) -> None:
    attributes = {"node_name": node_name, "node_uid": node_uid, "role": str(telemetry.TelemetryRole.WORKER)}
    if worker:
        attributes["worker"] = worker
    telemetry.configure(
        endpoint=endpoint,
        service="iris-node-agent",
        attributes=attributes,
    )


def _boot_id() -> str:
    try:
        return _BOOT_ID_PATH.read_text().strip()
    except OSError:
        return ""


def _local_metrics(collector: HostMetricsCollector, node_uid: str) -> NodeMetrics:
    snapshot = collector.collect()
    host_available = snapshot.memory_total_bytes > 0
    source_replica_uid = node_uid
    if boot_id := _boot_id():
        source_replica_uid = f"{node_uid}:{boot_id}"
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


def collect_gcp_once(
    collector: HostMetricsCollector,
    target: NodeTarget,
    hardware: HardwareProbe,
) -> None:
    """Publish one bounded local-host collection pass."""
    publish_node_telemetry(target, _local_metrics(collector, target.node_uid), time.time())
    _publish_tpu_inventory(hardware)
    telemetry.record_runtime_health()


def collect_k8s_once(scraper: NodeStatsScraper, target: NodeTarget) -> None:
    """Publish one same-node exporter collection pass."""
    metrics = scraper.scrape([target])[target.name]
    publish_node_telemetry(target, metrics, time.time())
    telemetry.record_runtime_health()


def _k8s_target(k8s: CloudK8sService, node_name: str) -> NodeTarget:
    node = k8s.get_json(K8sResource.NODES, node_name)
    if node is None:
        raise ConnectionError(f"Kubernetes node {node_name!r} is not visible")
    metadata = node.get("metadata", {})
    node_uid = metadata.get("uid", "")
    if not node_uid:
        raise ValueError(f"Kubernetes node {node_name!r} has no metadata.uid")
    labels = metadata.get("labels", {})
    allocatable = node.get("status", {}).get("allocatable", {})
    gpu_count = int(allocatable.get("nvidia.com/gpu", 0))
    gpu_model = next((labels[name] for name in _K8S_GPU_MODEL_LABELS if labels.get(name)), "")
    return NodeTarget(
        name=node_name,
        node_uid=node_uid,
        internal_ip=NODE_EXPORTER_ADDRESS,
        device_type="gpu" if gpu_count or gpu_model else "cpu",
        device_variant=gpu_model,
    )


def _install_signal_handlers(stop: threading.Event) -> None:
    def handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


def run_gcp(config_path: Path) -> None:
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
    runner = BoundedCommandRunner()
    stop = threading.Event()
    _install_signal_handlers(stop)
    next_nvidia_collection = 0.0
    try:
        while not stop.is_set():
            now = time.monotonic()
            collect_gcp_once(collector, target, hardware)
            if now >= next_nvidia_collection:
                nvidia.collect(runner)
                next_nvidia_collection = now + NVIDIA_COLLECTION_INTERVAL
            stop.wait(DEFAULT_COLLECTION_INTERVAL)
    finally:
        runner.cancel(2.0)
        telemetry.shutdown(5.0)


def run_k8s(config_path: Path, node_name: str, namespace: str) -> None:
    config = load_config(config_path)
    endpoint = _telemetry_endpoint(config)
    k8s = CloudK8sService(namespace=namespace, timeout=K8S_API_TIMEOUT)
    target = _k8s_target(k8s, node_name)
    _configure(endpoint, node_name=target.name, node_uid=target.node_uid)
    scraper = NodeStatsScraper(k8s)
    stop = threading.Event()
    _install_signal_handlers(stop)
    try:
        while not stop.is_set():
            collect_k8s_once(scraper, target)
            stop.wait(DEFAULT_COLLECTION_INTERVAL)
    finally:
        telemetry.shutdown(5.0)


@click.group()
def cli() -> None:
    """Run Iris physical node telemetry."""


@cli.command("gcp")
@click.option("--worker-config", type=click.Path(path_type=Path, exists=True), required=True)
def gcp_command(worker_config: Path) -> None:
    """Run beside an Iris worker on a GCP VM."""
    configure_logging(level=logging.INFO)
    run_gcp(worker_config)


@cli.command("k8s")
@click.option("--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--node-name", envvar="IRIS_NODE_NAME", required=True)
@click.option("--namespace", envvar="IRIS_NAMESPACE", required=True)
def k8s_command(config_path: Path, node_name: str, namespace: str) -> None:
    """Run once per Kubernetes node."""
    configure_logging(level=logging.INFO)
    run_k8s(config_path, node_name, namespace)


if __name__ == "__main__":
    cli()
