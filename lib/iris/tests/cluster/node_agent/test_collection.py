# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end node-agent telemetry publication tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from iris.cluster.node_agent import gcp, kubernetes
from iris.cluster.node_agent.metrics import NodeTarget
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.worker.env_probe import HardwareProbe
from iris.rpc import job_pb2
from rigging import telemetry
from rigging.testing import RecordingTelemetryTransport

NODE_EXPORTER_TEXT = """
node_memory_MemAvailable_bytes 100
node_memory_MemTotal_bytes 300
node_cpu_seconds_total{cpu="0",mode="idle"} 20
node_cpu_seconds_total{cpu="0",mode="user"} 10
node_filesystem_size_bytes{mountpoint="/"} 500
node_filesystem_avail_bytes{mountpoint="/"} 200
node_network_receive_bytes_total{device="eth0"} 1000
node_network_transmit_bytes_total{device="eth0"} 2000
node_boot_time_seconds 1752000000
"""
_DCGM_IDENTITY = (
    'gpu="0",UUID="GPU-aaa",pci_bus_id="00000000:1A:00.0",modelName="NVIDIA H100 80GB HBM3",' 'hostname="g83d142"'
)
DCGM_TEXT = f"""
DCGM_FI_DEV_FB_USED{{{_DCGM_IDENTITY},DCGM_FI_DRIVER_VERSION="570.86.15"}} 200
DCGM_FI_DEV_FB_TOTAL{{{_DCGM_IDENTITY}}} 81281
DCGM_FI_DEV_GPU_UTIL{{{_DCGM_IDENTITY}}} 40
DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{_DCGM_IDENTITY}}} 3
"""


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    telemetry.shutdown(0.01)
    yield
    telemetry.shutdown(0.1)


def _transport(monkeypatch: pytest.MonkeyPatch) -> RecordingTelemetryTransport:
    transport = RecordingTelemetryTransport()
    monkeypatch.setattr(telemetry, "_RequestsTransport", lambda: transport)
    telemetry.configure(
        endpoint="http://finelog/v1/telemetry",
        service="iris-node-agent",
        attributes={"node_name": "g83d142", "node_uid": "node-uid-1", "role": "worker"},
    )
    return transport


def _fetch_from(mapping: dict[str, str]):
    return lambda url: mapping.get(url)


def test_k8s_collection_exports_normalized_node_and_device_records(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    k8s = InMemoryK8sService(namespace="iris")
    k8s.seed_namespaced_pod(
        "cw-exporters",
        "dcgm-exporter-abc",
        {
            "metadata": {
                "name": "dcgm-exporter-abc",
                "uid": "dcgm-pod-uid-1",
                "labels": {"app.kubernetes.io/name": "dcgm-exporter"},
            },
            "spec": {"nodeName": "g83d142"},
            "status": {"podIP": "10.9.9.9"},
        },
    )
    scraper = kubernetes.NodeStatsScraper(
        k8s,
        fetch=_fetch_from(
            {
                "http://127.0.0.1:9100/metrics": NODE_EXPORTER_TEXT,
                "http://10.9.9.9:9400/metrics": DCGM_TEXT,
            }
        ),
    )
    target = NodeTarget(
        name="g83d142",
        node_uid="node-uid-1",
        internal_ip="127.0.0.1",
        device_type="gpu",
    )

    kubernetes.collect_once(scraper, target)

    host_memory = transport.record("node_memory_used_bytes", {"node_uid": "node-uid-1"})
    host_network = transport.record("node_network_receive_bytes", {"node_uid": "node-uid-1"})
    gpu_memory = transport.record("gpu_memory_used_bytes", {"gpu_uuid": "GPU-aaa"})
    pcie_errors = transport.record("gpu_pcie_replay_errors", {"gpu_uuid": "GPU-aaa"})
    inventory = transport.record("hardware_inventory", {"gpu_uuid": "GPU-aaa"})

    assert host_memory["attributes"] == {
        "node_name": "g83d142",
        "node_uid": "node-uid-1",
        "source_kind": "node_exporter",
        "source_temporality": telemetry.CURRENT_SNAPSHOT,
    }
    assert host_network["attributes"] == {
        "node_name": "g83d142",
        "node_uid": "node-uid-1",
        "source_kind": "node_exporter",
        "source_replica_uid": "node-uid-1:1752000000",
        "source_temporality": telemetry.CUMULATIVE_SNAPSHOT,
    }
    assert gpu_memory["value"] == 200 * 1024 * 1024
    assert gpu_memory["attributes"] == {
        "dcgm_exporter_uid": "dcgm-pod-uid-1",
        "gpu_index": "0",
        "gpu_uuid": "GPU-aaa",
        "node_name": "g83d142",
        "node_uid": "node-uid-1",
        "pci_bus_id": "00000000:1A:00.0",
        "source_kind": "dcgm",
        "source_replica_uid": "dcgm-pod-uid-1",
        "source_temporality": telemetry.CURRENT_SNAPSHOT,
    }
    assert pcie_errors["value"] == 3
    assert pcie_errors["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT
    assert pcie_errors["attributes"]["source_replica_uid"] == "dcgm-pod-uid-1"
    assert inventory["attributes"]["gpu_model"] == "NVIDIA H100 80GB HBM3"
    assert inventory["attributes"]["driver_version"] == "570.86.15"
    assert transport.resources[-1] == {
        "service": "iris-node-agent",
        "attributes": {"node_name": "g83d142", "node_uid": "node-uid-1", "role": "worker"},
    }


def test_k8s_collection_without_dcgm_exports_unavailable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _transport(monkeypatch)
    scraper = kubernetes.NodeStatsScraper(
        InMemoryK8sService(namespace="iris"),
        fetch=_fetch_from({"http://127.0.0.1:9100/metrics": NODE_EXPORTER_TEXT}),
    )
    target = NodeTarget(
        name="g83d142",
        node_uid="node-uid-1",
        internal_ip="127.0.0.1",
        device_type="gpu",
    )

    kubernetes.collect_once(scraper, target)

    available = transport.record("hardware_source_available", {"node_uid": "node-uid-1", "source_kind": "dcgm"})
    assert available["value"] == 0
    assert available["attributes"]["source_temporality"] == telemetry.CURRENT_SNAPSHOT


def test_gcp_collection_uses_host_boot_identity_and_publishes_tpu_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(monkeypatch)
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("boot-123")
    monkeypatch.setattr(gcp, "_BOOT_ID_PATH", boot_id)

    snapshot = job_pb2.WorkerResourceSnapshot(
        host_cpu_percent=25,
        memory_used_bytes=100,
        memory_total_bytes=200,
        disk_used_bytes=300,
        disk_total_bytes=400,
        net_recv_bytes=500,
        net_sent_bytes=600,
    )

    class HostCollector:
        def collect(self) -> job_pb2.WorkerResourceSnapshot:
            return snapshot

    hardware = HardwareProbe(
        hostname="gcp-node",
        ip_address="10.0.0.1",
        cpu_count=8,
        memory_bytes=200,
        disk_bytes=400,
        gpu_count=0,
        gpu_name="",
        gpu_memory_mb=0,
        tpu_name="slice-a",
        tpu_type="v6e-8",
        tpu_worker_hostnames="10.0.0.1",
        tpu_worker_id="0",
        tpu_chips_per_host_bounds="2,2,1",
        gce_instance_name="gcp-node",
        gce_instance_uid="123456789",
    )
    target = NodeTarget(name="gcp-node", node_uid="123456789", internal_ip="10.0.0.1")

    gcp.collect_once(HostCollector(), target, hardware)

    network = transport.record("node_network_receive_bytes", {"node_uid": "123456789"})
    inventory = transport.record("hardware_inventory", {"device_kind": "tpu"})
    assert network["attributes"]["source_kind"] == "procfs"
    assert network["attributes"]["source_replica_uid"] == "123456789:boot-123"
    assert network["attributes"]["source_temporality"] == telemetry.CUMULATIVE_SNAPSHOT
    assert inventory["attributes"]["tpu_type"] == "v6e-8"
    assert inventory["attributes"]["tpu_name"] == "slice-a"


def test_gcp_collection_rejects_missing_boot_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _transport(monkeypatch)
    monkeypatch.setattr(gcp, "_BOOT_ID_PATH", tmp_path / "missing-boot-id")
    snapshot = job_pb2.WorkerResourceSnapshot(memory_total_bytes=200)

    class HostCollector:
        def collect(self) -> job_pb2.WorkerResourceSnapshot:
            return snapshot

    target = NodeTarget(name="gcp-node", node_uid="123456789", internal_ip="10.0.0.1")
    hardware = HardwareProbe(
        hostname="gcp-node",
        ip_address="10.0.0.1",
        cpu_count=8,
        memory_bytes=200,
        disk_bytes=400,
        gpu_count=0,
        gpu_name="",
        gpu_memory_mb=0,
        tpu_name="",
        tpu_type="",
        tpu_worker_hostnames="",
        tpu_worker_id="",
        tpu_chips_per_host_bounds="",
    )

    with pytest.raises(FileNotFoundError):
        gcp.collect_once(HostCollector(), target, hardware)
