# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker environment probing."""

import time

import iris.cluster.worker.env_probe as env_probe
from iris.cluster.worker.env_probe import (
    DefaultEnvironmentProvider,
    HardwareProbe,
    HostMetricsCollector,
    _read_net_dev_bytes,
    build_worker_metadata,
)
from iris.rpc import config_pb2


def test_environment_provider_basic_probe(monkeypatch):
    """Test that DefaultEnvironmentProvider produces valid WorkerMetadata."""
    # Clear TPU environment variables to ensure CPU-only probing
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.delenv("TPU_WORKER_HOSTNAMES", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("IRIS_WORKER_ATTRIBUTES", raising=False)

    provider = DefaultEnvironmentProvider()
    metadata = provider.probe()

    # Verify basic fields are populated
    assert metadata.hostname
    assert metadata.ip_address
    assert metadata.cpu_count > 0
    assert metadata.memory_bytes > 0
    assert metadata.disk_bytes > 0

    # CPU-only device should be set
    assert metadata.device.HasField("cpu")

    # Preemptible attribute should be present and default to false on non-GCP
    assert "preemptible" in metadata.attributes
    assert metadata.attributes["preemptible"].string_value == "false"


def test_environment_provider_probes_tpu_metadata(monkeypatch):
    """Provider should resolve TPU metadata from GCP metadata service."""
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.delenv("TPU_WORKER_HOSTNAMES", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("TPU_CHIPS_PER_HOST_BOUNDS", raising=False)
    monkeypatch.delenv("IRIS_WORKER_ATTRIBUTES", raising=False)

    monkeypatch.setattr(env_probe, "_is_gcp_vm", lambda: True)
    metadata_values = {
        "name": "test-slice-w-3",
        "attributes/accelerator-type": "v5litepod-16",
        "attributes/agent-worker-number": "3",
        "attributes/worker-network-endpoints": "x:y:10.0.0.11,x:y:10.0.0.12",
        "attributes/tpu-env": "CHIPS_PER_HOST_BOUNDS: '2,2,1'\nOTHER: 'x'",
        "scheduling/preemptible": "FALSE",
    }
    monkeypatch.setattr(env_probe, "_get_gcp_metadata", lambda key: metadata_values.get(key))

    metadata = DefaultEnvironmentProvider().probe()

    assert metadata.tpu_name == "test-slice"
    assert metadata.tpu_worker_id == "3"
    assert metadata.tpu_worker_hostnames == "10.0.0.11,10.0.0.12"
    assert metadata.tpu_chips_per_host_bounds == "2,2,1"
    assert metadata.device.HasField("tpu")
    assert metadata.device.tpu.variant == "v5litepod-16"


def test_environment_provider_ignores_tpu_env_vars_without_metadata(monkeypatch):
    """TPU env vars alone should not trigger TPU detection."""
    monkeypatch.setattr(env_probe, "_is_gcp_vm", lambda: False)
    monkeypatch.setenv("TPU_NAME", "env-slice")
    monkeypatch.setenv("TPU_TYPE", "v5litepod-16")
    monkeypatch.setenv("TPU_WORKER_HOSTNAMES", "10.1.0.1,10.1.0.2")
    monkeypatch.setenv("TPU_WORKER_ID", "7")
    monkeypatch.setenv("TPU_CHIPS_PER_HOST_BOUNDS", "1,2,1")
    monkeypatch.setattr(env_probe, "_get_gcp_metadata", lambda _key: None)

    metadata = DefaultEnvironmentProvider().probe()

    assert metadata.tpu_name == ""
    assert metadata.tpu_worker_id == ""
    assert metadata.tpu_worker_hostnames == ""
    assert metadata.tpu_chips_per_host_bounds == ""
    assert metadata.device.HasField("cpu")


def test_build_worker_metadata_gpu_override():
    """build_worker_metadata should use accelerator_type/variant from config over hardware probe."""
    hardware = HardwareProbe(
        hostname="test-host",
        ip_address="10.0.0.1",
        cpu_count=4,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        gpu_count=0,
        gpu_name="",
        gpu_memory_mb=0,
        tpu_name="",
        tpu_type="",
        tpu_worker_hostnames="",
        tpu_worker_id="",
        tpu_chips_per_host_bounds="",
    )

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        gpu_count_override=8,
    )

    assert metadata.device.HasField("gpu")
    assert metadata.device.gpu.variant == "H100"
    assert metadata.device.gpu.count == 8
    assert metadata.gpu_count == 8
    assert metadata.gpu_name == "H100"


def test_build_worker_metadata_cpu_fallback():
    """build_worker_metadata should fall back to CPU when no accelerator is specified."""
    hardware = HardwareProbe(
        hostname="test-host",
        ip_address="10.0.0.1",
        cpu_count=4,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        gpu_count=0,
        gpu_name="",
        gpu_memory_mb=0,
        tpu_name="",
        tpu_type="",
        tpu_worker_hostnames="",
        tpu_worker_id="",
        tpu_chips_per_host_bounds="",
    )

    metadata = build_worker_metadata(hardware=hardware)

    assert metadata.device.HasField("cpu")
    assert metadata.hostname == "test-host"
    assert metadata.ip_address == "10.0.0.1"


def test_read_net_dev_bytes_returns_nonzero_on_linux():
    """On Linux, /proc/net/dev should be readable and return non-negative values."""
    try:
        recv, sent = _read_net_dev_bytes()
        assert recv >= 0
        assert sent >= 0
    except OSError:
        # Non-Linux systems won't have /proc/net/dev
        pass


def test_host_metrics_collector_network_first_call_returns_zero():
    """First network collection establishes baseline and reports 0 B/s."""
    collector = HostMetricsCollector()
    snapshot = collector.collect()
    # First call should report 0 since there's no previous measurement to delta against
    assert snapshot.net_recv_bps == 0
    assert snapshot.net_sent_bps == 0


def test_host_metrics_collector_network_delta(monkeypatch):
    """Second network collection computes bytes/sec from the delta."""
    fake_time = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

    # Simulate /proc/net/dev with known values
    call_count = [0]
    net_values = [
        (1000, 2000),  # First call: baseline
        (6000, 12000),  # Second call: 5000 recv, 10000 sent over 5s
    ]

    def fake_read_net():
        idx = min(call_count[0], len(net_values) - 1)
        call_count[0] += 1
        return net_values[idx]

    monkeypatch.setattr(env_probe, "_read_net_dev_bytes", fake_read_net)

    collector = HostMetricsCollector()

    # First collect: establishes baseline
    snapshot1 = collector.collect()
    assert snapshot1.net_recv_bps == 0
    assert snapshot1.net_sent_bps == 0

    # Advance time by 5 seconds
    fake_time[0] = 105.0

    # Second collect: should compute rates
    snapshot2 = collector.collect()
    assert snapshot2.net_recv_bps == 1000  # (6000-1000) / 5 = 1000 B/s
    assert snapshot2.net_sent_bps == 2000  # (12000-2000) / 5 = 2000 B/s


def test_host_metrics_collector_network_graceful_on_non_linux(monkeypatch):
    """Network collection silently returns 0 on systems without /proc/net/dev."""
    monkeypatch.setattr(env_probe, "_read_net_dev_bytes", lambda: (_ for _ in ()).throw(OSError("no /proc/net/dev")))

    collector = HostMetricsCollector()
    snapshot = collector.collect()
    assert snapshot.net_recv_bps == 0
    assert snapshot.net_sent_bps == 0
