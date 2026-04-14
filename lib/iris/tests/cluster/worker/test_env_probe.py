# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker environment probing."""

import sys
import time

import pytest

import iris.cluster.worker.env_probe as env_probe
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.worker.env_probe import (
    DefaultEnvironmentProvider,
    HardwareProbe,
    HostMetricsCollector,
    _read_net_dev_bytes,
    build_worker_metadata,
    check_worker_health,
    construct_worker_id,
    probe_disk_writable,
)
from iris.rpc import config_pb2


def _make_hardware(**overrides) -> HardwareProbe:
    """Create a HardwareProbe with sensible defaults, overridable per field."""
    defaults = dict(
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
    defaults.update(overrides)
    return HardwareProbe(**defaults)


def test_environment_provider_basic_probe(monkeypatch):
    """Test that DefaultEnvironmentProvider produces valid WorkerMetadata."""
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_TYPE", raising=False)
    monkeypatch.delenv("TPU_WORKER_HOSTNAMES", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("IRIS_WORKER_ATTRIBUTES", raising=False)

    provider = DefaultEnvironmentProvider()
    metadata = provider.probe()

    assert metadata.hostname
    assert metadata.ip_address
    assert metadata.cpu_count > 0
    assert metadata.memory_bytes > 0
    assert metadata.disk_bytes > 0
    assert metadata.device.HasField("cpu")

    # Attributes come from config defaults (no config = CPU, not preemptible)
    assert WellKnownAttribute.PREEMPTIBLE in metadata.attributes
    assert metadata.attributes[WellKnownAttribute.PREEMPTIBLE].string_value == "false"
    assert WellKnownAttribute.DEVICE_TYPE in metadata.attributes
    assert metadata.attributes[WellKnownAttribute.DEVICE_TYPE].string_value == "cpu"


def test_environment_provider_probes_tpu_metadata(monkeypatch):
    """Provider should resolve TPU diagnostic metadata from GCP metadata service."""
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

    # Diagnostic TPU fields are populated from probes
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


# --- Scheduling attributes from config ---


def test_gpu_worker_attributes_from_config():
    """Scheduling attributes (device-type, device-variant, preemptible) come from config, not probes."""
    hardware = _make_hardware()

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        gpu_count_override=8,
        capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
    )

    # Device config for capacity accounting
    assert metadata.device.HasField("gpu")
    assert metadata.device.gpu.variant == "H100"
    assert metadata.device.gpu.count == 8

    # Scheduling attributes from config
    attrs = metadata.attributes
    assert attrs[WellKnownAttribute.DEVICE_TYPE].string_value == "gpu"
    assert attrs[WellKnownAttribute.DEVICE_VARIANT].string_value == "h100"
    assert attrs[WellKnownAttribute.PREEMPTIBLE].string_value == "true"


def test_tpu_worker_attributes_from_config():
    """TPU scheduling attributes come from config; diagnostic fields from probes."""
    hardware = _make_hardware(
        tpu_name="my-tpu-slice",
        tpu_type="v5litepod-16",
        tpu_worker_hostnames="10.0.0.1,10.0.0.2",
        tpu_worker_id="2",
        tpu_chips_per_host_bounds="2,2,1",
    )

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        capacity_type=config_pb2.CAPACITY_TYPE_PREEMPTIBLE,
    )

    attrs = metadata.attributes

    # Scheduling attributes from config
    assert attrs[WellKnownAttribute.DEVICE_TYPE].string_value == "tpu"
    assert attrs[WellKnownAttribute.DEVICE_VARIANT].string_value == "v5litepod-16"
    assert attrs[WellKnownAttribute.PREEMPTIBLE].string_value == "true"

    # TPU multi-host identity from probes (not config)
    assert attrs[WellKnownAttribute.TPU_NAME].string_value == "my-tpu-slice"
    assert attrs[WellKnownAttribute.TPU_WORKER_ID].int_value == 2

    # TPU topology derived from config variant
    assert attrs[WellKnownAttribute.TPU_TOPOLOGY].string_value == "v5litepod-16"

    # Diagnostic fields on WorkerMetadata from probes
    assert metadata.tpu_name == "my-tpu-slice"
    assert metadata.tpu_worker_id == "2"
    assert metadata.tpu_worker_hostnames == "10.0.0.1,10.0.0.2"


def test_cpu_worker_attributes_from_config():
    """CPU workers get device-type=cpu from config, no device-variant."""
    hardware = _make_hardware()

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_CPU,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
    )

    attrs = metadata.attributes
    assert attrs[WellKnownAttribute.DEVICE_TYPE].string_value == "cpu"
    assert WellKnownAttribute.DEVICE_VARIANT not in attrs
    assert attrs[WellKnownAttribute.PREEMPTIBLE].string_value == "false"


def test_cpu_fallback_when_no_config():
    """When no accelerator_type is specified and no hardware detected, defaults to CPU."""
    hardware = _make_hardware()
    metadata = build_worker_metadata(hardware=hardware)

    assert metadata.device.HasField("cpu")
    attrs = metadata.attributes
    assert attrs[WellKnownAttribute.DEVICE_TYPE].string_value == "cpu"
    assert attrs[WellKnownAttribute.PREEMPTIBLE].string_value == "false"


def test_custom_worker_attributes_merged():
    """Custom user attributes from YAML worker.attributes are merged into the attributes map."""
    hardware = _make_hardware()

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        gpu_count_override=8,
        worker_attributes={"pool": "large-jobs", "custom-key": "custom-value"},
    )

    attrs = metadata.attributes
    assert attrs["pool"].string_value == "large-jobs"
    assert attrs["custom-key"].string_value == "custom-value"
    # Config-derived attributes still present
    assert attrs[WellKnownAttribute.DEVICE_TYPE].string_value == "gpu"


def test_preemptible_not_from_gcp_metadata():
    """Preemptible attribute comes from config, not GCP metadata probing."""
    hardware = _make_hardware()

    # Config says not preemptible -- even if GCP metadata would say TRUE,
    # the attribute should reflect config
    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        gpu_count_override=8,
        capacity_type=config_pb2.CAPACITY_TYPE_ON_DEMAND,
    )

    assert metadata.attributes[WellKnownAttribute.PREEMPTIBLE].string_value == "false"


def test_build_worker_metadata_gpu_diagnostic_fields():
    """GPU diagnostic fields (gpu_count, gpu_name) are still populated on WorkerMetadata."""
    hardware = _make_hardware()

    metadata = build_worker_metadata(
        hardware=hardware,
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="H100",
        gpu_count_override=8,
    )

    assert metadata.gpu_count == 8
    assert metadata.gpu_name == "H100"


# --- Worker ID resolution ---


def test_construct_worker_id_format():
    """construct_worker_id produces the expected '{slice_id}-worker-{index}' format."""
    assert construct_worker_id("marin-tpu_v6e_4-europe-west4-a-xxxx", 2) == (
        "marin-tpu_v6e_4-europe-west4-a-xxxx-worker-2"
    )
    assert construct_worker_id("my-slice", 0) == "my-slice-worker-0"


def test_worker_id_from_slice_id_and_tpu_worker_index(tmp_path, monkeypatch):
    """When WorkerConfig.slice_id is set and hardware reports a TPU worker index,
    the Worker resolves worker_id as construct_worker_id(slice_id, tpu_worker_index).

    This is the slice_id resolution path (priority 2), which takes precedence over
    GCP metadata inference but not an explicit config.worker_id.
    """
    from iris.cluster.worker import worker as worker_mod
    from iris.cluster.worker.worker import Worker, WorkerConfig

    hardware = _make_hardware(tpu_worker_id="2", tpu_name="some-tpu-slice")

    monkeypatch.setattr(worker_mod, "probe_hardware", lambda: hardware)

    config = WorkerConfig(
        cache_dir=tmp_path / "cache",
        slice_id="marin-tpu_v6e_4-europe-west4-a-xxxx",
        worker_id=None,
        default_task_image="mock-image",
    )
    worker = Worker(config, container_runtime=None)

    assert worker._worker_id == "marin-tpu_v6e_4-europe-west4-a-xxxx-worker-2"


def test_worker_id_explicit_config_overrides_slice_id(tmp_path, monkeypatch):
    """An explicit config.worker_id (priority 1) takes precedence over slice_id resolution."""
    from iris.cluster.worker import worker as worker_mod
    from iris.cluster.worker.worker import Worker, WorkerConfig

    hardware = _make_hardware(tpu_worker_id="2", tpu_name="some-tpu-slice")
    monkeypatch.setattr(worker_mod, "probe_hardware", lambda: hardware)

    config = WorkerConfig(
        cache_dir=tmp_path / "cache",
        slice_id="marin-tpu_v6e_4-europe-west4-a-xxxx",
        worker_id="my-explicit-id",
        default_task_image="mock-image",
    )
    worker = Worker(config, container_runtime=None)

    assert worker._worker_id == "my-explicit-id"


# --- Network metrics ---


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only")
def test_read_net_dev_bytes_returns_nonzero_on_linux():
    recv, sent = _read_net_dev_bytes()
    assert recv >= 0
    assert sent >= 0


def test_host_metrics_collector_network_first_call_returns_zero():
    """First network collection establishes baseline and reports 0 B/s."""
    collector = HostMetricsCollector()
    snapshot = collector.collect()
    assert snapshot.net_recv_bps == 0
    assert snapshot.net_sent_bps == 0


def test_host_metrics_collector_network_delta(monkeypatch):
    """Second network collection computes bytes/sec from the delta."""
    fake_time = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_time[0])

    call_count = [0]
    net_values = [
        (1000, 2000),
        (6000, 12000),
    ]

    def fake_read_net():
        idx = min(call_count[0], len(net_values) - 1)
        call_count[0] += 1
        return net_values[idx]

    monkeypatch.setattr(env_probe, "_read_net_dev_bytes", fake_read_net)

    collector = HostMetricsCollector()

    snapshot1 = collector.collect()
    assert snapshot1.net_recv_bps == 0
    assert snapshot1.net_sent_bps == 0

    fake_time[0] = 105.0

    snapshot2 = collector.collect()
    assert snapshot2.net_recv_bps == 1000
    assert snapshot2.net_sent_bps == 2000


# --- Network metrics ---


# --- Health check ---


def test_health_check_nonexistent_path():
    """Health check skips gracefully when disk_path does not exist."""
    result = check_worker_health(disk_path="/nonexistent/path/that/does/not/exist")
    assert result.healthy


def test_health_check_file_not_dir(tmp_path):
    """Health check skips gracefully when disk_path is a file, not a directory."""
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("hello")
    result = check_worker_health(disk_path=str(file_path))
    assert result.healthy


def test_health_check_writable_dir(tmp_path):
    """Health check succeeds on a writable directory."""
    result = check_worker_health(disk_path=str(tmp_path))
    assert result.healthy


def test_health_check_does_not_touch_filesystem(tmp_path):
    """Heartbeat-path health check must not create the probe file (issue #4732)."""
    for _ in range(5):
        result = check_worker_health(disk_path=str(tmp_path))
        assert result.healthy
    # No per-heartbeat file churn — the probe file is never created here.
    assert not (tmp_path / ".iris_health_probe").exists()


def test_health_check_surfaces_cached_writable_error(tmp_path):
    """A non-empty writable_error from startup propagates into heartbeat results."""
    result = check_worker_health(
        disk_path=str(tmp_path),
        writable_error="tempfile write failed: read-only mount",
    )
    assert not result.healthy
    assert "read-only mount" in result.error


def test_probe_disk_writable_success(tmp_path):
    """probe_disk_writable returns empty string and cleans up on a writable dir."""
    assert probe_disk_writable(str(tmp_path)) == ""
    assert not (tmp_path / ".iris_health_probe").exists()


def test_probe_disk_writable_missing_dir():
    """probe_disk_writable returns empty string when the path is not a directory."""
    assert probe_disk_writable("/nonexistent/path/that/does/not/exist") == ""


def test_probe_disk_writable_reports_error(tmp_path, monkeypatch):
    """probe_disk_writable surfaces OSError from the write as a descriptive string."""

    def _raise(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr("pathlib.Path.write_text", _raise)
    err = probe_disk_writable(str(tmp_path))
    assert err.startswith("tempfile write failed:")
    assert "read-only filesystem" in err


def test_host_metrics_collector_network_graceful_on_non_linux(monkeypatch):
    """Network collection silently returns 0 on systems without /proc/net/dev."""
    monkeypatch.setattr(env_probe, "_read_net_dev_bytes", lambda: (_ for _ in ()).throw(OSError("no /proc/net/dev")))

    collector = HostMetricsCollector()
    snapshot = collector.collect()
    assert snapshot.net_recv_bps == 0
    assert snapshot.net_sent_bps == 0
