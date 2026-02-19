# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker environment probing."""

import pytest

import iris.cluster.worker.env_probe as env_probe
from iris.cluster.worker.env_probe import DefaultEnvironmentProvider, _get_extra_attributes


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("", {}),
        ('{"key1":"value1"}', {"key1": "value1"}),
        ('{"key1":"value1","key2":"value2"}', {"key1": "value1", "key2": "value2"}),
        ('{"taint:maintenance":"true","pool":"large-jobs"}', {"taint:maintenance": "true", "pool": "large-jobs"}),
        ('{"key":""}', {"key": ""}),
    ],
)
def test_get_extra_attributes_parsing(monkeypatch, env_value, expected):
    """Test parsing of IRIS_WORKER_ATTRIBUTES environment variable."""
    monkeypatch.setenv("IRIS_WORKER_ATTRIBUTES", env_value)
    result = _get_extra_attributes()
    assert result == expected


def test_get_extra_attributes_raises_for_non_json(monkeypatch):
    monkeypatch.setenv("IRIS_WORKER_ATTRIBUTES", "key1=value1")
    with pytest.raises(ValueError):
        _get_extra_attributes()


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


def test_infer_worker_log_prefix_uses_region_bucket_mapping(monkeypatch):
    """europe-west4 must map to marin-tmp-eu-west4 bucket naming."""
    monkeypatch.setattr(env_probe, "_is_gcp_vm", lambda: True)
    monkeypatch.setattr(
        env_probe,
        "_get_gcp_metadata",
        lambda key: "projects/hai-gcp-models/zones/europe-west4-b" if key == "zone" else None,
    )

    prefix = env_probe._infer_worker_log_prefix()
    assert prefix == "gs://marin-tmp-eu-west4/ttl=30d/iris-logs"


def test_infer_worker_log_prefix_unknown_region_returns_none(monkeypatch):
    """Unknown regions should fail closed (no guessed bucket)."""
    monkeypatch.setattr(env_probe, "_is_gcp_vm", lambda: True)
    monkeypatch.setattr(
        env_probe,
        "_get_gcp_metadata",
        lambda key: "projects/hai-gcp-models/zones/antarctica-south1-a" if key == "zone" else None,
    )

    assert env_probe._infer_worker_log_prefix() is None


def test_log_prefix_prefers_env_override(monkeypatch):
    """IRIS_LOG_PREFIX overrides inferred values."""
    monkeypatch.setenv("IRIS_LOG_PREFIX", "gs://custom/ttl=30d/iris-logs")
    monkeypatch.setattr(env_probe, "_is_gcp_vm", lambda: True)
    monkeypatch.setattr(
        env_probe,
        "_get_gcp_metadata",
        lambda key: "projects/hai-gcp-models/zones/europe-west4-b" if key == "zone" else None,
    )

    assert DefaultEnvironmentProvider().log_prefix() == "gs://custom/ttl=30d/iris-logs"
