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

"""Tests for worker environment probing."""

import pytest

from iris.cluster.worker.env_probe import DefaultEnvironmentProvider, _get_extra_attributes


@pytest.mark.parametrize(
    "env_value,expected",
    [
        ("", {}),
        ("key1=value1", {"key1": "value1"}),
        ("key1=value1,key2=value2", {"key1": "value1", "key2": "value2"}),
        ("taint:maintenance=true,pool=large-jobs", {"taint:maintenance": "true", "pool": "large-jobs"}),
        ("key1=value1,  key2=value2  ", {"key1": "value1", "key2": "value2"}),  # whitespace
        ("key1=value1,malformed,key2=value2", {"key1": "value1", "key2": "value2"}),  # skip malformed
        ("=value", {}),  # empty key
        ("key=", {"key": ""}),  # empty value is valid
    ],
)
def test_get_extra_attributes_parsing(monkeypatch, env_value, expected):
    """Test parsing of IRIS_WORKER_ATTRIBUTES environment variable."""
    monkeypatch.setenv("IRIS_WORKER_ATTRIBUTES", env_value)
    result = _get_extra_attributes()
    assert result == expected


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

    # Preemptible attribute should be present (value depends on environment:
    # "false" on non-GCP, "true" on preemptible GCP VMs)
    assert "preemptible" in metadata.attributes
    assert metadata.attributes["preemptible"].string_value in ("true", "false")
