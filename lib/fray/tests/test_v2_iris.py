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

"""Unit tests for the fray v2 Iris backend.

Tests type conversions and handle serialization without requiring an Iris cluster.
Integration tests that need a running cluster are marked with @pytest.mark.iris.
"""

import pickle


from fray.v2.iris_backend import (
    IrisActorHandle,
    convert_constraints,
    convert_entrypoint,
    convert_resources,
    map_iris_job_state,
)
from fray.v2.types import (
    CpuConfig,
    Entrypoint,
    GpuConfig,
    JobStatus,
    ResourceConfig,
    TpuConfig,
)

# ---------------------------------------------------------------------------
# ResourceConfig → ResourceSpec conversion
# ---------------------------------------------------------------------------


class TestConvertResources:
    def test_cpu_defaults(self):
        resources = ResourceConfig()
        spec = convert_resources(resources)
        assert spec.cpu == 1
        assert spec.memory == "128m"
        assert spec.disk == "1g"
        assert spec.device is None
        assert spec.replicas == 1

    def test_replicas_passed_through(self):
        spec = convert_resources(ResourceConfig(), replicas=4)
        assert spec.replicas == 4

    def test_tpu_device(self):
        resources = ResourceConfig(device=TpuConfig(variant="v5litepod-16"))
        spec = convert_resources(resources)
        assert spec.device is not None
        assert spec.device.HasField("tpu")
        assert spec.device.tpu.variant == "v5litepod-16"

    def test_gpu_device(self):
        resources = ResourceConfig(device=GpuConfig(variant="H100", count=8))
        spec = convert_resources(resources)
        assert spec.device is not None
        assert spec.device.HasField("gpu")
        assert spec.device.gpu.variant == "H100"
        assert spec.device.gpu.count == 8

    def test_cpu_device_produces_no_device(self):
        resources = ResourceConfig(device=CpuConfig())
        spec = convert_resources(resources)
        assert spec.device is None

    def test_regions_passed_through(self):
        resources = ResourceConfig(regions=["us-central1", "us-east1"])
        spec = convert_resources(resources)
        assert list(spec.regions) == ["us-central1", "us-east1"]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConvertConstraints:
    def test_preemptible_true_produces_no_constraints(self):
        resources = ResourceConfig(preemptible=True)
        constraints = convert_constraints(resources)
        assert constraints == []

    def test_preemptible_false_adds_constraint(self):
        resources = ResourceConfig(preemptible=False)
        constraints = convert_constraints(resources)
        assert len(constraints) == 1
        c = constraints[0]
        assert c.key == "preemptible"
        assert c.value == "false"


# ---------------------------------------------------------------------------
# Entrypoint conversion
# ---------------------------------------------------------------------------


def _dummy_fn(x: int) -> int:
    return x + 1


class TestConvertEntrypoint:
    def test_callable_entrypoint(self):
        entry = Entrypoint.from_callable(_dummy_fn, args=(42,))
        iris_entry = convert_entrypoint(entry)
        assert iris_entry.is_callable

    def test_binary_entrypoint(self):
        entry = Entrypoint.from_binary("python", ["-c", "print('hi')"])
        iris_entry = convert_entrypoint(entry)
        assert iris_entry.is_command
        assert iris_entry.command == ["python", "-c", "print('hi')"]


# ---------------------------------------------------------------------------
# JobStatus mapping
# ---------------------------------------------------------------------------


class TestMapIrisJobState:
    def test_succeeded(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_SUCCEEDED) == JobStatus.SUCCEEDED

    def test_failed(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_FAILED) == JobStatus.FAILED

    def test_killed_maps_to_stopped(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_KILLED) == JobStatus.STOPPED

    def test_running(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_RUNNING) == JobStatus.RUNNING

    def test_pending(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_PENDING) == JobStatus.PENDING

    def test_worker_failed_maps_to_failed(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_WORKER_FAILED) == JobStatus.FAILED

    def test_unschedulable_maps_to_failed(self):
        from iris.rpc import cluster_pb2

        assert map_iris_job_state(cluster_pb2.JOB_STATE_UNSCHEDULABLE) == JobStatus.FAILED


# ---------------------------------------------------------------------------
# IrisActorHandle pickle round-trip
# ---------------------------------------------------------------------------


class TestIrisActorHandlePickle:
    def test_pickle_roundtrip_preserves_name(self):
        handle = IrisActorHandle("my-actor")
        data = pickle.dumps(handle)
        restored = pickle.loads(data)
        assert restored._actor_name == "my-actor"
        assert restored._client is None

    def test_pickle_drops_client(self):
        """Client is transient state — pickle should not carry it."""
        handle = IrisActorHandle("my-actor", client="fake-client")
        data = pickle.dumps(handle)
        restored = pickle.loads(data)
        assert restored._client is None


# ---------------------------------------------------------------------------
# FRAY_CLIENT_SPEC parsing for iris://
# ---------------------------------------------------------------------------


class TestIrisClientSpec:
    def test_iris_spec_parses(self):
        """Verify iris:// spec creates FrayIrisClient (mocked — no real controller)."""
        from unittest.mock import patch

        with patch("fray.v2.iris_backend.IrisClientLib") as mock_iris:
            from fray.v2.client import _parse_client_spec
            from fray.v2.iris_backend import FrayIrisClient

            client = _parse_client_spec("iris://controller:10000")
            assert isinstance(client, FrayIrisClient)
            mock_iris.remote.assert_called_once_with("controller:10000", workspace=None)

    def test_iris_spec_with_workspace(self):
        from pathlib import Path
        from unittest.mock import patch

        with patch("fray.v2.iris_backend.IrisClientLib") as mock_iris:
            from fray.v2.client import _parse_client_spec
            from fray.v2.iris_backend import FrayIrisClient

            client = _parse_client_spec("iris://controller:10000?ws=/tmp/my-workspace")
            assert isinstance(client, FrayIrisClient)
            mock_iris.remote.assert_called_once_with("controller:10000", workspace=Path("/tmp/my-workspace"))
