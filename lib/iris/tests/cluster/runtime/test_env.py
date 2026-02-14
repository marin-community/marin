# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from iris.cluster.runtime.env import build_device_env_vars
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2


def _tpu_container_config(worker_metadata: cluster_pb2.WorkerMetadata) -> ContainerConfig:
    resources = cluster_pb2.ResourceSpecProto()
    resources.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))
    return ContainerConfig(
        image="iris-task:latest",
        entrypoint=cluster_pb2.RuntimeEntrypoint(),
        env={},
        resources=resources,
        worker_metadata=worker_metadata,
    )


def test_build_device_env_vars_sets_jax_process_id_from_tpu_worker_id():
    worker_metadata = cluster_pb2.WorkerMetadata(
        tpu_worker_id="3",
        tpu_worker_hostnames="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
    )
    worker_metadata.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))

    env = build_device_env_vars(_tpu_container_config(worker_metadata))

    assert env["JAX_PROCESS_ID"] == "3"
    assert env["JAX_NUM_PROCESSES"] == "4"
    assert env["JAX_COORDINATOR_ADDRESS"] == "10.0.0.1:8476"


def test_build_device_env_vars_raises_for_missing_tpu_worker_id():
    worker_metadata = cluster_pb2.WorkerMetadata(
        tpu_worker_hostnames="10.0.0.1,10.0.0.2",
    )
    worker_metadata.device.tpu.CopyFrom(cluster_pb2.TpuDevice(variant="v5litepod-16", count=4))

    with pytest.raises(ValueError, match="TPU_WORKER_ID is required"):
        build_device_env_vars(_tpu_container_config(worker_metadata))
