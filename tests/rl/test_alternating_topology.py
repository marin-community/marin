# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from marin.rl.alternating.topology import local_vllm_topology


@pytest.mark.parametrize(
    ("tensor_parallel_size", "process_bounds", "visible_chips"),
    [
        (1, "1,1,1", "0"),
        (2, "2,1,1", "0,1"),
        (4, "2,2,1", "0,1,2,3"),
    ],
)
def test_local_vllm_topology_exports_explicit_single_host_tpu_metadata(
    tensor_parallel_size: int,
    process_bounds: str,
    visible_chips: str,
) -> None:
    topology = local_vllm_topology("v5p-8", tensor_parallel_size)

    assert topology.tensor_parallel_size == tensor_parallel_size
    assert topology.tpu_type == "v5p-8"

    env = topology.env()
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["TPU_BACKEND_TYPE"] == "jax"
    assert env["TPU_SKIP_MDS_QUERY"] == "1"
    assert env["TPU_ACCELERATOR_TYPE"] == "v5p-8"
    assert env["TPU_TYPE"] == "v5p-8"
    assert env["TPU_WORKER_ID"] == "0"
    assert env["TPU_WORKER_HOSTNAMES"] == "127.0.0.1"
    assert env["TPU_HOST_BOUNDS"] == "1,1,1"
    assert env["CLOUD_TPU_TASK_ID"] == "0"
    assert env["TPU_PROCESS_BOUNDS"] == "1,1,1"
    assert env["TPU_CHIPS_PER_PROCESS_BOUNDS"] == process_bounds
    assert env["TPU_CHIPS_PER_HOST_BOUNDS"] == process_bounds
    assert env["TPU_VISIBLE_CHIPS"] == visible_chips
