# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local TPU topology helpers for per-host vLLM sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalVllmTopology:
    """Environment contract for one host-local vLLM replica."""

    tpu_type: str
    tensor_parallel_size: int
    tpu_process_bounds: str
    tpu_chips_per_process_bounds: str
    tpu_visible_chips: str

    def env(self) -> dict[str, str]:
        """Return the environment variables needed before JAX/vLLM starts."""
        return {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TPU_BACKEND_TYPE": "jax",
            "PJRT_DEVICE": "TPU",
            # In Docker on TPU VMs, force libtpu/JAX to use explicit local
            # metadata instead of querying the full pod topology from metadata
            # service. This is what lets one host behave like an independent
            # single-host replica.
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": self.tpu_type,
            "TPU_TYPE": self.tpu_type,
            "TPU_NAME": "alternating-local-vllm",
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "127.0.0.1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_CHIPS_PER_HOST_BOUNDS": self.tpu_chips_per_process_bounds,
            "CLOUD_TPU_TASK_ID": "0",
            "TPU_PROCESS_BOUNDS": self.tpu_process_bounds,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": self.tpu_chips_per_process_bounds,
            "TPU_VISIBLE_CHIPS": self.tpu_visible_chips,
        }


def local_vllm_topology(tpu_type: str, tensor_parallel_size: int) -> LocalVllmTopology:
    """Return one-host TPU topology settings for supported vLLM replicas."""
    if tensor_parallel_size == 1:
        return LocalVllmTopology(
            tpu_type=tpu_type,
            tensor_parallel_size=1,
            tpu_process_bounds="1,1,1",
            tpu_chips_per_process_bounds="1,1,1",
            tpu_visible_chips="0",
        )
    if tensor_parallel_size == 2:
        return LocalVllmTopology(
            tpu_type=tpu_type,
            tensor_parallel_size=2,
            tpu_process_bounds="1,1,1",
            tpu_chips_per_process_bounds="2,1,1",
            tpu_visible_chips="0,1",
        )
    if tensor_parallel_size == 4:
        return LocalVllmTopology(
            tpu_type=tpu_type,
            tensor_parallel_size=4,
            tpu_process_bounds="1,1,1",
            tpu_chips_per_process_bounds="2,2,1",
            tpu_visible_chips="0,1,2,3",
        )
    raise ValueError(
        "Unsupported tensor_parallel_size for local TPU vLLM topology: " f"{tensor_parallel_size}. Expected 1, 2, or 4."
    )


def apply_local_vllm_topology(topology: LocalVllmTopology) -> None:
    """Install host-local TPU topology settings into the current environment."""
    for key, value in topology.env().items():
        os.environ[key] = value
