# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local TPU topology helpers for per-host vLLM sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalVllmTopology:
    """Environment contract for one host-local vLLM replica."""

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
            "CLOUD_TPU_TASK_ID": "0",
            "TPU_PROCESS_BOUNDS": self.tpu_process_bounds,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": self.tpu_chips_per_process_bounds,
            "TPU_VISIBLE_CHIPS": self.tpu_visible_chips,
        }


def local_vllm_topology(tpu_type: str, tensor_parallel_size: int) -> LocalVllmTopology:
    """Return one-host TPU topology settings for supported vLLM replicas."""
    del tpu_type
    if tensor_parallel_size == 1:
        return LocalVllmTopology(
            tensor_parallel_size=1,
            tpu_process_bounds="1,1,1",
            tpu_chips_per_process_bounds="1,1,1",
            tpu_visible_chips="0",
        )
    if tensor_parallel_size == 2:
        return LocalVllmTopology(
            tensor_parallel_size=2,
            tpu_process_bounds="1,1,1",
            tpu_chips_per_process_bounds="2,1,1",
            tpu_visible_chips="0,1",
        )
    if tensor_parallel_size == 4:
        return LocalVllmTopology(
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
