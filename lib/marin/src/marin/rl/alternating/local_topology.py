# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map TPU type to local per-host vLLM topology for isolated sampling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalVllmTopology:
    """Per-host vLLM topology for a given TPU type."""

    tensor_parallel_size: int
    tpu_process_bounds: str
    tpu_chips_per_process_bounds: str
    tpu_visible_chips: str

    def env_vars(self) -> dict[str, str]:
        """Return environment variables that isolate one host as a local JAX cluster."""
        return {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TPU_BACKEND_TYPE": "jax",
            "PJRT_DEVICE": "TPU",
            "CLOUD_TPU_TASK_ID": "0",
            "TPU_PROCESS_BOUNDS": self.tpu_process_bounds,
            "TPU_CHIPS_PER_PROCESS_BOUNDS": self.tpu_chips_per_process_bounds,
            "TPU_VISIBLE_CHIPS": self.tpu_visible_chips,
        }


# TPU type -> (chips_per_host, local topology)
# Chip counts and topology strings for common TPU types.
# Each host in a multi-host pod sees only its local chips.

_TOPOLOGY_TABLE: dict[str, LocalVllmTopology] = {
    # v5p: 4 chips per host
    "v5p-8": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v5p-16": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v5p-32": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v5p-64": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v5p-128": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    # v6e: 4 chips per host
    "v6e-8": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v6e-16": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v6e-32": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v6e-64": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v6e-128": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v6e-256": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    # v4: 4 chips per host
    "v4-8": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v4-16": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
    "v4-32": LocalVllmTopology(
        tensor_parallel_size=4,
        tpu_process_bounds="1,1,1",
        tpu_chips_per_process_bounds="2,2,1",
        tpu_visible_chips="0,1,2,3",
    ),
}


def local_vllm_topology(tpu_type: str) -> LocalVllmTopology:
    """Return the local per-host vLLM topology for a TPU type.

    Raises ValueError if the TPU type is unknown.
    """
    if tpu_type in _TOPOLOGY_TABLE:
        return _TOPOLOGY_TABLE[tpu_type]

    raise ValueError(
        f"Unknown TPU type {tpu_type!r}. Known types: {sorted(_TOPOLOGY_TABLE.keys())}. "
        "Add an entry to _TOPOLOGY_TABLE in local_topology.py."
    )


def num_hosts_for_tpu_type(tpu_type: str) -> int:
    """Return the number of hosts for a TPU type (chips / 4 for v5p/v6e)."""
    parts = tpu_type.split("-")
    if len(parts) != 2:
        raise ValueError(f"Cannot parse TPU type {tpu_type!r}")
    total_chips = int(parts[1])
    # All current TPU types have 4 chips per host
    chips_per_host = 4
    return max(1, total_chips // chips_per_host)
