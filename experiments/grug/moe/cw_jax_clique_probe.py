# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny CoreWeave/JAX multi-host clique probe.

This intentionally avoids the Grug model. It initializes JAX distributed via
Iris, builds a mesh over all visible devices, and runs a minimal collective.
"""

from __future__ import annotations

import os
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def _env_snapshot() -> dict[str, str]:
    keys = (
        "IRIS_TASK_ID",
        "IRIS_NUM_TASKS",
        "NCCL_SOCKET_IFNAME",
        "NCCL_SOCKET_FAMILY",
        "NCCL_IB_DISABLE",
        "NCCL_IB_HCA",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "XLA_FLAGS",
        "CUDA_VISIBLE_DEVICES",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def main() -> None:
    print(f"probe: hostname={socket.gethostname()} env={_env_snapshot()}", flush=True)
    start = time.time()
    initialize_jax()
    print(
        "probe: jax initialized "
        f"process={jax.process_index()}/{jax.process_count()} "
        f"local_devices={jax.local_device_count()} global_devices={jax.device_count()} "
        f"elapsed={time.time() - start:.3f}",
        flush=True,
    )

    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("d",))
    sharding = NamedSharding(mesh, P("d"))

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)
    def collective(x):
        return jax.lax.psum(x, "d")

    x = jax.device_put(jnp.arange(jax.device_count(), dtype=jnp.float32), sharding)
    print("probe: starting collective compile/run", flush=True)
    result = collective(x)
    result.block_until_ready()
    local_shards = [np.asarray(shard.data).reshape(-1).tolist() for shard in result.addressable_shards[:2]]
    print(f"probe: done process={jax.process_index()} local_shards={local_shards}", flush=True)


if __name__ == "__main__":
    main()
