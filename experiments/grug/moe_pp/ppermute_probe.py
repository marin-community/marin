# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does cross-host GPU-to-GPU ``ppermute`` work in multi-controller JAX, and how fast?

The device-group pipeline currently crosses the host boundary by staging the activation
through host memory (``device_get`` + ``broadcast_one_to_all`` or a TCP send). The right
move is an on-device NCCL send/recv -- ``jax.lax.ppermute`` -- which uses GPUDirect RDMA
over InfiniBand and never touches host memory, and which XLA overlaps with compute for
free. The open question is whether a ``shard_map``+``ppermute`` over a mesh that spans
both processes actually lowers and runs under multi-controller JAX (each process only
addresses its local devices), and whether it is meaningfully faster than the host hop.

This probe builds a ``[pp, *act]`` array on a ``(pp=process_count, data=local)`` mesh --
pp rank == process -- from on-device shards (no host staging), ``ppermute``s it one hop
along ``pp`` so each process receives its neighbour's slice, verifies the received tag,
and times the ``ppermute`` against ``broadcast_one_to_all`` of the same bytes.

    KUBECONFIG=~/.kube/coreweave-iris-gpu uv run iris --cluster=cw-us-east-02a job run \\
      --gpu H100x8 --enable-extra-resources --extra gpu --replicas 2 \\
      -- python -m experiments.grug.moe_pp.ppermute_probe
"""

from __future__ import annotations

import functools
import logging
import time

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_pp.benchmark import init_distributed

logger = logging.getLogger(__name__)

ACT = (8, 1024, 1536)  # one boundary activation, ~48 MiB f32


def _build_pp_array(mesh: Mesh, per_shape: tuple[int, ...], fill: float) -> jax.Array:
    """Assemble ``[pp, *per_shape]`` on ``mesh`` (spec ``P('pp','data')``) from on-device
    shards -- each process fills only its own ``pp`` slice, so nothing crosses the wire."""
    pc = mesh.devices.shape[0]
    shape = (pc, *per_shape)
    sharding = NamedSharding(mesh, P("pp", "data"))
    shard_shape = sharding.shard_shape(shape)
    shards = [
        jax.device_put(np.full(shard_shape, fill, np.float32), d)
        for d in mesh.devices.flat
        if d.process_index == jax.process_index()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    init_distributed()
    pc, pid, ld = jax.process_count(), jax.process_index(), jax.local_device_count()
    logger.info("ppermute_probe: %d process(es), %d local dev, platform=%s", pc, ld, jax.devices()[0].platform)
    if pc < 2:
        logger.info("single process -- nothing cross-host to probe; exiting")
        return 0

    mesh = Mesh(np.array(jax.devices()).reshape(pc, ld), ("pp", "data"))

    # Each process tags its slice with its own pid; after one ppermute hop each process
    # should hold its neighbour's tag.
    x = _build_pp_array(mesh, ACT, float(pid))
    perm = [(i, (i + 1) % pc) for i in range(pc)]

    @functools.partial(shard_map, mesh=mesh, in_specs=P("pp", "data"), out_specs=P("pp", "data"))
    def hop(a):
        return jax.lax.ppermute(a, "pp", perm)

    hop = jax.jit(hop)
    y = jax.block_until_ready(hop(x))

    # Verify: this process's pp slice now carries (pid-1) mod pc.
    expected = float((pid - 1) % pc)
    got = float(np.asarray(jax.device_get(y[pid])).flat[0])
    ok = abs(got - expected) < 1e-6
    logger.info(
        "ppermute correctness: process %d received tag %.0f (expected %.0f) -> %s",
        pid,
        got,
        expected,
        "OK" if ok else "WRONG",
    )

    iters = 50

    def time_ppermute():
        t0 = time.perf_counter()
        for _ in range(iters):
            out = hop(x)
        jax.block_until_ready(out)
        return (time.perf_counter() - t0) / iters

    def time_broadcast():
        host = np.full(ACT, float(pid), np.float32)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = multihost_utils.broadcast_one_to_all(host, is_source=(pid == 0))
        return (time.perf_counter() - t0) / iters

    for _ in range(3):
        hop(x)
    jax.block_until_ready(hop(x))
    pp_sec = time_ppermute()
    bc_sec = time_broadcast()
    mb = np.prod(ACT) * 4 / 2**20
    logger.info(
        "PERF process %d: ppermute=%.3f ms | broadcast_one_to_all=%.3f ms | %.0f MiB | speedup=%.1fx | correct=%s",
        pid,
        pp_sec * 1e3,
        bc_sec * 1e3,
        mb,
        bc_sec / pp_sec if pp_sec else float("nan"),
        ok,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
