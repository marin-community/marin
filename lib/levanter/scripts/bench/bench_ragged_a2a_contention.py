# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""One-node benchmark of the ragged all-to-all device kernel next to a concurrent QuACK grouped GEMM.

The GB200 MoE layer sends one expert chunk per ragged all-to-all: each rank sends
``TK / chunks = 262_144`` bf16 rows of width 3072 (1.61 GB) at (peer, local expert) granularity,
six local experts with three active per chunk, into a ``301_568``-row receiver buffer. The device
kernel runs 32 CTAs of 512 threads. Training overlaps most of these calls with the expert GEMMs,
which are persistent QuACK kernels sized to the whole GPU.

This times, on the node's GPUs as one expert axis with the production per-rank payload:

* ``a2a``: the all-to-all alone.
* ``gemm/r<N>``: the expert up-projection GEMM ``[C, 3072] @ [3, 3072, 4608]`` alone, keeping ``N`` SMs free.
* ``both/r<N>``: both in one program with no dependency between them, so XLA runs them concurrently.

A ``both`` time near ``a2a + gemm`` means the two serialize on SMs; near ``max(a2a, gemm)``, they overlap.

Example::

    XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \\
      --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true \\
      --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall" \\
      python lib/levanter/scripts/bench/bench_ragged_a2a_contention.py --reserve 0 32 40
"""

import argparse
import functools
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.gemm_sm_reservation import reserved_sms
from levanter.grug._moe.quack_moe_cute import quack_grouped_gemm

RESULT_PREFIX = "RESULT_JSON "

# Production per-rank shapes (gb200-d6144: 65_536 tokens x top-8, latent 3072, 2 chunks, capacity 1.15).
ASSIGNMENTS = 524_288
WIDTH = 3_072
CHUNK_ROWS = ASSIGNMENTS // 2
CAPACITY = 301_568
LOCAL_EXPERTS = 6
CHUNK_EXPERTS = 3
INTERMEDIATE = 4_608


def _a2a_params(num_shards: int) -> tuple[np.ndarray, ...]:
    """[S, S * LOCAL_EXPERTS] int32 offsets and sizes, the same on every shard but for its sender position."""
    per = CHUNK_ROWS // (num_shards * CHUNK_EXPERTS)
    sizes = np.zeros((num_shards, LOCAL_EXPERTS), np.int32)
    sizes[:, :CHUNK_EXPERTS] = per
    flat_sizes = sizes.reshape(-1)
    input_offsets = (np.cumsum(flat_sizes) - flat_sizes).astype(np.int32)
    rows = []
    for shard in range(num_shards):
        # Receiver packs arrivals expert-major, sender-major within an expert.
        output_offsets = np.zeros((num_shards, LOCAL_EXPERTS), np.int32)
        for expert in range(CHUNK_EXPERTS):
            output_offsets[:, expert] = expert * num_shards * per + shard * per
        rows.append(output_offsets.reshape(-1))
    out = np.stack(rows)
    tiled = lambda a: np.tile(a[None], (num_shards, 1))  # noqa: E731
    return tiled(input_offsets), tiled(flat_sizes), out, tiled(flat_sizes)


def _time(fn, args, iters: int) -> float:
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=int, nargs="+", default=[0, 32])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    devices = jax.devices()
    num_shards = len(devices)
    mesh = Mesh(np.array(devices), ("expert",))
    shard = NamedSharding(mesh, P("expert"))
    key = jax.random.key(0)
    x = jax.device_put(jax.random.normal(key, (num_shards * ASSIGNMENTS, WIDTH), jnp.bfloat16), shard)
    a = jax.device_put(jax.random.normal(key, (num_shards * CAPACITY, WIDTH), jnp.bfloat16), shard)
    w = jax.device_put(
        jax.random.normal(key, (num_shards * CHUNK_EXPERTS, WIDTH, INTERMEDIATE), jnp.bfloat16) * 0.02, shard
    )
    group = CHUNK_ROWS // CHUNK_EXPERTS
    cu = np.array([0, group, 2 * group, CHUNK_ROWS], np.int32)
    cu = jax.device_put(np.tile(cu[None], (num_shards, 1)), shard)
    params = tuple(jax.device_put(p, shard) for p in _a2a_params(num_shards))

    def a2a_local(x, in_off, send, out_off, recv):
        out = jnp.zeros((CAPACITY, WIDTH), x.dtype)
        return jax.lax.ragged_all_to_all(x, out, in_off[0], send[0], out_off[0], recv[0], axis_name="expert")

    def gemm_local(a, w, cu, reserve):
        with reserved_sms(reserve):
            return quack_grouped_gemm(a, w, cu[0])

    spec = P("expert")
    a2a = jax.jit(jax.shard_map(a2a_local, mesh=mesh, in_specs=(spec,) * 5, out_specs=spec, check_vma=False))
    results = {"shards": num_shards, "a2a_ms": _time(a2a, (x, *params), args.iters)}
    print(f"a2a alone: {results['a2a_ms']:.3f} ms", flush=True)
    for reserve in args.reserve:
        gemm = jax.jit(
            jax.shard_map(
                functools.partial(gemm_local, reserve=reserve),
                mesh=mesh,
                in_specs=(spec,) * 3,
                out_specs=spec,
                check_vma=False,
            )
        )

        def both_local(x, in_off, send, out_off, recv, a, w, cu, reserve=reserve):
            return a2a_local(x, in_off, send, out_off, recv), gemm_local(a, w, cu, reserve)

        both = jax.jit(
            jax.shard_map(both_local, mesh=mesh, in_specs=(spec,) * 8, out_specs=(spec, spec), check_vma=False)
        )
        g = _time(gemm, (a, w, cu), args.iters)
        b = _time(both, (x, *params, a, w, cu), args.iters)
        results[f"gemm_r{reserve}_ms"] = g
        results[f"both_r{reserve}_ms"] = b
        print(
            f"reserve {reserve:3d}: gemm {g:.3f} ms  both {b:.3f} ms  (sum {results['a2a_ms'] + g:.3f}, "
            f"max {max(results['a2a_ms'], g):.3f})",
            flush=True,
        )
    print(RESULT_PREFIX + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
