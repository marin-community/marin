"""Standalone jax.lax.ragged_all_to_all throughput benchmark (no framework deps).

Times the ragged all-to-all collective at the per-launch shape of a large MoE
expert-parallel training step: every rank exchanges `ROWS` rows of `HIDDEN`
bf16 with all peers, as `UPDATES_PER_PEER` equal updates per peer (the
per-(peer, expert, split) granularity an EP MoE layer produces).

Run one process per GPU across an NVLink domain (tested at 64 GPUs on a GB200
NVL72). The transport implementation is selected purely by XLA_FLAGS:

  one-shot (default path):
    XLA_FLAGS=""
    # needs kMaxPeers >= world size in multi_gpu_barrier_kernel.h; stock XLA
    # (kMaxPeers=32) reads out of bounds above 32 ranks
  device kernel:
    XLA_FLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true
               --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"

Prints per-call latency and effective per-rank egress bandwidth.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROWS = 261_120  # rows sent per rank per call (~3.2 GB at HIDDEN=6144 bf16)
HIDDEN = 6_144
UPDATES_PER_PEER = 30
WARMUP = 10
ITERS = 50


def main() -> None:
    if os.environ.get("JAX_COORDINATOR_ADDRESS") or os.environ.get("COORDINATOR_ADDRESS"):
        jax.distributed.initialize()
    world = jax.device_count()
    rank = jax.process_index()
    mesh = Mesh(np.array(jax.devices()), ("x",))
    sharded = NamedSharding(mesh, P("x"))

    assert ROWS % (world * UPDATES_PER_PEER) == 0, "ROWS must divide evenly into updates"
    rows_per_update = ROWS // (world * UPDATES_PER_PEER)
    updates = world * UPDATES_PER_PEER

    # Uniform layout: update u sends rows_per_update rows to peer u // UPDATES_PER_PEER.
    # output_offsets are receiver-frame landing offsets; each receiver packs one slot per
    # (sender, update) pair, so the received buffer is exactly full.
    input_offsets = np.arange(updates, dtype=np.int64) * rows_per_update
    send_sizes = np.full(updates, rows_per_update, dtype=np.int64)
    slot = np.arange(updates, dtype=np.int64) % UPDATES_PER_PEER
    recv_sizes = send_sizes.copy()

    def a2a(x, out, in_off, sizes, out_off, rsizes):
        return jax.lax.ragged_all_to_all(
            x[0], out[0], in_off[0], sizes[0], out_off[0], rsizes[0], axis_name="x"
        )[None]

    fn = jax.jit(
        jax.shard_map(
            a2a,
            mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"), P("x"), P("x"), P("x")),
            out_specs=P("x"),
        ),
        # Donate the output buffer and feed it back each call, as a training step
        # does; without donation every call pays a fresh 3.2 GB materialization.
        donate_argnums=(1,),
    )

    def shard(build_row):
        example = build_row(0)
        shape = (world,) + example.shape
        return jax.make_array_from_callback(
            shape, sharded, lambda idx: build_row(idx[0].start)[None, ...]
        )

    args = [
        shard(lambda r: np.random.default_rng(r).standard_normal((ROWS, HIDDEN)).astype(jnp.bfloat16)),
        shard(lambda r: np.zeros((ROWS, HIDDEN), dtype=jnp.bfloat16)),
        shard(lambda r: input_offsets),
        shard(lambda r: send_sizes),
        shard(lambda r: (r * UPDATES_PER_PEER + slot) * rows_per_update),
        shard(lambda r: recv_sizes),
    ]

    x, out, *meta = args
    for _ in range(WARMUP):
        out = fn(x, out, *meta)
    out.block_until_ready()

    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn(x, out, *meta)
    out.block_until_ready()
    elapsed = time.perf_counter() - start

    ms = elapsed / ITERS * 1e3
    gbytes = ROWS * HIDDEN * 2 / 1e9
    if rank == 0:
        print(f"world={world} rows={ROWS} hidden={HIDDEN} updates/peer={UPDATES_PER_PEER}")
        print(f"per-call: {ms:.2f} ms   payload/rank: {gbytes:.2f} GB   egress: {gbytes / ms * 1e3:.0f} GB/s")
        print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")


if __name__ == "__main__":
    main()
