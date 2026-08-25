"""Standalone jax.lax.ragged_all_to_all throughput benchmark with output validation.

Same shape and protocol as loop-260823-dknative/ragged_a2a_bench.py (hero-shape uniform
exchange, donated ping-pong output), plus a correctness check: the payload is an analytic
row pattern (value depends only on sender rank and source row, exactly representable in
bf16), so every receiver can reconstruct its expected buffer locally and compare bit-exact.
A barrier/synchronization bug in a patched kernel shows up as VALIDATION FAIL instead of a
silently wrong-but-fast timing.

Run one process per GPU across an NVLink domain. Transport selected purely by XLA_FLAGS
(see the leg-3 bench docstring for the flag recipes).
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ROWS = 261_120  # rows sent per rank per call (~3.2 GB at HIDDEN=6144 bf16)
HIDDEN = 6_144
UPDATES_PER_PEER = int(os.environ.get("BENCH_UPDATES_PER_PEER", "30"))
WARMUP = 10
ITERS = 50
VALUE_MOD = 251  # keep row values exactly representable in bf16


def row_values(sender: int) -> np.ndarray:
    """Analytic per-row payload value for a sender rank."""
    return ((sender * ROWS + np.arange(ROWS, dtype=np.int64)) % VALUE_MOD).astype(np.float32)


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
        donate_argnums=(1,),
    )

    def shard(build_row):
        example = build_row(0)
        shape = (world,) + example.shape
        return jax.make_array_from_callback(
            shape, sharded, lambda idx: build_row(idx[0].start)[None, ...]
        )

    def payload(r):
        return np.broadcast_to(row_values(r)[:, None], (ROWS, HIDDEN)).astype(jnp.bfloat16)

    args = [
        shard(payload),
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

    # Validate the final output buffer. Receiver r, landing rows
    # [(s*U+u)*rpu, ...+rpu) hold sender s's rows [(r*U+u)*rpu, ...+rpu).
    expected = np.empty(ROWS, dtype=np.float32)
    for s in range(world):
        sender_vals = row_values(s)
        for u in range(UPDATES_PER_PEER):
            dst0 = (s * UPDATES_PER_PEER + u) * rows_per_update
            src0 = (rank * UPDATES_PER_PEER + u) * rows_per_update
            expected[dst0 : dst0 + rows_per_update] = sender_vals[src0 : src0 + rows_per_update]
    local = np.asarray(jax.device_get(out.addressable_shards[0].data))[0]
    got = local[:, 0].astype(np.float32)
    mismatched = int(np.sum(got != expected))
    full_ok = bool(np.array_equal(local, np.broadcast_to(local[:, :1], local.shape)))
    if mismatched or not full_ok:
        print(f"[rank{rank}] VALIDATION FAIL: {mismatched} row mismatches, row-constant={full_ok}")
    elif rank == 0:
        print("VALIDATION OK")

    ms = elapsed / ITERS * 1e3
    gbytes = ROWS * HIDDEN * 2 / 1e9
    if rank == 0:
        print(f"world={world} rows={ROWS} hidden={HIDDEN} updates/peer={UPDATES_PER_PEER}")
        print(f"per-call: {ms:.2f} ms   payload/rank: {gbytes:.2f} GB   egress: {gbytes / ms * 1e3:.0f} GB/s")
        print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")


if __name__ == "__main__":
    main()
