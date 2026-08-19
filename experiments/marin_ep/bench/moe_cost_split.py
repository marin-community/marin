# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split the fused MoE layer cost into transport and expert-GEMM shares.

MEP-081's constant screen measured the whole fused layer (dispatch puts +
gate/up + down + combine, forward and backward) at 204 ms for the hero's
per-device 8/384 workload, which matches the EP64 hero's 202 ms/layer/step of
MoE device time. That makes this one-tray geometry a faithful proxy for the
rack.

This bench measures three costs at that same geometry so the 204 ms can be
attributed:

* ``fused``       -- ``marin_ep_moe_local`` with the fused Mosaic-GPU transport.
* ``unfused-put`` -- the same layer with the standalone put kernel.
* ``ragged-a2a``  -- the same layer over ``jax.lax.ragged_all_to_all``.
* ``gemm``        -- ``brd_expert_mlp`` alone on an already-local pool of the
  same row count, so no puts, no combine, no plan. The difference against a
  transport case is what that transport and the permute bookkeeping cost.
* ``dense``       -- the incumbent's formulation: a capacity-padded
  ``erh,ehi->eri`` batched einsum over the same rows, to price grouped ragged
  GEMMs against dense ones at this shape.

Run one process per GPU:

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<i> python .../moe_cost_split.py
"""

import os
import sys
import time

import jax

collective_mb = int(os.environ.get("MARIN_EP_COLLECTIVE_MEMORY_MB", "0"))
if collective_mb:
    # Bound the NCCL-window collective arena; must precede client creation.
    jax.config.update("jax_pjrt_client_create_options", {"collective_memory_size": collective_mb * 1024 * 1024})

jax.distributed.initialize(
    coordinator_address=os.environ["MARIN_EP_COORD"],
    num_processes=int(os.environ["MARIN_EP_NUM_PROCS"]),
    process_id=int(os.environ["MARIN_EP_PROC_ID"]),
)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import shard_map  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from levanter.grug._moe.brd_expert_mlp import ROW_ALIGN, brd_expert_mlp, brd_expert_mlp_padded  # noqa: E402
from levanter.grug._moe.ep_marin import marin_ep_moe_local  # noqa: E402

TOKENS, TOPK, HIDDEN, INTER = 65536, 8, 6144, 3072
LOCAL_EXPERTS = 6
CF = 1.1
ITERS = 6
# The grouped kernel tiles rows, so hold the pool at the tile boundary the
# fused path also pads to.
POOL_ROWS = -(-int(TOKENS * TOPK * CF) // ROW_ALIGN) * ROW_ALIGN


def bench(fn, *args) -> float:
    """Minimum wall time over ITERS calls, in milliseconds."""
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(ITERS):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def report(label: str, ms: float) -> None:
    if jax.process_index() == 0:
        print(f"{label}: {ms:.2f} ms fwd+bwd", flush=True)


def bench_transport(mesh, rng, transport: str, label: str) -> None:
    proc = jax.process_index()
    devices = jax.device_count()
    num_experts = devices * LOCAL_EXPERTS
    probs = rng.dirichlet(np.full(num_experts, 0.7))
    experts = rng.choice(num_experts, size=(devices * TOKENS, TOPK), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * TOKENS, HIDDEN)).astype(np.float32)
    weights = (rng.random((devices * TOKENS, TOPK)) + 0.05).astype(np.float32)
    w13 = (0.02 * rng.standard_normal((num_experts, HIDDEN, 2 * INTER))).astype(np.float32)
    w2 = (0.02 * rng.standard_normal((num_experts, INTER, HIDDEN))).astype(np.float32)
    cot = rng.standard_normal((devices * TOKENS, HIDDEN)).astype(np.float32)

    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)

    def put_batch(a, dtype=None):
        mine = a.reshape(devices, TOKENS, *a.shape[1:])[proc]
        if dtype is not None:
            mine = mine.astype(dtype)
        return jax.make_array_from_process_local_data(NamedSharding(mesh, batch_spec), mine, a.shape)

    def put_weight(a):
        mine = a.reshape(devices, LOCAL_EXPERTS, *a.shape[1:])[proc].astype(jnp.bfloat16)
        return jax.make_array_from_process_local_data(NamedSharding(mesh, weight_spec), mine, a.shape)

    xb, eb, wb = put_batch(x, jnp.bfloat16), put_batch(experts), put_batch(weights)
    cb, w13b, w2b = put_batch(cot, jnp.bfloat16), put_weight(w13), put_weight(w2)

    shard_fn = shard_map(
        lambda x_, e_, w_, w13_, w2_: marin_ep_moe_local(
            x_,
            e_,
            w_,
            w13_,
            w2_,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=CF,
            pool_group_size=LOCAL_EXPERTS,
            transport=transport,
            expert_mlp=brd_expert_mlp_padded,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def loss(x_, e_, w_, c_, w13_, w2_):
        y, _ = shard_fn(x_, e_, w_, w13_, w2_)
        return jnp.sum(y * c_)

    with jax.set_mesh(mesh):
        step = jax.jit(jax.grad(loss, argnums=(0, 4, 5)))
        report(label, bench(step, xb, eb, wb, cb, w13b, w2b))


def local_pool(rng):
    """Per-device pool rows, expert weights, and a cotangent, on one device."""
    x = jnp.asarray(rng.standard_normal((POOL_ROWS, HIDDEN)).astype(np.float32), dtype=jnp.bfloat16)
    w13 = jnp.asarray(0.02 * rng.standard_normal((LOCAL_EXPERTS, HIDDEN, 2 * INTER)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(0.02 * rng.standard_normal((LOCAL_EXPERTS, INTER, HIDDEN)), dtype=jnp.bfloat16)
    cot = jnp.asarray(rng.standard_normal((POOL_ROWS, HIDDEN)), dtype=jnp.bfloat16)
    return x, w13, w2, cot


def bench_gemm(rng) -> None:
    """Grouped ragged expert MLP with no transport, at the pooled row count."""
    x, w13, w2, cot = local_pool(rng)
    # Uneven groups, 64-aligned as cudnn_grouped_wgrad requires.
    shares = rng.dirichlet(np.full(LOCAL_EXPERTS, 4.0))
    counts = np.floor(shares * POOL_ROWS / 64).astype(np.int32) * 64
    counts[-1] = POOL_ROWS - counts[:-1].sum()
    group_sizes = jnp.asarray(counts, dtype=jnp.int32)

    def loss(x_, w13_, w2_, c_):
        return jnp.sum(brd_expert_mlp(x_, w13_, w2_, group_sizes) * c_)

    step = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    report("gemm", bench(step, x, w13, w2, cot))


def bench_dense(rng) -> None:
    """Capacity-padded dense batched einsum, the incumbent's expert-MLP form."""
    capacity = POOL_ROWS // LOCAL_EXPERTS
    x = jnp.asarray(rng.standard_normal((LOCAL_EXPERTS, capacity, HIDDEN)), dtype=jnp.bfloat16)
    w13 = jnp.asarray(0.02 * rng.standard_normal((LOCAL_EXPERTS, HIDDEN, 2 * INTER)), dtype=jnp.bfloat16)
    w2 = jnp.asarray(0.02 * rng.standard_normal((LOCAL_EXPERTS, INTER, HIDDEN)), dtype=jnp.bfloat16)
    cot = jnp.asarray(rng.standard_normal((LOCAL_EXPERTS, capacity, HIDDEN)), dtype=jnp.bfloat16)

    def loss(x_, w13_, w2_, c_):
        gu = jnp.einsum("erh,ehi->eri", x_, w13_)
        gate, up = gu[:, :, :INTER], gu[:, :, INTER:]
        act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
        y = jnp.einsum("eri,eih->erh", act, w2_)
        return jnp.sum(y * c_)

    step = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
    report("dense", bench(step, x, w13, w2, cot))


def main() -> None:
    devices = jax.device_count()
    assert jax.local_device_count() == 1
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    if jax.process_index() == 0:
        print(f"pool_rows={POOL_ROWS} hidden={HIDDEN} inter={INTER} local_experts={LOCAL_EXPERTS}", flush=True)
    bench_transport(mesh, np.random.default_rng(seed=7), "mgpu_fused", "fused")
    bench_transport(mesh, np.random.default_rng(seed=7), "mgpu", "unfused-put")
    bench_transport(mesh, np.random.default_rng(seed=7), "ragged", "ragged-a2a")
    bench_gemm(np.random.default_rng(seed=11))
    bench_dense(np.random.default_rng(seed=13))
    if jax.process_index() == 0:
        print("SPLIT_COMPLETED", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
