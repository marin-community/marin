# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process EP8 conformance smoke for the marin_ep ragged path.

Hero topology in miniature: one process per node with 4 local GPUs, expert
axis spanning both processes. Validates the waterfilling collectives,
ragged_all_to_all transport, drops, and gradients against the oracle in
the exact multi-controller regime the EP64 benchmark runs in (the fused
mgpu transport is single-process-only upstream, so this is the transport
the hero run uses).

Run once per NODE:
  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=2 MARIN_EP_PROC_ID=<i>
  uv run python experiments/marin_ep/bench/smoke_marin_ep_multiproc.py
"""

import os
from functools import partial

import jax

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
from levanter.grug._moe.ep_marin import _moe_mlp_ep_marin_local, _static_capacity  # noqa: E402

from experiments.marin_ep.oracle import moe_oracle, pooled_keep_mask  # noqa: E402


def main() -> None:
    proc = jax.process_index()
    devices = jax.device_count()
    local = jax.local_device_count()
    tokens, topk, hidden, intermediate = 512, 4, 512, 768
    num_experts = devices * 3  # El=3, like hero
    local_experts = 3
    capacity_factor = 1.2

    rng = np.random.default_rng(seed=11)
    probs = rng.dirichlet(np.full(num_experts, 2.0))
    experts = rng.choice(num_experts, size=(devices * tokens, topk), p=probs).astype(np.int32)
    x = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)
    weights = (rng.random((devices * tokens, topk)) + 0.05).astype(np.float32)
    w13 = (0.1 * rng.standard_normal((num_experts, hidden, 2 * intermediate))).astype(np.float32)
    w2 = (0.1 * rng.standard_normal((num_experts, intermediate, hidden))).astype(np.float32)
    cot = rng.standard_normal((devices * tokens, hidden)).astype(np.float32)

    mesh = Mesh(
        np.asarray(jax.devices()).reshape(1, 1, devices, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    batch_spec = P(("replica_dcn", "data", "expert"))
    weight_spec = P("expert", None, None)
    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_marin_local,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, weight_spec, weight_spec),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )

    def put_batch(a):
        mine = a.reshape(devices, tokens, *a.shape[1:])[proc * local : (proc + 1) * local]
        mine = mine.reshape(local * tokens, *a.shape[1:])
        return jax.make_array_from_process_local_data(NamedSharding(mesh, batch_spec), mine, a.shape)

    def put_weight(a):
        mine = a.reshape(devices, local_experts, *a.shape[1:])[proc * local : (proc + 1) * local]
        mine = mine.reshape(local * local_experts, *a.shape[1:])
        return jax.make_array_from_process_local_data(NamedSharding(mesh, weight_spec), mine, a.shape)

    args = (put_batch(x), put_batch(experts), put_batch(weights), put_weight(w13), put_weight(w2))
    cot_g = put_batch(cot)

    with jax.set_mesh(mesh):
        y, dropped = jax.jit(shard_fn)(*args)
        dropped = int(dropped)

        def loss(x_, e_, w_, w13_, w2_, cot_):
            y_, _ = shard_fn(x_, e_, w_, w13_, w2_)
            return jnp.sum((y_ * cot_).astype(jnp.float32))

        grads = jax.jit(jax.grad(loss, argnums=(0, 3, 4)))(*args, cot_g)
        jax.block_until_ready(grads)

    capacity = _static_capacity(devices * tokens * topk, num_experts, capacity_factor)
    keep, dropped_oracle = pooled_keep_mask(
        experts.reshape(devices, tokens, topk), num_experts=num_experts, capacity=capacity, group_size=local_experts
    )
    assert dropped == dropped_oracle, (dropped, dropped_oracle)
    assert dropped_oracle > 0

    keep_flat = jnp.asarray(keep.reshape(devices * tokens, topk))
    y_oracle = moe_oracle(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        keep_flat,
        activation_fn=jax.nn.silu,
    )
    y_local = np.concatenate([np.asarray(s.data) for s in y.addressable_shards])
    want = np.asarray(y_oracle).reshape(devices, tokens, hidden)[proc * local : (proc + 1) * local].reshape(-1, hidden)
    np.testing.assert_allclose(y_local, want, rtol=2e-2, atol=0.2)
    print(
        f"[proc {proc}] CONFORMANT: EP{devices} multi-process, dropped={dropped} "
        f"({100 * dropped / (devices * tokens * topk):.2f}%), grads computed",
        flush=True,
    )


if __name__ == "__main__":
    main()
