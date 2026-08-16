# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-process (1 GPU/process) conformance smoke of marin_ep with fused transport.

Same oracle checks as ``smoke_marin_ep_multiproc.py`` but forces
``transport="mgpu"`` in a multi-controller mesh: forward pools, drop counts,
and gradients ride ``put_segments``/``put_with_transpose`` whose remote puts
lower through Mosaic's collective-metadata / NCCL device API path (requires a
jax nightly > 0.11.0; no NVSHMEM, no extra XLA flags). Run once per GPU:

  MARIN_EP_COORD=<host:port> MARIN_EP_NUM_PROCS=<N> MARIN_EP_PROC_ID=<i>
  CUDA_VISIBLE_DEVICES=<local gpu> uv run python .../smoke_mgpu_train_multiproc.py
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
from levanter.grug._moe.ep_marin import _static_capacity, marin_ep_moe_local  # noqa: E402

from experiments.marin_ep.oracle import moe_oracle, pooled_keep_mask  # noqa: E402


def main() -> None:
    proc = jax.process_index()
    devices = jax.device_count()
    local = jax.local_device_count()
    assert local == 1, "this smoke wants the production 1 process/GPU topology"
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
            marin_ep_moe_local,
            activation_fn=jax.nn.silu,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            pool_group_size=local_experts,
            transport="mgpu",
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

    def loss_ref(x_, e_, w_, w13_, w2_, cot_, keep_):
        y_ = moe_oracle(x_, e_, w_, w13_, w2_, keep_, activation_fn=jax.nn.silu)
        return jnp.sum((y_ * cot_).astype(jnp.float32))

    gx_ref, gw13_ref, gw2_ref = jax.grad(loss_ref, argnums=(0, 3, 4))(
        jnp.asarray(x),
        jnp.asarray(experts),
        jnp.asarray(weights),
        jnp.asarray(w13),
        jnp.asarray(w2),
        jnp.asarray(cot),
        keep_flat,
    )
    gx_local = np.concatenate([np.asarray(s.data) for s in grads[0].addressable_shards])
    gx_want = np.asarray(gx_ref).reshape(devices, tokens, hidden)[proc * local : (proc + 1) * local].reshape(-1, hidden)
    np.testing.assert_allclose(gx_local, gx_want, rtol=5e-2, atol=0.5)
    gw13_local = np.concatenate([np.asarray(s.data) for s in grads[1].addressable_shards])
    gw13_want = (
        np.asarray(gw13_ref)
        .reshape(devices, local_experts, hidden, 2 * intermediate)[proc * local : (proc + 1) * local]
        .reshape(-1, hidden, 2 * intermediate)
    )
    np.testing.assert_allclose(gw13_local, gw13_want, rtol=5e-2, atol=0.5)
    gw2_local = np.concatenate([np.asarray(s.data) for s in grads[2].addressable_shards])
    gw2_want = (
        np.asarray(gw2_ref)
        .reshape(devices, local_experts, intermediate, hidden)[proc * local : (proc + 1) * local]
        .reshape(-1, intermediate, hidden)
    )
    np.testing.assert_allclose(gw2_local, gw2_want, rtol=5e-2, atol=0.5)

    print(f"[proc {proc}] MGPU TRAIN CONFORMANT (drops {dropped}, {devices} devices)", flush=True)


if __name__ == "__main__":
    main()
