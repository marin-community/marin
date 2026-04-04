# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the #3821-style expert-padded w13 path on top of the current Grug EP ring kernel.

This is a minimal exact-cap-style benchmark for the overlapping optimization
surface between:
- #4297: Triton ragged_dot backend (`RAGGED_DOT_IMPL=triton`)
- #3821: expert-padded w13 FC1 lowering

It intentionally benchmarks only the `current` EP=8 path, avoiding the broader
DeepEP benchmark stack that is not present on the PR 4297 branch.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Literal

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh

from levanter.grug import grug_moe as grug_moe_lib
from levanter.utils.activation import ActivationFunctionEnum

Distribution = Literal["random", "runs"]


def _round_up_capacity(value: int, *, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _sample_router_logits(
    key: jax.Array,
    *,
    tokens: int,
    experts: int,
    distribution: Distribution,
    run_alpha: float,
    run_noise_scale: float,
) -> jax.Array:
    if distribution == "random":
        return jax.random.normal(key, (tokens, experts), dtype=jnp.float32)

    if distribution == "runs":
        seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**31 - 1))
        rng = np.random.default_rng(seed)
        mean_run = max(2.0, 1.0 / max(1e-6, 1.0 - run_alpha))
        p = min(0.9, max(0.01, 1.0 / mean_run))

        assigned = np.empty((tokens,), dtype=np.int32)
        loads = np.zeros((experts,), dtype=np.int32)
        prev_expert = -1
        pos = 0
        while pos < tokens:
            run_len = int(rng.geometric(p))
            run_len = min(run_len, tokens - pos)
            min_load = int(np.min(loads))
            candidates = np.flatnonzero(loads == min_load)
            if prev_expert in candidates and candidates.size > 1:
                candidates = candidates[candidates != prev_expert]
            expert = int(rng.choice(candidates))
            assigned[pos : pos + run_len] = expert
            loads[expert] += run_len
            prev_expert = expert
            pos += run_len

        logits = rng.normal(loc=0.0, scale=float(run_noise_scale), size=(tokens, experts)).astype(np.float32)
        logits[np.arange(tokens), assigned] += 6.0
        return jnp.asarray(logits, dtype=jnp.float32)

    raise ValueError(f"Unknown distribution: {distribution}")


def _route_topk(router_logits: jax.Array, *, topk: int) -> tuple[jax.Array, jax.Array]:
    topk_logits, topk_idx = jax.lax.top_k(router_logits, topk)
    topk_weights = jax.nn.softmax(topk_logits, axis=-1)
    return topk_idx.astype(jnp.int32), topk_weights.astype(router_logits.dtype)


def _shared_mlp(x: jax.Array, shared_w13: jax.Array, shared_w2: jax.Array) -> jax.Array:
    shared_dim = shared_w2.shape[0]
    if shared_dim == 0:
        return jnp.zeros_like(x)

    batch_spec = grug_moe_lib._batch_spec_from_x(x, get_abstract_mesh())
    shared13 = jnp.einsum("td,dm->tm", x, shared_w13, out_sharding=batch_spec, preferred_element_type=jnp.float32)
    gate, up = jnp.split(shared13, [shared_dim], axis=-1)
    shared_gated = jax.nn.silu(gate) * up
    shared_out = jnp.einsum(
        "tm,md->td",
        shared_gated,
        shared_w2,
        out_sharding=batch_spec,
        preferred_element_type=jnp.float32,
    )
    return shared_out.astype(x.dtype)


def _make_mesh(ep_size: int) -> Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    if len(devices) % ep_size != 0:
        raise ValueError(f"ep_size={ep_size} must divide local device count={len(devices)}")

    mesh_devices = np.array(devices).reshape(len(devices) // ep_size, ep_size, 1)
    return Mesh(
        mesh_devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _shard_inputs(
    mesh: Mesh,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    return (
        jax.sharding.reshard(x, batch_sharding),
        jax.sharding.reshard(selected_experts, batch_sharding),
        jax.sharding.reshard(combine_weights, batch_sharding),
        jax.sharding.reshard(w_up_gate, expert_sharding),
        jax.sharding.reshard(w_down, expert_sharding),
    )


def _shard_shared_weights(mesh: Mesh, shared_w13: jax.Array, shared_w2: jax.Array) -> tuple[jax.Array, jax.Array]:
    replicated = NamedSharding(mesh, P(None, None))
    return (
        jax.sharding.reshard(shared_w13, replicated),
        jax.sharding.reshard(shared_w2, replicated),
    )


def _current_ring_w13_cap_metadata(
    selected_experts: jax.Array,
    *,
    mesh: Mesh,
    num_experts: int,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
    capacity_multiple: int = 128,
) -> int:
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    if expert_axis_size <= 1:
        raise ValueError("This benchmark is intended for expert-parallel runs (ep_size > 1)")
    if num_experts % expert_axis_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size} for current metadata"
        )

    selected_experts_host = np.asarray(jax.device_get(selected_experts), dtype=np.int32)
    local_experts = num_experts // expert_axis_size
    assignments = selected_experts_host.shape[0] * selected_experts_host.shape[1]
    local_capacity = max(local_experts, int(math.ceil(capacity_factor * assignments / expert_axis_size)))
    expert_counts = np.bincount(selected_experts_host.reshape(-1), minlength=num_experts).reshape(expert_axis_size, local_experts)

    max_local_expert_assignments = 0
    for counts in expert_counts:
        remaining = local_capacity
        for count in counts:
            take = min(int(count), remaining)
            max_local_expert_assignments = max(max_local_expert_assignments, take)
            remaining = max(remaining - take, 0)

    return _round_up_capacity(max_local_expert_assignments, multiple=capacity_multiple)


def _forward_current(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    w13_local_expert_capacity: int | None = None,
) -> jax.Array:
    routed = grug_moe_lib.moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=ActivationFunctionEnum.silu,
        w13_local_expert_capacity=w13_local_expert_capacity,
    )
    return routed + _shared_mlp(x, shared_w13, shared_w2)


def _time_fn(fn, *args, warmup: int, iters: int) -> float:
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / iters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=262_144)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--shared-expert-dim", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--distribution", choices=["random", "runs"], default="random")
    parser.add_argument("--run-alpha", type=float, default=0.98)
    parser.add_argument("--run-noise-scale", type=float, default=0.35)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w13-expert-padded", action="store_true")
    parser.add_argument("--ragged-dot-impl", choices=["xla", "triton"], default="xla")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["RAGGED_DOT_IMPL"] = args.ragged_dot_impl

    dtype = jnp.dtype(args.dtype)
    key = jax.random.PRNGKey(args.seed)
    key_x, key_router, key_w13, key_w2, key_sw13, key_sw2 = jax.random.split(key, 6)

    x = jax.random.normal(key_x, (args.tokens, args.hidden), dtype=dtype)
    router_logits = _sample_router_logits(
        key_router,
        tokens=args.tokens,
        experts=args.experts,
        distribution=args.distribution,
        run_alpha=args.run_alpha,
        run_noise_scale=args.run_noise_scale,
    )
    selected_experts, combine_weights = _route_topk(router_logits, topk=args.topk)
    combine_weights = combine_weights.astype(dtype)

    w_up_gate = jax.random.normal(key_w13, (args.experts, args.hidden, 2 * args.mlp_dim), dtype=dtype)
    w_down = jax.random.normal(key_w2, (args.experts, args.mlp_dim, args.hidden), dtype=dtype)
    shared_w13 = jax.random.normal(key_sw13, (args.hidden, 2 * args.shared_expert_dim), dtype=dtype)
    shared_w2 = jax.random.normal(key_sw2, (args.shared_expert_dim, args.hidden), dtype=dtype)

    print(f"DEVICES {jax.devices()}", flush=True)
    print(
        "CONFIG "
        f"tokens={args.tokens} hidden={args.hidden} mlp_dim={args.mlp_dim} experts={args.experts} "
        f"topk={args.topk} shared_expert_dim={args.shared_expert_dim} dtype={dtype} "
        f"distribution={args.distribution} ep_size={args.ep_size} warmup={args.warmup} iters={args.iters} "
        f"ragged_dot_impl={args.ragged_dot_impl} w13_expert_padded={int(args.w13_expert_padded)}",
        flush=True,
    )

    mesh = _make_mesh(args.ep_size)
    with jax.set_mesh(mesh):
        x_sharded, selected_sharded, weights_sharded, w13_sharded, w2_sharded = _shard_inputs(
            mesh,
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
        shared_w13_sharded, shared_w2_sharded = _shard_shared_weights(mesh, shared_w13, shared_w2)
        w13_local_expert_capacity = None
        if args.w13_expert_padded:
            w13_local_expert_capacity = _current_ring_w13_cap_metadata(
                selected_sharded,
                mesh=mesh,
                num_experts=args.experts,
            )
            print(f"W13_CAP local_expert_capacity={w13_local_expert_capacity}", flush=True)

        dt = _time_fn(
            lambda x_, s_, c_, w13_, w2_, sw13_, sw2_: _forward_current(
                x_,
                s_,
                c_,
                w13_,
                w2_,
                sw13_,
                sw2_,
                w13_local_expert_capacity=w13_local_expert_capacity,
            ),
            x_sharded,
            selected_sharded,
            weights_sharded,
            w13_sharded,
            w2_sharded,
            shared_w13_sharded,
            shared_w2_sharded,
            warmup=args.warmup,
            iters=args.iters,
        )

    tokens_per_s = args.tokens / dt
    print(
        "RESULT "
        f"ragged_dot_impl={args.ragged_dot_impl} "
        f"w13_expert_padded={int(args.w13_expert_padded)} "
        f"ep_size={args.ep_size} "
        f"time_s={dt:.6f} "
        f"tokens_per_s={tokens_per_s:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
