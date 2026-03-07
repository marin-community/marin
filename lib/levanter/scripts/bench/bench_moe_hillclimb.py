# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused MoE hillclimb harness for the functional Grug MoE kernels.

This harness compares the current `levanter.grug.grug_moe.moe_mlp` path against
the pre-optimization EP ring implementation that globally sorted all gathered
assignments before filtering to local experts. It keeps routing fixed so the
timing reflects kernel/collective work rather than router noise.
"""

import argparse
import time
from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np

import jax
import jax.distributed
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh

from levanter.grug import grug_moe as grug_moe_lib
from levanter.utils.activation import ActivationFunctionEnum


Distribution = Literal["random", "runs"]
Kernel = Literal["legacy", "current"]
BenchPass = Literal["forward", "forward_backward"]


def _print0(*args, **kwargs) -> None:
    if jax.process_index() == 0:
        print(*args, **kwargs)


def _time_fn(fn: Callable, *args, warmup: int = 2, iters: int = 5) -> float:
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / iters


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


def _shared_mlp(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
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


def _moe_mlp_ep_ring_local_legacy(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[jax.Array, jax.Array]:
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_experts_global = jax.lax.all_gather(selected_experts_local, "expert", tiled=True)
        combine_weights_global = jax.lax.all_gather(combine_weights_local, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)
        token_flat = jnp.arange(assignments, dtype=jnp.int32) // topk

        sort_idx = jnp.argsort(expert_flat, axis=0)
        expert_sorted = jnp.take(expert_flat, sort_idx, axis=0)
        token_sorted = jnp.take(token_flat, sort_idx, axis=0)
        weight_sorted = jnp.take(weight_flat, sort_idx, axis=0).astype(x_local.dtype)

        local_experts = moe_w13_local.shape[0]
        if num_experts % local_experts != 0:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
            )

        ep_size = num_experts // local_experts
        local_capacity = int(np.ceil(capacity_factor * assignments / ep_size))
        local_capacity = max(local_experts, local_capacity)

        expert_axis = jax.lax.axis_index("expert")
        expert_start = expert_axis * local_experts
        expert_end = expert_start + local_experts
        local_mask = jnp.logical_and(expert_sorted >= expert_start, expert_sorted < expert_end)

        local_idx = jnp.nonzero(local_mask, size=local_capacity, fill_value=0)[0]
        local_count = jnp.sum(local_mask, dtype=jnp.int32)
        dropped_local = jnp.maximum(local_count - local_capacity, 0)
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < local_count
        valid_weight = valid.astype(jnp.float32)

        token_local = jnp.take(token_sorted, local_idx, axis=0)
        expert_local = jnp.take(expert_sorted, local_idx, axis=0) - expert_start
        weight_local = jnp.take(weight_sorted, local_idx, axis=0)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
        expert_local = jnp.where(valid, expert_local, 0)

    group_sizes = jnp.bincount(expert_local, weights=valid_weight, length=local_experts).astype(jnp.int32)
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("scatter"):
        out_global = jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


def _moe_mlp_legacy(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()
    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        out, _ = grug_moe_lib._moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=num_experts,
        )
        return out

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
    local_expert_spec = P("expert", None, None) if has_expert_axis else P(None, None, None)

    if has_expert_axis and expert_axis_size > 1:
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_ring_local_legacy,
                activation_fn=activation_fn,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            ),
            mesh=mesh,
            in_specs=(
                batch_spec,
                batch_spec,
                batch_spec,
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=(batch_spec, P()),
            check_vma=False,
        )
        out, _ = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
        return out

    shard_fn = shard_map(
        partial(
            grug_moe_lib._moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            local_expert_spec,
            local_expert_spec,
        ),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    out, _ = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    return out


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


def _forward(
    kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    if kernel == "legacy":
        routed = _moe_mlp_legacy(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "current":
        routed = grug_moe_lib.moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return routed + _shared_mlp(x, shared_w13, shared_w2)


def _loss_and_grads(
    kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = _forward(
            kernel,
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(x, w_up_gate, w_down, shared_w13, shared_w2)


def _flatten_tree_max_abs(tree_a, tree_b) -> float:
    leaves_a = jax.tree.leaves(tree_a)
    leaves_b = jax.tree.leaves(tree_b)
    return max(
        float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) for a, b in zip(leaves_a, leaves_b)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark legacy vs current functional Grug MoE kernels.")
    parser.add_argument("--coordinator-address", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--process-id", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=32_768)
    parser.add_argument("--hidden", type=int, default=2_048)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--shared-expert-dim", type=int, default=2_048)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--distribution", choices=["random", "runs"], default="random")
    parser.add_argument("--run-alpha", type=float, default=0.98)
    parser.add_argument("--run-noise-scale", type=float, default=0.35)
    parser.add_argument("--bench-pass", choices=["forward", "forward_backward"], default="forward_backward")
    parser.add_argument("--kernel", choices=["legacy", "current", "both"], default="both")
    parser.add_argument("--ep-list", type=str, default="1,2,4,8")
    parser.add_argument("--capacity-factor", type=float, default=grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-equivalence", action="store_true")
    args = parser.parse_args()

    if args.coordinator_address is not None or args.num_processes is not None or args.process_id is not None:
        if args.coordinator_address is None or args.num_processes is None or args.process_id is None:
            raise ValueError(
                "--coordinator-address, --num-processes, and --process-id must be set together for multihost runs"
            )
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )

    dtype = jnp.dtype(args.dtype)
    eps = [int(tok.strip()) for tok in args.ep_list.split(",") if tok.strip()]
    kernels: list[Kernel] = ["legacy", "current"] if args.kernel == "both" else [args.kernel]

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

    _print0(f"devices={jax.devices()}")
    _print0(
        "shape "
        f"tokens={args.tokens} hidden={args.hidden} mlp_dim={args.mlp_dim} experts={args.experts} "
        f"topk={args.topk} shared_expert_dim={args.shared_expert_dim} dtype={dtype} "
        f"distribution={args.distribution} bench_pass={args.bench_pass} capacity_factor={args.capacity_factor}"
    )

    for ep_size in eps:
        mesh = _make_mesh(ep_size)
        with jax.set_mesh(mesh):
            x_sharded, selected_sharded, weights_sharded, w13_sharded, w2_sharded = _shard_inputs(
                mesh, x, selected_experts, combine_weights, w_up_gate, w_down
            )
            shared_w13_sharded, shared_w2_sharded = _shard_shared_weights(mesh, shared_w13, shared_w2)

            if args.check_equivalence and set(kernels) == {"legacy", "current"}:
                legacy_out = jax.jit(_forward, static_argnums=0)(
                    "legacy",
                    x_sharded,
                    selected_sharded,
                    weights_sharded,
                    w13_sharded,
                    w2_sharded,
                    shared_w13_sharded,
                    shared_w2_sharded,
                )
                current_out = jax.jit(_forward, static_argnums=0)(
                    "current",
                    x_sharded,
                    selected_sharded,
                    weights_sharded,
                    w13_sharded,
                    w2_sharded,
                    shared_w13_sharded,
                    shared_w2_sharded,
                )
                out_max_abs = float(jnp.max(jnp.abs(legacy_out.astype(jnp.float32) - current_out.astype(jnp.float32))))
                (_, legacy_grads) = jax.jit(_loss_and_grads, static_argnums=0)(
                    "legacy",
                    x_sharded,
                    selected_sharded,
                    weights_sharded,
                    w13_sharded,
                    w2_sharded,
                    shared_w13_sharded,
                    shared_w2_sharded,
                )
                (_, current_grads) = jax.jit(_loss_and_grads, static_argnums=0)(
                    "current",
                    x_sharded,
                    selected_sharded,
                    weights_sharded,
                    w13_sharded,
                    w2_sharded,
                    shared_w13_sharded,
                    shared_w2_sharded,
                )
                grad_max_abs = _flatten_tree_max_abs(legacy_grads, current_grads)
                _print0(f"CHECK ep={ep_size} out_max_abs={out_max_abs:.6e} grad_max_abs={grad_max_abs:.6e}")

            for kernel in kernels:
                forward_fn = partial(_forward, kernel)
                grad_fn = partial(_loss_and_grads, kernel)
                if args.bench_pass == "forward":
                    dt = _time_fn(
                        forward_fn,
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
                else:
                    dt = _time_fn(
                        grad_fn,
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

                _print0(
                    "RESULT "
                    f"kernel={kernel} ep={ep_size} pass={args.bench_pass} "
                    f"time_s={dt:.6f} tokens_per_s={args.tokens / dt:.2f}"
                )


if __name__ == "__main__":
    main()
