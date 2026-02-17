# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from haliax.nn.linear import gmm_sharded


Distribution = Literal["random", "runs"]
Backend = Literal["gmm", "ragged_dot"]
Impl = Literal[
    "baseline",
    "fused_w13",
    "preweight",
    "fused_w13_preweight",
    "scatter",
    "fast",
]
BenchPass = Literal["forward", "forward_backward"]
CapacityPolicy = Literal["none", "drop", "pad"]
QueueMode = Literal["full", "prequeue", "both"]
RoutingPackStrategy = Literal["argsort", "tuple_sort", "expert_score_sort"]
RematMode = Literal["none", "expert_mlp", "combine"]
ParallelMode = Literal["none", "ep"]
DispatchPermuteMode = Literal["direct_take", "repeat_sort"]
EpCommPath = Literal["compact_psum", "ring_ag_rs"]


def _time_fn(fn: Callable, *args, warmup: int = 2, iters: int = 5) -> float:
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / iters


def _ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(((1,), (1,)), ((), ())),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=dim_numbers,
    )


def _sort_by_permutation(
    inputs: jax.Array,
    sort_indices: jax.Array,
    *,
    use_custom_sort_vjp: bool,
) -> jax.Array:
    """Sorts the leading axis by a permutation, with an optional explicit VJP."""
    if use_custom_sort_vjp:
        return _sort_by_permutation_custom(inputs, sort_indices)
    return jnp.take(inputs, sort_indices, axis=0)


@jax.custom_vjp
def _sort_by_permutation_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    return jnp.take(inputs, sort_indices, axis=0)


def _sort_by_permutation_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _sort_by_permutation_custom(inputs, sort_indices), sort_indices


def _sort_by_permutation_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
    sort_indices = residuals
    # Explicit inverse-permutation backward mirrors MaxText's custom sort VJP.
    return _sort_by_permutation_custom(grads, jnp.argsort(sort_indices, axis=0)), None


_sort_by_permutation_custom.defvjp(
    _sort_by_permutation_custom_fwd,
    _sort_by_permutation_custom_bwd,
)


def _prepare_dispatch(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    capacity_policy: CapacityPolicy,
    routing_pack_strategy: RoutingPackStrategy,
    dispatch_permute_mode: DispatchPermuteMode,
    use_custom_sort_vjp: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens = x.shape[0]
    topk = topk_idx.shape[1]
    assignments = tokens * topk

    topk_idx_flat = topk_idx.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)

    tok_id = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), topk)
    if routing_pack_strategy == "argsort":
        sort_idx = jnp.argsort(topk_idx_flat, axis=0)
        token_idx_sort = jnp.take(tok_id, sort_idx, axis=0)
    elif routing_pack_strategy == "tuple_sort":
        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        _, sort_idx, token_idx_sort = jax.lax.sort((topk_idx_flat, flat_pos, tok_id), dimension=0)
    elif routing_pack_strategy == "expert_score_sort":
        # Jaxformer-style score-priority within each expert group:
        # sort first by expert id, then by descending token top-1 score.
        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        top1_score_flat = jnp.repeat(topk_weights[:, 0].astype(jnp.float32), topk)
        _, _, sort_idx, token_idx_sort = jax.lax.sort(
            (
                topk_idx_flat,
                -top1_score_flat,
                flat_pos,
                tok_id,
            ),
            dimension=0,
        )
    else:
        raise ValueError(f"Unknown routing pack strategy: {routing_pack_strategy}")

    if dispatch_permute_mode == "repeat_sort":
        x_repeat = jnp.repeat(x, topk, axis=0)
        x_repeat_sort = _sort_by_permutation(
            x_repeat,
            sort_idx,
            use_custom_sort_vjp=use_custom_sort_vjp,
        )
    else:
        x_repeat_sort = jnp.take(x, token_idx_sort, axis=0)
    sorted_weights = jnp.take(topk_weights_flat, sort_idx, axis=0).astype(x.dtype)
    group_sizes_raw = jnp.bincount(topk_idx_flat, length=num_experts).astype(jnp.int32)

    if capacity_policy == "none" or capacity_factor <= 0.0:
        return x_repeat_sort, sorted_weights, token_idx_sort, sort_idx, group_sizes_raw

    capacity = int(np.ceil(capacity_factor * assignments / num_experts))
    capacity = max(1, capacity)
    starts = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes_raw, dtype=jnp.int32)[:-1]],
        axis=0,
    )

    if capacity_policy == "drop":
        sorted_expert = jnp.take(topk_idx_flat, sort_idx, axis=0)
        rank_in_expert = jnp.arange(assignments, dtype=jnp.int32) - jnp.take(starts, sorted_expert, axis=0)
        valid = rank_in_expert < capacity
        x_repeat_sort = jnp.where(valid[:, None], x_repeat_sort, 0)
        sorted_weights = jnp.where(valid, sorted_weights, 0)
        return x_repeat_sort, sorted_weights, token_idx_sort, sort_idx, group_sizes_raw

    if capacity_policy == "pad":
        slot_idx = jnp.arange(num_experts * capacity, dtype=jnp.int32)
        slot_expert = slot_idx // capacity
        slot_offset = slot_idx % capacity
        src_idx = jnp.take(starts, slot_expert, axis=0) + slot_offset
        valid = slot_offset < jnp.take(group_sizes_raw, slot_expert, axis=0)

        x_sel = jnp.take(x_repeat_sort, jnp.minimum(src_idx, assignments - 1), axis=0)
        w_sel = jnp.take(sorted_weights, jnp.minimum(src_idx, assignments - 1), axis=0)
        t_sel = jnp.take(token_idx_sort, jnp.minimum(src_idx, assignments - 1), axis=0)

        x_pad = jnp.where(valid[:, None], x_sel, 0)
        w_pad = jnp.where(valid, w_sel, 0)
        t_pad = jnp.where(valid, t_sel, 0)
        g_pad = jnp.full((num_experts,), capacity, dtype=jnp.int32)
        # sort_idx is unused in pad mode because combine always uses scatter.
        return x_pad, w_pad, t_pad, jnp.zeros((1,), dtype=jnp.int32), g_pad

    raise ValueError(f"Unknown capacity policy: {capacity_policy}")


def _moe_block(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    w13: jax.Array,
    shared_w1: jax.Array,
    shared_w2: jax.Array,
    shared_w3: jax.Array,
    shared_w13: jax.Array,
    *,
    backend: Backend,
    impl: Impl,
    shared_fused: bool,
    capacity_factor: float,
    capacity_policy: CapacityPolicy,
    routing_pack_strategy: RoutingPackStrategy,
    dispatch_permute_mode: DispatchPermuteMode,
    use_custom_sort_vjp: bool,
    remat_mode: RematMode,
    force_scatter_combine: bool = False,
) -> jax.Array:
    tokens, hidden = x.shape
    num_experts = w1.shape[0]

    x_repeat_sort, sorted_weights, token_idx_sort, sort_idx, group_sizes = _prepare_dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        capacity_policy=capacity_policy,
        routing_pack_strategy=routing_pack_strategy,
        dispatch_permute_mode=dispatch_permute_mode,
        use_custom_sort_vjp=use_custom_sort_vjp,
    )

    return _moe_from_dispatch(
        x,
        topk_weights,
        x_repeat_sort,
        sorted_weights,
        token_idx_sort,
        sort_idx,
        group_sizes,
        w1,
        w2,
        w3,
        w13,
        shared_w1,
        shared_w2,
        shared_w3,
        shared_w13,
        backend=backend,
        impl=impl,
        shared_fused=shared_fused,
        force_scatter=(capacity_policy == "pad") or force_scatter_combine,
        use_custom_sort_vjp=use_custom_sort_vjp,
        remat_mode=remat_mode,
    )


def _moe_from_dispatch(
    x: jax.Array,
    topk_weights: jax.Array,
    x_repeat_sort: jax.Array,
    sorted_weights: jax.Array,
    token_idx_sort: jax.Array,
    sort_idx: jax.Array,
    group_sizes: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    w13: jax.Array,
    shared_w1: jax.Array,
    shared_w2: jax.Array,
    shared_w3: jax.Array,
    shared_w13: jax.Array,
    *,
    backend: Backend,
    impl: Impl,
    shared_fused: bool,
    force_scatter: bool = False,
    use_custom_sort_vjp: bool = False,
    remat_mode: RematMode = "none",
) -> jax.Array:
    tokens, hidden = x.shape
    topk = topk_weights.shape[1]
    fuse_w13 = impl in ("fused_w13", "fast")
    preweight = impl in ("preweight", "fused_w13_preweight")
    if impl == "fused_w13_preweight":
        fuse_w13 = True
    use_scatter = (impl in ("scatter", "fast")) or force_scatter

    def expert_mlp_path(
        x_repeat_sort_in: jax.Array,
        sorted_weights_in: jax.Array,
        group_sizes_in: jax.Array,
        w1_in: jax.Array,
        w2_in: jax.Array,
        w3_in: jax.Array,
        w13_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if fuse_w13:
            if backend == "gmm":
                w13_out = gmm_sharded(x_repeat_sort_in, w13_in, group_sizes_in)
            else:
                w13_out = _ragged_dot(x_repeat_sort_in, w13_in, group_sizes_in)
            w1_out, w3_out = jnp.split(w13_out, [w1_in.shape[2]], axis=-1)
        else:
            if backend == "gmm":
                w1_out = gmm_sharded(x_repeat_sort_in, w1_in, group_sizes_in)
                w3_out = gmm_sharded(x_repeat_sort_in, w3_in, group_sizes_in)
            else:
                w1_out = _ragged_dot(x_repeat_sort_in, w1_in, group_sizes_in)
                w3_out = _ragged_dot(x_repeat_sort_in, w3_in, group_sizes_in)

        gated = jax.nn.silu(w1_out) * w3_out
        sorted_weights_cast = sorted_weights_in.astype(gated.dtype)
        if preweight:
            gated = gated * sorted_weights_cast[:, None]

        if backend == "gmm":
            out_repeat_sort_out = gmm_sharded(gated, w2_in, group_sizes_in)
        else:
            out_repeat_sort_out = _ragged_dot(gated, w2_in, group_sizes_in)
        return out_repeat_sort_out, sorted_weights_cast

    if remat_mode == "expert_mlp":
        out_repeat_sort, sorted_weights = jax.checkpoint(expert_mlp_path)(
            x_repeat_sort, sorted_weights, group_sizes, w1, w2, w3, w13
        )
    else:
        out_repeat_sort, sorted_weights = expert_mlp_path(x_repeat_sort, sorted_weights, group_sizes, w1, w2, w3, w13)

    def combine_path(
        out_repeat_sort_in: jax.Array,
        sorted_weights_in: jax.Array,
        token_idx_sort_in: jax.Array,
        sort_idx_in: jax.Array,
    ) -> jax.Array:
        if use_scatter:
            weighted = out_repeat_sort_in if preweight else (out_repeat_sort_in * sorted_weights_in[:, None])
            return jnp.zeros((tokens, hidden), dtype=weighted.dtype).at[token_idx_sort_in].add(weighted)

        inv_sort_idx = jnp.argsort(sort_idx_in, axis=0)
        out_repeat = _sort_by_permutation(
            out_repeat_sort_in,
            inv_sort_idx,
            use_custom_sort_vjp=use_custom_sort_vjp,
        )
        out_repeat = out_repeat.reshape(tokens, topk, hidden)
        return jnp.sum(out_repeat, axis=1) if preweight else jnp.sum(out_repeat * topk_weights[..., None], axis=1)

    if remat_mode == "combine":
        out = jax.checkpoint(combine_path)(out_repeat_sort, sorted_weights, token_idx_sort, sort_idx)
    else:
        out = combine_path(out_repeat_sort, sorted_weights, token_idx_sort, sort_idx)

    shared_dim = shared_w2.shape[0]
    if shared_dim > 0:
        if shared_fused:
            shared13 = jax.lax.dot_general(
                x,
                shared_w13,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            shared1, shared3 = jnp.split(shared13, [shared_dim], axis=-1)
        else:
            shared1 = jax.lax.dot_general(
                x,
                shared_w1,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            shared3 = jax.lax.dot_general(
                x,
                shared_w3,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        shared_gated = jax.nn.silu(shared1) * shared3
        shared_out = jax.lax.dot_general(
            shared_gated,
            shared_w2,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out + shared_out.astype(out.dtype)

    return out.astype(x.dtype)


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


def _apply_null_routing(
    key: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    experts: int,
    null_route_frac: float,
    null_experts: int,
    renormalize_real: bool,
) -> tuple[jax.Array, jax.Array, float, int]:
    if null_route_frac <= 0.0:
        return topk_idx, topk_weights, 0.0, 0

    if null_route_frac >= 1.0:
        raise ValueError(f"null_route_frac must be in [0, 1), got {null_route_frac}")

    if null_experts <= 0:
        # rho = 1 - null_frac -> M = N * (1-rho)/rho = N * null_frac / (1-null_frac)
        null_experts = max(1, int(round(experts * null_route_frac / (1.0 - null_route_frac))))

    null_mask = jax.random.bernoulli(key, p=null_route_frac, shape=topk_idx.shape)
    # Keep at least one routed assignment to avoid degenerate all-null dispatch.
    if bool(jnp.all(null_mask)):
        null_mask = null_mask.at[0, 0].set(False)

    null_slots = experts + (jnp.arange(topk_idx.size, dtype=jnp.int32) % null_experts).reshape(topk_idx.shape)
    routed_idx = jnp.where(null_mask, null_slots, topk_idx).astype(jnp.int32)
    routed_weights = jnp.where(null_mask, jnp.zeros_like(topk_weights), topk_weights)
    if renormalize_real:
        denom = jnp.sum(routed_weights, axis=-1, keepdims=True)
        routed_weights = jnp.where(denom > 0.0, routed_weights / denom, routed_weights)

    realized_frac = float(np.asarray(jnp.mean(null_mask.astype(jnp.float32))))
    return routed_idx, routed_weights.astype(topk_weights.dtype), realized_frac, null_experts


def _run_stats(top1_experts: jax.Array) -> dict[str, float]:
    arr = np.asarray(top1_experts, dtype=np.int32)
    if arr.size == 0:
        return {
            "runs": 0.0,
            "mean_run_len": 0.0,
            "p95_run_len": 0.0,
            "max_run_len": 0.0,
        }

    change_idx = np.nonzero(arr[1:] != arr[:-1])[0] + 1
    run_starts = np.concatenate([np.array([0], dtype=np.int32), change_idx, np.array([arr.size], dtype=np.int32)])
    run_lengths = np.diff(run_starts)
    return {
        "runs": float(run_lengths.size),
        "mean_run_len": float(np.mean(run_lengths)),
        "p95_run_len": float(np.percentile(run_lengths, 95)),
        "max_run_len": float(np.max(run_lengths)),
    }


def _load_stats(topk_idx: jax.Array, *, num_experts: int) -> dict[str, float]:
    counts = np.asarray(jnp.bincount(topk_idx.reshape(-1), length=num_experts), dtype=np.float64)
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    cv = std / mean if mean > 0 else 0.0
    return {
        "min_load": float(np.min(counts)),
        "max_load": float(np.max(counts)),
        "std_load": std,
        "cv_load": cv,
    }


def _bench_one_distribution(
    *,
    distribution: Distribution,
    key: jax.Array,
    x: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    w13: jax.Array,
    shared_w1: jax.Array,
    shared_w2: jax.Array,
    shared_w3: jax.Array,
    shared_w13: jax.Array,
    topk: int,
    backend: Backend,
    impl: Impl,
    shared_fused: bool,
    bench_pass: BenchPass,
    stage_timing: bool,
    capacity_factor: float,
    capacity_policy: CapacityPolicy,
    routing_pack_strategy: RoutingPackStrategy,
    dispatch_permute_mode: DispatchPermuteMode,
    use_custom_sort_vjp: bool,
    ep_comm_path: EpCommPath,
    remat_mode: RematMode,
    parallel_mode: ParallelMode,
    queue_mode: QueueMode,
    warmup: int,
    iters: int,
    run_alpha: float,
    run_noise_scale: float,
    null_route_frac: float,
    null_experts: int,
    null_route_seed: int,
    renormalize_real_after_null: bool,
) -> None:
    tokens = x.shape[0]
    hidden = x.shape[1]
    mlp_dim = w1.shape[2]
    experts = w1.shape[0]

    key_router, key_null = jax.random.split(key)
    key_null = jax.random.fold_in(key_null, null_route_seed)

    router_logits = _sample_router_logits(
        key_router,
        tokens=tokens,
        experts=experts,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
    )
    topk_idx, topk_weights = _route_topk(router_logits, topk=topk)
    topk_idx, topk_weights, realized_null_frac, active_null_experts = _apply_null_routing(
        key_null,
        topk_idx,
        topk_weights,
        experts=experts,
        null_route_frac=null_route_frac,
        null_experts=null_experts,
        renormalize_real=renormalize_real_after_null,
    )
    null_enabled = realized_null_frac > 0.0
    if null_enabled and capacity_policy != "none":
        raise ValueError("Null routing currently requires capacity_policy='none' in this harness.")

    run_stats = _run_stats(topk_idx[:, 0])
    load_stats = _load_stats(topk_idx, num_experts=experts)
    assignments = tokens * topk
    null_assignments = int(np.asarray(jnp.sum(topk_idx >= experts)))
    real_assignments = assignments - null_assignments
    cap = int(np.ceil(capacity_factor * assignments / experts)) if capacity_factor > 0.0 else 0
    if capacity_policy == "none" or cap <= 0:
        dropped = 0
        drop_frac = 0.0
    else:
        counts = np.asarray(jnp.bincount(topk_idx.reshape(-1), length=experts), dtype=np.int32)
        dropped = int(np.maximum(counts - cap, 0).sum())
        drop_frac = float(dropped / max(1, assignments))

    modes: list[QueueMode]
    if queue_mode == "both":
        modes = ["full", "prequeue"]
    else:
        modes = [queue_mode]
    if bench_pass != "forward":
        modes = ["full"]

    dispatch = _prepare_dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=experts,
        capacity_factor=capacity_factor,
        capacity_policy=capacity_policy,
        routing_pack_strategy=routing_pack_strategy,
        dispatch_permute_mode=dispatch_permute_mode,
        use_custom_sort_vjp=use_custom_sort_vjp,
    )

    if parallel_mode == "ep" and capacity_policy == "pad":
        raise ValueError("parallel_mode='ep' currently does not support capacity_policy='pad'")

    x_dispatch, w_dispatch, token_dispatch, sort_dispatch, group_dispatch = dispatch
    if capacity_policy == "pad" and cap > 0:
        sorted_expert_dispatch = jnp.repeat(jnp.arange(experts, dtype=jnp.int32), cap)
    else:
        sorted_expert_dispatch = jnp.take(topk_idx.reshape(-1), sort_dispatch, axis=0)

    def forward_fn(
        x_in,
        w1_in,
        w2_in,
        w3_in,
        w13_in,
        sw1_in,
        sw2_in,
        sw3_in,
        sw13_in,
    ):
        return _moe_block(
            x_in,
            topk_idx,
            topk_weights,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            sw1_in,
            sw2_in,
            sw3_in,
            sw13_in,
            backend=backend,
            impl=impl,
            shared_fused=shared_fused,
            capacity_factor=capacity_factor,
            capacity_policy=capacity_policy,
            routing_pack_strategy=routing_pack_strategy,
            dispatch_permute_mode=dispatch_permute_mode,
            use_custom_sort_vjp=use_custom_sort_vjp,
            remat_mode=remat_mode,
            force_scatter_combine=null_enabled,
        )

    if parallel_mode == "ep":
        num_devices = jax.local_device_count()
        if experts % num_devices != 0:
            raise ValueError(
                f"parallel_mode='ep' requires experts divisible by local device count: experts={experts}, devices={num_devices}"
            )
        ep_mesh = jax.make_mesh((num_devices,), ("ep",))
        local_experts = experts // num_devices

        sorted_expert_np = np.asarray(sorted_expert_dispatch, dtype=np.int32)
        token_dispatch_np = np.asarray(token_dispatch, dtype=np.int32)
        w_dispatch_np = np.asarray(w_dispatch)
        group_dispatch_np = np.asarray(group_dispatch, dtype=np.int32)

        # Build compact per-shard dispatch once from fixed routing so each shard processes
        # only its local assignments instead of a dense global-length buffer.
        per_device_indices: list[np.ndarray] = []
        per_device_counts: list[int] = []
        for dev_idx in range(num_devices):
            start = dev_idx * local_experts
            end = start + local_experts
            local_idx = np.nonzero((sorted_expert_np >= start) & (sorted_expert_np < end))[0].astype(np.int32)
            per_device_indices.append(local_idx)
            per_device_counts.append(int(local_idx.size))

        local_cap_raw = max(per_device_counts) if per_device_counts else 0
        if local_cap_raw <= 0:
            raise ValueError("parallel_mode='ep' found no routed assignments to any local expert shard.")
        # gmm_sharded currently pads using a float32 literal; keep row count aligned to avoid that path.
        local_cap = ((local_cap_raw + 511) // 512) * 512

        valid_mask_per_dev = np.zeros((num_devices, local_cap), dtype=bool)
        token_local_per_dev = np.zeros((num_devices, local_cap), dtype=np.int32)
        weight_local_per_dev = np.zeros((num_devices, local_cap), dtype=w_dispatch_np.dtype)
        group_local_per_dev = np.zeros((num_devices, local_experts), dtype=np.int32)

        for dev_idx in range(num_devices):
            local_idx = per_device_indices[dev_idx]
            local_count = per_device_counts[dev_idx]
            if local_count > 0:
                valid_mask_per_dev[dev_idx, :local_count] = True
                token_local_per_dev[dev_idx, :local_count] = token_dispatch_np[local_idx]
                weight_local_per_dev[dev_idx, :local_count] = w_dispatch_np[local_idx]

            start = dev_idx * local_experts
            end = start + local_experts
            local_sizes = group_dispatch_np[start:end].copy()
            local_sizes[0] += local_cap - local_count
            group_local_per_dev[dev_idx] = local_sizes

        valid_mask_per_dev = jnp.asarray(valid_mask_per_dev)
        token_local_per_dev = jnp.asarray(token_local_per_dev)
        weight_local_per_dev = jnp.asarray(weight_local_per_dev)
        group_local_per_dev = jnp.asarray(group_local_per_dev)

        @partial(
            jax.shard_map,
            mesh=ep_mesh,
            in_specs=(
                P(None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None),
                P("ep", None),
                P("ep", None),
                P("ep", None),
            ),
            out_specs=P(None, None),
            check_vma=False,
        )
        def ep_routed_only(
            x_in,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            valid_local_in,
            token_local_in,
            weight_local_in,
            group_local_in,
        ):
            x_work = x_in[0] if x_in.ndim == 3 and x_in.shape[0] == 1 else x_in
            valid_local = valid_local_in.reshape(-1)
            token_local = token_local_in.reshape(-1)
            weight_local = weight_local_in.reshape(-1)
            group_local = group_local_in.reshape(-1)

            x_take = jnp.take(x_work, token_local, axis=0)
            x_local = jnp.where(valid_local[:, None], x_take, jnp.zeros_like(x_take))
            weight_local = jnp.where(valid_local, weight_local, jnp.zeros_like(weight_local))

            zero_sw1 = jnp.zeros((x_work.shape[1], 0), dtype=x_work.dtype)
            zero_sw2 = jnp.zeros((0, x_work.shape[1]), dtype=x_work.dtype)
            zero_sw3 = jnp.zeros((x_work.shape[1], 0), dtype=x_work.dtype)
            zero_sw13 = jnp.zeros((x_work.shape[1], 0), dtype=x_work.dtype)

            out_local = _moe_from_dispatch(
                x_work,
                topk_weights,
                x_local,
                weight_local,
                token_local,
                jnp.zeros((1,), dtype=jnp.int32),
                group_local,
                w1_in,
                w2_in,
                w3_in,
                w13_in,
                zero_sw1,
                zero_sw2,
                zero_sw3,
                zero_sw13,
                backend=backend,
                impl=impl,
                shared_fused=shared_fused,
                force_scatter=True,
                use_custom_sort_vjp=use_custom_sort_vjp,
                remat_mode=remat_mode,
            )
            out_local = jax.lax.psum(out_local, "ep")
            if x_in.ndim == 3 and x_in.shape[0] == 1:
                return out_local[None, ...]
            return out_local

        if tokens % num_devices != 0 and ep_comm_path == "ring_ag_rs":
            raise ValueError(
                f"ep_comm_path='ring_ag_rs' requires tokens divisible by local device count: "
                f"tokens={tokens}, devices={num_devices}"
            )

        @partial(
            jax.shard_map,
            mesh=ep_mesh,
            in_specs=(
                P("ep", None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P(None, None),
                P(None, None),
                P(None, None),
                P(None, None),
                P("ep", None),
                P("ep", None),
                P("ep", None),
                P("ep", None),
            ),
            out_specs=P("ep", None),
            check_vma=False,
        )
        def ep_routed_ring(
            x_in,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            sw1_in,
            sw2_in,
            sw3_in,
            sw13_in,
            valid_local_in,
            token_local_in,
            weight_local_in,
            group_local_in,
        ):
            x_local = x_in[0] if x_in.ndim == 3 and x_in.shape[0] == 1 else x_in
            # MaxText-style ring dispatch: all-gather token activations to all expert shards.
            x_global = jax.lax.all_gather(x_local, "ep", tiled=True)

            valid_local = valid_local_in.reshape(-1)
            token_local = token_local_in.reshape(-1)
            weight_local = weight_local_in.reshape(-1)
            group_local = group_local_in.reshape(-1)

            x_take = jnp.take(x_global, token_local, axis=0)
            x_dispatch_local = jnp.where(valid_local[:, None], x_take, jnp.zeros_like(x_take))
            weight_dispatch_local = jnp.where(valid_local, weight_local, jnp.zeros_like(weight_local))

            zero_sw1 = jnp.zeros((x_global.shape[1], 0), dtype=x_global.dtype)
            zero_sw2 = jnp.zeros((0, x_global.shape[1]), dtype=x_global.dtype)
            zero_sw3 = jnp.zeros((x_global.shape[1], 0), dtype=x_global.dtype)
            zero_sw13 = jnp.zeros((x_global.shape[1], 0), dtype=x_global.dtype)

            out_local_full = _moe_from_dispatch(
                x_global,
                topk_weights,
                x_dispatch_local,
                weight_dispatch_local,
                token_local,
                jnp.zeros((1,), dtype=jnp.int32),
                group_local,
                w1_in,
                w2_in,
                w3_in,
                w13_in,
                zero_sw1,
                zero_sw2,
                zero_sw3,
                zero_sw13,
                backend=backend,
                impl=impl,
                shared_fused=shared_fused,
                force_scatter=True,
                use_custom_sort_vjp=use_custom_sort_vjp,
                remat_mode=remat_mode,
            )
            # MaxText-style ring collect: reduce-scatter token outputs across expert shards.
            out_shard = jax.lax.psum_scatter(out_local_full, "ep", scatter_dimension=0, tiled=True)

            shared_dim = sw2_in.shape[0]
            if shared_dim > 0:
                if shared_fused:
                    shared13 = jax.lax.dot_general(
                        x_local,
                        sw13_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    shared1, shared3 = jnp.split(shared13, [shared_dim], axis=-1)
                else:
                    shared1 = jax.lax.dot_general(
                        x_local,
                        sw1_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    shared3 = jax.lax.dot_general(
                        x_local,
                        sw3_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                shared_gated = jax.nn.silu(shared1) * shared3
                shared_out = jax.lax.dot_general(
                    shared_gated,
                    sw2_in,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                out_shard = out_shard + shared_out.astype(out_shard.dtype)

            if x_in.ndim == 3 and x_in.shape[0] == 1:
                return out_shard[None, ...].astype(x_local.dtype)
            return out_shard.astype(x_local.dtype)

        def ep_forward_fn_compact(
            x_in,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            sw1_in,
            sw2_in,
            sw3_in,
            sw13_in,
        ):
            out = ep_routed_only(
                x_in,
                w1_in,
                w2_in,
                w3_in,
                w13_in,
                valid_mask_per_dev,
                token_local_per_dev,
                weight_local_per_dev,
                group_local_per_dev,
            )
            shared_dim = sw2_in.shape[0]
            if shared_dim > 0:
                if shared_fused:
                    shared13 = jax.lax.dot_general(
                        x_in,
                        sw13_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    shared1, shared3 = jnp.split(shared13, [shared_dim], axis=-1)
                else:
                    shared1 = jax.lax.dot_general(
                        x_in,
                        sw1_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    shared3 = jax.lax.dot_general(
                        x_in,
                        sw3_in,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                shared_gated = jax.nn.silu(shared1) * shared3
                shared_out = jax.lax.dot_general(
                    shared_gated,
                    sw2_in,
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                out = out + shared_out.astype(out.dtype)
            return out.astype(x_in.dtype)

        def ep_forward_fn_ring(
            x_in,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            sw1_in,
            sw2_in,
            sw3_in,
            sw13_in,
        ):
            return ep_routed_ring(
                x_in,
                w1_in,
                w2_in,
                w3_in,
                w13_in,
                sw1_in,
                sw2_in,
                sw3_in,
                sw13_in,
                valid_mask_per_dev,
                token_local_per_dev,
                weight_local_per_dev,
                group_local_per_dev,
            )

        forward_fn = ep_forward_fn_ring if ep_comm_path == "ring_ag_rs" else ep_forward_fn_compact

    for mode in modes:
        fn: Callable
        if bench_pass == "forward":
            if mode == "full":
                fn = forward_fn
                run_args = (
                    x,
                    w1,
                    w2,
                    w3,
                    w13,
                    shared_w1,
                    shared_w2,
                    shared_w3,
                    shared_w13,
                )
            else:
                if parallel_mode == "ep":
                    fn = forward_fn
                    run_args = (
                        x,
                        w1,
                        w2,
                        w3,
                        w13,
                        shared_w1,
                        shared_w2,
                        shared_w3,
                        shared_w13,
                    )
                else:

                    def prequeue_fn(
                        x_in,
                        w_dispatch_in,
                        token_dispatch_in,
                        sort_dispatch_in,
                        group_dispatch_in,
                        w1_in,
                        w2_in,
                        w3_in,
                        w13_in,
                        sw1_in,
                        sw2_in,
                        sw3_in,
                        sw13_in,
                    ):
                        return _moe_from_dispatch(
                            x_in,
                            topk_weights,
                            x_dispatch,
                            w_dispatch_in,
                            token_dispatch_in,
                            sort_dispatch_in,
                            group_dispatch_in,
                            w1_in,
                            w2_in,
                            w3_in,
                            w13_in,
                            sw1_in,
                            sw2_in,
                            sw3_in,
                            sw13_in,
                            backend=backend,
                            impl=impl,
                            shared_fused=shared_fused,
                            force_scatter=(capacity_policy == "pad") or null_enabled,
                            use_custom_sort_vjp=use_custom_sort_vjp,
                            remat_mode=remat_mode,
                        )

                    fn = prequeue_fn
                    run_args = (
                        x,
                        w_dispatch,
                        token_dispatch,
                        sort_dispatch,
                        group_dispatch,
                        w1,
                        w2,
                        w3,
                        w13,
                        shared_w1,
                        shared_w2,
                        shared_w3,
                        shared_w13,
                    )
        else:
            # Fixed-routing train-like step: includes full backward for x and expert weights.
            def loss_fn(x_in, w1_in, w2_in, w3_in, sw1_in, sw2_in, sw3_in):
                w13_in = jnp.concatenate([w1_in, w3_in], axis=-1)
                sw13_in = jnp.concatenate([sw1_in, sw3_in], axis=-1)
                y = forward_fn(x_in, w1_in, w2_in, w3_in, w13_in, sw1_in, sw2_in, sw3_in, sw13_in)
                return jnp.mean(jnp.square(y.astype(jnp.float32)))

            fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))
            run_args = (
                x,
                w1,
                w2,
                w3,
                shared_w1,
                shared_w2,
                shared_w3,
            )

        dt = _time_fn(
            fn,
            *run_args,
            warmup=warmup,
            iters=iters,
        )

        shared_dim = shared_w2.shape[0]
        if capacity_policy == "pad" and cap > 0:
            routed_flops = 6.0 * hidden * mlp_dim * experts * cap
        else:
            routed_flops = 6.0 * hidden * mlp_dim * real_assignments
        shared_flops = 6.0 * tokens * hidden * shared_dim
        fwd_flops = routed_flops + shared_flops
        flops = fwd_flops if bench_pass == "forward" else 3.0 * fwd_flops
        tflops = flops / dt / 1e12
        tokens_per_sec = tokens / dt

        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} backend={backend}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} remat_mode={remat_mode}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} parallel_mode={parallel_mode}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} ep_comm_path={ep_comm_path}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} time_s={dt:.6f}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} tflops={tflops:.3f}")
        print(
            f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} tokens_per_s={tokens_per_sec:.2f}"
        )
        print(
            f"[{distribution}] topk={topk} impl={impl} run_stats runs={run_stats['runs']:.0f} "
            f"mean={run_stats['mean_run_len']:.2f} p95={run_stats['p95_run_len']:.2f} "
            f"max={run_stats['max_run_len']:.0f}"
        )
        print(
            f"[{distribution}] topk={topk} impl={impl} load_stats min={load_stats['min_load']:.0f} "
            f"max={load_stats['max_load']:.0f} std={load_stats['std_load']:.2f} "
            f"cv={load_stats['cv_load']:.4f}"
        )
        print(
            f"[{distribution}] topk={topk} impl={impl} capacity policy={capacity_policy} "
            f"factor={capacity_factor:.3f} cap={cap} dropped={dropped} drop_frac={drop_frac:.4f}"
        )
        print(
            f"[{distribution}] topk={topk} impl={impl} null_route "
            f"target={null_route_frac:.3f} realized={realized_null_frac:.3f} "
            f"null_experts={active_null_experts} null_assignments={null_assignments} "
            f"real_assignments={real_assignments}"
        )
        print(
            "RESULT "
            f"distribution={distribution} mode={mode} pass={bench_pass} impl={impl} "
            f"backend={backend} topk={topk} null_target={null_route_frac:.3f} "
            f"null_realized={realized_null_frac:.3f} null_assignments={null_assignments} "
            f"real_assignments={real_assignments} tflops={tflops:.6f} tokens_per_s={tokens_per_sec:.6f}"
        )

    if stage_timing and bench_pass == "forward":
        if parallel_mode == "ep":
            print(f"[{distribution}] pass=forward stage_timing skipped for parallel_mode=ep")
            return

        def pack_fn(x_in):
            return _prepare_dispatch(
                x_in,
                topk_idx,
                topk_weights,
                num_experts=experts,
                capacity_factor=capacity_factor,
                capacity_policy=capacity_policy,
                routing_pack_strategy=routing_pack_strategy,
                dispatch_permute_mode=dispatch_permute_mode,
                use_custom_sort_vjp=use_custom_sort_vjp,
            )

        pack_dt = _time_fn(pack_fn, x, warmup=warmup, iters=iters)
        x_dispatch, w_dispatch, token_dispatch, sort_dispatch, group_dispatch = jax.jit(pack_fn)(x)
        use_scatter = impl in ("scatter", "fast") or capacity_policy == "pad" or null_enabled
        preweight = impl in ("preweight", "fused_w13_preweight")
        fuse_w13 = impl in ("fused_w13", "fast", "fused_w13_preweight")

        def up_fn(x_in, groups_in, w_in):
            if fuse_w13:
                if backend == "gmm":
                    w13_out = gmm_sharded(x_in, w13, groups_in)
                else:
                    w13_out = _ragged_dot(x_in, w13, groups_in)
                w1_out, w3_out = jnp.split(w13_out, [w1.shape[2]], axis=-1)
            else:
                if backend == "gmm":
                    w1_out = gmm_sharded(x_in, w1, groups_in)
                    w3_out = gmm_sharded(x_in, w3, groups_in)
                else:
                    w1_out = _ragged_dot(x_in, w1, groups_in)
                    w3_out = _ragged_dot(x_in, w3, groups_in)
            gated = jax.nn.silu(w1_out) * w3_out
            return gated * w_in[:, None] if preweight else gated

        up_dt = _time_fn(up_fn, x_dispatch, group_dispatch, w_dispatch, warmup=warmup, iters=iters)
        gated_dispatch = jax.jit(up_fn)(x_dispatch, group_dispatch, w_dispatch)

        def down_fn(gated_in, groups_in):
            if backend == "gmm":
                return gmm_sharded(gated_in, w2, groups_in)
            return _ragged_dot(gated_in, w2, groups_in)

        down_dt = _time_fn(down_fn, gated_dispatch, group_dispatch, warmup=warmup, iters=iters)
        out_dispatch = jax.jit(down_fn)(gated_dispatch, group_dispatch)

        def combine_fn(out_in, tok_in, sort_in, w_in):
            if use_scatter:
                weighted = out_in if preweight else (out_in * w_in[:, None])
                return jnp.zeros((tokens, hidden), dtype=weighted.dtype).at[tok_in].add(weighted)
            inv_sort_idx = jnp.argsort(sort_in, axis=0)
            out_repeat = _sort_by_permutation(
                out_in,
                inv_sort_idx,
                use_custom_sort_vjp=use_custom_sort_vjp,
            ).reshape(tokens, topk, hidden)
            return jnp.sum(out_repeat, axis=1) if preweight else jnp.sum(out_repeat * topk_weights[..., None], axis=1)

        combine_dt = _time_fn(
            combine_fn, out_dispatch, token_dispatch, sort_dispatch, w_dispatch, warmup=warmup, iters=iters
        )
        stage_total = pack_dt + up_dt + down_dt + combine_dt
        print(f"[{distribution}] pass=forward stage pack_s={pack_dt:.6f}")
        print(f"[{distribution}] pass=forward stage up_s={up_dt:.6f}")
        print(f"[{distribution}] pass=forward stage down_s={down_dt:.6f}")
        print(f"[{distribution}] pass=forward stage combine_s={combine_dt:.6f}")
        print(
            f"[{distribution}] pass=forward stage_pct pack={100.0 * pack_dt / stage_total:.1f} "
            f"up={100.0 * up_dt / stage_total:.1f} down={100.0 * down_dt / stage_total:.1f} "
            f"combine={100.0 * combine_dt / stage_total:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple MoE block harness for hillclimbing across routing distributions."
    )
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--mlp-dim", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--topk-list", type=str, default="")
    parser.add_argument("--shared-expert-dim", type=int, default=0)
    parser.add_argument("--shared-fused", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--backend", choices=["gmm", "ragged_dot"], default="gmm")
    parser.add_argument(
        "--impl",
        choices=["baseline", "fused_w13", "preweight", "fused_w13_preweight", "scatter", "fast", "both"],
        default="both",
    )
    parser.add_argument("--distribution", choices=["random", "runs", "both"], default="both")
    parser.add_argument("--bench-pass", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--parallel-mode", choices=["none", "ep"], default="none")
    parser.add_argument(
        "--ep-comm-path",
        choices=["compact_psum", "ring_ag_rs"],
        default="compact_psum",
        help="Communication strategy for parallel_mode=ep.",
    )
    parser.add_argument("--queue-mode", choices=["full", "prequeue", "both"], default="full")
    parser.add_argument(
        "--routing-pack-strategy",
        choices=["argsort", "tuple_sort", "expert_score_sort"],
        default="argsort",
    )
    parser.add_argument(
        "--dispatch-permute-mode",
        choices=["direct_take", "repeat_sort"],
        default="direct_take",
        help="How token-to-expert dispatch activations are materialized.",
    )
    parser.add_argument(
        "--use-custom-sort-vjp",
        action="store_true",
        help="Use explicit inverse-permutation VJP for permutation sorts (MaxText style).",
    )
    parser.add_argument("--remat-mode", choices=["none", "expert_mlp", "combine"], default="none")
    parser.add_argument("--stage-timing", action="store_true")
    parser.add_argument("--capacity-factor", type=float, default=0.0)
    parser.add_argument("--capacity-policy", choices=["none", "drop", "pad"], default="none")
    parser.add_argument(
        "--run-alpha",
        type=float,
        default=0.98,
        help="AR(1) stickiness for runs distribution (higher => longer runs).",
    )
    parser.add_argument(
        "--run-noise-scale",
        type=float,
        default=0.35,
        help="Noise scale for runs distribution (lower => longer runs).",
    )
    parser.add_argument(
        "--null-route-frac",
        type=float,
        default=0.0,
        help="Target fraction of top-k assignments routed to null slots (0 disables null routing).",
    )
    parser.add_argument(
        "--null-route-frac-list",
        type=str,
        default="",
        help="Comma-separated null-route fractions to sweep (overrides --null-route-frac).",
    )
    parser.add_argument(
        "--null-route-sweep",
        action="store_true",
        help="Shortcut for sweeping null-route fractions 0.1,0.2,...,0.9.",
    )
    parser.add_argument(
        "--null-experts",
        type=int,
        default=0,
        help="Number of null expert slots to model (0 = derive from null fraction).",
    )
    parser.add_argument(
        "--null-route-seed",
        type=int,
        default=0,
        help="Seed offset for sampling which routing assignments are null.",
    )
    parser.add_argument(
        "--renormalize-real-after-null",
        action="store_true",
        help="Renormalize per-token top-k weights over remaining real (non-null) assignments.",
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default="",
        help="Optional base directory for JAX profiler traces (one subdirectory per run).",
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dtype = jnp.dtype(args.dtype)
    key = jax.random.PRNGKey(args.seed)
    key_x, key_w1, key_w2, key_w3, key_route = jax.random.split(key, 5)

    x = jax.random.normal(key_x, (args.tokens, args.hidden), dtype=dtype)
    w1 = jax.random.normal(key_w1, (args.experts, args.hidden, args.mlp_dim), dtype=dtype)
    w2 = jax.random.normal(key_w2, (args.experts, args.mlp_dim, args.hidden), dtype=dtype)
    w3 = jax.random.normal(key_w3, (args.experts, args.hidden, args.mlp_dim), dtype=dtype)
    w13 = jnp.concatenate([w1, w3], axis=-1)
    shared_w1 = jax.random.normal(key_w1, (args.hidden, args.shared_expert_dim), dtype=dtype)
    shared_w2 = jax.random.normal(key_w2, (args.shared_expert_dim, args.hidden), dtype=dtype)
    shared_w3 = jax.random.normal(key_w3, (args.hidden, args.shared_expert_dim), dtype=dtype)
    shared_w13 = jnp.concatenate([shared_w1, shared_w3], axis=-1)

    print(f"devices={jax.devices()}")
    print(
        "shape "
        f"tokens={args.tokens} hidden={args.hidden} mlp_dim={args.mlp_dim} "
        f"experts={args.experts} topk={args.topk} dtype={dtype} backend={args.backend} "
        f"impl={args.impl} shared_expert_dim={args.shared_expert_dim} "
        f"shared_fused={args.shared_fused} bench_pass={args.bench_pass} "
        f"parallel_mode={args.parallel_mode} "
        f"ep_comm_path={args.ep_comm_path} "
        f"queue_mode={args.queue_mode} "
        f"routing_pack_strategy={args.routing_pack_strategy} "
        f"dispatch_permute_mode={args.dispatch_permute_mode} "
        f"use_custom_sort_vjp={args.use_custom_sort_vjp} "
        f"remat_mode={args.remat_mode} "
        f"capacity_policy={args.capacity_policy} capacity_factor={args.capacity_factor} "
        f"null_route_frac={args.null_route_frac} "
        f"null_experts={args.null_experts} "
        f"renormalize_real_after_null={args.renormalize_real_after_null} "
        f"stage_timing={args.stage_timing}"
    )

    distributions: list[Distribution]
    if args.distribution == "both":
        distributions = ["random", "runs"]
    else:
        distributions = [args.distribution]

    topks: list[int]
    if args.topk_list:
        topks = [int(tok.strip()) for tok in args.topk_list.split(",") if tok.strip()]
    else:
        topks = [args.topk]
    for tk in topks:
        if tk <= 0 or tk > args.experts:
            raise ValueError(f"Invalid topk={tk}; must be in [1, experts].")

    impls: list[Impl]
    if args.impl == "both":
        impls = ["baseline", "fused_w13", "preweight", "fused_w13_preweight", "scatter", "fast"]
    else:
        impls = [args.impl]

    null_fracs: list[float]
    if args.null_route_frac_list:
        null_fracs = [float(tok.strip()) for tok in args.null_route_frac_list.split(",") if tok.strip()]
    elif args.null_route_sweep:
        null_fracs = [idx / 10.0 for idx in range(1, 10)]
    else:
        null_fracs = [args.null_route_frac]
    for frac in null_fracs:
        if frac < 0.0 or frac >= 1.0:
            raise ValueError(f"Invalid null-route fraction {frac}; expected in [0, 1).")

    trace_dir = args.trace_dir.strip()

    route_keys = jax.random.split(key_route, len(distributions))
    for dist, dist_key in zip(distributions, route_keys, strict=True):
        for tk in topks:
            for impl in impls:
                for null_frac in null_fracs:
                    bench_kwargs = dict(
                        distribution=dist,
                        key=dist_key,
                        x=x,
                        w1=w1,
                        w2=w2,
                        w3=w3,
                        w13=w13,
                        shared_w1=shared_w1,
                        shared_w2=shared_w2,
                        shared_w3=shared_w3,
                        shared_w13=shared_w13,
                        topk=tk,
                        backend=args.backend,
                        impl=impl,
                        shared_fused=args.shared_fused,
                        bench_pass=args.bench_pass,
                        parallel_mode=args.parallel_mode,
                        queue_mode=args.queue_mode,
                        stage_timing=args.stage_timing,
                        capacity_factor=args.capacity_factor,
                        capacity_policy=args.capacity_policy,
                        routing_pack_strategy=args.routing_pack_strategy,
                        dispatch_permute_mode=args.dispatch_permute_mode,
                        use_custom_sort_vjp=args.use_custom_sort_vjp,
                        ep_comm_path=args.ep_comm_path,
                        remat_mode=args.remat_mode,
                        warmup=args.warmup,
                        iters=args.iters,
                        run_alpha=args.run_alpha,
                        run_noise_scale=args.run_noise_scale,
                        null_route_frac=null_frac,
                        null_experts=args.null_experts,
                        null_route_seed=args.null_route_seed,
                        renormalize_real_after_null=args.renormalize_real_after_null,
                    )
                    if not trace_dir:
                        _bench_one_distribution(**bench_kwargs)
                        continue

                    trace_name = (
                        f"{int(time.time_ns())}_{dist}_topk{tk}_{impl}_null{null_frac:.2f}_"
                        f"{args.parallel_mode}_{args.ep_comm_path}_{args.bench_pass}_{args.queue_mode}"
                    )
                    trace_path = f"{trace_dir.rstrip('/')}/{trace_name}"
                    print(f"trace_dir={trace_path}")
                    with jax.profiler.trace(trace_path, create_perfetto_trace=True):
                        _bench_one_distribution(**bench_kwargs)


if __name__ == "__main__":
    main()
