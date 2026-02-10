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
from haliax.quantization import Int8DotGeneralOp


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
RoutingPackStrategy = Literal["argsort", "tuple_sort"]
RematMode = Literal["none", "expert_mlp", "combine"]
ParallelMode = Literal["none", "ep"]
QuantMode = Literal["none", "int8", "int8_routed", "int8_all", "int8_routed_prequant"]


_INT8_DOT_GENERAL = Int8DotGeneralOp.init()
_RAGGED_DOT_DIM_NUMBERS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((1,), (1,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=(0,),
)


def _uses_routed_int8(quant_mode: QuantMode) -> bool:
    return quant_mode in ("int8_routed", "int8_all", "int8_routed_prequant")


def _uses_dense_int8(quant_mode: QuantMode) -> bool:
    return quant_mode in ("int8", "int8_all")


def _uses_prequant_routed_weights(quant_mode: QuantMode) -> bool:
    return quant_mode == "int8_routed_prequant"


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
    return jax.lax.ragged_dot_general(
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=_RAGGED_DOT_DIM_NUMBERS,
    )


def _int8_dot_general(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
    return _INT8_DOT_GENERAL(
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=jnp.float32,
    )


def _symmetric_int8_quantize(
    x: jax.Array,
    *,
    axis: int | tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    if not isinstance(axis, tuple):
        axis = (axis,)
    axis = tuple(a % x.ndim for a in axis)
    scale_dtype = jnp.float32
    amax = jnp.max(jnp.abs(x.astype(scale_dtype)), axis=axis, keepdims=True)
    scale = (amax / 127.0 + jnp.finfo(scale_dtype).tiny).astype(scale_dtype)
    q = jnp.clip(jnp.round(x.astype(scale_dtype) / scale), -127, 127).astype(jnp.int8)
    return q, scale


def _expert_matmul_int8_routed(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    lhs_q: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    rhs_q: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
) -> jax.Array:
    rows = lhs.shape[0]
    experts = rhs.shape[0]
    if rows == 0:
        return jnp.zeros((0, rhs.shape[2]), dtype=lhs.dtype)
    if experts <= 0:
        raise ValueError(f"Expected positive number of experts, got {experts}")
    if lhs_q is None or lhs_scale is None:
        lhs_q, lhs_scale = _symmetric_int8_quantize(lhs, axis=1)
    if rhs_q is None or rhs_scale is None:
        rhs_q, rhs_scale = _symmetric_int8_quantize(rhs, axis=(1, 2))

    out_i32 = jax.lax.ragged_dot_general(
        lhs=lhs_q,
        rhs=rhs_q,
        group_sizes=group_sizes,
        ragged_dot_dimension_numbers=_RAGGED_DOT_DIM_NUMBERS,
        preferred_element_type=jnp.int32,
    )

    # rhs_scale is [experts, 1, 1]; expand to one scale per routed row.
    rhs_scale_per_expert = rhs_scale.reshape(experts)
    expert_ids = jnp.repeat(jnp.arange(experts, dtype=jnp.int32), group_sizes, total_repeat_length=rows)
    rhs_scale_per_row = jnp.take(rhs_scale_per_expert, expert_ids, axis=0)

    out = out_i32.astype(jnp.float32) * lhs_scale.astype(jnp.float32) * rhs_scale_per_row[:, None]
    return out.astype(lhs.dtype)


@jax.custom_vjp
def _expert_matmul_int8_routed_ste(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    lhs_q: jax.Array | None,
    lhs_scale: jax.Array | None,
    rhs_q: jax.Array | None,
    rhs_scale: jax.Array | None,
) -> jax.Array:
    return _expert_matmul_int8_routed(
        lhs,
        rhs,
        group_sizes,
        lhs_q=lhs_q,
        lhs_scale=lhs_scale,
        rhs_q=rhs_q,
        rhs_scale=rhs_scale,
    )


def _expert_matmul_int8_routed_ste_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    lhs_q: jax.Array | None,
    lhs_scale: jax.Array | None,
    rhs_q: jax.Array | None,
    rhs_scale: jax.Array | None,
) -> tuple[
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None],
]:
    out = _expert_matmul_int8_routed(
        lhs,
        rhs,
        group_sizes,
        lhs_q=lhs_q,
        lhs_scale=lhs_scale,
        rhs_q=rhs_q,
        rhs_scale=rhs_scale,
    )
    return out, (lhs, rhs, group_sizes, lhs_q, lhs_scale, rhs_q, rhs_scale)


def _expert_matmul_int8_routed_ste_bwd(
    residual: tuple[
        jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None
    ],
    grad_out: jax.Array,
) -> tuple[jax.Array, jax.Array, None, None, None, None, None]:
    lhs, rhs, group_sizes, lhs_q, lhs_scale, rhs_q, rhs_scale = residual

    rhs_t = jnp.swapaxes(rhs, 1, 2)
    rhs_t_q: jax.Array | None = None
    rhs_t_scale: jax.Array | None = None
    if rhs_q is not None and rhs_scale is not None:
        rhs_t_q = jnp.swapaxes(rhs_q, 1, 2)
        rhs_t_scale = rhs_scale

    grad_lhs = _expert_matmul_int8_routed(
        grad_out.astype(lhs.dtype),
        rhs_t,
        group_sizes,
        rhs_q=rhs_t_q,
        rhs_scale=rhs_t_scale,
    )

    # STE backward for rhs: use gmm pullback (float weights) to avoid requantizing rhs.
    def _surrogate_rhs(rhs_in: jax.Array) -> jax.Array:
        return gmm_sharded(lhs, rhs_in, group_sizes).astype(grad_out.dtype)

    _, pullback_rhs = jax.vjp(_surrogate_rhs, rhs)
    (grad_rhs,) = pullback_rhs(grad_out)
    return grad_lhs, grad_rhs, None, None, None, None, None


_expert_matmul_int8_routed_ste.defvjp(_expert_matmul_int8_routed_ste_fwd, _expert_matmul_int8_routed_ste_bwd)


def _expert_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    backend: Backend,
    quant_mode: QuantMode,
    lhs_q: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    rhs_q: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
) -> jax.Array:
    if quant_mode == "int8_routed_prequant":
        return _expert_matmul_int8_routed_ste(
            lhs,
            rhs,
            group_sizes,
            lhs_q,
            lhs_scale,
            rhs_q,
            rhs_scale,
        )
    if _uses_routed_int8(quant_mode):
        return _expert_matmul_int8_routed(
            lhs,
            rhs,
            group_sizes,
            lhs_q=lhs_q,
            lhs_scale=lhs_scale,
            rhs_q=rhs_q,
            rhs_scale=rhs_scale,
        )
    if quant_mode == "int8":
        if backend == "gmm":
            return gmm_sharded(lhs, rhs, group_sizes)
        return _ragged_dot(lhs, rhs, group_sizes)
    if backend == "gmm":
        return gmm_sharded(lhs, rhs, group_sizes)
    return _ragged_dot(lhs, rhs, group_sizes)


def _dense_dot(lhs: jax.Array, rhs: jax.Array, *, quant_mode: QuantMode) -> jax.Array:
    if _uses_dense_int8(quant_mode):
        return _int8_dot_general(lhs, rhs, (((1,), (0,)), ((), ())))
    return jax.lax.dot_general(lhs, rhs, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _prepare_dispatch(
    x: jax.Array,
    topk_idx: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float,
    capacity_policy: CapacityPolicy,
    routing_pack_strategy: RoutingPackStrategy,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens = x.shape[0]
    topk = topk_idx.shape[1]
    assignments = tokens * topk

    topk_idx_flat = topk_idx.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)

    if routing_pack_strategy == "argsort":
        sort_idx = jnp.argsort(topk_idx_flat, axis=0)
        token_idx_sort = sort_idx // topk
    elif routing_pack_strategy == "tuple_sort":
        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        tok_id = jnp.repeat(jnp.arange(tokens, dtype=jnp.int32), topk)
        _, sort_idx, token_idx_sort = jax.lax.sort((topk_idx_flat, flat_pos, tok_id), dimension=0)
    else:
        raise ValueError(f"Unknown routing pack strategy: {routing_pack_strategy}")

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
    remat_mode: RematMode,
    quant_mode: QuantMode,
    w1_q: jax.Array | None = None,
    w1_scale: jax.Array | None = None,
    w2_q: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w3_q: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    w13_q: jax.Array | None = None,
    w13_scale: jax.Array | None = None,
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
    )

    x_repeat_sort_q: jax.Array | None = None
    x_repeat_sort_scale: jax.Array | None = None
    if _uses_routed_int8(quant_mode):
        x_q, x_scale = _symmetric_int8_quantize(x, axis=1)
        x_repeat_sort_q = jnp.take(x_q, token_idx_sort, axis=0)
        x_repeat_sort_scale = jnp.take(x_scale, token_idx_sort, axis=0)

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
        force_scatter=(capacity_policy == "pad"),
        remat_mode=remat_mode,
        quant_mode=quant_mode,
        x_repeat_sort_q=x_repeat_sort_q,
        x_repeat_sort_scale=x_repeat_sort_scale,
        w1_q=w1_q,
        w1_scale=w1_scale,
        w2_q=w2_q,
        w2_scale=w2_scale,
        w3_q=w3_q,
        w3_scale=w3_scale,
        w13_q=w13_q,
        w13_scale=w13_scale,
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
    remat_mode: RematMode = "none",
    quant_mode: QuantMode = "none",
    x_repeat_sort_q: jax.Array | None = None,
    x_repeat_sort_scale: jax.Array | None = None,
    w1_q: jax.Array | None = None,
    w1_scale: jax.Array | None = None,
    w2_q: jax.Array | None = None,
    w2_scale: jax.Array | None = None,
    w3_q: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    w13_q: jax.Array | None = None,
    w13_scale: jax.Array | None = None,
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
        use_prequant_lhs = (
            _uses_routed_int8(quant_mode) and x_repeat_sort_q is not None and x_repeat_sort_scale is not None
        )
        use_prequant_rhs = _uses_prequant_routed_weights(quant_mode)
        if fuse_w13:
            w13_out = _expert_matmul(
                x_repeat_sort_in,
                w13_in,
                group_sizes_in,
                backend=backend,
                quant_mode=quant_mode,
                lhs_q=x_repeat_sort_q if use_prequant_lhs else None,
                lhs_scale=x_repeat_sort_scale if use_prequant_lhs else None,
                rhs_q=w13_q if use_prequant_rhs else None,
                rhs_scale=w13_scale if use_prequant_rhs else None,
            )
            w1_out, w3_out = jnp.split(w13_out, [w1_in.shape[2]], axis=-1)
        else:
            w1_out = _expert_matmul(
                x_repeat_sort_in,
                w1_in,
                group_sizes_in,
                backend=backend,
                quant_mode=quant_mode,
                lhs_q=x_repeat_sort_q if use_prequant_lhs else None,
                lhs_scale=x_repeat_sort_scale if use_prequant_lhs else None,
                rhs_q=w1_q if use_prequant_rhs else None,
                rhs_scale=w1_scale if use_prequant_rhs else None,
            )
            w3_out = _expert_matmul(
                x_repeat_sort_in,
                w3_in,
                group_sizes_in,
                backend=backend,
                quant_mode=quant_mode,
                lhs_q=x_repeat_sort_q if use_prequant_lhs else None,
                lhs_scale=x_repeat_sort_scale if use_prequant_lhs else None,
                rhs_q=w3_q if use_prequant_rhs else None,
                rhs_scale=w3_scale if use_prequant_rhs else None,
            )

        gated = jax.nn.silu(w1_out) * w3_out
        sorted_weights_cast = sorted_weights_in.astype(gated.dtype)
        if preweight:
            gated = gated * sorted_weights_cast[:, None]

        out_repeat_sort_out = _expert_matmul(
            gated,
            w2_in,
            group_sizes_in,
            backend=backend,
            quant_mode=quant_mode,
            rhs_q=w2_q if use_prequant_rhs else None,
            rhs_scale=w2_scale if use_prequant_rhs else None,
        )
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
        out_repeat = jnp.take(out_repeat_sort_in, inv_sort_idx, axis=0)
        out_repeat = out_repeat.reshape(tokens, topk, hidden)
        return jnp.sum(out_repeat, axis=1) if preweight else jnp.sum(out_repeat * topk_weights[..., None], axis=1)

    if remat_mode == "combine":
        out = jax.checkpoint(combine_path)(out_repeat_sort, sorted_weights, token_idx_sort, sort_idx)
    else:
        out = combine_path(out_repeat_sort, sorted_weights, token_idx_sort, sort_idx)

    shared_dim = shared_w2.shape[0]
    if shared_dim > 0:
        if shared_fused:
            shared13 = _dense_dot(
                x,
                shared_w13,
                quant_mode=quant_mode,
            )
            shared1, shared3 = jnp.split(shared13, [shared_dim], axis=-1)
        else:
            shared1 = _dense_dot(
                x,
                shared_w1,
                quant_mode=quant_mode,
            )
            shared3 = _dense_dot(
                x,
                shared_w3,
                quant_mode=quant_mode,
            )
        shared_gated = jax.nn.silu(shared1) * shared3
        shared_out = _dense_dot(
            shared_gated,
            shared_w2,
            quant_mode=quant_mode,
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


def _prequantize_expert_weights(
    w1: jax.Array,
    w2: jax.Array,
    w3: jax.Array,
    w13: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    w1_q, w1_scale = _symmetric_int8_quantize(w1, axis=(1, 2))
    w2_q, w2_scale = _symmetric_int8_quantize(w2, axis=(1, 2))
    w3_q, w3_scale = _symmetric_int8_quantize(w3, axis=(1, 2))
    w13_q, w13_scale = _symmetric_int8_quantize(w13, axis=(1, 2))
    return w1_q, w1_scale, w2_q, w2_scale, w3_q, w3_scale, w13_q, w13_scale


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
    remat_mode: RematMode,
    parallel_mode: ParallelMode,
    queue_mode: QueueMode,
    warmup: int,
    iters: int,
    run_alpha: float,
    run_noise_scale: float,
    quant_mode: QuantMode,
) -> None:
    tokens = x.shape[0]
    hidden = x.shape[1]
    mlp_dim = w1.shape[2]
    experts = w1.shape[0]

    router_logits = _sample_router_logits(
        key,
        tokens=tokens,
        experts=experts,
        distribution=distribution,
        run_alpha=run_alpha,
        run_noise_scale=run_noise_scale,
    )
    topk_idx, topk_weights = _route_topk(router_logits, topk=topk)

    run_stats = _run_stats(topk_idx[:, 0])
    load_stats = _load_stats(topk_idx, num_experts=experts)
    assignments = tokens * topk
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

    use_prequant_routed_weights = _uses_prequant_routed_weights(quant_mode)

    w1_q: jax.Array | None = None
    w1_scale: jax.Array | None = None
    w2_q: jax.Array | None = None
    w2_scale: jax.Array | None = None
    w3_q: jax.Array | None = None
    w3_scale: jax.Array | None = None
    w13_q: jax.Array | None = None
    w13_scale: jax.Array | None = None
    if use_prequant_routed_weights or parallel_mode == "ep":
        w1_q, w1_scale, w2_q, w2_scale, w3_q, w3_scale, w13_q, w13_scale = _prequantize_expert_weights(
            w1,
            w2,
            w3,
            w13,
        )

    dispatch = _prepare_dispatch(
        x,
        topk_idx,
        topk_weights,
        num_experts=experts,
        capacity_factor=capacity_factor,
        capacity_policy=capacity_policy,
        routing_pack_strategy=routing_pack_strategy,
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
        w1_q_in=None,
        w1_scale_in=None,
        w2_q_in=None,
        w2_scale_in=None,
        w3_q_in=None,
        w3_scale_in=None,
        w13_q_in=None,
        w13_scale_in=None,
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
            remat_mode=remat_mode,
            quant_mode=quant_mode,
            w1_q=w1_q_in,
            w1_scale=w1_scale_in,
            w2_q=w2_q_in,
            w2_scale=w2_scale_in,
            w3_q=w3_q_in,
            w3_scale=w3_scale_in,
            w13_q=w13_q_in,
            w13_scale=w13_scale_in,
        )

    prequant_forward_args: tuple[jax.Array, ...] = ()
    if use_prequant_routed_weights or parallel_mode == "ep":
        assert w1_q is not None and w1_scale is not None
        assert w2_q is not None and w2_scale is not None
        assert w3_q is not None and w3_scale is not None
        assert w13_q is not None and w13_scale is not None
        prequant_forward_args = (
            w1_q,
            w1_scale,
            w2_q,
            w2_scale,
            w3_q,
            w3_scale,
            w13_q,
            w13_scale,
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
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
                P("ep", None, None),
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
            w1_q_in,
            w1_scale_in,
            w2_q_in,
            w2_scale_in,
            w3_q_in,
            w3_scale_in,
            w13_q_in,
            w13_scale_in,
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
                remat_mode=remat_mode,
                quant_mode=quant_mode,
                w1_q=w1_q_in,
                w1_scale=w1_scale_in,
                w2_q=w2_q_in,
                w2_scale=w2_scale_in,
                w3_q=w3_q_in,
                w3_scale=w3_scale_in,
                w13_q=w13_q_in,
                w13_scale=w13_scale_in,
            )
            out_local = jax.lax.psum(out_local, "ep")
            if x_in.ndim == 3 and x_in.shape[0] == 1:
                return out_local[None, ...]
            return out_local

        def ep_forward_fn(
            x_in,
            w1_in,
            w2_in,
            w3_in,
            w13_in,
            sw1_in,
            sw2_in,
            sw3_in,
            sw13_in,
            w1_q_in=None,
            w1_scale_in=None,
            w2_q_in=None,
            w2_scale_in=None,
            w3_q_in=None,
            w3_scale_in=None,
            w13_q_in=None,
            w13_scale_in=None,
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
                w1_q_in,
                w1_scale_in,
                w2_q_in,
                w2_scale_in,
                w3_q_in,
                w3_scale_in,
                w13_q_in,
                w13_scale_in,
            )
            shared_dim = sw2_in.shape[0]
            if shared_dim > 0:
                if shared_fused:
                    shared13 = _dense_dot(
                        x_in,
                        sw13_in,
                        quant_mode=quant_mode,
                    )
                    shared1, shared3 = jnp.split(shared13, [shared_dim], axis=-1)
                else:
                    shared1 = _dense_dot(
                        x_in,
                        sw1_in,
                        quant_mode=quant_mode,
                    )
                    shared3 = _dense_dot(
                        x_in,
                        sw3_in,
                        quant_mode=quant_mode,
                    )
                shared_gated = jax.nn.silu(shared1) * shared3
                shared_out = _dense_dot(
                    shared_gated,
                    sw2_in,
                    quant_mode=quant_mode,
                )
                out = out + shared_out.astype(out.dtype)
            return out.astype(x_in.dtype)

        forward_fn = ep_forward_fn

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
                ) + prequant_forward_args
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
                    ) + prequant_forward_args
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
                        w1_q_in=None,
                        w1_scale_in=None,
                        w2_q_in=None,
                        w2_scale_in=None,
                        w3_q_in=None,
                        w3_scale_in=None,
                        w13_q_in=None,
                        w13_scale_in=None,
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
                            force_scatter=(capacity_policy == "pad"),
                            remat_mode=remat_mode,
                            quant_mode=quant_mode,
                            w1_q=w1_q_in,
                            w1_scale=w1_scale_in,
                            w2_q=w2_q_in,
                            w2_scale=w2_scale_in,
                            w3_q=w3_q_in,
                            w3_scale=w3_scale_in,
                            w13_q=w13_q_in,
                            w13_scale=w13_scale_in,
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
                    ) + prequant_forward_args
        else:
            # Fixed-routing train-like step: includes full backward for x and expert weights.
            def loss_fn(x_in, w1_in, w2_in, w3_in, sw1_in, sw2_in, sw3_in):
                w13_in = jnp.concatenate([w1_in, w3_in], axis=-1)
                sw13_in = jnp.concatenate([sw1_in, sw3_in], axis=-1)
                y = forward_fn(
                    x_in,
                    w1_in,
                    w2_in,
                    w3_in,
                    w13_in,
                    sw1_in,
                    sw2_in,
                    sw3_in,
                    sw13_in,
                    *prequant_forward_args,
                )
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
            routed_flops = 6.0 * tokens * hidden * mlp_dim * topk
        shared_flops = 6.0 * tokens * hidden * shared_dim
        fwd_flops = routed_flops + shared_flops
        flops = fwd_flops if bench_pass == "forward" else 3.0 * fwd_flops
        tflops = flops / dt / 1e12
        tokens_per_sec = tokens / dt

        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} backend={backend}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} quant_mode={quant_mode}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} remat_mode={remat_mode}")
        print(f"[{distribution}] mode={mode} pass={bench_pass} topk={topk} impl={impl} parallel_mode={parallel_mode}")
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
            )

        pack_dt = _time_fn(pack_fn, x, warmup=warmup, iters=iters)
        x_dispatch, w_dispatch, token_dispatch, sort_dispatch, group_dispatch = jax.jit(pack_fn)(x)
        use_scatter = impl in ("scatter", "fast") or capacity_policy == "pad"
        preweight = impl in ("preweight", "fused_w13_preweight")
        fuse_w13 = impl in ("fused_w13", "fast", "fused_w13_preweight")

        def up_fn(x_in, groups_in, w_in):
            if fuse_w13:
                w13_out = _expert_matmul(
                    x_in,
                    w13,
                    groups_in,
                    backend=backend,
                    quant_mode=quant_mode,
                    rhs_q=w13_q if use_prequant_routed_weights else None,
                    rhs_scale=w13_scale if use_prequant_routed_weights else None,
                )
                w1_out, w3_out = jnp.split(w13_out, [w1.shape[2]], axis=-1)
            else:
                w1_out = _expert_matmul(
                    x_in,
                    w1,
                    groups_in,
                    backend=backend,
                    quant_mode=quant_mode,
                    rhs_q=w1_q if use_prequant_routed_weights else None,
                    rhs_scale=w1_scale if use_prequant_routed_weights else None,
                )
                w3_out = _expert_matmul(
                    x_in,
                    w3,
                    groups_in,
                    backend=backend,
                    quant_mode=quant_mode,
                    rhs_q=w3_q if use_prequant_routed_weights else None,
                    rhs_scale=w3_scale if use_prequant_routed_weights else None,
                )
            gated = jax.nn.silu(w1_out) * w3_out
            return gated * w_in[:, None] if preweight else gated

        up_dt = _time_fn(up_fn, x_dispatch, group_dispatch, w_dispatch, warmup=warmup, iters=iters)
        gated_dispatch = jax.jit(up_fn)(x_dispatch, group_dispatch, w_dispatch)

        def down_fn(gated_in, groups_in):
            return _expert_matmul(
                gated_in,
                w2,
                groups_in,
                backend=backend,
                quant_mode=quant_mode,
                rhs_q=w2_q if use_prequant_routed_weights else None,
                rhs_scale=w2_scale if use_prequant_routed_weights else None,
            )

        down_dt = _time_fn(down_fn, gated_dispatch, group_dispatch, warmup=warmup, iters=iters)
        out_dispatch = jax.jit(down_fn)(gated_dispatch, group_dispatch)

        def combine_fn(out_in, tok_in, sort_in, w_in):
            if use_scatter:
                weighted = out_in if preweight else (out_in * w_in[:, None])
                return jnp.zeros((tokens, hidden), dtype=weighted.dtype).at[tok_in].add(weighted)
            inv_sort_idx = jnp.argsort(sort_in, axis=0)
            out_repeat = jnp.take(out_in, inv_sort_idx, axis=0).reshape(tokens, topk, hidden)
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
        "--quant-mode",
        choices=["none", "int8", "int8_routed", "int8_all", "int8_routed_prequant"],
        default="none",
    )
    parser.add_argument("--queue-mode", choices=["full", "prequeue", "both"], default="full")
    parser.add_argument("--routing-pack-strategy", choices=["argsort", "tuple_sort"], default="argsort")
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
        f"quant_mode={args.quant_mode} "
        f"queue_mode={args.queue_mode} "
        f"routing_pack_strategy={args.routing_pack_strategy} "
        f"remat_mode={args.remat_mode} "
        f"capacity_policy={args.capacity_policy} capacity_factor={args.capacity_factor} "
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

    route_keys = jax.random.split(key_route, len(distributions))
    for dist, dist_key in zip(distributions, route_keys, strict=True):
        for tk in topks:
            for impl in impls:
                _bench_one_distribution(
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
                    remat_mode=args.remat_mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    run_alpha=args.run_alpha,
                    run_noise_scale=args.run_noise_scale,
                    quant_mode=args.quant_mode,
                )


if __name__ == "__main__":
    main()
