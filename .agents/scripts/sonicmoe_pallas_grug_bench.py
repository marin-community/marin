#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Microbench Grug local MoE scatter combine vs SonicMoE gather-sum combine."""

# ruff: noqa: E731

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

try:
    from jax.experimental.pallas import triton as pltriton
except (ImportError, ModuleNotFoundError):
    pltriton = None  # type: ignore[assignment]

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.grug_moe import (
    _ragged_dot_expert_padded_batched,
    _ragged_dot_expert_padded_batched_custom_vjp,
    moe_mlp,
)
from levanter.grug.sonic_moe import (
    SonicGatherRaggedDotBlockSizes,
    SonicGatherSumBlockSizes,
    SonicMetadataBlockSizes,
    _sonic_gather_ragged_dot_backward,
    _sonic_gather_sum_backward,
    _sonic_gather_sum_pallas_mgpu_call,
    _sonic_gather_sum_pallas_triton_call,
    _sonic_gather_sum_pallas_triton_token_kblock_call,
    _sonic_gather_sum_pallas_triton_token_loop_call,
    sonic_gather_ragged_dot_pallas_triton,
    sonic_gather_ragged_dot_reference,
    sonic_gather_sum_pallas_mgpu,
    sonic_gather_sum_pallas_triton,
    sonic_gather_sum_reference,
    sonic_topk_metadata_pallas_triton,
    sonic_topk_metadata_reference,
)


def _dtype(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "fp32":
        return jnp.float32
    raise ValueError(f"Unknown dtype={name!r}")


def _block_until_ready(x):
    leaves = jax.tree.leaves(x)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def _time_call(fn: Callable[[], jax.Array], *, warmup: int, steps: int) -> tuple[float, float, jax.Array]:
    start = time.perf_counter()
    out = _block_until_ready(fn())
    compile_inclusive = time.perf_counter() - start

    for _ in range(warmup):
        out = _block_until_ready(fn())

    start = time.perf_counter()
    for _ in range(steps):
        out = _block_until_ready(fn())
    steady = (time.perf_counter() - start) / steps
    return compile_inclusive, steady, out


def _make_inputs(args: argparse.Namespace):
    key = jax.random.key(args.seed)
    k_x, k_sel, k_weights, k_w13, k_w2 = jax.random.split(key, 5)
    dtype = _dtype(args.dtype)
    tokens = args.tokens
    if tokens is None:
        tokens = args.batch_size * args.seq_len

    x = jax.random.normal(k_x, (tokens, args.hidden), dtype=jnp.float32).astype(dtype)
    if args.unique_topk:
        router_logits = jax.random.normal(k_sel, (tokens, args.experts), dtype=jnp.float32)
        _, selected_experts = jax.lax.top_k(router_logits, args.topk)
        selected_experts = selected_experts.astype(jnp.int32)
    else:
        selected_experts = jax.random.randint(
            k_sel,
            (tokens, args.topk),
            0,
            args.experts,
            dtype=jnp.int32,
        )
    combine_logits = jax.random.normal(k_weights, (tokens, args.topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(dtype)
    w_up_gate = jax.random.normal(
        k_w13,
        (args.experts, args.hidden, 2 * args.intermediate),
        dtype=jnp.float32,
    ).astype(dtype)
    w_down = jax.random.normal(
        k_w2,
        (args.experts, args.intermediate, args.hidden),
        dtype=jnp.float32,
    ).astype(dtype)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _relative_error(a: jax.Array, b: jax.Array) -> tuple[float, float, float]:
    diff = jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))
    denom = jnp.maximum(jnp.abs(b.astype(jnp.float32)), 1e-6)
    return float(jnp.max(diff)), float(jnp.mean(diff)), float(jnp.max(diff / denom))


def _max_abs_tree(a, b) -> float:
    a_leaves = jax.tree.leaves(a)
    b_leaves = jax.tree.leaves(b)
    if len(a_leaves) != len(b_leaves):
        return float("nan")
    diffs = []
    for a_leaf, b_leaf in zip(a_leaves, b_leaves, strict=True):
        if a_leaf.shape != b_leaf.shape:
            return float("nan")
        diffs.append(jnp.max(jnp.abs(a_leaf.astype(jnp.float32) - b_leaf.astype(jnp.float32))))
    if not diffs:
        return 0.0
    return float(jnp.max(jnp.stack(diffs)))


def _relative_error_tree(a, b) -> tuple[float, float, float]:
    a_leaves = jax.tree.leaves(a)
    b_leaves = jax.tree.leaves(b)
    if len(a_leaves) != len(b_leaves):
        return float("nan"), float("nan"), float("nan")
    max_abs = []
    sum_abs = []
    max_rel = []
    size = 0
    for a_leaf, b_leaf in zip(a_leaves, b_leaves, strict=True):
        if a_leaf.shape != b_leaf.shape:
            return float("nan"), float("nan"), float("nan")
        diff = jnp.abs(a_leaf.astype(jnp.float32) - b_leaf.astype(jnp.float32))
        denom = jnp.maximum(jnp.abs(b_leaf.astype(jnp.float32)), 1e-6)
        max_abs.append(jnp.max(diff))
        sum_abs.append(jnp.sum(diff))
        max_rel.append(jnp.max(diff / denom))
        size += diff.size
    if size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(jnp.max(jnp.stack(max_abs))),
        float(jnp.sum(jnp.stack(sum_abs)) / size),
        float(jnp.max(jnp.stack(max_rel))),
    )


def _run_moe_backend(
    backend: str,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    x, selected_experts, combine_weights, w_up_gate, w_down = inputs

    def fn():
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=None,
            local_implementation=backend,
        )

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _tree_l2_norm(tree) -> float:
    leaves = [
        leaf for leaf in jax.tree.leaves(tree) if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.number)
    ]
    if not leaves:
        return 0.0
    sums = [jnp.sum(leaf.astype(jnp.float32) * leaf.astype(jnp.float32)) for leaf in leaves]
    return float(jnp.sqrt(jnp.sum(jnp.stack(sums))))


def _run_moe_block_backend(
    backend: str,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    measure: str,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    x, selected_experts, combine_weights, w_up_gate, w_down = inputs

    def forward(x_arg, combine_weights_arg, w_up_gate_arg, w_down_arg):
        return moe_mlp(
            x_arg,
            selected_experts,
            combine_weights_arg,
            w_up_gate_arg,
            w_down_arg,
            mesh=None,
            local_implementation=backend,
        )

    def loss_fn(x_arg, combine_weights_arg, w_up_gate_arg, w_down_arg):
        if measure == "fwd2_bwd":
            extra = forward(
                jax.lax.optimization_barrier(x_arg),
                jax.lax.optimization_barrier(combine_weights_arg),
                jax.lax.optimization_barrier(w_up_gate_arg),
                jax.lax.optimization_barrier(w_down_arg),
            )
            extra_term = jax.lax.stop_gradient(jnp.mean(jax.lax.optimization_barrier(extra).astype(jnp.float32)))
        else:
            extra_term = jnp.array(0.0, dtype=jnp.float32)
        out = forward(x_arg, combine_weights_arg, w_up_gate_arg, w_down_arg)
        loss = jnp.mean(out.astype(jnp.float32) * out.astype(jnp.float32))
        return loss + jnp.array(1e-7, dtype=jnp.float32) * extra_term

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))

    def fn():
        return grad_fn(x, combine_weights, w_up_gate, w_down)

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    loss, grads = out
    return {
        "backend": backend,
        "measure": measure,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "loss": float(loss),
        "grad_l2_norm": _tree_l2_norm(grads),
        "output": out,
    }


def _make_moe_bwd_part_inputs(args: argparse.Namespace) -> dict[str, jax.Array]:
    x, selected_experts, combine_weights, w_up_gate, w_down = _make_inputs(args)
    token_ids_sort, dispatch_positions, group_sizes = _make_routing_metadata(selected_experts, num_experts=args.experts)
    dtype = _dtype(args.dtype)
    tokens = args.tokens if args.tokens is not None else args.batch_size * args.seq_len
    assignments = tokens * args.topk
    keys = jax.random.split(jax.random.key(args.seed + 20), 7)
    x_dispatch = jnp.take(x, token_ids_sort, axis=0)
    w13_out = jax.random.normal(keys[0], (assignments, 2 * args.intermediate), dtype=jnp.float32).astype(dtype)
    expert_hidden = jax.random.normal(keys[1], (assignments, args.intermediate), dtype=jnp.float32).astype(dtype)
    dispatch_output = jax.random.normal(keys[2], (assignments, args.hidden), dtype=jnp.float32).astype(dtype)
    dout = jax.random.normal(keys[3], (tokens, args.hidden), dtype=jnp.float32).astype(dtype)
    d_dispatch = jax.random.normal(keys[4], (assignments, args.hidden), dtype=jnp.float32).astype(dtype)
    d_expert_hidden = jax.random.normal(keys[5], (assignments, args.intermediate), dtype=jnp.float32).astype(dtype)
    d_w13_out = jax.random.normal(keys[6], (assignments, 2 * args.intermediate), dtype=jnp.float32).astype(dtype)
    return {
        "x": x,
        "selected_experts": selected_experts,
        "combine_weights": combine_weights,
        "w_up_gate": w_up_gate,
        "w_down": w_down,
        "token_ids_sort": token_ids_sort,
        "dispatch_positions": dispatch_positions,
        "group_sizes": group_sizes,
        "x_dispatch": x_dispatch,
        "w13_out": w13_out,
        "expert_hidden": expert_hidden,
        "dispatch_output": dispatch_output,
        "dout": dout,
        "d_dispatch": d_dispatch,
        "d_expert_hidden": d_expert_hidden,
        "d_w13_out": d_w13_out,
    }


def _run_moe_bwd_part_backend(
    backend: str,
    inputs: dict[str, jax.Array],
    *,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    x = inputs["x"]
    combine_weights = inputs["combine_weights"]
    w_up_gate = inputs["w_up_gate"]
    w_down = inputs["w_down"]
    token_ids_sort = inputs["token_ids_sort"]
    dispatch_positions = inputs["dispatch_positions"]
    group_sizes = inputs["group_sizes"]
    x_dispatch = inputs["x_dispatch"]
    w13_out = inputs["w13_out"]
    expert_hidden = inputs["expert_hidden"]
    dispatch_output = inputs["dispatch_output"]
    dout = inputs["dout"]
    d_dispatch = inputs["d_dispatch"]
    d_expert_hidden = inputs["d_expert_hidden"]
    d_w13_out = inputs["d_w13_out"]

    if backend == "combine_bwd":
        fn = lambda: _sonic_gather_sum_backward(dispatch_output, dispatch_positions, combine_weights, dout)
    elif backend == "w2_bwd":
        fn = lambda: jax.grad(
            lambda expert_hidden_arg, w_down_arg: jnp.sum(
                ragged_dot(expert_hidden_arg, w_down_arg, group_sizes) * d_dispatch
            ),
            argnums=(0, 1),
        )(expert_hidden, w_down)
    elif backend == "activation_bwd":
        fn = lambda: jax.grad(
            lambda w13_out_arg: jnp.sum(
                (jax.nn.silu(w13_out_arg[:, : w_down.shape[1]]) * w13_out_arg[:, w_down.shape[1] :]) * d_expert_hidden
            )
        )(w13_out)
    elif backend == "w13_bwd_sonic":
        fn = lambda: _sonic_gather_ragged_dot_backward(x, token_ids_sort, w_up_gate, group_sizes, d_w13_out)
    elif backend == "w13_bwd_materialized":
        fn = lambda: jax.grad(
            lambda x_dispatch_arg, w_up_gate_arg: jnp.sum(
                ragged_dot(x_dispatch_arg, w_up_gate_arg, group_sizes) * d_w13_out
            ),
            argnums=(0, 1),
        )(x_dispatch, w_up_gate)
    elif backend == "w13_bwd_materialized_with_xscatter":
        fn = lambda: (
            lambda dx_dispatch, dw: (
                jnp.zeros_like(x).at[token_ids_sort].add(dx_dispatch, mode="drop"),
                dw,
            )
        )(
            *jax.grad(
                lambda x_dispatch_arg, w_up_gate_arg: jnp.sum(
                    ragged_dot(x_dispatch_arg, w_up_gate_arg, group_sizes) * d_w13_out
                ),
                argnums=(0, 1),
            )(x_dispatch, w_up_gate)
        )
    else:
        raise ValueError(f"Unknown moe_bwd_parts backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _make_combine_inputs(args: argparse.Namespace) -> tuple[jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(args.seed + 1)
    k_y, k_pos, k_weights = jax.random.split(key, 3)
    dtype = _dtype(args.dtype)
    tokens = args.tokens if args.tokens is not None else args.batch_size * args.seq_len
    assignments = tokens * args.topk
    dispatch_output = jax.random.normal(k_y, (assignments, args.hidden), dtype=jnp.float32).astype(dtype)
    if args.sonic_routing_positions:
        selected_experts = jax.random.randint(k_pos, (tokens, args.topk), 0, args.experts, dtype=jnp.int32)
        _, _, s_reverse_scatter_idx, _, _ = sonic_topk_metadata_reference(
            selected_experts,
            num_experts=args.experts,
        )
        dispatch_positions = s_reverse_scatter_idx.reshape(tokens, args.topk)
    else:
        random_positions = jax.random.permutation(k_pos, jnp.arange(assignments, dtype=jnp.int32))
        dispatch_positions = random_positions.reshape(tokens, args.topk)
    if args.unit_weights:
        combine_weights = jnp.ones((tokens, args.topk), dtype=dtype)
    else:
        combine_logits = jax.random.normal(k_weights, (tokens, args.topk), dtype=jnp.float32)
        combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(dtype)
    return dispatch_output, dispatch_positions, combine_weights


def _run_combine_backend(
    backend: str,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    *,
    block_sizes: SonicGatherSumBlockSizes,
    interpret_pallas: bool,
    repeat_placement: str,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    dispatch_output, dispatch_positions, combine_weights = inputs

    def replicated_repeat_inputs() -> tuple[jax.Array, jax.Array]:
        assignments = dispatch_output.shape[0]
        if block_sizes.kernel_repeat == 1:
            return dispatch_output, jnp.zeros((1,), dtype=jnp.int32)
        repeated_output = jnp.concatenate([dispatch_output] * block_sizes.kernel_repeat, axis=0)
        repeat_offsets = jnp.arange(block_sizes.kernel_repeat, dtype=jnp.int32) * assignments
        return repeated_output, repeat_offsets

    def repeated_xla_reference() -> jax.Array:
        acc = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
        for _ in range(block_sizes.kernel_repeat):
            repeated_output = jax.lax.optimization_barrier(dispatch_output)
            repeated_positions = jax.lax.optimization_barrier(dispatch_positions)
            repeated_weights = jax.lax.optimization_barrier(combine_weights)
            out = sonic_gather_sum_reference(repeated_output, repeated_positions, repeated_weights)
            acc = jax.lax.optimization_barrier(acc + out)
        if block_sizes.kernel_repeat == 1:
            return acc
        return (acc.astype(jnp.float32) * (1.0 / block_sizes.kernel_repeat)).astype(dispatch_output.dtype)

    if backend == "xla":
        fn = repeated_xla_reference
    elif backend == "pallas_triton":
        if repeat_placement == "replicated":
            repeated_output, repeat_offsets = replicated_repeat_inputs()
            fn = lambda: _sonic_gather_sum_pallas_triton_call(
                repeated_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
                repeat_offsets=repeat_offsets,
            )
        elif repeat_placement == "outer":
            inner_block_sizes = dataclasses.replace(block_sizes, kernel_repeat=1)

            def fn():
                acc = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
                for _ in range(block_sizes.kernel_repeat):
                    out = sonic_gather_sum_pallas_triton(
                        jax.lax.optimization_barrier(dispatch_output),
                        jax.lax.optimization_barrier(dispatch_positions),
                        jax.lax.optimization_barrier(combine_weights),
                        block_sizes=inner_block_sizes,
                        interpret=interpret_pallas,
                    )
                    out = jax.lax.optimization_barrier(out)
                    acc = jax.lax.optimization_barrier(acc + out)
                if block_sizes.kernel_repeat == 1:
                    return acc
                return (acc.astype(jnp.float32) * (1.0 / block_sizes.kernel_repeat)).astype(dispatch_output.dtype)

        else:
            fn = lambda: sonic_gather_sum_pallas_triton(
                dispatch_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
            )
    elif backend == "pallas_mgpu":
        if repeat_placement == "replicated":
            repeated_output, repeat_offsets = replicated_repeat_inputs()
            fn = lambda: _sonic_gather_sum_pallas_mgpu_call(
                repeated_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
                repeat_offsets=repeat_offsets,
            )
        elif repeat_placement == "outer":
            inner_block_sizes = dataclasses.replace(block_sizes, kernel_repeat=1)

            def fn():
                acc = jnp.zeros((dispatch_positions.shape[0], dispatch_output.shape[1]), dtype=dispatch_output.dtype)
                for _ in range(block_sizes.kernel_repeat):
                    out = sonic_gather_sum_pallas_mgpu(
                        jax.lax.optimization_barrier(dispatch_output),
                        jax.lax.optimization_barrier(dispatch_positions),
                        jax.lax.optimization_barrier(combine_weights),
                        block_sizes=inner_block_sizes,
                        interpret=interpret_pallas,
                    )
                    out = jax.lax.optimization_barrier(out)
                    acc = jax.lax.optimization_barrier(acc + out)
                if block_sizes.kernel_repeat == 1:
                    return acc
                return (acc.astype(jnp.float32) * (1.0 / block_sizes.kernel_repeat)).astype(dispatch_output.dtype)

        else:
            fn = lambda: sonic_gather_sum_pallas_mgpu(
                dispatch_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
            )
    elif backend == "pallas_triton_token_loop":
        if repeat_placement == "replicated":
            repeated_output, repeat_offsets = replicated_repeat_inputs()
            fn = lambda: _sonic_gather_sum_pallas_triton_token_loop_call(
                repeated_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
                repeat_offsets=repeat_offsets,
            )
        else:
            fn = lambda: _sonic_gather_sum_pallas_triton_token_loop_call(
                dispatch_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
            )
    elif backend == "pallas_triton_token_kblock":
        if repeat_placement == "replicated":
            repeated_output, repeat_offsets = replicated_repeat_inputs()
            fn = lambda: _sonic_gather_sum_pallas_triton_token_kblock_call(
                repeated_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
                repeat_offsets=repeat_offsets,
            )
        else:
            fn = lambda: _sonic_gather_sum_pallas_triton_token_kblock_call(
                dispatch_output,
                dispatch_positions,
                combine_weights,
                block_sizes=block_sizes,
                interpret=interpret_pallas,
            )
    else:
        raise ValueError(f"Unknown combine backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "kernel_repeat": block_sizes.kernel_repeat,
        "repeat_placement": repeat_placement if backend != "xla" else "outer",
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _make_routing_metadata(
    selected_experts: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    expert_ids = selected_experts.reshape(assignments)
    sort_idx = jnp.argsort(expert_ids, axis=0)
    token_ids = jnp.arange(assignments, dtype=jnp.int32) // topk
    token_ids_sort = token_ids[sort_idx]
    sorted_positions = jnp.arange(assignments, dtype=jnp.int32)
    dispatch_positions = jnp.zeros((assignments,), dtype=jnp.int32).at[sort_idx].set(sorted_positions)
    dispatch_positions = dispatch_positions.reshape(tokens, topk)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return token_ids_sort, dispatch_positions, group_sizes


def _make_routing_metadata_with_assignment_ids(
    selected_experts: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    tokens, topk = selected_experts.shape
    assignments = tokens * topk
    expert_ids = selected_experts.reshape(assignments)
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    sort_idx = jnp.argsort(expert_ids, axis=0)
    sorted_assignment_ids = assignment_ids[sort_idx]
    token_ids_sort = sorted_assignment_ids // topk
    sorted_positions = jnp.arange(assignments, dtype=jnp.int32)
    dispatch_positions = jnp.zeros((assignments,), dtype=jnp.int32).at[sort_idx].set(sorted_positions)
    dispatch_positions = dispatch_positions.reshape(tokens, topk)
    group_sizes = jnp.bincount(expert_ids, length=num_experts).astype(jnp.int32)
    return token_ids_sort, dispatch_positions, group_sizes, sorted_assignment_ids


def _make_w13_inputs(args: argparse.Namespace) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    x, selected_experts, _, w_up_gate, _ = _make_inputs(args)
    token_ids_sort, _, group_sizes = _make_routing_metadata(selected_experts, num_experts=args.experts)
    return x, token_ids_sort, w_up_gate, group_sizes


def _run_w13_backend(
    backend: str,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    block_sizes: SonicGatherRaggedDotBlockSizes,
    interpret_pallas: bool,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    x, token_ids_sort, w_up_gate, group_sizes = inputs
    if backend == "materialized":
        fn = lambda: ragged_dot(jnp.take(x, token_ids_sort, axis=0), w_up_gate, group_sizes)
    elif backend == "gather_xla":
        fn = lambda: sonic_gather_ragged_dot_reference(x, token_ids_sort, w_up_gate, group_sizes)
    elif backend == "gather_pallas":
        fn = lambda: sonic_gather_ragged_dot_pallas_triton(
            x,
            token_ids_sort,
            w_up_gate,
            group_sizes,
            block_sizes=block_sizes,
            interpret=interpret_pallas,
        )
    else:
        raise ValueError(f"Unknown w13 backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _w2_combine_token_pallas_triton_kernel(
    expert_hidden_ref,
    selected_experts_ref,
    w_down_ref,
    dispatch_positions_ref,
    combine_weights_ref,
    output_ref,
    *,
    topk: int,
    contraction_block_size: int,
    hidden_block_size: int,
    intermediate_tiles: int,
) -> None:
    token_index = pl.program_id(0)
    hidden_block = pl.program_id(1)
    hidden_offsets = hidden_block * hidden_block_size + jnp.arange(hidden_block_size)
    hidden_mask = hidden_offsets < output_ref.shape[1]
    acc = jnp.zeros((hidden_block_size,), dtype=jnp.float32)

    for topk_index in range(topk):
        row = pltriton.load(dispatch_positions_ref.at[token_index, topk_index])
        expert = pltriton.load(selected_experts_ref.at[token_index, topk_index])
        combine_weight = pltriton.load(combine_weights_ref.at[token_index, topk_index]).astype(jnp.float32)
        slot_acc = jnp.zeros((hidden_block_size,), dtype=jnp.float32)

        def body(k_block: jax.Array, inner_acc: jax.Array, *, row=row, expert=expert) -> jax.Array:
            k_offsets = k_block * contraction_block_size + jnp.arange(contraction_block_size)
            k_mask = k_offsets < expert_hidden_ref.shape[1]
            values = pltriton.load(
                expert_hidden_ref.at[row, k_offsets],
                mask=k_mask,
                other=0.0,
            )
            weights = pltriton.load(
                w_down_ref.at[expert, k_offsets[:, None], hidden_offsets[None, :]],
                mask=k_mask[:, None] & hidden_mask[None, :],
                other=0.0,
            )
            dtype = jnp.result_type(values, weights)
            return inner_acc + pl.dot(values[None, :].astype(dtype), weights.astype(dtype))[0]

        slot_acc = jax.lax.fori_loop(0, intermediate_tiles, body, slot_acc)
        acc += slot_acc * combine_weight

    pltriton.store(output_ref.at[token_index, hidden_offsets], acc.astype(output_ref.dtype), mask=hidden_mask)


def _w2_combine_token_pallas_triton(
    expert_hidden: jax.Array,
    selected_experts: jax.Array,
    w_down: jax.Array,
    dispatch_positions: jax.Array,
    combine_weights: jax.Array,
    *,
    contraction_block_size: int,
    hidden_block_size: int,
    num_warps: int,
    interpret: bool,
) -> jax.Array:
    if pltriton is None:
        raise RuntimeError("Pallas Triton backend is not available")
    tokens = dispatch_positions.shape[0]
    hidden = w_down.shape[2]
    contraction_block_size = min(contraction_block_size, int(pl.next_power_of_2(expert_hidden.shape[1])))
    hidden_block_size = min(hidden_block_size, int(pl.next_power_of_2(hidden)))
    intermediate_tiles = math.ceil(expert_hidden.shape[1] / contraction_block_size)

    def kernel(
        expert_hidden_ref,
        selected_experts_ref,
        w_down_ref,
        dispatch_positions_ref,
        combine_weights_ref,
        output_ref,
    ) -> None:
        _w2_combine_token_pallas_triton_kernel(
            expert_hidden_ref,
            selected_experts_ref,
            w_down_ref,
            dispatch_positions_ref,
            combine_weights_ref,
            output_ref,
            topk=dispatch_positions.shape[1],
            contraction_block_size=contraction_block_size,
            hidden_block_size=hidden_block_size,
            intermediate_tiles=intermediate_tiles,
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), expert_hidden.dtype),
        grid=(tokens, pl.cdiv(hidden, hidden_block_size)),
        compiler_params=pltriton.CompilerParams(num_warps=num_warps, num_stages=4),
        interpret=interpret,
        name="w2_combine_token_pallas_triton",
    )(expert_hidden, selected_experts, w_down, dispatch_positions, combine_weights)


def _w2_combine_expert_atomic_pallas_triton_kernel(
    expert_hidden_ref,
    sorted_assignment_ids_ref,
    expert_offsets_ref,
    group_sizes_ref,
    w_down_ref,
    combine_weights_ref,
    output_init_ref,
    output_ref,
    *,
    topk: int,
    row_block_size: int,
    contraction_block_size: int,
    hidden_block_size: int,
    intermediate_tiles: int,
) -> None:
    del output_init_ref
    row_block = pl.program_id(0)
    hidden_block = pl.program_id(1)
    expert = pl.program_id(2)
    local_rows = row_block * row_block_size + jnp.arange(row_block_size)
    hidden_offsets = hidden_block * hidden_block_size + jnp.arange(hidden_block_size)
    expert_start = pltriton.load(expert_offsets_ref.at[expert])
    expert_size = pltriton.load(group_sizes_ref.at[expert])
    row_offsets = expert_start + local_rows
    row_mask = local_rows < expert_size
    hidden_mask = hidden_offsets < output_ref.shape[1]
    assignment_ids = pltriton.load(sorted_assignment_ids_ref.at[row_offsets], mask=row_mask, other=0)
    token_offsets = assignment_ids // topk
    topk_offsets = assignment_ids - token_offsets * topk
    combine_weights = pltriton.load(
        combine_weights_ref.at[token_offsets, topk_offsets],
        mask=row_mask,
        other=0.0,
    ).astype(jnp.float32)
    acc = jnp.zeros((row_block_size, hidden_block_size), dtype=jnp.float32)

    def body(k_block: jax.Array, inner_acc: jax.Array) -> jax.Array:
        k_offsets = k_block * contraction_block_size + jnp.arange(contraction_block_size)
        k_mask = k_offsets < expert_hidden_ref.shape[1]
        values = pltriton.load(
            expert_hidden_ref.at[row_offsets[:, None], k_offsets[None, :]],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weights = pltriton.load(
            w_down_ref.at[expert, k_offsets[:, None], hidden_offsets[None, :]],
            mask=k_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        dtype = jnp.result_type(values, weights)
        return inner_acc + pl.dot(values.astype(dtype), weights.astype(dtype))

    acc = jax.lax.fori_loop(0, intermediate_tiles, body, acc)
    values = (acc * combine_weights[:, None]).astype(output_ref.dtype)
    pltriton.atomic_add(
        output_ref,
        (token_offsets[:, None], hidden_offsets[None, :]),
        values,
        mask=row_mask[:, None] & hidden_mask[None, :],
    )


def _w2_combine_expert_atomic_pallas_triton(
    expert_hidden: jax.Array,
    sorted_assignment_ids: jax.Array,
    group_sizes: jax.Array,
    w_down: jax.Array,
    combine_weights: jax.Array,
    *,
    max_group_size: int,
    row_block_size: int,
    contraction_block_size: int,
    hidden_block_size: int,
    num_warps: int,
    interpret: bool,
) -> jax.Array:
    if pltriton is None:
        raise RuntimeError("Pallas Triton backend is not available")
    tokens = combine_weights.shape[0]
    hidden = w_down.shape[2]
    experts = w_down.shape[0]
    row_block_size = min(row_block_size, int(pl.next_power_of_2(expert_hidden.shape[0])))
    contraction_block_size = min(contraction_block_size, int(pl.next_power_of_2(expert_hidden.shape[1])))
    hidden_block_size = min(hidden_block_size, int(pl.next_power_of_2(hidden)))
    intermediate_tiles = math.ceil(expert_hidden.shape[1] / contraction_block_size)
    expert_offsets = jnp.cumulative_sum(group_sizes, include_initial=True)[:-1].astype(jnp.int32)
    output_init = jnp.zeros((tokens, hidden), dtype=expert_hidden.dtype)

    def kernel(
        expert_hidden_ref,
        sorted_assignment_ids_ref,
        expert_offsets_ref,
        group_sizes_ref,
        w_down_ref,
        combine_weights_ref,
        output_init_ref,
        output_ref,
    ) -> None:
        _w2_combine_expert_atomic_pallas_triton_kernel(
            expert_hidden_ref,
            sorted_assignment_ids_ref,
            expert_offsets_ref,
            group_sizes_ref,
            w_down_ref,
            combine_weights_ref,
            output_init_ref,
            output_ref,
            topk=combine_weights.shape[1],
            row_block_size=row_block_size,
            contraction_block_size=contraction_block_size,
            hidden_block_size=hidden_block_size,
            intermediate_tiles=intermediate_tiles,
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden), expert_hidden.dtype),
        grid=(pl.cdiv(max_group_size, row_block_size), pl.cdiv(hidden, hidden_block_size), experts),
        input_output_aliases={6: 0},
        compiler_params=pltriton.CompilerParams(num_warps=num_warps, num_stages=4),
        interpret=interpret,
        name="w2_combine_expert_atomic_pallas_triton",
    )(
        expert_hidden,
        sorted_assignment_ids,
        expert_offsets,
        group_sizes,
        w_down,
        combine_weights,
        output_init,
    )


def _make_w2_combine_inputs(
    args: argparse.Namespace,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    _, selected_experts, combine_weights, _, w_down = _make_inputs(args)
    _, dispatch_positions, group_sizes, sorted_assignment_ids = _make_routing_metadata_with_assignment_ids(
        selected_experts,
        num_experts=args.experts,
    )
    dtype = _dtype(args.dtype)
    tokens = args.tokens if args.tokens is not None else args.batch_size * args.seq_len
    assignments = tokens * args.topk
    key = jax.random.key(args.seed + 3)
    expert_hidden = jax.random.normal(
        key,
        (assignments, args.intermediate),
        dtype=jnp.float32,
    ).astype(dtype)
    max_group_size = int(_block_until_ready(jnp.max(group_sizes)))
    return (
        expert_hidden,
        selected_experts,
        w_down,
        dispatch_positions,
        combine_weights,
        group_sizes,
        sorted_assignment_ids,
        max_group_size,
    )


def _run_w2_combine_backend(
    backend: str,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int],
    *,
    gather_block_sizes: SonicGatherSumBlockSizes,
    row_block_size: int,
    contraction_block_size: int,
    interpret_pallas: bool,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    (
        expert_hidden,
        selected_experts,
        w_down,
        dispatch_positions,
        combine_weights,
        group_sizes,
        sorted_assignment_ids,
        max_group_size,
    ) = inputs

    if backend == "composed_xla":
        fn = lambda: sonic_gather_sum_reference(
            ragged_dot(expert_hidden, w_down, group_sizes),
            dispatch_positions,
            combine_weights,
        )
    elif backend == "composed_pallas_combine":
        fn = lambda: sonic_gather_sum_pallas_triton(
            ragged_dot(expert_hidden, w_down, group_sizes),
            dispatch_positions,
            combine_weights,
            block_sizes=gather_block_sizes,
            interpret=interpret_pallas,
        )
    elif backend == "fused_token_pallas":
        fn = lambda: _w2_combine_token_pallas_triton(
            expert_hidden,
            selected_experts,
            w_down,
            dispatch_positions,
            combine_weights,
            contraction_block_size=contraction_block_size,
            hidden_block_size=gather_block_sizes.hidden_block_size,
            num_warps=gather_block_sizes.num_warps,
            interpret=interpret_pallas,
        )
    elif backend == "fused_expert_atomic_pallas":
        fn = lambda: _w2_combine_expert_atomic_pallas_triton(
            expert_hidden,
            sorted_assignment_ids,
            group_sizes,
            w_down,
            combine_weights,
            max_group_size=max_group_size,
            row_block_size=row_block_size,
            contraction_block_size=contraction_block_size,
            hidden_block_size=gather_block_sizes.hidden_block_size,
            num_warps=gather_block_sizes.num_warps,
            interpret=interpret_pallas,
        )
    else:
        raise ValueError(f"Unknown w2combine backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _run_routing_backend(
    backend: str,
    x: jax.Array,
    selected_experts: jax.Array,
    *,
    num_experts: int,
    block_sizes: SonicMetadataBlockSizes,
    interpret_pallas: bool,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    if backend == "metadata":
        fn = lambda: _make_routing_metadata(selected_experts, num_experts=num_experts)
    elif backend == "metadata_xsort":
        fn = lambda: (
            lambda token_ids_sort, dispatch_positions, group_sizes: (
                jnp.take(x, token_ids_sort, axis=0),
                token_ids_sort,
                dispatch_positions,
                group_sizes,
            )
        )(*_make_routing_metadata(selected_experts, num_experts=num_experts))
    elif backend == "sonic_metadata_xla":
        fn = lambda: sonic_topk_metadata_reference(selected_experts, num_experts=num_experts)
    elif backend == "sonic_metadata_pallas":
        fn = lambda: sonic_topk_metadata_pallas_triton(
            selected_experts,
            num_experts=num_experts,
            block_sizes=block_sizes,
            interpret=interpret_pallas,
        )
    else:
        raise ValueError(f"Unknown routing backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _expert_padded_indices(
    group_sizes: jax.Array,
    *,
    total_rows: int,
    local_expert_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    local_experts = group_sizes.shape[0]
    row_ids = jnp.arange(total_rows, dtype=jnp.int32)
    segment_ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
    total_valid = jnp.sum(group_sizes, dtype=jnp.int32)
    valid = row_ids < total_valid
    expert_ids = jnp.searchsorted(segment_ends, row_ids, side="right").astype(jnp.int32)
    segment_starts = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), segment_ends[:-1].astype(jnp.int32)],
        axis=0,
    )
    expert_ids_clipped = jnp.clip(expert_ids, 0, max(0, local_experts - 1))
    within_expert = row_ids - jnp.take(segment_starts, expert_ids_clipped, axis=0)
    valid = valid & (expert_ids < local_experts) & (within_expert < local_expert_capacity)

    flat_capacity = local_experts * local_expert_capacity
    flat_indices = expert_ids * local_expert_capacity + within_expert
    scatter_indices = jnp.where(valid, flat_indices, flat_capacity)
    gather_indices = jnp.where(valid, flat_indices, 0)
    return valid, scatter_indices, gather_indices


def _expert_padded_pack(lhs: jax.Array, scatter_indices: jax.Array, *, flat_capacity: int) -> jax.Array:
    packed_lhs = jnp.zeros((flat_capacity, lhs.shape[-1]), dtype=lhs.dtype)
    return packed_lhs.at[scatter_indices].add(lhs, mode="drop")


def _expert_padded_pack_set(lhs: jax.Array, scatter_indices: jax.Array, *, flat_capacity: int) -> jax.Array:
    packed_lhs = jnp.zeros((flat_capacity, lhs.shape[-1]), dtype=lhs.dtype)
    return packed_lhs.at[scatter_indices].set(lhs, mode="drop")


def _expert_padded_bmm(packed_lhs: jax.Array, rhs: jax.Array, *, local_expert_capacity: int) -> jax.Array:
    local_experts = rhs.shape[0]
    packed_lhs = packed_lhs.reshape(local_experts, local_expert_capacity, packed_lhs.shape[-1])
    return jax.lax.dot_general(
        packed_lhs,
        rhs,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
    )


def _expert_padded_unpack(packed_out: jax.Array, gather_indices: jax.Array, valid: jax.Array) -> jax.Array:
    packed_out_flat = packed_out.reshape(-1, packed_out.shape[-1])
    out = jnp.take(packed_out_flat, gather_indices, axis=0)
    return jnp.where(valid[:, None], out, 0)


def _expert_padded_unpack_bwd_scatter(
    dout: jax.Array,
    gather_indices: jax.Array,
    valid: jax.Array,
    *,
    flat_capacity: int,
) -> jax.Array:
    scatter_values = jnp.where(valid[:, None], dout, 0)
    packed_grad = jnp.zeros((flat_capacity, dout.shape[-1]), dtype=dout.dtype)
    return packed_grad.at[gather_indices].add(scatter_values, mode="drop")


def _expert_padded_unpack_bwd_set(
    dout: jax.Array,
    gather_indices: jax.Array,
    valid: jax.Array,
    *,
    flat_capacity: int,
) -> jax.Array:
    scatter_indices = jnp.where(valid, gather_indices, flat_capacity)
    packed_grad = jnp.zeros((flat_capacity, dout.shape[-1]), dtype=dout.dtype)
    return packed_grad.at[scatter_indices].set(dout, mode="drop")


def _expert_padded_unpack_bwd_pallas_triton_kernel(
    dout_ref,
    gather_indices_ref,
    valid_ref,
    packed_grad_ref,
    *,
    row_block_size: int,
    hidden_block_size: int,
    flat_capacity: int,
):
    row_offsets = pl.program_id(0) * row_block_size + jnp.arange(row_block_size)
    hidden_offsets = pl.program_id(1) * hidden_block_size + jnp.arange(hidden_block_size)
    row_mask = row_offsets < dout_ref.shape[0]
    hidden_mask = hidden_offsets < dout_ref.shape[1]
    flat_indices = pltriton.load(gather_indices_ref.at[row_offsets], mask=row_mask, other=flat_capacity)
    valid = pltriton.load(valid_ref.at[row_offsets], mask=row_mask, other=False)
    values = pltriton.load(
        dout_ref.at[row_offsets[:, None], hidden_offsets[None, :]],
        mask=row_mask[:, None] & hidden_mask[None, :],
        other=0.0,
    )
    pltriton.store(
        packed_grad_ref.at[flat_indices[:, None], hidden_offsets[None, :]],
        values,
        mask=(row_mask & valid & (flat_indices < flat_capacity))[:, None] & hidden_mask[None, :],
    )


def _expert_padded_unpack_bwd_pallas_triton(
    dout: jax.Array,
    gather_indices: jax.Array,
    valid: jax.Array,
    *,
    flat_capacity: int,
    row_block_size: int,
    hidden_block_size: int,
    interpret: bool,
) -> jax.Array:
    if pltriton is None:
        raise RuntimeError("Pallas Triton backend is not available")
    packed_grad = pl.pallas_call(
        lambda dout_ref, gather_indices_ref, valid_ref, packed_grad_ref: (
            _expert_padded_unpack_bwd_pallas_triton_kernel(
                dout_ref,
                gather_indices_ref,
                valid_ref,
                packed_grad_ref,
                row_block_size=row_block_size,
                hidden_block_size=hidden_block_size,
                flat_capacity=flat_capacity,
            )
        ),
        out_shape=jax.ShapeDtypeStruct((flat_capacity, dout.shape[-1]), dout.dtype),
        grid=(pl.cdiv(dout.shape[0], row_block_size), pl.cdiv(dout.shape[1], hidden_block_size)),
        compiler_params=pltriton.CompilerParams(num_warps=4, num_stages=4),
        interpret=interpret,
        name="expert_padded_unpack_bwd_pallas_triton",
    )(dout, gather_indices, valid)
    used = jnp.zeros((flat_capacity,), dtype=bool).at[gather_indices].set(valid, mode="drop")
    return jnp.where(used[:, None], packed_grad, 0)


def _expert_padded_dot_autodiff(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    valid: jax.Array,
    scatter_indices: jax.Array,
    gather_indices: jax.Array,
    flat_capacity: int,
    local_expert_capacity: int,
) -> jax.Array:
    packed_lhs = _expert_padded_pack(lhs, scatter_indices, flat_capacity=flat_capacity)
    packed_out = _expert_padded_bmm(packed_lhs, rhs, local_expert_capacity=local_expert_capacity)
    return _expert_padded_unpack(packed_out, gather_indices, valid)


def _valid_indices_are_unique(valid: jax.Array, gather_indices: jax.Array, *, flat_capacity: int) -> jax.Array:
    scatter_indices = jnp.where(valid, gather_indices, flat_capacity)
    counts = jnp.zeros((flat_capacity + 1,), dtype=jnp.int32).at[scatter_indices].add(1)
    return jnp.max(counts[:flat_capacity]) <= 1


def _make_padded_inputs(args: argparse.Namespace) -> dict[str, jax.Array | int]:
    x, selected_experts, _, w_up_gate, _ = _make_inputs(args)
    _, _, group_sizes = _make_routing_metadata(selected_experts, num_experts=args.experts)
    assignments = x.shape[0] * args.topk
    local_expert_capacity = args.local_expert_capacity
    if local_expert_capacity is None:
        local_expert_capacity = math.ceil(assignments / args.experts * 1.5)
    flat_capacity = args.experts * local_expert_capacity
    valid, scatter_indices, gather_indices = _expert_padded_indices(
        group_sizes,
        total_rows=assignments,
        local_expert_capacity=local_expert_capacity,
    )
    dtype = _dtype(args.dtype)
    key = jax.random.key(args.seed + 2)
    k_packed, k_dout = jax.random.split(key)
    packed_out = jax.random.normal(
        k_packed,
        (args.experts, local_expert_capacity, 2 * args.intermediate),
        dtype=jnp.float32,
    ).astype(dtype)
    dout = jax.random.normal(k_dout, (assignments, 2 * args.intermediate), dtype=jnp.float32).astype(dtype)
    lhs = jnp.repeat(x, args.topk, axis=0)
    packed_lhs = _block_until_ready(_expert_padded_pack(lhs, scatter_indices, flat_capacity=flat_capacity))
    unique_indices = bool(
        _block_until_ready(_valid_indices_are_unique(valid, gather_indices, flat_capacity=flat_capacity))
    )
    return {
        "lhs": lhs,
        "rhs": w_up_gate,
        "group_sizes": group_sizes,
        "packed_lhs": packed_lhs,
        "packed_out": packed_out,
        "dout": dout,
        "valid": valid,
        "scatter_indices": scatter_indices,
        "gather_indices": gather_indices,
        "flat_capacity": flat_capacity,
        "local_expert_capacity": local_expert_capacity,
        "unique_indices": int(unique_indices),
    }


def _run_padded_backend(
    backend: str,
    inputs: dict[str, jax.Array | int],
    *,
    row_block_size: int,
    hidden_block_size: int,
    interpret_pallas: bool,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    lhs = inputs["lhs"]
    rhs = inputs["rhs"]
    group_sizes = inputs["group_sizes"]
    packed_lhs = inputs["packed_lhs"]
    packed_out = inputs["packed_out"]
    dout = inputs["dout"]
    valid = inputs["valid"]
    scatter_indices = inputs["scatter_indices"]
    gather_indices = inputs["gather_indices"]
    flat_capacity = int(inputs["flat_capacity"])
    local_expert_capacity = int(inputs["local_expert_capacity"])
    assert isinstance(lhs, jax.Array)
    assert isinstance(rhs, jax.Array)
    assert isinstance(group_sizes, jax.Array)
    assert isinstance(packed_lhs, jax.Array)
    assert isinstance(packed_out, jax.Array)
    assert isinstance(dout, jax.Array)
    assert isinstance(valid, jax.Array)
    assert isinstance(scatter_indices, jax.Array)
    assert isinstance(gather_indices, jax.Array)

    if backend == "pack":
        fn = lambda: _expert_padded_pack(lhs, scatter_indices, flat_capacity=flat_capacity)
    elif backend == "pack_set":
        fn = lambda: _expert_padded_pack_set(lhs, scatter_indices, flat_capacity=flat_capacity)
    elif backend == "bmm":
        fn = lambda: _expert_padded_bmm(
            packed_lhs,
            rhs,
            local_expert_capacity=local_expert_capacity,
        )
    elif backend == "unpack":
        fn = lambda: _expert_padded_unpack(packed_out, gather_indices, valid)
    elif backend == "unpack_bwd":
        fn = lambda: _expert_padded_unpack_bwd_scatter(
            dout,
            gather_indices,
            valid,
            flat_capacity=flat_capacity,
        )
    elif backend == "unpack_bwd_set":
        fn = lambda: _expert_padded_unpack_bwd_set(
            dout,
            gather_indices,
            valid,
            flat_capacity=flat_capacity,
        )
    elif backend == "unpack_bwd_pallas":
        fn = lambda: _expert_padded_unpack_bwd_pallas_triton(
            dout,
            gather_indices,
            valid,
            flat_capacity=flat_capacity,
            row_block_size=row_block_size,
            hidden_block_size=hidden_block_size,
            interpret=interpret_pallas,
        )
    elif backend == "padded_dot_bwd_autodiff":
        fn = lambda: jax.grad(
            lambda lhs, rhs: jnp.sum(
                _expert_padded_dot_autodiff(
                    lhs,
                    rhs,
                    group_sizes,
                    valid=valid,
                    scatter_indices=scatter_indices,
                    gather_indices=gather_indices,
                    flat_capacity=flat_capacity,
                    local_expert_capacity=local_expert_capacity,
                )
                * dout
            ),
            argnums=(0, 1),
        )(lhs, rhs)
    elif backend == "padded_dot_bwd_custom":
        fn = lambda: jax.grad(
            lambda lhs, rhs: jnp.sum(
                _ragged_dot_expert_padded_batched_custom_vjp(
                    lhs,
                    rhs,
                    group_sizes,
                    local_expert_capacity=local_expert_capacity,
                    rhs_contract_axis=1,
                )
                * dout
            ),
            argnums=(0, 1),
        )(lhs, rhs)
    elif backend == "padded_dot_bwd_production":
        fn = lambda: jax.grad(
            lambda lhs, rhs: jnp.sum(
                _ragged_dot_expert_padded_batched(
                    lhs,
                    rhs,
                    group_sizes,
                    local_expert_capacity=local_expert_capacity,
                    rhs_contract_axis=1,
                )
                * dout
            ),
            argnums=(0, 1),
        )(lhs, rhs)
    else:
        raise ValueError(f"Unknown padded backend={backend!r}")

    jitted = jax.jit(fn)
    compile_inclusive, steady, out = _time_call(jitted, warmup=warmup, steps=steps)
    return {
        "backend": backend,
        "compile_inclusive_s": compile_inclusive,
        "steady_s": steady,
        "output": out,
    }


def _parse_backends(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("moe", "moe_block", "moe_bwd_parts", "combine", "w13", "w2combine", "routing", "padded"),
        default="moe",
    )
    parser.add_argument("--backends", default="scatter,sonic_xla,sonic_pallas")
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--block-measure",
        choices=("fwd_bwd", "fwd2_bwd"),
        default="fwd2_bwd",
        help="For --mode moe_block, time forward+backward or an extra forward plus forward+backward.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unique-topk",
        action="store_true",
        help="Generate selected experts using top-k over router logits, so each token has unique experts.",
    )
    parser.add_argument("--token-block", type=int, default=16)
    parser.add_argument("--hidden-block", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--row-block", type=int, default=128)
    parser.add_argument("--contraction-block", type=int, default=32)
    parser.add_argument("--metadata-tile", type=int, default=1024)
    parser.add_argument("--local-expert-capacity", type=int, default=None)
    parser.add_argument("--kernel-repeat", type=int, default=1)
    parser.add_argument(
        "--repeat-placement",
        choices=("inner", "outer", "replicated"),
        default="inner",
        help=(
            "Place repeat count inside the Pallas kernel, around repeated Pallas calls, "
            "or inside one kernel with replicated inputs."
        ),
    )
    parser.add_argument(
        "--unit-weights",
        action="store_true",
        help="Use all-one combine weights; matches SonicMoE's unweighted token-broadcast backward path.",
    )
    parser.add_argument(
        "--sonic-routing-positions",
        action="store_true",
        help="Use SonicMoE-style reverse-scatter positions from random top-k experts instead of a flat permutation.",
    )
    parser.add_argument("--interpret-pallas", action="store_true")
    args = parser.parse_args(argv)

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "mode": args.mode,
                "shape": {
                    "tokens": args.tokens if args.tokens is not None else args.batch_size * args.seq_len,
                    "hidden": args.hidden,
                    "intermediate": args.intermediate,
                    "experts": args.experts,
                    "topk": args.topk,
                    "dtype": args.dtype,
                    "kernel_repeat": args.kernel_repeat,
                    "repeat_placement": args.repeat_placement,
                    "block_measure": args.block_measure,
                    "unique_topk": args.unique_topk,
                },
            },
            sort_keys=True,
        )
    )

    results: list[dict[str, object]] = []
    if args.mode == "moe":
        inputs = _make_inputs(args)
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_moe_backend(backend, inputs, warmup=args.warmup, steps=args.steps)
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
                result["mean_abs_vs_first"] = 0.0
                result["max_rel_vs_first"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error(out, reference_out)
                result["max_abs_vs_first"] = max_abs
                result["mean_abs_vs_first"] = mean_abs
                result["max_rel_vs_first"] = max_rel
            results.append(result)
    elif args.mode == "moe_block":
        inputs = _make_inputs(args)
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_moe_block_backend(
                backend,
                inputs,
                measure=args.block_measure,
                warmup=args.warmup,
                steps=args.steps,
            )
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
                result["mean_abs_vs_first"] = 0.0
                result["max_rel_vs_first"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error_tree(out, reference_out)
                result["max_abs_vs_first"] = max_abs
                result["mean_abs_vs_first"] = mean_abs
                result["max_rel_vs_first"] = max_rel
            results.append(result)
    elif args.mode == "moe_bwd_parts":
        inputs = _make_moe_bwd_part_inputs(args)
        references: dict[str, object] = {}
        reference_keys = {
            "combine_bwd": "combine_bwd",
            "w2_bwd": "w2_bwd",
            "activation_bwd": "activation_bwd",
            "w13_bwd_sonic": "w13_bwd",
            "w13_bwd_materialized": "w13_bwd_materialized",
            "w13_bwd_materialized_with_xscatter": "w13_bwd",
        }
        for backend in _parse_backends(args.backends):
            result = _run_moe_bwd_part_backend(backend, inputs, warmup=args.warmup, steps=args.steps)
            out = result.pop("output")
            reference_key = reference_keys.get(backend, backend)
            if reference_key not in references:
                references[reference_key] = out
                result["max_abs_vs_reference"] = 0.0
                result["mean_abs_vs_reference"] = 0.0
                result["max_rel_vs_reference"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error_tree(out, references[reference_key])
                result["max_abs_vs_reference"] = max_abs
                result["mean_abs_vs_reference"] = mean_abs
                result["max_rel_vs_reference"] = max_rel
            results.append(result)
    elif args.mode == "combine":
        inputs = _make_combine_inputs(args)
        block_sizes = SonicGatherSumBlockSizes(
            token_block_size=args.token_block,
            hidden_block_size=args.hidden_block,
            kernel_repeat=args.kernel_repeat,
            num_warps=args.num_warps,
        )
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_combine_backend(
                backend,
                inputs,
                block_sizes=block_sizes,
                interpret_pallas=args.interpret_pallas,
                repeat_placement=args.repeat_placement,
                warmup=args.warmup,
                steps=args.steps,
            )
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
                result["mean_abs_vs_first"] = 0.0
                result["max_rel_vs_first"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error(out, reference_out)
                result["max_abs_vs_first"] = max_abs
                result["mean_abs_vs_first"] = mean_abs
                result["max_rel_vs_first"] = max_rel
            results.append(result)
    elif args.mode == "w13":
        inputs = _make_w13_inputs(args)
        block_sizes = SonicGatherRaggedDotBlockSizes(
            row_block_size=args.row_block,
            contraction_block_size=args.contraction_block,
        )
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_w13_backend(
                backend,
                inputs,
                block_sizes=block_sizes,
                interpret_pallas=args.interpret_pallas,
                warmup=args.warmup,
                steps=args.steps,
            )
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
                result["mean_abs_vs_first"] = 0.0
                result["max_rel_vs_first"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error(out, reference_out)
                result["max_abs_vs_first"] = max_abs
                result["mean_abs_vs_first"] = mean_abs
                result["max_rel_vs_first"] = max_rel
            results.append(result)
    elif args.mode == "w2combine":
        inputs = _make_w2_combine_inputs(args)
        block_sizes = SonicGatherSumBlockSizes(
            token_block_size=args.token_block,
            hidden_block_size=args.hidden_block,
            kernel_repeat=args.kernel_repeat,
            num_warps=args.num_warps,
        )
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_w2_combine_backend(
                backend,
                inputs,
                gather_block_sizes=block_sizes,
                row_block_size=args.row_block,
                contraction_block_size=args.contraction_block,
                interpret_pallas=args.interpret_pallas,
                warmup=args.warmup,
                steps=args.steps,
            )
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
                result["mean_abs_vs_first"] = 0.0
                result["max_rel_vs_first"] = 0.0
            else:
                max_abs, mean_abs, max_rel = _relative_error(out, reference_out)
                result["max_abs_vs_first"] = max_abs
                result["mean_abs_vs_first"] = mean_abs
                result["max_rel_vs_first"] = max_rel
            results.append(result)
    else:
        if args.mode == "padded":
            inputs = _make_padded_inputs(args)
            references: dict[str, jax.Array] = {}
            reference_keys = {
                "pack": "pack",
                "pack_set": "pack",
                "unpack_bwd": "unpack_bwd",
                "unpack_bwd_set": "unpack_bwd",
                "unpack_bwd_pallas": "unpack_bwd",
                "padded_dot_bwd_autodiff": "padded_dot_bwd",
                "padded_dot_bwd_custom": "padded_dot_bwd",
                "padded_dot_bwd_production": "padded_dot_bwd",
            }
            for backend in _parse_backends(args.backends):
                result = _run_padded_backend(
                    backend,
                    inputs,
                    row_block_size=args.row_block,
                    hidden_block_size=args.hidden_block,
                    interpret_pallas=args.interpret_pallas,
                    warmup=args.warmup,
                    steps=args.steps,
                )
                out = result.pop("output")
                reference_key = reference_keys.get(backend)
                if reference_key is None:
                    result["max_abs_vs_reference"] = 0.0
                    result["mean_abs_vs_reference"] = 0.0
                    result["max_rel_vs_reference"] = 0.0
                elif reference_key not in references:
                    references[reference_key] = out
                    result["max_abs_vs_reference"] = 0.0
                    result["mean_abs_vs_reference"] = 0.0
                    result["max_rel_vs_reference"] = 0.0
                else:
                    max_abs, mean_abs, max_rel = _relative_error_tree(out, references[reference_key])
                    result["max_abs_vs_reference"] = max_abs
                    result["mean_abs_vs_reference"] = mean_abs
                    result["max_rel_vs_reference"] = max_rel
                result["local_expert_capacity"] = inputs["local_expert_capacity"]
                result["flat_capacity"] = inputs["flat_capacity"]
                result["unique_valid_indices"] = bool(inputs["unique_indices"])
                results.append(result)
            for result in results:
                print(json.dumps(result, sort_keys=True))
            return

        x, selected_experts, _, _, _ = _make_inputs(args)
        block_sizes = SonicMetadataBlockSizes(assignments_per_tile=args.metadata_tile)
        reference_out = None
        for backend in _parse_backends(args.backends):
            result = _run_routing_backend(
                backend,
                x,
                selected_experts,
                num_experts=args.experts,
                block_sizes=block_sizes,
                interpret_pallas=args.interpret_pallas,
                warmup=args.warmup,
                steps=args.steps,
            )
            out = result.pop("output")
            if reference_out is None:
                reference_out = out
                result["max_abs_vs_first"] = 0.0
            else:
                result["max_abs_vs_first"] = _max_abs_tree(out, reference_out)
            results.append(result)

    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
