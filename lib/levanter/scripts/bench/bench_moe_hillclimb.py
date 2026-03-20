# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused MoE hillclimb harness for the functional Grug MoE kernels.

This harness compares the current `levanter.grug.grug_moe.moe_mlp` path against
the pre-optimization EP ring implementation that globally sorted all gathered
assignments before filtering to local experts. It keeps routing fixed so the
timing reflects kernel/collective work rather than router noise.
"""

import argparse
import math
import os
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np

import jax
import jax.distributed
import jax.numpy as jnp
from haliax.nn.ragged_dot import ragged_dot
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh

from levanter.grug import grug_moe as grug_moe_lib
from levanter.kernels.deepep import (
    IntranodeConfig,
    deepep_combine_intranode,
    deepep_dispatch_intranode,
    deepep_get_dispatch_layout,
    set_intranode_config_overrides,
)
from levanter.kernels.deepep import transport_ffi as deepep_transport_ffi
from levanter.utils.activation import ActivationFunctionEnum

Distribution = Literal["random", "runs"]
Kernel = Literal[
    "legacy",
    "current",
    "shared_mlp_only_probe",
    "deepep_transport_capped_prewarmed_shared_detached_probe",
    "deepep_transport_capped_prewarmed_routed_detached_probe",
    "deepep_transport_capped_prewarmed_shared_dx_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw_psum_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw_psum_splitvjp_probe",
    "deepep_transport_capped_prewarmed_shared_dw13_psum_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw2_psum_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw13_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw2_only_probe",
    "deepep_transport_capped_prewarmed_shared_dw_only_probe",
    "deepep_transport_capped_prewarmed_split_loss_probe",
    "deepep_transport_capped_prewarmed_separate_bwd_probe",
    "cumsum",
    "packed_return",
    "stream_ring",
    "ragged_a2a",
    "deepep_layout_ragged_a2a",
    "deepep_transport_identity",
    "deepep_transport_assignments_identity",
    "deepep_transport_first_ragged_dot_probe",
    "deepep_transport_gate_probe",
    "deepep_transport_second_ragged_dot_probe",
    "deepep_transport_w13_only_probe",
    "deepep_transport_w2_only_probe",
    "deepep_transport_local_compute_only_probe",
    "deepep_transport_collapse_only_probe",
    "deepep_transport_combine_only_probe",
    "deepep_transport_w13_only_bwd_probe",
    "deepep_transport_w2_only_bwd_probe",
    "deepep_transport_local_compute_bwd_probe",
    "deepep_transport_combine_bwd_cached_dispatch_probe",
    "deepep_transport_dispatch_bwd_combine_probe",
    "deepep_transport",
    "deepep_transport_prewarmed",
    "deepep_transport_capped",
    "deepep_transport_capped_prewarmed",
    "deepep_transport_staged",
    "prefix_counts",
    "vector_prefix",
    "onehot_counts",
    "padded_take",
    "segment_sum",
    "sorted_segment_sum",
    "prefix_segment_sum",
    "vector_sorted_segment_sum",
    "owner_local_scatter",
    "lax_scatter",
    "take_segment_bwd",
    "take_sorted_segment_bwd",
    "owner_local_take",
    "weight_cast",
    "narrow_meta",
]

_USE_SHARED_MLP_EXPLICIT_BWD = False
_USE_SHARED_MLP_FUSED_DW_PSUM_BWD = False
_USE_SHARED_MLP_GRADX_FIRST_BWD = False
_USE_SHARED_MLP_FAST_ACCUM = False
_USE_COMBINE_FAST_ACCUM = False
BenchPass = Literal["forward", "forward_backward"]
CollapseImpl = Literal["segment_sum", "sorted_segment_sum", "scatter_add", "lax_scatter"]


def _print0(*args, **kwargs) -> None:
    if jax.process_index() == 0:
        print(*args, **kwargs)


def _deepep_config_payload(config: IntranodeConfig | None) -> str:
    if config is None:
        return "default"
    return (
        f"num_sms={config.num_sms} "
        f"num_max_send_tokens={config.num_max_send_tokens} "
        f"num_max_recv_tokens={config.num_max_recv_tokens}"
    )


def _deepep_dispatch_config_from_args(args: argparse.Namespace, ep_size: int) -> IntranodeConfig | None:
    if (
        args.deepep_dispatch_num_sms is None
        and args.deepep_dispatch_num_max_send_tokens is None
        and args.deepep_dispatch_num_max_recv_tokens is None
    ):
        return None

    config = deepep_transport_ffi._DEFAULT_DISPATCH_CONFIGS[ep_size]
    return IntranodeConfig(
        num_sms=args.deepep_dispatch_num_sms or config.num_sms,
        num_max_send_tokens=args.deepep_dispatch_num_max_send_tokens or config.num_max_send_tokens,
        num_max_recv_tokens=args.deepep_dispatch_num_max_recv_tokens or config.num_max_recv_tokens,
    )


def _deepep_combine_config_from_args(args: argparse.Namespace, ep_size: int) -> IntranodeConfig | None:
    if (
        args.deepep_combine_num_sms is None
        and args.deepep_combine_num_max_send_tokens is None
        and args.deepep_combine_num_max_recv_tokens is None
        and args.deepep_dispatch_num_sms is None
    ):
        return None

    config = deepep_transport_ffi._DEFAULT_COMBINE_CONFIGS[ep_size]
    if args.deepep_combine_num_sms is None and args.deepep_dispatch_num_sms is not None:
        combine_num_sms = args.deepep_dispatch_num_sms
    else:
        combine_num_sms = args.deepep_combine_num_sms or config.num_sms
    return IntranodeConfig(
        num_sms=combine_num_sms,
        num_max_send_tokens=args.deepep_combine_num_max_send_tokens or config.num_max_send_tokens,
        num_max_recv_tokens=args.deepep_combine_num_max_recv_tokens or config.num_max_recv_tokens,
    )


def _round_up_capacity(value: int, *, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _moe_intermediate_dim_from_w13_out(w13_out: jax.Array, moe_w2_local: jax.Array) -> int:
    out_width = int(w13_out.shape[-1])
    if moe_w2_local.shape[1] * 2 == out_width:
        return int(moe_w2_local.shape[1])
    if moe_w2_local.ndim > 2 and moe_w2_local.shape[2] * 2 == out_width:
        return int(moe_w2_local.shape[2])
    raise ValueError(
        f"Could not infer MoE intermediate dim from w13_out.shape={w13_out.shape} and moe_w2_local.shape={moe_w2_local.shape}"
    )


def _time_fn(fn: Callable, *args, warmup: int = 2, iters: int = 5) -> float:
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(compiled(*args))
    return (time.perf_counter() - start) / iters


def _sort_activations(
    inputs: jax.Array,
    sort_indices: jax.Array,
) -> jax.Array:
    if inputs.shape[0] != sort_indices.shape[0]:
        raise ValueError(f"Expected matching leading dims, got {inputs.shape[0]} and {sort_indices.shape[0]}")
    return _sort_activations_custom(inputs, sort_indices)


@jax.custom_vjp
def _sort_activations_custom(inputs: jax.Array, sort_indices: jax.Array) -> jax.Array:
    return inputs[sort_indices, ...]


def _sort_activations_custom_fwd(inputs: jax.Array, sort_indices: jax.Array) -> tuple[jax.Array, jax.Array]:
    return _sort_activations_custom(inputs, sort_indices), sort_indices


def _sort_activations_custom_bwd(residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
    sort_indices = residuals
    return _sort_activations_custom(grads, jnp.argsort(sort_indices)), None


_sort_activations_custom.defvjp(_sort_activations_custom_fwd, _sort_activations_custom_bwd)


@jax.custom_vjp
def _take_tokens_segment_sum(inputs: jax.Array, token_indices: jax.Array) -> jax.Array:
    return inputs[token_indices, ...]


def _take_tokens_segment_sum_fwd(
    inputs: jax.Array, token_indices: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, int]]:
    return _take_tokens_segment_sum(inputs, token_indices), (token_indices, inputs.shape[0])


def _take_tokens_segment_sum_bwd(residuals: tuple[jax.Array, int], grads: jax.Array) -> tuple[jax.Array, None]:
    token_indices, tokens = residuals
    grad_inputs = jax.ops.segment_sum(grads, token_indices, num_segments=tokens, indices_are_sorted=False)
    return grad_inputs, None


_take_tokens_segment_sum.defvjp(_take_tokens_segment_sum_fwd, _take_tokens_segment_sum_bwd)


@jax.custom_vjp
def _take_tokens_sorted_segment_sum(inputs: jax.Array, token_indices: jax.Array) -> jax.Array:
    return inputs[token_indices, ...]


def _take_tokens_sorted_segment_sum_fwd(
    inputs: jax.Array, token_indices: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, int]]:
    return _take_tokens_sorted_segment_sum(inputs, token_indices), (token_indices, inputs.shape[0])


def _take_tokens_sorted_segment_sum_bwd(residuals: tuple[jax.Array, int], grads: jax.Array) -> tuple[jax.Array, None]:
    token_indices, tokens = residuals
    sort_idx = jnp.argsort(token_indices)
    token_sorted = jnp.take(token_indices, sort_idx, axis=0)
    grads_sorted = _sort_activations(grads, sort_idx)
    grad_inputs = jax.ops.segment_sum(grads_sorted, token_sorted, num_segments=tokens, indices_are_sorted=True)
    return grad_inputs, None


_take_tokens_sorted_segment_sum.defvjp(_take_tokens_sorted_segment_sum_fwd, _take_tokens_sorted_segment_sum_bwd)


def _profile_fn(
    fn: Callable,
    *args,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
) -> float:
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(*args))
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))

    os.makedirs(profile_dir, exist_ok=True)
    jax.profiler.start_trace(str(profile_dir), create_perfetto_link=False, create_perfetto_trace=True)
    start = time.perf_counter()
    try:
        for step in range(iters):
            with jax.profiler.StepTraceAnnotation(profile_name, step_num=step):
                jax.block_until_ready(compiled(*args))
    finally:
        jax.profiler.stop_trace()
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


def _set_shared_mlp_explicit_bwd(enabled: bool) -> None:
    global _USE_SHARED_MLP_EXPLICIT_BWD
    _USE_SHARED_MLP_EXPLICIT_BWD = enabled


def _set_shared_mlp_fused_dw_psum_bwd(enabled: bool) -> None:
    global _USE_SHARED_MLP_FUSED_DW_PSUM_BWD
    _USE_SHARED_MLP_FUSED_DW_PSUM_BWD = enabled


def _set_shared_mlp_gradx_first_bwd(enabled: bool) -> None:
    global _USE_SHARED_MLP_GRADX_FIRST_BWD
    _USE_SHARED_MLP_GRADX_FIRST_BWD = enabled


def _set_shared_mlp_fast_accum(enabled: bool) -> None:
    global _USE_SHARED_MLP_FAST_ACCUM
    _USE_SHARED_MLP_FAST_ACCUM = enabled


def _set_combine_fast_accum(enabled: bool) -> None:
    global _USE_COMBINE_FAST_ACCUM
    _USE_COMBINE_FAST_ACCUM = enabled


def _batch_axis_names(x: jax.Array) -> tuple[str, ...]:
    x_type = jax.typeof(x)
    sharding = getattr(x_type, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) == 0:
        sharding = getattr(x, "sharding", None)
        spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) == 0:
        return ()

    axis_spec = spec[0]
    if axis_spec is None:
        return ()
    if isinstance(axis_spec, tuple):
        return tuple(str(name) for name in axis_spec if name is not None)
    return (str(axis_spec),)


def _mesh_reduction_axes(mesh: jax.sharding.AbstractMesh | None) -> tuple[str, ...]:
    if mesh is None or mesh.empty:
        return ()
    return tuple(str(name) for name, size in mesh.shape.items() if int(size) > 1)


def _silu_grad(x: jax.Array) -> jax.Array:
    sigmoid = jax.nn.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def _shared_mlp_preferred_element_type():
    return None if _USE_SHARED_MLP_FAST_ACCUM else jnp.float32


def _combine_preferred_element_type():
    return None if _USE_COMBINE_FAST_ACCUM else jnp.float32


def _shared_mlp_reference(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    shared_dim = shared_w2.shape[0]
    if shared_dim == 0:
        return jnp.zeros_like(x)

    batch_spec = grug_moe_lib._batch_spec_from_x(x, get_abstract_mesh())
    preferred = _shared_mlp_preferred_element_type()
    shared13 = jnp.einsum("td,dm->tm", x, shared_w13, out_sharding=batch_spec, preferred_element_type=preferred)
    gate, up = jnp.split(shared13, [shared_dim], axis=-1)
    shared_gated = jax.nn.silu(gate) * up
    if _USE_SHARED_MLP_FAST_ACCUM:
        shared_gated = shared_gated.astype(x.dtype)
    shared_out = jnp.einsum(
        "tm,md->td",
        shared_gated,
        shared_w2,
        out_sharding=batch_spec,
        preferred_element_type=preferred,
    )
    return shared_out.astype(x.dtype)


def _psum_flattened_pair(
    left: jax.Array,
    right: jax.Array,
    reduction_axes: tuple[str, ...],
) -> tuple[jax.Array, jax.Array]:
    if not reduction_axes:
        return left, right

    left_size = left.size
    flat = jnp.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)
    flat = jax.lax.psum(flat, reduction_axes)
    return flat[:left_size].reshape(left.shape), flat[left_size:].reshape(right.shape)


@jax.custom_vjp
def _shared_mlp_explicit_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    return _shared_mlp_reference(x, shared_w13, shared_w2)


def _shared_mlp_explicit_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array | None, ...]]:
    shared_dim = shared_w2.shape[0]
    if shared_dim == 0:
        return jnp.zeros_like(x), (None, None, None, None, None)

    batch_spec = grug_moe_lib._batch_spec_from_x(x, get_abstract_mesh())
    preferred = _shared_mlp_preferred_element_type()
    shared13 = jnp.einsum("td,dm->tm", x, shared_w13, out_sharding=batch_spec, preferred_element_type=preferred)
    gate, up = jnp.split(shared13, [shared_dim], axis=-1)
    gate_f32 = gate.astype(jnp.float32)
    up_f32 = up.astype(jnp.float32)
    shared_gated = jax.nn.silu(gate_f32) * up_f32
    if _USE_SHARED_MLP_FAST_ACCUM:
        shared_gated = shared_gated.astype(x.dtype)
    shared_out = jnp.einsum(
        "tm,md->td",
        shared_gated,
        shared_w2,
        out_sharding=batch_spec,
        preferred_element_type=preferred,
    )
    residuals = (x, shared_w13, shared_w2, gate_f32, up_f32)
    return shared_out.astype(x.dtype), residuals


def _shared_mlp_explicit_bwd_bwd(
    residuals: tuple[jax.Array | None, ...],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2, gate_f32, up_f32 = residuals
    if x is None or shared_w13 is None or shared_w2 is None or gate_f32 is None or up_f32 is None:
        return jnp.zeros_like(g), jnp.zeros((g.shape[-1], 0), dtype=g.dtype), jnp.zeros((0, g.shape[-1]), dtype=g.dtype)

    x_f32 = x.astype(jnp.float32)
    shared_w13_f32 = shared_w13.astype(jnp.float32)
    shared_w2_f32 = shared_w2.astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)
    preferred = _shared_mlp_preferred_element_type()

    def _shared_bwd_local(
        x_local: jax.Array,
        gate_local: jax.Array,
        up_local: jax.Array,
        g_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        shared_gated_local = jax.nn.silu(gate_local) * up_local
        if _USE_SHARED_MLP_FAST_ACCUM:
            shared_gated_mat = shared_gated_local.astype(jnp.bfloat16)
            g_mat = g_local.astype(jnp.bfloat16)
            x_mat = x_local.astype(jnp.bfloat16)
            shared_w13_mat = shared_w13_local.astype(jnp.bfloat16)
            grad_shared13_dtype = shared_w13_local.dtype
        else:
            shared_gated_mat = shared_gated_local
            g_mat = g_local
            x_mat = x_local
            shared_w13_mat = shared_w13_local
            grad_shared13_dtype = shared_w13_local.dtype
        grad_shared_w2_local = jnp.einsum(
            "tm,td->md",
            shared_gated_mat,
            g_mat,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = jnp.einsum(
            "td,md->tm",
            g_mat,
            shared_w2_local,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = grad_shared_gated_local.astype(jnp.float32)
        grad_gate_local = grad_shared_gated_local * up_local * _silu_grad(gate_local)
        grad_up_local = grad_shared_gated_local * jax.nn.silu(gate_local)
        grad_shared13_local = jnp.concatenate([grad_gate_local, grad_up_local], axis=-1)
        grad_shared13_mat = (
            grad_shared13_local.astype(grad_shared13_dtype) if _USE_SHARED_MLP_FAST_ACCUM else grad_shared13_local
        )
        grad_shared_w13_local = jnp.einsum(
            "td,tm->dm",
            x_mat,
            grad_shared13_mat,
            preferred_element_type=preferred,
        )
        if reduction_axes:
            grad_shared_w13_local = jax.lax.psum(grad_shared_w13_local, reduction_axes)
            grad_shared_w2_local = jax.lax.psum(grad_shared_w2_local, reduction_axes)
        grad_x_local = jnp.einsum(
            "tm,dm->td",
            grad_shared13_mat,
            shared_w13_mat,
            preferred_element_type=preferred,
        )
        return grad_x_local, grad_shared_w13_local, grad_shared_w2_local

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_explicit_bwd.defvjp(_shared_mlp_explicit_bwd_fwd, _shared_mlp_explicit_bwd_bwd)


@jax.custom_vjp
def _shared_mlp_fused_dw_psum_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    return _shared_mlp_reference(x, shared_w13, shared_w2)


def _shared_mlp_fused_dw_psum_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array | None, ...]]:
    return _shared_mlp_explicit_bwd_fwd(x, shared_w13, shared_w2)


def _shared_mlp_fused_dw_psum_bwd_bwd(
    residuals: tuple[jax.Array | None, ...],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2, gate_f32, up_f32 = residuals
    if x is None or shared_w13 is None or shared_w2 is None or gate_f32 is None or up_f32 is None:
        return jnp.zeros_like(g), jnp.zeros((g.shape[-1], 0), dtype=g.dtype), jnp.zeros((0, g.shape[-1]), dtype=g.dtype)

    x_f32 = x.astype(jnp.float32)
    shared_w13_f32 = shared_w13.astype(jnp.float32)
    shared_w2_f32 = shared_w2.astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)
    preferred = _shared_mlp_preferred_element_type()

    def _shared_bwd_local(
        x_local: jax.Array,
        gate_local: jax.Array,
        up_local: jax.Array,
        g_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        shared_gated_local = jax.nn.silu(gate_local) * up_local
        if _USE_SHARED_MLP_FAST_ACCUM:
            shared_gated_mat = shared_gated_local.astype(jnp.bfloat16)
            g_mat = g_local.astype(jnp.bfloat16)
            x_mat = x_local.astype(jnp.bfloat16)
            shared_w13_mat = shared_w13_local.astype(jnp.bfloat16)
            grad_shared13_dtype = shared_w13_local.dtype
        else:
            shared_gated_mat = shared_gated_local
            g_mat = g_local
            x_mat = x_local
            shared_w13_mat = shared_w13_local
            grad_shared13_dtype = shared_w13_local.dtype
        grad_shared_w2_local = jnp.einsum(
            "tm,td->md",
            shared_gated_mat,
            g_mat,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = jnp.einsum(
            "td,md->tm",
            g_mat,
            shared_w2_local,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = grad_shared_gated_local.astype(jnp.float32)
        grad_gate_local = grad_shared_gated_local * up_local * _silu_grad(gate_local)
        grad_up_local = grad_shared_gated_local * jax.nn.silu(gate_local)
        grad_shared13_local = jnp.concatenate([grad_gate_local, grad_up_local], axis=-1)
        grad_shared13_mat = (
            grad_shared13_local.astype(grad_shared13_dtype) if _USE_SHARED_MLP_FAST_ACCUM else grad_shared13_local
        )
        grad_shared_w13_local = jnp.einsum(
            "td,tm->dm",
            x_mat,
            grad_shared13_mat,
            preferred_element_type=preferred,
        )
        grad_shared_w13_local, grad_shared_w2_local = _psum_flattened_pair(
            grad_shared_w13_local,
            grad_shared_w2_local,
            reduction_axes,
        )
        grad_x_local = jnp.einsum(
            "tm,dm->td",
            grad_shared13_mat,
            shared_w13_mat,
            preferred_element_type=preferred,
        )
        return grad_x_local, grad_shared_w13_local, grad_shared_w2_local

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_fused_dw_psum_bwd.defvjp(_shared_mlp_fused_dw_psum_bwd_fwd, _shared_mlp_fused_dw_psum_bwd_bwd)


@jax.custom_vjp
def _shared_mlp_gradx_first_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    return _shared_mlp_reference(x, shared_w13, shared_w2)


def _shared_mlp_gradx_first_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array | None, ...]]:
    return _shared_mlp_explicit_bwd_fwd(x, shared_w13, shared_w2)


def _shared_mlp_gradx_first_bwd_bwd(
    residuals: tuple[jax.Array | None, ...],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2, gate_f32, up_f32 = residuals
    if x is None or shared_w13 is None or shared_w2 is None or gate_f32 is None or up_f32 is None:
        return jnp.zeros_like(g), jnp.zeros((g.shape[-1], 0), dtype=g.dtype), jnp.zeros((0, g.shape[-1]), dtype=g.dtype)

    x_f32 = x.astype(jnp.float32)
    shared_w13_f32 = shared_w13.astype(jnp.float32)
    shared_w2_f32 = shared_w2.astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)
    preferred = _shared_mlp_preferred_element_type()

    def _shared_bwd_local(
        x_local: jax.Array,
        gate_local: jax.Array,
        up_local: jax.Array,
        g_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        shared_gated_local = jax.nn.silu(gate_local) * up_local
        if _USE_SHARED_MLP_FAST_ACCUM:
            shared_gated_mat = shared_gated_local.astype(jnp.bfloat16)
            g_mat = g_local.astype(jnp.bfloat16)
            x_mat = x_local.astype(jnp.bfloat16)
            shared_w13_mat = shared_w13_local.astype(jnp.bfloat16)
            grad_shared13_dtype = shared_w13_local.dtype
        else:
            shared_gated_mat = shared_gated_local
            g_mat = g_local
            x_mat = x_local
            shared_w13_mat = shared_w13_local
            grad_shared13_dtype = shared_w13_local.dtype
        grad_shared_w2_local = jnp.einsum(
            "tm,td->md",
            shared_gated_mat,
            g_mat,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = jnp.einsum(
            "td,md->tm",
            g_mat,
            shared_w2_local,
            preferred_element_type=preferred,
        )
        grad_shared_gated_local = grad_shared_gated_local.astype(jnp.float32)
        grad_gate_local = grad_shared_gated_local * up_local * _silu_grad(gate_local)
        grad_up_local = grad_shared_gated_local * jax.nn.silu(gate_local)
        grad_shared13_local = jnp.concatenate([grad_gate_local, grad_up_local], axis=-1)
        grad_shared13_mat = (
            grad_shared13_local.astype(grad_shared13_dtype) if _USE_SHARED_MLP_FAST_ACCUM else grad_shared13_local
        )
        grad_x_local = jnp.einsum(
            "tm,dm->td",
            grad_shared13_mat,
            shared_w13_mat,
            preferred_element_type=preferred,
        )
        grad_shared_w13_local = jnp.einsum(
            "td,tm->dm",
            x_mat,
            grad_shared13_mat,
            preferred_element_type=preferred,
        )
        grad_shared_w13_local, grad_shared_w2_local = _psum_flattened_pair(
            grad_shared_w13_local,
            grad_shared_w2_local,
            reduction_axes,
        )
        return grad_x_local, grad_shared_w13_local, grad_shared_w2_local

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(
            x_f32,
            gate_f32,
            up_f32,
            g_f32,
            shared_w13_f32,
            shared_w2_f32,
        )
    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_gradx_first_bwd.defvjp(_shared_mlp_gradx_first_bwd_fwd, _shared_mlp_gradx_first_bwd_bwd)


def _shared_mlp(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    if _USE_SHARED_MLP_GRADX_FIRST_BWD:
        return _shared_mlp_gradx_first_bwd(x, shared_w13, shared_w2)
    if _USE_SHARED_MLP_FUSED_DW_PSUM_BWD:
        return _shared_mlp_fused_dw_psum_bwd(x, shared_w13, shared_w2)
    if _USE_SHARED_MLP_EXPLICIT_BWD:
        return _shared_mlp_explicit_bwd(x, shared_w13, shared_w2)
    return _shared_mlp_reference(x, shared_w13, shared_w2)


@jax.custom_vjp
def _shared_mlp_psum_only_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    return _shared_mlp_reference(x, shared_w13, shared_w2)


def _shared_mlp_psum_only_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    return _shared_mlp_reference(x, shared_w13, shared_w2), (x, shared_w13, shared_w2)


def _shared_mlp_psum_only_bwd_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2 = residuals
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)

    def _shared_bwd_local(
        g_local: jax.Array,
        x_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Use a data-dependent scalar so the all-reduce payload stays live while
        # stripping out the expensive local shared-dw GEMMs.
        scale = jnp.mean(g_local.astype(jnp.float32)) + jnp.mean(x_local.astype(jnp.float32))
        grad_shared_w13_local = jnp.broadcast_to(scale.astype(shared_w13_local.dtype), shared_w13_local.shape)
        grad_shared_w2_local = jnp.broadcast_to(scale.astype(shared_w2_local.dtype), shared_w2_local.shape)
        if reduction_axes:
            grad_shared_w13_local = jax.lax.psum(grad_shared_w13_local, reduction_axes)
            grad_shared_w2_local = jax.lax.psum(grad_shared_w2_local, reduction_axes)
        grad_x_local = jnp.zeros_like(g_local)
        return grad_x_local, grad_shared_w13_local, grad_shared_w2_local

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(g, x, shared_w13, shared_w2)
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(g, x, shared_w13, shared_w2)

    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_psum_only_bwd.defvjp(_shared_mlp_psum_only_bwd_fwd, _shared_mlp_psum_only_bwd_bwd)


@jax.custom_vjp
def _shared_mlp_psum_only_w13_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    half = jnp.asarray(0.5, dtype=x.dtype)
    return _shared_mlp_reference(x, shared_w13, shared_w2) * half


def _shared_mlp_psum_only_w13_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    half = jnp.asarray(0.5, dtype=x.dtype)
    return _shared_mlp_reference(x, shared_w13, shared_w2) * half, (x, shared_w13, shared_w2)


def _shared_mlp_psum_only_w13_bwd_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2 = residuals
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)

    def _shared_bwd_local(
        g_local: jax.Array,
        x_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        scale = jnp.mean(g_local.astype(jnp.float32)) + jnp.mean(x_local.astype(jnp.float32))
        grad_shared_w13_local = jnp.broadcast_to(scale.astype(shared_w13_local.dtype), shared_w13_local.shape)
        if reduction_axes:
            grad_shared_w13_local = jax.lax.psum(grad_shared_w13_local, reduction_axes)
        return jnp.zeros_like(g_local), grad_shared_w13_local, jnp.zeros_like(shared_w2_local)

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(g, x, shared_w13, shared_w2)
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(g, x, shared_w13, shared_w2)

    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_psum_only_w13_bwd.defvjp(_shared_mlp_psum_only_w13_bwd_fwd, _shared_mlp_psum_only_w13_bwd_bwd)


@jax.custom_vjp
def _shared_mlp_psum_only_w2_bwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> jax.Array:
    half = jnp.asarray(0.5, dtype=x.dtype)
    return _shared_mlp_reference(x, shared_w13, shared_w2) * half


def _shared_mlp_psum_only_w2_bwd_fwd(
    x: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    half = jnp.asarray(0.5, dtype=x.dtype)
    return _shared_mlp_reference(x, shared_w13, shared_w2) * half, (x, shared_w13, shared_w2)


def _shared_mlp_psum_only_w2_bwd_bwd(
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x, shared_w13, shared_w2 = residuals
    mesh = get_abstract_mesh()
    reduction_axes = _mesh_reduction_axes(mesh)

    def _shared_bwd_local(
        g_local: jax.Array,
        x_local: jax.Array,
        shared_w13_local: jax.Array,
        shared_w2_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        scale = jnp.mean(g_local.astype(jnp.float32)) + jnp.mean(x_local.astype(jnp.float32))
        grad_shared_w2_local = jnp.broadcast_to(scale.astype(shared_w2_local.dtype), shared_w2_local.shape)
        if reduction_axes:
            grad_shared_w2_local = jax.lax.psum(grad_shared_w2_local, reduction_axes)
        return jnp.zeros_like(g_local), jnp.zeros_like(shared_w13_local), grad_shared_w2_local

    if mesh is not None and not mesh.empty:
        batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
        shard_fn = shard_map(
            _shared_bwd_local,
            mesh=mesh,
            in_specs=(batch_spec, batch_spec, P(None, None), P(None, None)),
            out_specs=(batch_spec, P(None, None), P(None, None)),
            check_vma=False,
        )
        grad_x, grad_shared_w13_local, grad_shared_w2_local = shard_fn(g, x, shared_w13, shared_w2)
    else:
        grad_x, grad_shared_w13_local, grad_shared_w2_local = _shared_bwd_local(g, x, shared_w13, shared_w2)

    return (
        grad_x.astype(x.dtype),
        grad_shared_w13_local.astype(shared_w13.dtype),
        grad_shared_w2_local.astype(shared_w2.dtype),
    )


_shared_mlp_psum_only_w2_bwd.defvjp(_shared_mlp_psum_only_w2_bwd_fwd, _shared_mlp_psum_only_w2_bwd_bwd)


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
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
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


def _prefix_cap_counts(counts: jax.Array, *, capacity: int) -> jax.Array:
    accepted = []
    remaining = jnp.array(capacity, dtype=jnp.int32)
    for expert in range(int(counts.shape[0])):
        take = jnp.minimum(counts[expert], remaining)
        accepted.append(take)
        remaining = jnp.maximum(remaining - take, 0)
    return jnp.stack(accepted, axis=0)


def _compact_local_assignments_cumsum(
    x_source: jax.Array,
    token_flat: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    safe_expert = jnp.where(local_mask, local_expert, 0)
    expert_mask = local_mask[:, None] & (safe_expert[:, None] == expert_ids[None, :])
    counts = jnp.sum(expert_mask, axis=0, dtype=jnp.int32)
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - jnp.sum(accepted_counts, dtype=jnp.int32)

    pos_within = jnp.cumsum(expert_mask.astype(jnp.int32), axis=0) - 1
    pos_within_local = jnp.take_along_axis(pos_within, safe_expert[:, None], axis=1).squeeze(1)
    expert_offsets = jnp.cumsum(accepted_counts, dtype=jnp.int32) - accepted_counts
    accepted_limit = jnp.take(accepted_counts, safe_expert, axis=0)
    dest_pos = jnp.take(expert_offsets, safe_expert, axis=0) + pos_within_local
    accepted_mask = local_mask & (pos_within_local < accepted_limit)
    scatter_idx = jnp.where(accepted_mask, dest_pos, local_capacity)

    token_local = jnp.zeros((local_capacity,), dtype=jnp.int32).at[scatter_idx].set(token_flat, mode="drop")
    expert_local = jnp.zeros((local_capacity,), dtype=jnp.int32).at[scatter_idx].set(safe_expert, mode="drop")
    weight_local = jnp.zeros((local_capacity,), dtype=weight_flat.dtype).at[scatter_idx].set(weight_flat, mode="drop")
    x_take = jnp.take(x_source, token_flat, axis=0)
    x_dispatch = (
        jnp.zeros((local_capacity, x_source.shape[1]), dtype=x_source.dtype).at[scatter_idx].set(x_take, mode="drop")
    )

    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, expert_local, weight_local.astype(x_source.dtype), x_dispatch, group_sizes, dropped_local


def _moe_mlp_ep_ring_local_cumsum(
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
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        token_local, expert_local, weight_dispatch, x_dispatch, group_sizes, dropped_local = (
            _compact_local_assignments_cumsum(
                x_global,
                token_flat,
                local_expert,
                local_mask,
                weight_flat,
                local_experts=local_experts,
                local_capacity=local_capacity,
            )
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("scatter"):
        out_global = jnp.zeros_like(x_global).at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


def _moe_mlp_cumsum(
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
                _moe_mlp_ep_ring_local_cumsum,
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


def _pack_by_shard(
    payload: jax.Array,
    token_local: jax.Array,
    shard_ids: jax.Array,
    *,
    num_shards: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    flat_size = int(token_local.shape[0])
    flat_pos = jnp.arange(flat_size, dtype=jnp.int32)

    packed_payload = []
    packed_tokens = []
    packed_valid = []
    dropped_total = jnp.array(0, dtype=jnp.int32)

    for shard in range(num_shards):
        shard_mask = shard_ids == shard
        shard_count = jnp.sum(shard_mask, dtype=jnp.int32)
        dropped_total = dropped_total + jnp.maximum(shard_count - capacity, 0)
        valid = jnp.arange(capacity, dtype=jnp.int32) < shard_count
        selection_key = jnp.where(shard_mask, flat_size - flat_pos, -1)
        _, shard_idx = jax.lax.top_k(selection_key, capacity)

        shard_payload = jnp.take(payload, shard_idx, axis=0)
        shard_tokens = jnp.take(token_local, shard_idx, axis=0)
        shard_payload = jnp.where(valid[:, None], shard_payload, jnp.zeros_like(shard_payload))
        shard_tokens = jnp.where(valid, shard_tokens, 0)

        packed_payload.append(shard_payload)
        packed_tokens.append(shard_tokens)
        packed_valid.append(valid)

    return (
        jnp.stack(packed_payload, axis=0),
        jnp.stack(packed_tokens, axis=0),
        jnp.stack(packed_valid, axis=0),
        dropped_total,
    )


def _compact_local_assignments_topk(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if take_fn is None:
        take_fn = _take_with_gather
    local_count = jnp.sum(local_mask, dtype=jnp.int32)
    dropped_local = jnp.maximum(local_count - local_capacity, 0)
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < local_count
    valid_weight = valid.astype(jnp.float32)

    local_expert = jnp.where(local_mask, local_expert, 0)
    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    token_local = jnp.floor_divide(local_idx, topk)
    expert_local = jnp.take(local_expert, local_idx, axis=0)
    weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_source.dtype)
    x_take = take_fn(x_source, token_local)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    expert_local = jnp.where(valid, expert_local, 0)
    group_sizes = jnp.bincount(expert_local, weights=valid_weight, length=local_experts).astype(jnp.int32)
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _compact_local_assignments_prefix_counts(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if take_fn is None:
        take_fn = _take_with_gather
    local_expert = jnp.where(local_mask, local_expert, 0)
    counts = jnp.bincount(local_expert, weights=local_mask.astype(jnp.int32), length=local_experts).astype(jnp.int32)
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    token_local = jnp.floor_divide(local_idx, topk)
    weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_source.dtype)
    x_take = take_fn(x_source, token_local)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _prefix_cap_counts_vectorized(counts: jax.Array, *, capacity: int) -> jax.Array:
    capacity_i32 = jnp.array(capacity, dtype=jnp.int32)
    prefix_before = jnp.cumsum(counts, dtype=jnp.int32) - counts
    remaining = jnp.maximum(capacity_i32 - prefix_before, 0)
    return jnp.minimum(counts, remaining)


def _compact_local_assignments_vector_prefix(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if take_fn is None:
        take_fn = _take_with_gather
    local_expert = jnp.where(local_mask, local_expert, 0)
    counts = jnp.bincount(local_expert, weights=local_mask.astype(jnp.int32), length=local_experts).astype(jnp.int32)
    accepted_counts = _prefix_cap_counts_vectorized(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    token_local = jnp.floor_divide(local_idx, topk)
    weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_source.dtype)
    x_take = take_fn(x_source, token_local)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _compact_local_assignments_onehot_counts(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if take_fn is None:
        take_fn = _take_with_gather
    local_expert = jnp.where(local_mask, local_expert, 0)
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    local_mask_i32 = local_mask.astype(jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
        axis=0,
        dtype=jnp.int32,
    )
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    token_local = jnp.floor_divide(local_idx, topk)
    weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_source.dtype)
    x_take = take_fn(x_source, token_local)
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _compact_local_assignments_padded_take(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    local_expert = jnp.where(local_mask, local_expert, 0)
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    local_mask_i32 = local_mask.astype(jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
        axis=0,
        dtype=jnp.int32,
    )
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    tokens = x_source.shape[0]
    token_local = jnp.floor_divide(local_idx, topk)
    token_take = jnp.where(valid, token_local, tokens)
    weight_take = jnp.where(valid, local_idx, assignments)

    zero_x = jnp.zeros((1, x_source.shape[1]), dtype=x_source.dtype)
    x_padded = jnp.concatenate([x_source, zero_x], axis=0)
    x_dispatch = jnp.take(x_padded, token_take, axis=0)

    zero_w = jnp.zeros((1,), dtype=weight_flat.dtype)
    weight_padded = jnp.concatenate([weight_flat, zero_w], axis=0)
    weight_dispatch = jnp.take(weight_padded, weight_take, axis=0).astype(x_source.dtype)

    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _compact_local_assignments_owner_local_take(
    x_source: jax.Array,
    local_expert: jax.Array,
    local_mask: jax.Array,
    weight_flat: jax.Array,
    *,
    local_experts: int,
    local_capacity: int,
    topk: int,
    ep_size: int | None = None,
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if ep_size is None:
        raise ValueError("owner_local_take requires ep_size")

    local_expert = jnp.where(local_mask, local_expert, 0)
    expert_ids = jnp.arange(local_experts, dtype=jnp.int32)
    local_mask_i32 = local_mask.astype(jnp.int32)
    counts = jnp.sum(
        (local_expert[:, None] == expert_ids[None, :]).astype(jnp.int32) * local_mask_i32[:, None],
        axis=0,
        dtype=jnp.int32,
    )
    accepted_counts = _prefix_cap_counts(counts, capacity=local_capacity)
    accepted_total = jnp.sum(accepted_counts, dtype=jnp.int32)
    dropped_local = jnp.sum(counts, dtype=jnp.int32) - accepted_total
    valid = jnp.arange(local_capacity, dtype=jnp.int32) < accepted_total

    assignments = int(local_expert.shape[0])
    flat_pos = jnp.arange(assignments, dtype=jnp.int32)
    order_key = local_expert * assignments + flat_pos
    max_order_key = local_experts * assignments
    selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
    _, local_idx = jax.lax.top_k(selection_key, local_capacity)

    token_local = jnp.floor_divide(local_idx, topk)
    weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_source.dtype)

    tokens = x_source.shape[0]
    tokens_per_shard = tokens // ep_size
    owner = jnp.floor_divide(token_local, tokens_per_shard)
    local_token = jnp.remainder(token_local, tokens_per_shard)
    x_by_owner = x_source.reshape(ep_size, tokens_per_shard, x_source.shape[1])
    x_take = x_by_owner[owner, local_token]
    x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
    weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
    group_sizes = accepted_counts.at[-1].add(local_capacity - jnp.sum(accepted_counts, dtype=jnp.int32))
    return token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local


def _scatter_with_add(
    out_dispatch: jax.Array,
    weight_dispatch: jax.Array,
    token_local: jax.Array,
    *,
    tokens: int,
    ep_size: int,
) -> jax.Array:
    out_weighted = out_dispatch * weight_dispatch[:, None]
    out_shape = (tokens, out_weighted.shape[1])
    return jnp.zeros(out_shape, dtype=out_weighted.dtype).at[token_local].add(out_weighted, mode="drop")


def _scatter_with_segment_sum(
    out_dispatch: jax.Array,
    weight_dispatch: jax.Array,
    token_local: jax.Array,
    *,
    tokens: int,
    ep_size: int,
) -> jax.Array:
    out_weighted = out_dispatch * weight_dispatch[:, None]
    return jax.ops.segment_sum(out_weighted, token_local, num_segments=tokens, indices_are_sorted=False)


def _scatter_with_sorted_segment_sum(
    out_dispatch: jax.Array,
    weight_dispatch: jax.Array,
    token_local: jax.Array,
    *,
    tokens: int,
    ep_size: int,
) -> jax.Array:
    out_weighted = out_dispatch * weight_dispatch[:, None]
    sort_idx = jnp.argsort(token_local)
    token_sorted = jnp.take(token_local, sort_idx, axis=0)
    out_sorted = _sort_activations(out_weighted, sort_idx)
    return jax.ops.segment_sum(out_sorted, token_sorted, num_segments=tokens, indices_are_sorted=True)


def _scatter_with_owner_local_add(
    out_dispatch: jax.Array,
    weight_dispatch: jax.Array,
    token_local: jax.Array,
    *,
    tokens: int,
    ep_size: int,
) -> jax.Array:
    out_weighted = out_dispatch * weight_dispatch[:, None]
    tokens_per_shard = tokens // ep_size
    owner = jnp.floor_divide(token_local, tokens_per_shard)
    local_token = jnp.remainder(token_local, tokens_per_shard)
    out_shape = (ep_size, tokens_per_shard, out_weighted.shape[1])
    return jnp.zeros(out_shape, dtype=out_weighted.dtype).at[owner, local_token].add(out_weighted, mode="drop")


def _scatter_with_lax_scatter_add(
    out_dispatch: jax.Array,
    weight_dispatch: jax.Array,
    token_local: jax.Array,
    *,
    tokens: int,
    ep_size: int,
) -> jax.Array:
    out_weighted = out_dispatch * weight_dispatch[:, None]
    indices = token_local[:, None]
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1,),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    init = jnp.zeros((tokens, out_weighted.shape[1]), dtype=out_weighted.dtype)
    return jax.lax.scatter_add(init, indices, out_weighted, dnums, indices_are_sorted=False, unique_indices=False)


def _take_with_gather(x_source: jax.Array, token_local: jax.Array) -> jax.Array:
    return jnp.take(x_source, token_local, axis=0)


def _take_with_segment_sum_bwd(x_source: jax.Array, token_local: jax.Array) -> jax.Array:
    return _take_tokens_segment_sum(x_source, token_local)


def _take_with_sorted_segment_sum_bwd(x_source: jax.Array, token_local: jax.Array) -> jax.Array:
    return _take_tokens_sorted_segment_sum(x_source, token_local)


def _moe_mlp_ep_ring_local_variant(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    compact_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]],
    scatter_fn: Callable[..., jax.Array],
    take_fn: Callable[[jax.Array, jax.Array], jax.Array] = _take_with_gather,
    gather_weight_dtype: jnp.dtype | None = None,
    gather_expert_dtype: jnp.dtype | None = None,
) -> tuple[jax.Array, jax.Array]:
    with jax.named_scope("gather"):
        x_global = jax.lax.all_gather(x_local, "expert", tiled=True)
        selected_gather = (
            selected_experts_local.astype(gather_expert_dtype)
            if gather_expert_dtype is not None
            else selected_experts_local
        )
        weight_gather = (
            combine_weights_local.astype(gather_weight_dtype)
            if gather_weight_dtype is not None
            else combine_weights_local
        )
        selected_experts_global = jax.lax.all_gather(selected_gather, "expert", tiled=True).astype(jnp.int32)
        combine_weights_global = jax.lax.all_gather(weight_gather, "expert", tiled=True)

        tokens = x_global.shape[0]
        topk = selected_experts_global.shape[1]
        assignments = tokens * topk
        expert_flat = selected_experts_global.reshape(assignments)
        weight_flat = combine_weights_global.reshape(assignments)

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
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local = compact_fn(
            x_global,
            local_expert,
            local_mask,
            weight_flat,
            local_experts=local_experts,
            local_capacity=local_capacity,
            topk=topk,
            ep_size=ep_size,
            take_fn=take_fn,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("scatter"):
        out_global = scatter_fn(
            out_dispatch,
            weight_dispatch,
            token_local,
            tokens=tokens,
            ep_size=ep_size,
        )
        out_local = jax.lax.psum_scatter(out_global, "expert", scatter_dimension=0, tiled=True)
        out_local = jnp.reshape(out_local, (x_local.shape[0], x_local.shape[1]))
        dropped_total = jax.lax.psum(dropped_local, ("data", "expert"))
    return out_local, dropped_total


def _permute_by_global_expert(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    *,
    num_experts: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    topk = selected_experts_local.shape[1]
    flat_selected = selected_experts_local.reshape(-1)
    sorted_indices = jnp.argsort(flat_selected)
    repeated_x = jnp.repeat(x_local, topk, axis=0)
    sorted_x = _sort_activations(repeated_x, sorted_indices)
    group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)
    return sorted_x, sorted_indices, flat_selected, group_sizes


def _unpermute_from_global_expert(
    intermediate: jax.Array,
    sorted_indices: jax.Array,
    combine_weights_local: jax.Array,
    *,
    tokens_per_shard: int,
    topk: int,
) -> jax.Array:
    unsorted = _sort_activations(intermediate, jnp.argsort(sorted_indices))
    reshaped = unsorted.reshape(tokens_per_shard, topk, -1)
    preferred = _combine_preferred_element_type()
    return jnp.einsum(
        "tkd,tk->td",
        reshaped,
        combine_weights_local.astype(reshaped.dtype),
        preferred_element_type=preferred,
    )


def _shard_a2a_params(
    shard_counts: jax.Array, shard_id: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    row = shard_counts[shard_id]
    input_offsets = jnp.cumsum(jnp.concatenate((jnp.array([0], dtype=row.dtype), row[:-1])))
    send_sizes = row

    zero_row = jnp.zeros((1, shard_counts.shape[1]), dtype=shard_counts.dtype)
    cumulative = jnp.cumsum(jnp.concatenate((zero_row, shard_counts), axis=0), axis=0, dtype=shard_counts.dtype)
    output_offsets = cumulative[shard_id]
    recv_sizes = shard_counts[:, shard_id]
    return input_offsets, send_sizes, output_offsets, recv_sizes


def _local_permute_from_counts(
    inputs: jax.Array,
    global_group_sizes: jax.Array,
    *,
    local_expert_size: int,
    shard_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    all_shard_local_sizes = jax.lax.dynamic_slice_in_dim(
        global_group_sizes,
        start_index=shard_index * local_expert_size,
        slice_size=local_expert_size,
        axis=1,
    )
    local_group_sizes = jnp.sum(all_shard_local_sizes, axis=0)
    local_sizes = all_shard_local_sizes.reshape(-1)
    total_valid = jnp.sum(local_sizes, dtype=jnp.int32)
    segment_ends = jnp.cumsum(local_sizes, dtype=jnp.int32)
    positions = jnp.arange(inputs.shape[0], dtype=jnp.int32)
    segment_index = jnp.searchsorted(segment_ends, positions, side="right")
    local_expert_ids = jnp.where(positions < total_valid, segment_index % local_expert_size, local_expert_size)
    sorted_indices = jnp.argsort(local_expert_ids)
    sorted_inputs = _sort_activations(inputs, sorted_indices)
    sorted_inputs = jnp.where((jnp.arange(inputs.shape[0], dtype=jnp.int32) < total_valid)[:, None], sorted_inputs, 0)
    group_sizes = local_group_sizes.at[-1].add(inputs.shape[0] - total_valid)
    return sorted_inputs, sorted_indices, group_sizes


def _deepep_transport_layout_counts_local(
    selected_experts_local: jax.Array,
    *,
    num_experts: int,
    num_ranks: int,
) -> tuple[jax.Array, jax.Array]:
    num_tokens_per_rank, num_tokens_per_expert, _ = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=num_ranks,
        num_experts=num_experts,
    )
    return num_tokens_per_rank[None, :], num_tokens_per_expert[None, :]


def _pack_deepep_local_assignments(
    recv_x: jax.Array,
    recv_topk_idx: jax.Array,
    recv_topk_weights: jax.Array,
    *,
    expert_start: jax.Array,
    local_experts: int,
    num_recv_tokens: jax.Array,
    max_local_assignments: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    max_recv_tokens, topk = recv_topk_idx.shape
    total_assignments = max_recv_tokens * topk
    packed_assignments = (
        total_assignments if max_local_assignments is None else min(max_local_assignments, total_assignments)
    )

    recv_token_indices = jnp.repeat(jnp.arange(max_recv_tokens, dtype=jnp.int32), topk)
    expert_flat = (recv_topk_idx.reshape(-1) - expert_start).astype(jnp.int32)
    recv_valid = jnp.arange(max_recv_tokens, dtype=jnp.int32) < num_recv_tokens
    local_mask = recv_valid[:, None] & (recv_topk_idx >= expert_start) & (recv_topk_idx < expert_start + local_experts)
    local_mask_flat = local_mask.reshape(-1)
    local_bucket = jnp.where(local_mask_flat, expert_flat, local_experts)
    local_group_sizes = jnp.bincount(local_bucket, length=local_experts + 1).astype(jnp.int32)[:-1]
    total_valid = jnp.sum(local_group_sizes, dtype=jnp.int32)

    flat_positions = jnp.arange(total_assignments, dtype=jnp.int32)
    order_key = local_bucket * total_assignments + flat_positions
    max_order_key = (local_experts + 1) * total_assignments
    selection_key = jnp.where(local_mask_flat, max_order_key - order_key, -1)
    _, sorted_assignment_indices = jax.lax.top_k(selection_key, total_assignments)
    sorted_assignment_indices = sorted_assignment_indices[:packed_assignments]

    recv_token_indices = jnp.take(recv_token_indices, sorted_assignment_indices, axis=0)
    x_dispatch = jnp.take(recv_x, recv_token_indices, axis=0)
    assignment_weights = jnp.take(recv_topk_weights.reshape(-1), sorted_assignment_indices, axis=0).astype(
        recv_x.dtype
    )
    valid_sorted = jnp.arange(packed_assignments, dtype=jnp.int32) < total_valid
    x_dispatch = jnp.where(valid_sorted[:, None], x_dispatch, 0)
    assignment_weights = jnp.where(valid_sorted, assignment_weights, 0)
    return x_dispatch, assignment_weights, recv_token_indices, local_group_sizes


def _collapse_deepep_local_assignments(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    *,
    recv_capacity: int,
    num_recv_tokens: jax.Array,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    out_weighted = out_dispatch * assignment_weights[:, None]
    if collapse_impl == "segment_sum":
        recv_out = jax.ops.segment_sum(
            out_weighted,
            recv_token_indices,
            num_segments=recv_capacity,
            indices_are_sorted=False,
        )
    elif collapse_impl == "sorted_segment_sum":
        sort_idx = jnp.argsort(recv_token_indices)
        token_sorted = jnp.take(recv_token_indices, sort_idx, axis=0)
        out_sorted = _sort_activations(out_weighted, sort_idx)
        recv_out = jax.ops.segment_sum(
            out_sorted,
            token_sorted,
            num_segments=recv_capacity,
            indices_are_sorted=True,
        )
    elif collapse_impl == "scatter_add":
        recv_out = jnp.zeros((recv_capacity, out_weighted.shape[1]), dtype=out_weighted.dtype).at[recv_token_indices].add(
            out_weighted,
            mode="drop",
        )
    elif collapse_impl == "lax_scatter":
        indices = recv_token_indices[:, None]
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(1,),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,),
        )
        init = jnp.zeros((recv_capacity, out_weighted.shape[1]), dtype=out_weighted.dtype)
        recv_out = jax.lax.scatter_add(
            init,
            indices,
            out_weighted,
            dnums,
            indices_are_sorted=False,
            unique_indices=False,
        )
    else:
        raise ValueError(f"Unsupported collapse_impl={collapse_impl}")
    recv_valid = jnp.arange(recv_capacity, dtype=jnp.int32) < num_recv_tokens
    return jnp.where(recv_valid[:, None], recv_out, 0)


def _deepep_transport_exact_caps(
    selected_experts: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh,
    num_experts: int,
    capacity_multiple: int = 128,
) -> tuple[int, int]:
    max_recv_tokens, max_local_assignments, _max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
        capacity_multiple=capacity_multiple,
    )
    return max_recv_tokens, max_local_assignments


def _deepep_transport_exact_cap_metadata(
    selected_experts: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh,
    num_experts: int,
    capacity_multiple: int = 128,
) -> tuple[int, int, int]:
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    if num_experts % expert_axis_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size} for capped DeepEP"
        )

    batch_spec = grug_moe_lib._batch_spec_from_x(selected_experts, mesh)
    shard_fn = shard_map(
        partial(
            _deepep_transport_layout_counts_local,
            num_experts=num_experts,
            num_ranks=expert_axis_size,
        ),
        mesh=mesh,
        in_specs=(batch_spec,),
        out_specs=(P("expert", None), P("expert", None)),
        check_vma=False,
    )
    num_tokens_per_rank, num_tokens_per_expert = jax.jit(shard_fn)(selected_experts)
    num_tokens_per_rank_host = np.asarray(jax.device_get(num_tokens_per_rank), dtype=np.int32)
    num_tokens_per_expert_host = np.asarray(jax.device_get(num_tokens_per_expert), dtype=np.int32)
    local_experts = num_experts // expert_axis_size
    recv_tokens_per_rank = np.sum(num_tokens_per_rank_host, axis=0, dtype=np.int64)
    global_assignments_per_expert = np.sum(num_tokens_per_expert_host, axis=0, dtype=np.int64)
    local_assignments_per_rank = global_assignments_per_expert.reshape(expert_axis_size, local_experts).sum(
        axis=1, dtype=np.int64
    )
    max_local_expert_assignments = _round_up_capacity(int(np.max(global_assignments_per_expert)), multiple=capacity_multiple)
    max_recv_tokens = _round_up_capacity(int(np.max(recv_tokens_per_rank)), multiple=capacity_multiple)
    max_local_assignments = _round_up_capacity(int(np.max(local_assignments_per_rank)), multiple=capacity_multiple)
    _print0(
        "DEEPEP_EXACT_CAPS "
        f"max_recv_tokens={max_recv_tokens} "
        f"max_local_assignments={max_local_assignments} "
        f"max_local_expert_assignments={max_local_expert_assignments} "
        f"recv_factor={(selected_experts.shape[0] / expert_axis_size * expert_axis_size) / max_recv_tokens:.6f} "
        f"assign_factor={((selected_experts.shape[0] / expert_axis_size) * expert_axis_size * selected_experts.shape[1]) / max_local_assignments:.6f}"
    )
    return max_recv_tokens, max_local_assignments, max_local_expert_assignments


def _current_ring_w13_cap_metadata(
    selected_experts: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh,
    num_experts: int,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
    capacity_multiple: int = 128,
) -> int:
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    if expert_axis_size <= 1:
        return _round_up_capacity(int(selected_experts.shape[0] * selected_experts.shape[1]), multiple=capacity_multiple)
    if num_experts % expert_axis_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size} for current ring metadata"
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

    max_local_expert_assignments = _round_up_capacity(max_local_expert_assignments, multiple=capacity_multiple)
    _print0(
        "CURRENT_RING_W13_CAPS "
        f"local_capacity={local_capacity} "
        f"max_local_expert_assignments={max_local_expert_assignments}"
    )
    return max_local_expert_assignments


def _fit_probe_output_to_hidden(probe_out: jax.Array, *, hidden_dim: int) -> jax.Array:
    if probe_out.shape[1] == hidden_dim:
        return probe_out
    if probe_out.shape[1] > hidden_dim:
        return probe_out[:, :hidden_dim]

    pad_width = hidden_dim - probe_out.shape[1]
    return jnp.pad(probe_out, ((0, 0), (0, pad_width)))


def _moe_mlp_ep_ragged_a2a_local(
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
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    recv_capacity = max(local_experts, int(np.ceil(capacity_factor * assignments_per_shard)))

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, _, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        shard_counts = jnp.sum(group_sizes.reshape(ep_size, local_experts), axis=1).astype(jnp.int32)
        all_shard_counts = jax.lax.all_gather(shard_counts, "expert")
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        x_dispatched = jax.lax.ragged_all_to_all(
            sorted_x,
            dispatch_out_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        global_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
        x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
            x_dispatched,
            global_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        local_output = _sort_activations(out_dispatch, jnp.argsort(local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            all_shard_counts.T, shard_id
        )
        returned = jax.lax.ragged_all_to_all(
            local_output,
            return_out_shape,
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_total = jnp.array(0, dtype=jnp.int32)
    return out_local, dropped_total


def _moe_mlp_ragged_a2a(
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
                _moe_mlp_ep_ragged_a2a_local,
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


def _moe_mlp_ep_ragged_a2a_deepep_layout_local(
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
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    recv_capacity = max(local_experts, int(np.ceil(capacity_factor * assignments_per_shard)))

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, _, _ = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        _, group_sizes, _ = deepep_get_dispatch_layout(
            selected_experts_local,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
        group_sizes = group_sizes.astype(jnp.int32)
        # DeepEP exposes token-per-rank reachability separately, but this ragged_a2a path
        # still dispatches repeated token assignments, so the send sizes must come from
        # per-expert assignment counts.
        shard_counts = jnp.sum(group_sizes.reshape(ep_size, local_experts), axis=1).astype(jnp.int32)
        all_shard_counts = jax.lax.all_gather(shard_counts, "expert")
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        x_dispatched = jax.lax.ragged_all_to_all(
            sorted_x,
            dispatch_out_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )
        global_group_sizes = jax.lax.all_gather(group_sizes, "expert")
        x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
            x_dispatched,
            global_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)

    with jax.named_scope("combine"):
        local_output = _sort_activations(out_dispatch, jnp.argsort(local_sorted_indices))
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=local_output.dtype)
        return_input_offsets, return_send_sizes, return_output_offsets, return_recv_sizes = _shard_a2a_params(
            all_shard_counts.T, shard_id
        )
        returned = jax.lax.ragged_all_to_all(
            local_output,
            return_out_shape,
            return_input_offsets,
            return_send_sizes,
            return_output_offsets,
            return_recv_sizes,
            axis_name="expert",
        )
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_total = jnp.array(0, dtype=jnp.int32)
    return out_local, dropped_total


def _moe_mlp_ep_deepep_transport_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    with jax.named_scope("dispatch"):
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
            selected_experts_local,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            _local_expert_counts,
            num_recv_tokens,
        ) = deepep_dispatch_intranode(
            x_local,
            selected_experts_local,
            combine_weights_local,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            max_recv_tokens=max_recv_tokens,
        )
        num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
        x_dispatch, assignment_weights, recv_token_indices, local_group_sizes = _pack_deepep_local_assignments(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            expert_start=expert_start,
            local_experts=local_experts,
            num_recv_tokens=num_recv_tokens_scalar,
            max_local_assignments=max_local_assignments,
        )

    with jax.named_scope("moe_up_down"):
        if w13_local_expert_capacity is None:
            w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        else:
            w13_out = _ragged_dot_expert_padded_batched(
                x_dispatch,
                moe_w13_local,
                local_group_sizes,
                local_expert_capacity=w13_local_expert_capacity,
            )
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        gate_up = activation_fn(gate) * up
        if w2_local_expert_capacity is None:
            out_dispatch = ragged_dot(gate_up, moe_w2_local, local_group_sizes)
        else:
            out_dispatch = _ragged_dot_expert_padded_batched(
                gate_up,
                moe_w2_local,
                local_group_sizes,
                local_expert_capacity=w2_local_expert_capacity,
            )

    with jax.named_scope("combine"):
        recv_out = _collapse_deepep_local_assignments(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            recv_capacity=recv_x.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
            collapse_impl=collapse_impl,
        )
        out_local, _ = deepep_combine_intranode(
            recv_out,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        )
        dropped_total = jnp.array(0, dtype=jnp.int32)
    return out_local.astype(x_local.dtype), dropped_total


def _moe_mlp_ragged_a2a_deepep_layout(
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
                _moe_mlp_ep_ragged_a2a_deepep_layout_local,
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


def _moe_mlp_deepep_transport(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
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
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport currently requires the expert group to span all visible local GPUs; "
                f"got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_local,
                activation_fn=activation_fn,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
                w13_local_expert_capacity=w13_local_expert_capacity,
                w2_local_expert_capacity=w2_local_expert_capacity,
                collapse_impl=collapse_impl,
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


def _moe_mlp_ep_deepep_transport_dispatch_pack_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    *,
    num_experts: int,
    local_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
]:
    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    with jax.named_scope("dispatch_layout"):
        num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
            selected_experts_local,
            num_ranks=ep_size,
            num_experts=num_experts,
        )
    with jax.named_scope("dispatch_intranode"):
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            _local_expert_counts,
            num_recv_tokens,
        ) = deepep_dispatch_intranode(
            x_local,
            selected_experts_local,
            combine_weights_local,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            num_experts=num_experts,
            max_recv_tokens=max_recv_tokens,
        )
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    with jax.named_scope("pack_local_assignments"):
        x_dispatch, assignment_weights, recv_token_indices, local_group_sizes = _pack_deepep_local_assignments(
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            expert_start=expert_start,
            local_experts=local_experts,
            num_recv_tokens=num_recv_tokens_scalar,
            max_local_assignments=max_local_assignments,
        )
    return (
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )


def _moe_mlp_deepep_transport_dispatch_pack(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    if mesh is None:
        mesh = get_abstract_mesh()

    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    if mesh is None or mesh.empty or not has_expert_axis or expert_axis_size <= 1:
        raise ValueError("deepep_transport_staged requires expert parallel mesh")
    if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
        raise ValueError(
            "deepep_transport_staged currently requires the expert group to span all visible local GPUs; "
            f"got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis size={expert_axis_size}"
        )
    if num_experts % expert_axis_size != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by expert axis size={expert_axis_size} for staged DeepEP"
        )

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)
    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_deepep_transport_dispatch_pack_local,
            num_experts=num_experts,
            local_experts=num_experts // expert_axis_size,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec),
        out_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert"),
            P("expert", None),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
            batch_spec,
        ),
        check_vma=False,
    )
    return shard_fn(x, selected_experts, combine_weights)


def _moe_mlp_ep_deepep_transport_local_compute_local(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    with jax.named_scope("w13_ragged_dot"):
        if w13_local_expert_capacity is None:
            w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        else:
            w13_out = _ragged_dot_expert_padded_batched(
                x_dispatch,
                moe_w13_local,
                local_group_sizes,
                local_expert_capacity=w13_local_expert_capacity,
            )
    moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
    with jax.named_scope("gate_up_split"):
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    with jax.named_scope("gate_activation"):
        gate_up = activation_fn(gate) * up
    with jax.named_scope("w2_ragged_dot"):
        if w2_local_expert_capacity is None:
            return ragged_dot(gate_up, moe_w2_local, local_group_sizes)
        return _ragged_dot_expert_padded_batched(
            gate_up,
            moe_w2_local,
            local_group_sizes,
            local_expert_capacity=w2_local_expert_capacity,
        )


def _moe_mlp_ep_deepep_transport_w13_only_local(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    moe_w13_local: jax.Array,
    *,
    w13_local_expert_capacity: int | None = None,
) -> jax.Array:
    with jax.named_scope("w13_ragged_dot"):
        if w13_local_expert_capacity is None:
            return ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        return _ragged_dot_expert_padded_batched(
            x_dispatch,
            moe_w13_local,
            local_group_sizes,
            local_expert_capacity=w13_local_expert_capacity,
        )


def _ragged_dot_expert_padded_batched(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    local_expert_capacity: int,
) -> jax.Array:
    if local_expert_capacity <= 0:
        raise ValueError(f"local_expert_capacity must be positive, got {local_expert_capacity}")

    hidden = lhs.shape[-1]
    if rhs.shape[1] == hidden:
        rhs_contract_axis = 1
    elif rhs.ndim > 2 and rhs.shape[2] == hidden:
        rhs_contract_axis = 2
    else:
        raise ValueError(
            f"ragged expert batched dot requires rhs to contract over hidden={hidden}, got rhs.shape={rhs.shape}"
        )
    return grug_moe_lib._ragged_dot_expert_padded_batched(
        lhs,
        rhs,
        group_sizes,
        local_expert_capacity=local_expert_capacity,
        rhs_contract_axis=rhs_contract_axis,
    )


def _moe_mlp_deepep_transport_w13_only(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    w_up_gate: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    w13_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_deepep_transport_w13_only_local,
            w13_local_expert_capacity=w13_local_expert_capacity,
        ),
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert", None, None),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )
    return shard_fn(x_dispatch, local_group_sizes, w_up_gate)


def _moe_mlp_ep_deepep_transport_gate_up_only_local(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    with jax.named_scope("w13_ragged_dot"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
    moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
    with jax.named_scope("gate_up_split"):
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    with jax.named_scope("gate_activation"):
        return activation_fn(gate) * up


def _moe_mlp_deepep_transport_gate_up_only(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()
    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_deepep_transport_gate_up_only_local,
            activation_fn=activation_fn,
        ),
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert", None, None),
            P("expert", None, None),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )
    return shard_fn(x_dispatch, local_group_sizes, w_up_gate, w_down)


def _moe_mlp_ep_deepep_transport_w2_only_local(
    gate_up: jax.Array,
    local_group_sizes: jax.Array,
    moe_w2_local: jax.Array,
    *,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    with jax.named_scope("w2_ragged_dot"):
        if w2_local_expert_capacity is None:
            return ragged_dot(gate_up, moe_w2_local, local_group_sizes)
        return _ragged_dot_expert_padded_batched(
            gate_up,
            moe_w2_local,
            local_group_sizes,
            local_expert_capacity=w2_local_expert_capacity,
        )


def _moe_mlp_deepep_transport_w2_only(
    gate_up: jax.Array,
    local_group_sizes: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_deepep_transport_w2_only_local,
            w2_local_expert_capacity=w2_local_expert_capacity,
        ),
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert", None, None),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )
    return shard_fn(gate_up, local_group_sizes, w_down)


def _moe_mlp_deepep_transport_local_compute(
    x_dispatch: jax.Array,
    local_group_sizes: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()
    shard_fn = shard_map(
        partial(
            _moe_mlp_ep_deepep_transport_local_compute_local,
            activation_fn=activation_fn,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
        ),
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert", None, None),
            P("expert", None, None),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )
    return shard_fn(x_dispatch, local_group_sizes, w_up_gate, w_down)


def _moe_mlp_ep_deepep_transport_collapse_combine_local(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> jax.Array:
    recv_out = _moe_mlp_ep_deepep_transport_collapse_only_local(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_topk_weights,
        num_recv_tokens,
    )
    return _moe_mlp_ep_deepep_transport_combine_only_local(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )


def _moe_mlp_ep_deepep_transport_collapse_only_local(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
) -> jax.Array:
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    with jax.named_scope("collapse_local_assignments"):
        return _collapse_deepep_local_assignments(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            recv_capacity=recv_topk_weights.shape[0],
            num_recv_tokens=num_recv_tokens_scalar,
        )


def _moe_mlp_ep_deepep_transport_combine_only_local(
    recv_out: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
) -> jax.Array:
    with jax.named_scope("combine_intranode"):
        out_local, _ = deepep_combine_intranode(
            recv_out,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        )
    return out_local


def _moe_mlp_deepep_transport_collapse_combine(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    batch_spec = grug_moe_lib._batch_spec(mesh)
    shard_fn = shard_map(
        _moe_mlp_ep_deepep_transport_collapse_combine_local,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert", None),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
            batch_spec,
        ),
        out_specs=batch_spec,
        check_vma=False,
    )
    return shard_fn(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )


def _moe_mlp_deepep_transport_collapse_only(
    out_dispatch: jax.Array,
    assignment_weights: jax.Array,
    recv_token_indices: jax.Array,
    recv_topk_weights: jax.Array,
    num_recv_tokens: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    shard_fn = shard_map(
        _moe_mlp_ep_deepep_transport_collapse_only_local,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert"),
            P("expert"),
            P("expert", None),
            P("expert"),
        ),
        out_specs=P("expert", None),
        check_vma=False,
    )
    return shard_fn(out_dispatch, assignment_weights, recv_token_indices, recv_topk_weights, num_recv_tokens)


def _moe_mlp_deepep_transport_combine_only(
    recv_out: jax.Array,
    recv_topk_weights: jax.Array,
    recv_src_idx: jax.Array,
    rank_prefix_matrix: jax.Array,
    channel_prefix_matrix: jax.Array,
    recv_channel_prefix_matrix: jax.Array,
    send_head: jax.Array,
    num_recv_tokens: jax.Array,
    is_token_in_rank: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    batch_spec = grug_moe_lib._batch_spec(mesh)
    shard_fn = shard_map(
        _moe_mlp_ep_deepep_transport_combine_only_local,
        mesh=mesh,
        in_specs=(
            P("expert", None),
            P("expert", None),
            P("expert"),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
            batch_spec,
        ),
        out_specs=batch_spec,
        check_vma=False,
    )
    return shard_fn(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )


def _moe_mlp_ep_deepep_transport_identity_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    _moe_w13_local: jax.Array,
    _moe_w2_local: jax.Array,
    *,
    num_experts: int,
    max_recv_tokens: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_experts = _moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    ep_size = num_experts // local_experts
    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        _recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
    )
    out_local, _ = deepep_combine_intranode(
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    fanout = jnp.maximum(jnp.sum(is_token_in_rank.astype(jnp.int32), axis=1), 1)
    return (out_local / fanout[:, None]).astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _moe_mlp_deepep_transport_identity(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        return x

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport_identity currently requires the expert group to span all visible local GPUs; "
                f"got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_identity_local,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
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

    return x


def _moe_mlp_ep_deepep_transport_assignments_identity_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    _moe_w2_local: jax.Array,
    *,
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
    )
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    x_dispatch, assignment_weights, recv_token_indices, _local_group_sizes = _pack_deepep_local_assignments(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        expert_start=expert_start,
        local_experts=local_experts,
        num_recv_tokens=num_recv_tokens_scalar,
        max_local_assignments=max_local_assignments,
    )
    recv_out = _collapse_deepep_local_assignments(
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_x.shape[0],
        num_recv_tokens=num_recv_tokens_scalar,
    )
    out_local, _ = deepep_combine_intranode(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return out_local.astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _moe_mlp_deepep_transport_assignments_identity(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        return x

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport_assignments_identity currently requires the expert group to span all visible local "
                f"GPUs; got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_assignments_identity_local,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
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

    return x


def _moe_mlp_ep_deepep_transport_first_ragged_dot_probe_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    _moe_w2_local: jax.Array,
    *,
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
    )
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    x_dispatch, assignment_weights, recv_token_indices, local_group_sizes = _pack_deepep_local_assignments(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        expert_start=expert_start,
        local_experts=local_experts,
        num_recv_tokens=num_recv_tokens_scalar,
        max_local_assignments=max_local_assignments,
    )
    probe_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
    recv_out = _collapse_deepep_local_assignments(
        _fit_probe_output_to_hidden(probe_out, hidden_dim=x_local.shape[1]),
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_x.shape[0],
        num_recv_tokens=num_recv_tokens_scalar,
    )
    out_local, _ = deepep_combine_intranode(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return out_local.astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _moe_mlp_deepep_transport_first_ragged_dot_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")

    if mesh is None or mesh.empty:
        return x

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport_first_ragged_dot_probe currently requires the expert group to span all visible "
                f"local GPUs; got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis "
                f"size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_first_ragged_dot_probe_local,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
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

    return x


def _moe_mlp_ep_deepep_transport_gate_probe_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
    )
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    x_dispatch, assignment_weights, recv_token_indices, local_group_sizes = _pack_deepep_local_assignments(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        expert_start=expert_start,
        local_experts=local_experts,
        num_recv_tokens=num_recv_tokens_scalar,
        max_local_assignments=max_local_assignments,
    )
    w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
    moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    probe_out = activation_fn(gate) * up
    recv_out = _collapse_deepep_local_assignments(
        _fit_probe_output_to_hidden(probe_out, hidden_dim=x_local.shape[1]),
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_x.shape[0],
        num_recv_tokens=num_recv_tokens_scalar,
    )
    out_local, _ = deepep_combine_intranode(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return out_local.astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _moe_mlp_deepep_transport_gate_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()

    if mesh is None or mesh.empty:
        return x

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport_gate_probe currently requires the expert group to span all visible local GPUs; "
                f"got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_gate_probe_local,
                activation_fn=activation_fn,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
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

    return x


def _moe_mlp_ep_deepep_transport_second_ragged_dot_probe_local(
    x_local: jax.Array,
    selected_experts_local: jax.Array,
    combine_weights_local: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )
    if x_local.shape[1] % 8 != 0:
        raise ValueError(f"DeepEP transport requires hidden % 8 == 0, got hidden={x_local.shape[1]}")

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    expert_start = shard_id * local_experts

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        _local_expert_counts,
        num_recv_tokens,
    ) = deepep_dispatch_intranode(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
    )
    num_recv_tokens_scalar = jnp.squeeze(num_recv_tokens, axis=0)
    x_dispatch, assignment_weights, recv_token_indices, local_group_sizes = _pack_deepep_local_assignments(
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        expert_start=expert_start,
        local_experts=local_experts,
        num_recv_tokens=num_recv_tokens_scalar,
        max_local_assignments=max_local_assignments,
    )
    w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
    moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    probe_out = ragged_dot(activation_fn(gate) * up, moe_w2_local, local_group_sizes)
    recv_out = _collapse_deepep_local_assignments(
        probe_out,
        assignment_weights,
        recv_token_indices,
        recv_capacity=recv_x.shape[0],
        num_recv_tokens=num_recv_tokens_scalar,
    )
    out_local, _ = deepep_combine_intranode(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return out_local.astype(x_local.dtype), jnp.array(0, dtype=jnp.int32)


def _moe_mlp_deepep_transport_second_ragged_dot_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int | None = None,
    max_local_assignments: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()

    num_experts = int(w_up_gate.shape[0])
    has_expert_axis = grug_moe_lib._mesh_has_axis(mesh, "expert")
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()

    if mesh is None or mesh.empty:
        return x

    batch_spec = grug_moe_lib._batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        if grug_moe_lib._mesh_axis_size(mesh, "data") != 1:
            raise ValueError(
                "deepep_transport_second_ragged_dot_probe currently requires the expert group to span all visible "
                f"local GPUs; got data axis size={grug_moe_lib._mesh_axis_size(mesh, 'data')} and expert axis "
                f"size={expert_axis_size}"
            )
        shard_fn = shard_map(
            partial(
                _moe_mlp_ep_deepep_transport_second_ragged_dot_probe_local,
                activation_fn=activation_fn,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
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

    return x


def _forward_deepep_transport_local_compute_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()
    num_experts = int(w_up_gate.shape[0])
    (
        x_dispatch,
        _assignment_weights,
        _recv_token_indices,
        local_group_sizes,
        _recv_topk_weights,
        _recv_src_idx,
        _rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        _send_head,
        _num_recv_tokens,
        _is_token_in_rank,
    ) = _moe_mlp_deepep_transport_dispatch_pack(
        x,
        selected_experts,
        combine_weights,
        mesh=mesh,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
    )
    return _moe_mlp_deepep_transport_local_compute(
        x_dispatch,
        local_group_sizes,
        w_up_gate,
        w_down,
        mesh=mesh,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )


def _forward_deepep_transport_w13_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()
    num_experts = int(w_up_gate.shape[0])
    (
        x_dispatch,
        _assignment_weights,
        _recv_token_indices,
        local_group_sizes,
        _recv_topk_weights,
        _recv_src_idx,
        _rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        _send_head,
        _num_recv_tokens,
        _is_token_in_rank,
    ) = _moe_mlp_deepep_transport_dispatch_pack(
        x,
        selected_experts,
        combine_weights,
        mesh=mesh,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
    )
    del w_down
    return _moe_mlp_deepep_transport_w13_only(
        x_dispatch,
        local_group_sizes,
        w_up_gate,
        mesh=mesh,
        w13_local_expert_capacity=w13_local_expert_capacity,
    )


def _forward_deepep_transport_w2_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int,
    max_local_assignments: int,
    w2_local_expert_capacity: int | None = None,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()
    num_experts = int(w_up_gate.shape[0])
    (
        x_dispatch,
        _assignment_weights,
        _recv_token_indices,
        local_group_sizes,
        _recv_topk_weights,
        _recv_src_idx,
        _rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        _send_head,
        _num_recv_tokens,
        _is_token_in_rank,
    ) = _moe_mlp_deepep_transport_dispatch_pack(
        x,
        selected_experts,
        combine_weights,
        mesh=mesh,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
    )
    gate_up = _moe_mlp_deepep_transport_gate_up_only(
        x_dispatch,
        local_group_sizes,
        w_up_gate,
        w_down,
        mesh=mesh,
    )
    return _moe_mlp_deepep_transport_w2_only(
        gate_up,
        local_group_sizes,
        w_down,
        mesh=mesh,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )


def _forward_deepep_transport_collapse_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    max_recv_tokens: int,
    max_local_assignments: int,
) -> jax.Array:
    if mesh is None:
        mesh = get_abstract_mesh()
    num_experts = int(w_up_gate.shape[0])
    (
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        local_group_sizes,
        recv_topk_weights,
        _recv_src_idx,
        _rank_prefix_matrix,
        _channel_prefix_matrix,
        _recv_channel_prefix_matrix,
        _send_head,
        num_recv_tokens,
        _is_token_in_rank,
    ) = _moe_mlp_deepep_transport_dispatch_pack(
        x,
        selected_experts,
        combine_weights,
        mesh=mesh,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
    )
    out_dispatch = _moe_mlp_deepep_transport_local_compute(
        x_dispatch,
        local_group_sizes,
        w_up_gate,
        w_down,
        mesh=mesh,
    )
    return _moe_mlp_deepep_transport_collapse_only(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        recv_topk_weights,
        num_recv_tokens,
        mesh=mesh,
    )


def _moe_mlp_ep_ring_local_prefix_counts(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_prefix_counts,
        scatter_fn=_scatter_with_add,
    )


def _moe_mlp_ep_ring_local_vector_prefix(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_vector_prefix,
        scatter_fn=_scatter_with_add,
    )


def _moe_mlp_ep_ring_local_onehot_counts(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_onehot_counts,
        scatter_fn=_scatter_with_add,
    )


def _moe_mlp_ep_ring_local_padded_take(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_padded_take,
        scatter_fn=_scatter_with_add,
    )


def _moe_mlp_ep_ring_local_segment_sum(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_topk,
        scatter_fn=_scatter_with_segment_sum,
    )


def _moe_mlp_ep_ring_local_sorted_segment_sum(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_topk,
        scatter_fn=_scatter_with_sorted_segment_sum,
    )


def _moe_mlp_ep_ring_local_prefix_segment_sum(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_prefix_counts,
        scatter_fn=_scatter_with_segment_sum,
    )


def _moe_mlp_ep_ring_local_vector_sorted_segment_sum(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_vector_prefix,
        scatter_fn=_scatter_with_sorted_segment_sum,
    )


def _moe_mlp_ep_ring_local_owner_local_scatter(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_onehot_counts,
        scatter_fn=_scatter_with_owner_local_add,
    )


def _moe_mlp_ep_ring_local_lax_scatter(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_onehot_counts,
        scatter_fn=_scatter_with_lax_scatter_add,
    )


def _moe_mlp_ep_ring_local_take_segment_bwd(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_onehot_counts,
        scatter_fn=_scatter_with_add,
        take_fn=_take_with_segment_sum_bwd,
    )


def _moe_mlp_ep_ring_local_take_sorted_segment_bwd(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_onehot_counts,
        scatter_fn=_scatter_with_add,
        take_fn=_take_with_sorted_segment_sum_bwd,
    )


def _moe_mlp_ep_ring_local_owner_local_take(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_owner_local_take,
        scatter_fn=_scatter_with_add,
    )


def _moe_mlp_ep_ring_local_weight_cast(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_prefix_counts,
        scatter_fn=_scatter_with_add,
        gather_weight_dtype=x_local.dtype,
    )


def _moe_mlp_ep_ring_local_narrow_meta(
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
    return _moe_mlp_ep_ring_local_variant(
        x_local,
        selected_experts_local,
        combine_weights_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        compact_fn=_compact_local_assignments_prefix_counts,
        scatter_fn=_scatter_with_add,
        gather_weight_dtype=x_local.dtype,
        gather_expert_dtype=jnp.uint16,
    )


def _moe_mlp_ring_variant(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    local_ring_fn: Callable[..., tuple[jax.Array, jax.Array]],
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
                local_ring_fn,
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


def _moe_mlp_prefix_counts(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_prefix_counts,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_vector_prefix(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_vector_prefix,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_onehot_counts(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_onehot_counts,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_padded_take(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_padded_take,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_segment_sum(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_segment_sum,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_sorted_segment_sum(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_sorted_segment_sum,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_prefix_segment_sum(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_prefix_segment_sum,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_vector_sorted_segment_sum(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_vector_sorted_segment_sum,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_owner_local_scatter(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_owner_local_scatter,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_lax_scatter(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_lax_scatter,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_take_segment_bwd(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_take_segment_bwd,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_take_sorted_segment_bwd(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_take_sorted_segment_bwd,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_owner_local_take(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_owner_local_take,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_weight_cast(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_weight_cast,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_narrow_meta(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR,
) -> jax.Array:
    return _moe_mlp_ring_variant(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        local_ring_fn=_moe_mlp_ep_ring_local_narrow_meta,
        mesh=mesh,
        capacity_factor=capacity_factor,
    )


def _moe_mlp_ep_ring_local_packed_return(
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
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)
        local_count = jnp.sum(local_mask, dtype=jnp.int32)
        dropped_local = jnp.maximum(local_count - local_capacity, 0)
        valid = jnp.arange(local_capacity, dtype=jnp.int32) < local_count
        valid_weight = valid.astype(jnp.float32)

        local_expert = jnp.where(local_mask, local_expert, 0)
        flat_pos = jnp.arange(assignments, dtype=jnp.int32)
        order_key = local_expert * assignments + flat_pos
        max_order_key = local_experts * assignments
        selection_key = jnp.where(local_mask, max_order_key - order_key, -1)
        _, local_idx = jax.lax.top_k(selection_key, local_capacity)

        token_local = jnp.take(token_flat, local_idx, axis=0)
        expert_local = jnp.take(local_expert, local_idx, axis=0)
        weight_local = jnp.take(weight_flat, local_idx, axis=0).astype(x_local.dtype)

        x_take = jnp.take(x_global, token_local, axis=0)
        x_dispatch = jnp.where(valid[:, None], x_take, jnp.zeros_like(x_take))
        weight_dispatch = jnp.where(valid, weight_local, jnp.zeros_like(weight_local))
        expert_local = jnp.where(valid, expert_local, 0)

    group_sizes = jnp.bincount(expert_local, weights=valid_weight, length=local_experts).astype(jnp.int32)
    group_sizes = group_sizes.at[-1].add(local_capacity - jnp.sum(group_sizes, dtype=jnp.int32))

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)

    with jax.named_scope("scatter"):
        tokens_per_shard = x_local.shape[0]
        owner_shard = token_local // tokens_per_shard
        owner_token = token_local % tokens_per_shard
        out_weighted = out_dispatch * weight_dispatch[:, None]
        per_owner_capacity = max(1, int(np.ceil(capacity_factor * local_capacity / ep_size)))
        packed_out, packed_token, packed_valid, dropped_return = _pack_by_shard(
            out_weighted,
            owner_token,
            owner_shard,
            num_shards=ep_size,
            capacity=per_owner_capacity,
        )

        returned_out = jax.lax.all_to_all(packed_out, "expert", split_axis=0, concat_axis=0, tiled=False)
        returned_token = jax.lax.all_to_all(packed_token, "expert", split_axis=0, concat_axis=0, tiled=False)
        returned_valid = jax.lax.all_to_all(packed_valid, "expert", split_axis=0, concat_axis=0, tiled=False)

        out_local = (
            jnp.zeros_like(x_local)
            .at[returned_token]
            .add(
                jnp.where(returned_valid[..., None], returned_out, jnp.zeros_like(returned_out)),
                mode="drop",
            )
        )
        dropped_total = jax.lax.psum(dropped_local + dropped_return, ("data", "expert"))
    return out_local, dropped_total


def _moe_mlp_ep_stream_ring_local(
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
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    ep_size = num_experts // local_experts
    expert_axis = jax.lax.axis_index("expert")
    expert_start = expert_axis * local_experts

    tokens_per_chunk = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_chunk = tokens_per_chunk * topk
    local_capacity = int(np.ceil(capacity_factor * assignments_per_chunk / ep_size))
    local_capacity = max(local_experts, local_capacity)

    next_perm = [(src, (src + 1) % ep_size) for src in range(ep_size)]

    def ring_step(carry, _):
        x_chunk, selected_chunk, weight_chunk, out_chunk, dropped_acc = carry

        expert_flat = selected_chunk.reshape(assignments_per_chunk)
        weight_flat = weight_chunk.reshape(assignments_per_chunk)
        local_expert = expert_flat - expert_start
        local_mask = jnp.logical_and(local_expert >= 0, local_expert < local_experts)

        token_local, weight_dispatch, x_dispatch, group_sizes, dropped_local = _compact_local_assignments_topk(
            x_chunk,
            local_expert,
            local_mask,
            weight_flat,
            local_experts=local_experts,
            local_capacity=local_capacity,
            topk=topk,
        )

        w13_out = ragged_dot(x_dispatch, moe_w13_local, group_sizes)
        moe_dim = _moe_intermediate_dim_from_w13_out(w13_out, moe_w2_local)
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out_dispatch = ragged_dot(activation_fn(gate) * up, moe_w2_local, group_sizes)
        out_chunk = out_chunk.at[token_local].add(out_dispatch * weight_dispatch[:, None], mode="drop")

        return (
            jax.lax.ppermute(x_chunk, "expert", next_perm),
            jax.lax.ppermute(selected_chunk, "expert", next_perm),
            jax.lax.ppermute(weight_chunk, "expert", next_perm),
            jax.lax.ppermute(out_chunk, "expert", next_perm),
            dropped_acc + dropped_local,
        ), None

    init_carry = (
        x_local,
        selected_experts_local,
        combine_weights_local,
        jnp.zeros_like(x_local),
        jnp.array(0, dtype=jnp.int32),
    )
    final_carry, _ = jax.lax.scan(ring_step, init_carry, xs=None, length=ep_size)
    _, _, _, out_local, dropped_local_total = final_carry
    dropped_total = jax.lax.psum(dropped_local_total, ("data", "expert"))
    return out_local, dropped_total


def _moe_mlp_stream_ring(
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
                _moe_mlp_ep_stream_ring_local,
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


def _moe_mlp_packed_return(
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
                _moe_mlp_ep_ring_local_packed_return,
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
    if kernel == "shared_mlp_only_probe":
        return _shared_mlp(x, shared_w13, shared_w2)
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
    elif kernel == "cumsum":
        routed = _moe_mlp_cumsum(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "packed_return":
        routed = _moe_mlp_packed_return(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "stream_ring":
        routed = _moe_mlp_stream_ring(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "ragged_a2a":
        routed = _moe_mlp_ragged_a2a(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_layout_ragged_a2a":
        routed = _moe_mlp_ragged_a2a_deepep_layout(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_identity":
        routed = _moe_mlp_deepep_transport_identity(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_assignments_identity":
        routed = _moe_mlp_deepep_transport_assignments_identity(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_first_ragged_dot_probe":
        routed = _moe_mlp_deepep_transport_first_ragged_dot_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_gate_probe":
        routed = _moe_mlp_deepep_transport_gate_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_second_ragged_dot_probe":
        routed = _moe_mlp_deepep_transport_second_ragged_dot_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "deepep_transport_w13_only_probe":
        mesh = get_abstract_mesh()
        num_experts = int(w_up_gate.shape[0])
        max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(
            selected_experts,
            mesh=mesh,
            num_experts=num_experts,
        )
        routed = _forward_deepep_transport_w13_only_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        )
    elif kernel == "deepep_transport_w2_only_probe":
        mesh = get_abstract_mesh()
        num_experts = int(w_up_gate.shape[0])
        max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(
            selected_experts,
            mesh=mesh,
            num_experts=num_experts,
        )
        routed = _forward_deepep_transport_w2_only_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        )
    elif kernel == "deepep_transport_local_compute_only_probe":
        mesh = get_abstract_mesh()
        num_experts = int(w_up_gate.shape[0])
        max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(
            selected_experts,
            mesh=mesh,
            num_experts=num_experts,
        )
        routed = _forward_deepep_transport_local_compute_only_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        )
    elif kernel == "deepep_transport_collapse_only_probe":
        mesh = get_abstract_mesh()
        num_experts = int(w_up_gate.shape[0])
        max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(
            selected_experts,
            mesh=mesh,
            num_experts=num_experts,
        )
        routed = _forward_deepep_transport_collapse_only_probe(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        )
    elif kernel == "deepep_transport":
        routed = _moe_mlp_deepep_transport(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "prefix_counts":
        routed = _moe_mlp_prefix_counts(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "vector_prefix":
        routed = _moe_mlp_vector_prefix(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "onehot_counts":
        routed = _moe_mlp_onehot_counts(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "padded_take":
        routed = _moe_mlp_padded_take(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "segment_sum":
        routed = _moe_mlp_segment_sum(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "sorted_segment_sum":
        routed = _moe_mlp_sorted_segment_sum(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "prefix_segment_sum":
        routed = _moe_mlp_prefix_segment_sum(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "vector_sorted_segment_sum":
        routed = _moe_mlp_vector_sorted_segment_sum(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "owner_local_scatter":
        routed = _moe_mlp_owner_local_scatter(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "lax_scatter":
        routed = _moe_mlp_lax_scatter(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "take_segment_bwd":
        routed = _moe_mlp_take_segment_bwd(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "take_sorted_segment_bwd":
        routed = _moe_mlp_take_sorted_segment_bwd(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "owner_local_take":
        routed = _moe_mlp_owner_local_take(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "weight_cast":
        routed = _moe_mlp_weight_cast(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    elif kernel == "narrow_meta":
        routed = _moe_mlp_narrow_meta(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    if kernel in {
        "deepep_transport_identity",
        "deepep_transport_assignments_identity",
        "deepep_transport_first_ragged_dot_probe",
        "deepep_transport_gate_probe",
        "deepep_transport_second_ragged_dot_probe",
        "deepep_transport_w13_only_probe",
        "deepep_transport_w2_only_probe",
        "deepep_transport_local_compute_only_probe",
    }:
        return routed
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


def _time_deepep_transport_staged_forward(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> float:
    _prewarm_deepep_transport_local_compute(x, selected_experts, w_up_gate, w_down, shared_w13, shared_w2)

    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    dispatch_pack = jax.jit(partial(_moe_mlp_deepep_transport_dispatch_pack, mesh=mesh, num_experts=num_experts))
    local_compute = jax.jit(partial(_moe_mlp_deepep_transport_local_compute, mesh=mesh))
    collapse_combine = jax.jit(partial(_moe_mlp_deepep_transport_collapse_combine, mesh=mesh))
    shared_mlp = jax.jit(_shared_mlp)

    def run_once() -> jax.Array:
        (
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            local_group_sizes,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        out_dispatch = local_compute(x_dispatch, local_group_sizes, w_up_gate, w_down)
        routed = collapse_combine(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        )
        return routed + shared_mlp(x, shared_w13, shared_w2)

    jax.block_until_ready(run_once())
    for _ in range(warmup):
        jax.block_until_ready(run_once())
    start = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(run_once())
    return (time.perf_counter() - start) / iters


def _prewarm_deepep_transport_local_compute(
    x: jax.Array,
    selected_experts: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_local_assignments: int | None = None,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
) -> None:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    expert_axis_size = grug_moe_lib._mesh_axis_size(mesh, "expert")
    local_assignments = (
        x.shape[0] * selected_experts.shape[1] if max_local_assignments is None else max_local_assignments
    )
    local_compute = jax.jit(
        partial(
            _moe_mlp_deepep_transport_local_compute,
            mesh=mesh,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
        )
    )
    shared_mlp = jax.jit(_shared_mlp)
    dispatch_spec = jax.ShapeDtypeStruct(
        (expert_axis_size * local_assignments, x.shape[1]),
        x.dtype,
        sharding=NamedSharding(mesh, P("expert", None)),
    )
    group_sizes_spec = jax.ShapeDtypeStruct(
        (num_experts,),
        jnp.int32,
        sharding=NamedSharding(mesh, P("expert")),
    )

    local_compute.lower(dispatch_spec, group_sizes_spec, w_up_gate, w_down).compile()
    jax.block_until_ready(shared_mlp(x, shared_w13, shared_w2))


def _forward_deepep_transport_capped(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed, shared = _forward_deepep_transport_capped_split(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    return routed + shared


def _forward_deepep_transport_capped_split(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> tuple[jax.Array, jax.Array]:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp(x, shared_w13, shared_w2)
    return routed, shared


def _forward_deepep_transport_capped_shared_detached_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed, shared = _forward_deepep_transport_capped_split(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    return routed + jax.lax.stop_gradient(shared)


def _forward_deepep_transport_capped_routed_detached_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed, shared = _forward_deepep_transport_capped_split(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    return jax.lax.stop_gradient(routed) + shared


def _forward_deepep_transport_capped_shared_dx_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp(x, jax.lax.stop_gradient(shared_w13), jax.lax.stop_gradient(shared_w2))
    return routed + shared


def _forward_deepep_transport_capped_shared_dw_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp(jax.lax.stop_gradient(x), shared_w13, shared_w2)
    return routed + shared


def _forward_deepep_transport_capped_shared_dw_psum_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp_psum_only_bwd(jax.lax.stop_gradient(x), shared_w13, shared_w2)
    return routed + shared


def _forward_deepep_transport_capped_shared_dw_psum_splitvjp_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    x_stop = jax.lax.stop_gradient(x)
    shared = _shared_mlp_psum_only_w13_bwd(
        x_stop,
        shared_w13,
        jax.lax.stop_gradient(shared_w2),
    ) + _shared_mlp_psum_only_w2_bwd(
        x_stop,
        jax.lax.stop_gradient(shared_w13),
        shared_w2,
    )
    return routed + shared


def _forward_deepep_transport_capped_shared_dw13_psum_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp_psum_only_bwd(
        jax.lax.stop_gradient(x),
        shared_w13,
        jax.lax.stop_gradient(shared_w2),
    )
    return routed + shared


def _forward_deepep_transport_capped_shared_dw2_psum_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp_psum_only_bwd(
        jax.lax.stop_gradient(x),
        jax.lax.stop_gradient(shared_w13),
        shared_w2,
    )
    return routed + shared


def _forward_deepep_transport_capped_shared_dw13_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp(jax.lax.stop_gradient(x), shared_w13, jax.lax.stop_gradient(shared_w2))
    return routed + shared


def _forward_deepep_transport_capped_shared_dw2_only_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> jax.Array:
    routed = _moe_mlp_deepep_transport(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    shared = _shared_mlp(jax.lax.stop_gradient(x), jax.lax.stop_gradient(shared_w13), shared_w2)
    return routed + shared


def _split_coupled_square_loss(routed: jax.Array, shared: jax.Array) -> jax.Array:
    routed_loss = jnp.mean(jnp.square((routed + jax.lax.stop_gradient(shared)).astype(jnp.float32)))
    shared_loss = jnp.mean(jnp.square((jax.lax.stop_gradient(routed) + shared).astype(jnp.float32)))
    return routed_loss + shared_loss


def _mean_square_loss(output: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(output.astype(jnp.float32)))


def _separate_branch_square_loss_and_grads(
    routed: jax.Array,
    routed_pullback: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
    shared: jax.Array,
    shared_pullback: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    total = routed + shared
    loss, loss_pullback = jax.vjp(_mean_square_loss, total)
    (grad_total,) = loss_pullback(jnp.array(1.0, dtype=loss.dtype))
    dx_routed, dw_up_gate, dw_down = routed_pullback(grad_total)
    dx_shared, dshared_w13, dshared_w2 = shared_pullback(grad_total)
    return loss, (dx_routed + dx_shared, dw_up_gate, dw_down, dshared_w13, dshared_w2)


def _make_deepep_transport_capped_prewarmed_separate_bwd_grad_fn(
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    *,
    max_recv_tokens: int,
    max_local_assignments: int,
    w13_local_expert_capacity: int | None = None,
    w2_local_expert_capacity: int | None = None,
    collapse_impl: CollapseImpl = "segment_sum",
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, tuple[jax.Array, ...]]]:
    def routed_forward(x_in: jax.Array, w_up_gate_in: jax.Array, w_down_in: jax.Array) -> jax.Array:
        return _moe_mlp_deepep_transport(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )

    def shared_forward(x_in: jax.Array, shared_w13_in: jax.Array, shared_w2_in: jax.Array) -> jax.Array:
        return _shared_mlp(x_in, shared_w13_in, shared_w2_in)

    def grad_fn(
        x_in: jax.Array,
        w_up_gate_in: jax.Array,
        w_down_in: jax.Array,
        shared_w13_in: jax.Array,
        shared_w2_in: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        routed, routed_pullback = jax.vjp(routed_forward, x_in, w_up_gate_in, w_down_in)
        shared, shared_pullback = jax.vjp(shared_forward, x_in, shared_w13_in, shared_w2_in)
        return _separate_branch_square_loss_and_grads(routed, routed_pullback, shared, shared_pullback)

    return grad_fn


def _make_deepep_transport_probe_forward_runner(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> tuple[Callable[..., jax.Array], tuple[jax.Array, ...]]:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None

    if probe_kernel == "deepep_transport_identity":
        return (
            partial(
                _moe_mlp_deepep_transport_identity,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_assignments_identity":
        return (
            partial(
                _moe_mlp_deepep_transport_assignments_identity,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_first_ragged_dot_probe":
        return (
            partial(
                _moe_mlp_deepep_transport_first_ragged_dot_probe,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_gate_probe":
        return (
            partial(
                _moe_mlp_deepep_transport_gate_probe,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_second_ragged_dot_probe":
        return (
            partial(
                _moe_mlp_deepep_transport_second_ragged_dot_probe,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_w13_only_probe":
        dispatch_pack = jax.jit(
            partial(
                _moe_mlp_deepep_transport_dispatch_pack,
                mesh=mesh,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            )
        )
        (
            x_dispatch,
            _assignment_weights,
            _recv_token_indices,
            local_group_sizes,
            _recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            _num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        jax.block_until_ready(x_dispatch)
        return (
            partial(
                _moe_mlp_deepep_transport_w13_only,
                mesh=mesh,
                w13_local_expert_capacity=w13_local_expert_capacity,
            ),
            (x_dispatch, local_group_sizes, w_up_gate),
        )
    if probe_kernel == "deepep_transport_w2_only_probe":
        dispatch_pack = jax.jit(
            partial(
                _moe_mlp_deepep_transport_dispatch_pack,
                mesh=mesh,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            )
        )
        gate_up_only = jax.jit(partial(_moe_mlp_deepep_transport_gate_up_only, mesh=mesh))
        (
            x_dispatch,
            _assignment_weights,
            _recv_token_indices,
            local_group_sizes,
            _recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            _num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        gate_up = gate_up_only(x_dispatch, local_group_sizes, w_up_gate, w_down)
        jax.block_until_ready(gate_up)
        return (
            partial(
                _moe_mlp_deepep_transport_w2_only,
                mesh=mesh,
                w2_local_expert_capacity=w2_local_expert_capacity,
            ),
            (gate_up, local_group_sizes, w_down),
        )
    if probe_kernel == "deepep_transport_local_compute_only_probe":
        return (
            partial(
                _forward_deepep_transport_local_compute_only_probe,
                mesh=mesh,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
                w13_local_expert_capacity=w13_local_expert_capacity,
                w2_local_expert_capacity=w2_local_expert_capacity,
            ),
            (x, selected_experts, combine_weights, w_up_gate, w_down),
        )
    if probe_kernel == "deepep_transport_collapse_only_probe":
        dispatch_pack = jax.jit(
            partial(
                _moe_mlp_deepep_transport_dispatch_pack,
                mesh=mesh,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            )
        )
        local_compute = jax.jit(partial(_moe_mlp_deepep_transport_local_compute, mesh=mesh))
        (
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            local_group_sizes,
            recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        out_dispatch = local_compute(x_dispatch, local_group_sizes, w_up_gate, w_down)
        jax.block_until_ready(out_dispatch)
        return (
            partial(_moe_mlp_deepep_transport_collapse_only, mesh=mesh),
            (out_dispatch, assignment_weights, recv_token_indices, recv_topk_weights, num_recv_tokens),
        )
    if probe_kernel == "deepep_transport_combine_only_probe":
        dispatch_pack = jax.jit(
            partial(
                _moe_mlp_deepep_transport_dispatch_pack,
                mesh=mesh,
                num_experts=num_experts,
                max_recv_tokens=max_recv_tokens,
                max_local_assignments=max_local_assignments,
            )
        )
        local_compute = jax.jit(partial(_moe_mlp_deepep_transport_local_compute, mesh=mesh))
        collapse_only = jax.jit(partial(_moe_mlp_deepep_transport_collapse_only, mesh=mesh))
        (
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            local_group_sizes,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        out_dispatch = local_compute(x_dispatch, local_group_sizes, w_up_gate, w_down)
        recv_out = collapse_only(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            recv_topk_weights,
            num_recv_tokens,
        )
        jax.block_until_ready(recv_out)
        return (
            partial(_moe_mlp_deepep_transport_combine_only, mesh=mesh),
            (
                recv_out,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                num_recv_tokens,
                is_token_in_rank,
            ),
        )
    raise ValueError(f"Unsupported DeepEP probe kernel for exact-cap timing: {probe_kernel}")


def _time_deepep_transport_probe_forward(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> float:
    probe_fn, probe_args = _make_deepep_transport_probe_forward_runner(
        probe_kernel,
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        w13_expert_padded=w13_expert_padded,
        w2_expert_padded=w2_expert_padded,
    )

    return _time_fn(
        probe_fn,
        *probe_args,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_probe_forward(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> float:
    probe_fn, probe_args = _make_deepep_transport_probe_forward_runner(
        probe_kernel,
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        w13_expert_padded=w13_expert_padded,
        w2_expert_padded=w2_expert_padded,
    )
    return _profile_fn(
        probe_fn,
        *probe_args,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _make_deepep_transport_probe_forward_backward_runner(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> tuple[Callable[[], jax.Array], tuple[()]]:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None

    dispatch_pack = partial(
        _moe_mlp_deepep_transport_dispatch_pack,
        mesh=mesh,
        num_experts=num_experts,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
    )
    local_compute = partial(
        _moe_mlp_deepep_transport_local_compute,
        mesh=mesh,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )
    collapse_only = partial(_moe_mlp_deepep_transport_collapse_only, mesh=mesh)
    combine_only = partial(_moe_mlp_deepep_transport_combine_only, mesh=mesh)

    if probe_kernel == "deepep_transport_combine_bwd_cached_dispatch_probe":
        (
            x_dispatch,
            assignment_weights,
            recv_token_indices,
            local_group_sizes,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        out_dispatch = local_compute(x_dispatch, local_group_sizes, w_up_gate, w_down)
        recv_out = collapse_only(
            out_dispatch,
            assignment_weights,
            recv_token_indices,
            recv_topk_weights,
            num_recv_tokens,
        )
        out_local, pullback = jax.vjp(
            combine_only,
            recv_out,
            recv_topk_weights,
            recv_src_idx,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            send_head,
            num_recv_tokens,
            is_token_in_rank,
        )
        cotangent = jnp.ones_like(out_local, dtype=out_local.dtype)
        jax.block_until_ready(out_local)

        def run() -> jax.Array:
            with jax.named_scope("combine_bwd_cached_dispatch_probe"):
                grad_recv_out, *_ = pullback(cotangent)
                return grad_recv_out

        return run, tuple()

    if probe_kernel == "deepep_transport_w13_only_bwd_probe":
        (
            x_dispatch,
            _assignment_weights,
            _recv_token_indices,
            local_group_sizes,
            _recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            _num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)

        def w13_only_wrt(
            x_dispatch_in: jax.Array,
            w_up_gate_in: jax.Array,
        ) -> jax.Array:
            return _moe_mlp_deepep_transport_w13_only(
                x_dispatch_in,
                local_group_sizes,
                w_up_gate_in,
                mesh=mesh,
                w13_local_expert_capacity=w13_local_expert_capacity,
            )

        w13_out, pullback = jax.vjp(w13_only_wrt, x_dispatch, w_up_gate)
        cotangent = jnp.ones_like(w13_out, dtype=w13_out.dtype)
        jax.block_until_ready(w13_out)

        def run() -> jax.Array:
            with jax.named_scope("w13_only_bwd_probe"):
                grad_x_dispatch, _ = pullback(cotangent)
                return grad_x_dispatch

        return run, tuple()

    if probe_kernel == "deepep_transport_w2_only_bwd_probe":
        (
            x_dispatch,
            _assignment_weights,
            _recv_token_indices,
            local_group_sizes,
            _recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            _num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)
        gate_up = _moe_mlp_deepep_transport_gate_up_only(
            x_dispatch,
            local_group_sizes,
            w_up_gate,
            w_down,
            mesh=mesh,
        )

        def w2_only_wrt(
            gate_up_in: jax.Array,
            w_down_in: jax.Array,
        ) -> jax.Array:
            return _moe_mlp_deepep_transport_w2_only(
                gate_up_in,
                local_group_sizes,
                w_down_in,
                mesh=mesh,
                w2_local_expert_capacity=w2_local_expert_capacity,
            )

        out_dispatch, pullback = jax.vjp(w2_only_wrt, gate_up, w_down)
        cotangent = jnp.ones_like(out_dispatch, dtype=out_dispatch.dtype)
        jax.block_until_ready(out_dispatch)

        def run() -> jax.Array:
            with jax.named_scope("w2_only_bwd_probe"):
                grad_gate_up, _ = pullback(cotangent)
                return grad_gate_up

        return run, tuple()

    if probe_kernel == "deepep_transport_local_compute_bwd_probe":
        (
            x_dispatch,
            _assignment_weights,
            _recv_token_indices,
            local_group_sizes,
            _recv_topk_weights,
            _recv_src_idx,
            _rank_prefix_matrix,
            _channel_prefix_matrix,
            _recv_channel_prefix_matrix,
            _send_head,
            _num_recv_tokens,
            _is_token_in_rank,
        ) = dispatch_pack(x, selected_experts, combine_weights)

        def local_compute_wrt(
            x_dispatch_in: jax.Array,
            w_up_gate_in: jax.Array,
            w_down_in: jax.Array,
        ) -> jax.Array:
            return local_compute(
                x_dispatch_in,
                local_group_sizes,
                w_up_gate_in,
                w_down_in,
            )

        out_dispatch, pullback = jax.vjp(local_compute_wrt, x_dispatch, w_up_gate, w_down)
        cotangent = jnp.ones_like(out_dispatch, dtype=out_dispatch.dtype)
        jax.block_until_ready(out_dispatch)

        def run() -> jax.Array:
            with jax.named_scope("local_compute_bwd_probe"):
                grad_x_dispatch, _, _ = pullback(cotangent)
                return grad_x_dispatch

        return run, tuple()

    if probe_kernel == "deepep_transport_dispatch_bwd_combine_probe":
        def dispatch_pack_wrt(
            x_in: jax.Array,
            combine_weights_in: jax.Array,
        ):
            return dispatch_pack(x_in, selected_experts, combine_weights_in)

        dispatch_outputs, dispatch_pullback = jax.vjp(dispatch_pack_wrt, x, combine_weights)

        def downstream_from_dispatch(
            x_dispatch: jax.Array,
            assignment_weights: jax.Array,
            recv_token_indices: jax.Array,
            local_group_sizes: jax.Array,
            recv_topk_weights: jax.Array,
            recv_src_idx: jax.Array,
            rank_prefix_matrix: jax.Array,
            channel_prefix_matrix: jax.Array,
            recv_channel_prefix_matrix: jax.Array,
            send_head: jax.Array,
            num_recv_tokens: jax.Array,
            is_token_in_rank: jax.Array,
        ) -> jax.Array:
            out_dispatch = local_compute(x_dispatch, local_group_sizes, w_up_gate, w_down)
            recv_out = collapse_only(
                out_dispatch,
                assignment_weights,
                recv_token_indices,
                recv_topk_weights,
                num_recv_tokens,
            )
            return combine_only(
                recv_out,
                recv_topk_weights,
                recv_src_idx,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                send_head,
                num_recv_tokens,
                is_token_in_rank,
            )

        out_local, downstream_pullback = jax.vjp(downstream_from_dispatch, *dispatch_outputs)
        cotangent = jnp.ones_like(out_local, dtype=out_local.dtype)
        dispatch_output_cotangents = downstream_pullback(cotangent)
        jax.block_until_ready(out_local)

        def run() -> jax.Array:
            with jax.named_scope("dispatch_bwd_combine_probe"):
                grad_x, _grad_combine_weights = dispatch_pullback(dispatch_output_cotangents)
                return grad_x

        return run, tuple()

    raise ValueError(f"Unsupported DeepEP forward_backward probe kernel: {probe_kernel}")


def _time_deepep_transport_probe_forward_backward(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> float:
    probe_fn, probe_args = _make_deepep_transport_probe_forward_backward_runner(
        probe_kernel,
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        w13_expert_padded=w13_expert_padded,
        w2_expert_padded=w2_expert_padded,
    )
    return _time_fn(
        probe_fn,
        *probe_args,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_probe_forward_backward(
    probe_kernel: Kernel,
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
) -> float:
    probe_fn, probe_args = _make_deepep_transport_probe_forward_backward_runner(
        probe_kernel,
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        w13_expert_padded=w13_expert_padded,
        w2_expert_padded=w2_expert_padded,
    )
    return _profile_fn(
        probe_fn,
        *probe_args,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _forward_current_w13_expert_padded(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    w13_local_expert_capacity: int,
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


def _time_current_forward_w13_expert_padded(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    w13_local_expert_capacity = _current_ring_w13_cap_metadata(selected_experts, mesh=mesh, num_experts=num_experts)
    return _time_fn(
        partial(_forward_current_w13_expert_padded, w13_local_expert_capacity=w13_local_expert_capacity),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_current_forward_w13_expert_padded(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    w13_local_expert_capacity = _current_ring_w13_cap_metadata(selected_experts, mesh=mesh, num_experts=num_experts)
    return _profile_fn(
        partial(_forward_current_w13_expert_padded, w13_local_expert_capacity=w13_local_expert_capacity),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_current_forward_backward_w13_expert_padded(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    w13_local_expert_capacity = _current_ring_w13_cap_metadata(selected_experts, mesh=mesh, num_experts=num_experts)

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = _forward_current_w13_expert_padded(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            w13_local_expert_capacity=w13_local_expert_capacity,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _time_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_current_forward_backward_w13_expert_padded(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    w13_local_expert_capacity = _current_ring_w13_cap_metadata(selected_experts, mesh=mesh, num_experts=num_experts)

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = _forward_current_w13_expert_padded(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            w13_local_expert_capacity=w13_local_expert_capacity,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _profile_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_deepep_transport_forward_prewarmed(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> float:
    _prewarm_deepep_transport_local_compute(x, selected_experts, w_up_gate, w_down, shared_w13, shared_w2)
    return _time_fn(
        partial(_forward, "deepep_transport"),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _time_deepep_transport_forward_capped_prewarmed(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )
    return _time_fn(
        partial(
            _forward_deepep_transport_capped,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        ),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_forward_capped_prewarmed(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )
    return _profile_fn(
        partial(
            _forward_deepep_transport_capped,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        ),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_deepep_transport_forward_capped(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    return _time_fn(
        partial(
            _forward_deepep_transport_capped,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
        ),
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _time_deepep_transport_forward_backward_capped_prewarmed(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = _forward_deepep_transport_capped(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _time_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_forward_backward_capped_prewarmed(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = _forward_deepep_transport_capped(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _profile_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
    forward_impl: Callable[..., jax.Array],
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = forward_impl(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _time_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
    forward_impl: Callable[..., jax.Array],
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        y = forward_impl(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return jnp.mean(jnp.square(y.astype(jnp.float32)))

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _profile_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_deepep_transport_forward_backward_capped_prewarmed_split_loss_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        routed, shared = _forward_deepep_transport_capped_split(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return _split_coupled_square_loss(routed, shared)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _time_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_forward_backward_capped_prewarmed_split_loss_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )

    def loss_fn(x_in, w_up_gate_in, w_down_in, shared_w13_in, shared_w2_in):
        routed, shared = _forward_deepep_transport_capped_split(
            x_in,
            selected_experts,
            combine_weights,
            w_up_gate_in,
            w_down_in,
            shared_w13_in,
            shared_w2_in,
            max_recv_tokens=max_recv_tokens,
            max_local_assignments=max_local_assignments,
            w13_local_expert_capacity=w13_local_expert_capacity,
            w2_local_expert_capacity=w2_local_expert_capacity,
            collapse_impl=collapse_impl,
        )
        return _split_coupled_square_loss(routed, shared)

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))
    return _profile_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


def _time_deepep_transport_forward_backward_capped_prewarmed_separate_bwd_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )
    grad_fn = _make_deepep_transport_capped_prewarmed_separate_bwd_grad_fn(
        selected_experts,
        combine_weights,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    return _time_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
    )


def _profile_deepep_transport_forward_backward_capped_prewarmed_separate_bwd_probe(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    shared_w13: jax.Array,
    shared_w2: jax.Array,
    *,
    warmup: int,
    iters: int,
    profile_dir: Path,
    profile_name: str,
    w13_expert_padded: bool = False,
    w2_expert_padded: bool = False,
    collapse_impl: CollapseImpl = "segment_sum",
) -> float:
    mesh = x.sharding.mesh
    num_experts = int(w_up_gate.shape[0])
    max_recv_tokens, max_local_assignments, max_local_expert_assignments = _deepep_transport_exact_cap_metadata(
        selected_experts,
        mesh=mesh,
        num_experts=num_experts,
    )
    w13_local_expert_capacity = max_local_expert_assignments if w13_expert_padded else None
    w2_local_expert_capacity = max_local_expert_assignments if w2_expert_padded else None
    _prewarm_deepep_transport_local_compute(
        x,
        selected_experts,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
    )
    grad_fn = _make_deepep_transport_capped_prewarmed_separate_bwd_grad_fn(
        selected_experts,
        combine_weights,
        max_recv_tokens=max_recv_tokens,
        max_local_assignments=max_local_assignments,
        w13_local_expert_capacity=w13_local_expert_capacity,
        w2_local_expert_capacity=w2_local_expert_capacity,
        collapse_impl=collapse_impl,
    )
    return _profile_fn(
        grad_fn,
        x,
        w_up_gate,
        w_down,
        shared_w13,
        shared_w2,
        warmup=warmup,
        iters=iters,
        profile_dir=profile_dir,
        profile_name=profile_name,
    )


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
    parser.add_argument(
        "--kernel",
        choices=[
            "legacy",
            "current",
            "shared_mlp_only_probe",
            "deepep_transport_capped_prewarmed_shared_detached_probe",
            "deepep_transport_capped_prewarmed_routed_detached_probe",
            "deepep_transport_capped_prewarmed_shared_dx_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw_psum_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw_psum_splitvjp_probe",
            "deepep_transport_capped_prewarmed_shared_dw13_psum_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw2_psum_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw13_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw2_only_probe",
            "deepep_transport_capped_prewarmed_shared_dw_only_probe",
            "deepep_transport_capped_prewarmed_split_loss_probe",
            "deepep_transport_capped_prewarmed_separate_bwd_probe",
            "cumsum",
            "packed_return",
            "stream_ring",
            "ragged_a2a",
            "deepep_layout_ragged_a2a",
            "deepep_transport_identity",
            "deepep_transport_assignments_identity",
            "deepep_transport_first_ragged_dot_probe",
            "deepep_transport_gate_probe",
            "deepep_transport_second_ragged_dot_probe",
            "deepep_transport_w13_only_probe",
            "deepep_transport_w2_only_probe",
            "deepep_transport_local_compute_only_probe",
            "deepep_transport_collapse_only_probe",
            "deepep_transport_combine_only_probe",
            "deepep_transport_w13_only_bwd_probe",
            "deepep_transport_w2_only_bwd_probe",
            "deepep_transport_local_compute_bwd_probe",
            "deepep_transport_combine_bwd_cached_dispatch_probe",
            "deepep_transport_dispatch_bwd_combine_probe",
            "deepep_transport",
            "deepep_transport_prewarmed",
            "deepep_transport_capped",
            "deepep_transport_capped_prewarmed",
            "deepep_transport_staged",
            "prefix_counts",
            "vector_prefix",
            "onehot_counts",
            "padded_take",
            "segment_sum",
            "sorted_segment_sum",
            "prefix_segment_sum",
            "vector_sorted_segment_sum",
            "owner_local_scatter",
            "lax_scatter",
            "take_segment_bwd",
            "take_sorted_segment_bwd",
            "owner_local_take",
            "weight_cast",
            "narrow_meta",
            "both",
        ],
        default="both",
    )
    parser.add_argument("--ep-list", type=str, default="1,2,4,8")
    parser.add_argument("--capacity-factor", type=float, default=grug_moe_lib._DEFAULT_EP_CAPACITY_FACTOR)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-equivalence", action="store_true")
    parser.add_argument("--profile-root", type=Path, default=None)
    parser.add_argument("--w13-out-first", action="store_true")
    parser.add_argument("--w2-out-first", action="store_true")
    parser.add_argument("--w13-expert-padded", action="store_true")
    parser.add_argument("--w2-expert-padded", action="store_true")
    parser.add_argument("--shared-mlp-explicit-bwd", action="store_true")
    parser.add_argument("--shared-mlp-fused-dw-psum-bwd", action="store_true")
    parser.add_argument("--shared-mlp-gradx-first-bwd", action="store_true")
    parser.add_argument("--shared-mlp-fast-accum", action="store_true")
    parser.add_argument("--combine-fast-accum", action="store_true")
    parser.add_argument("--deepep-dispatch-num-sms", type=int)
    parser.add_argument("--deepep-dispatch-num-max-send-tokens", type=int)
    parser.add_argument("--deepep-dispatch-num-max-recv-tokens", type=int)
    parser.add_argument("--deepep-combine-num-sms", type=int)
    parser.add_argument("--deepep-combine-num-max-send-tokens", type=int)
    parser.add_argument("--deepep-combine-num-max-recv-tokens", type=int)
    parser.add_argument(
        "--deepep-collapse-impl",
        choices=["segment_sum", "sorted_segment_sum", "scatter_add", "lax_scatter"],
        default="segment_sum",
    )
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
    _set_shared_mlp_explicit_bwd(args.shared_mlp_explicit_bwd)
    _set_shared_mlp_fused_dw_psum_bwd(args.shared_mlp_fused_dw_psum_bwd)
    _set_shared_mlp_gradx_first_bwd(args.shared_mlp_gradx_first_bwd)
    _set_shared_mlp_fast_accum(args.shared_mlp_fast_accum)
    _set_combine_fast_accum(args.combine_fast_accum)
    eps = [int(tok.strip()) for tok in args.ep_list.split(",") if tok.strip()]
    kernels: list[Kernel] = ["legacy", "current"] if args.kernel == "both" else [args.kernel]

    if args.profile_root is not None:
        if len(eps) != 1:
            raise ValueError("--profile-root requires exactly one EP value in --ep-list")
        if len(kernels) != 1:
            raise ValueError("--profile-root requires exactly one kernel via --kernel")
        if args.check_equivalence:
            raise ValueError("--profile-root cannot be combined with --check-equivalence")

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

    w13_shape = (
        (args.experts, 2 * args.mlp_dim, args.hidden)
        if args.w13_out_first
        else (args.experts, args.hidden, 2 * args.mlp_dim)
    )
    w2_shape = (
        (args.experts, args.hidden, args.mlp_dim)
        if args.w2_out_first
        else (args.experts, args.mlp_dim, args.hidden)
    )
    w_up_gate = jax.random.normal(key_w13, w13_shape, dtype=dtype)
    w_down = jax.random.normal(key_w2, w2_shape, dtype=dtype)
    shared_w13 = jax.random.normal(key_sw13, (args.hidden, 2 * args.shared_expert_dim), dtype=dtype)
    shared_w2 = jax.random.normal(key_sw2, (args.shared_expert_dim, args.hidden), dtype=dtype)

    _print0(f"devices={jax.devices()}")
    _print0(
        "shape "
        f"tokens={args.tokens} hidden={args.hidden} mlp_dim={args.mlp_dim} experts={args.experts} "
        f"topk={args.topk} shared_expert_dim={args.shared_expert_dim} dtype={dtype} "
        f"distribution={args.distribution} bench_pass={args.bench_pass} capacity_factor={args.capacity_factor} "
        f"w13_out_first={args.w13_out_first} w2_out_first={args.w2_out_first} "
        f"w13_expert_padded={args.w13_expert_padded} "
        f"w2_expert_padded={args.w2_expert_padded} "
        f"shared_mlp_explicit_bwd={args.shared_mlp_explicit_bwd} "
        f"shared_mlp_fused_dw_psum_bwd={args.shared_mlp_fused_dw_psum_bwd} "
        f"shared_mlp_gradx_first_bwd={args.shared_mlp_gradx_first_bwd} "
        f"shared_mlp_fast_accum={args.shared_mlp_fast_accum} "
        f"combine_fast_accum={args.combine_fast_accum} "
        f"deepep_collapse_impl={args.deepep_collapse_impl}"
    )

    for ep_size in eps:
        deepep_dispatch_config = _deepep_dispatch_config_from_args(args, ep_size)
        deepep_combine_config = _deepep_combine_config_from_args(args, ep_size)
        set_intranode_config_overrides(
            dispatch_config=deepep_dispatch_config,
            combine_config=deepep_combine_config,
        )
        _print0(
            "DEEPEP_TRANSPORT_CONFIG "
            f"ep={ep_size} dispatch={_deepep_config_payload(deepep_dispatch_config)} "
            f"combine={_deepep_config_payload(deepep_combine_config)}"
        )
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
                _, legacy_grads = jax.jit(_loss_and_grads, static_argnums=0)(
                    "legacy",
                    x_sharded,
                    selected_sharded,
                    weights_sharded,
                    w13_sharded,
                    w2_sharded,
                    shared_w13_sharded,
                    shared_w2_sharded,
                )
                _, current_grads = jax.jit(_loss_and_grads, static_argnums=0)(
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
                if args.bench_pass == "forward" and kernel in {
                    "deepep_transport_w13_only_bwd_probe",
                    "deepep_transport_w2_only_bwd_probe",
                    "deepep_transport_combine_bwd_cached_dispatch_probe",
                    "deepep_transport_dispatch_bwd_combine_probe",
                }:
                    raise ValueError(f"{kernel} currently supports only --bench-pass=forward_backward")
                forward_fn = partial(_forward, kernel)
                grad_fn = partial(_loss_and_grads, kernel)
                profile_name = f"moe_{kernel}_ep{ep_size}_{args.bench_pass}"
                if args.bench_pass == "forward":
                    if kernel == "current" and args.w13_expert_padded:
                        if args.profile_root is not None:
                            dt = _profile_current_forward_w13_expert_padded(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                            )
                        else:
                            dt = _time_current_forward_w13_expert_padded(
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
                    elif kernel in {
                        "deepep_transport_identity",
                        "deepep_transport_assignments_identity",
                        "deepep_transport_first_ragged_dot_probe",
                        "deepep_transport_gate_probe",
                        "deepep_transport_second_ragged_dot_probe",
                        "deepep_transport_w13_only_probe",
                        "deepep_transport_w2_only_probe",
                        "deepep_transport_local_compute_only_probe",
                        "deepep_transport_collapse_only_probe",
                        "deepep_transport_combine_only_probe",
                    }:
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_probe_forward(
                                kernel,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                            )
                        else:
                            dt = _time_deepep_transport_probe_forward(
                                kernel,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                            )
                    elif kernel == "deepep_transport_prewarmed":
                        if args.profile_root is not None:
                            raise ValueError("deepep_transport_prewarmed does not yet support --profile-root")
                        dt = _time_deepep_transport_forward_prewarmed(
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
                    elif kernel == "deepep_transport_capped":
                        if args.profile_root is not None:
                            raise ValueError("deepep_transport_capped does not yet support --profile-root")
                        dt = _time_deepep_transport_forward_capped(
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
                    elif kernel == "deepep_transport_capped_prewarmed":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_capped_prewarmed(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_capped_prewarmed(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_staged":
                        if args.profile_root is not None:
                            raise ValueError("deepep_transport_staged does not yet support --profile-root")
                        dt = _time_deepep_transport_staged_forward(
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
                    elif args.profile_root is not None:
                        dt = _profile_fn(
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
                            profile_dir=args.profile_root,
                            profile_name=profile_name,
                        )
                    else:
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
                    if kernel in {"deepep_transport_prewarmed", "deepep_transport_capped", "deepep_transport_staged"}:
                        raise ValueError(f"{kernel} currently supports only --bench-pass=forward")
                    if kernel in {
                        "deepep_transport_w13_only_bwd_probe",
                        "deepep_transport_w2_only_bwd_probe",
                        "deepep_transport_local_compute_bwd_probe",
                        "deepep_transport_combine_bwd_cached_dispatch_probe",
                        "deepep_transport_dispatch_bwd_combine_probe",
                    }:
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_probe_forward_backward(
                                kernel,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                            )
                        else:
                            dt = _time_deepep_transport_probe_forward_backward(
                                kernel,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                            )
                    elif kernel == "current" and args.w13_expert_padded:
                        if args.profile_root is not None:
                            dt = _profile_current_forward_backward_w13_expert_padded(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                            )
                        else:
                            dt = _time_current_forward_backward_w13_expert_padded(
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
                    elif kernel == "deepep_transport_capped_prewarmed":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_detached_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_detached_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_detached_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_routed_detached_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_routed_detached_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_routed_detached_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dx_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dx_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dx_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw_psum_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw13_psum_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw13_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw13_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw_psum_splitvjp_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_psum_splitvjp_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw_psum_splitvjp_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw2_psum_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw2_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw2_psum_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw13_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw13_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw13_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_shared_dw2_only_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw2_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_detach_probe(
                                _forward_deepep_transport_capped_shared_dw2_only_probe,
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_split_loss_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_split_loss_probe(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_split_loss_probe(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif kernel == "deepep_transport_capped_prewarmed_separate_bwd_probe":
                        if args.profile_root is not None:
                            dt = _profile_deepep_transport_forward_backward_capped_prewarmed_separate_bwd_probe(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                profile_dir=args.profile_root,
                                profile_name=profile_name,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                        else:
                            dt = _time_deepep_transport_forward_backward_capped_prewarmed_separate_bwd_probe(
                                x_sharded,
                                selected_sharded,
                                weights_sharded,
                                w13_sharded,
                                w2_sharded,
                                shared_w13_sharded,
                                shared_w2_sharded,
                                warmup=args.warmup,
                                iters=args.iters,
                                w13_expert_padded=args.w13_expert_padded,
                                w2_expert_padded=args.w2_expert_padded,
                                collapse_impl=args.deepep_collapse_impl,
                            )
                    elif args.profile_root is not None:
                        dt = _profile_fn(
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
                            profile_dir=args.profile_root,
                            profile_name=profile_name,
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
                if args.profile_root is not None:
                    _print0(f"PROFILE kernel={kernel} ep={ep_size} dir={args.profile_root}")

    set_intranode_config_overrides()


if __name__ == "__main__":
    main()
