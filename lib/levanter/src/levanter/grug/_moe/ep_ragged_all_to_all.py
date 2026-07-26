# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import logging
import math
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import _CHECKPOINT_DISPATCH_INPUT, _CHECKPOINT_MOE_OUTPUT
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expand_from_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
    _sort_activations,
    _unpermute_from_global_expert,
)
from levanter.grug.sharding import _batch_axes

logger = logging.getLogger(__name__)

_FP8_WIRE_MAX = {
    jnp.float8_e4m3fn: 448.0,
    jnp.float8_e5m2: 57344.0,
}
_WIRE_EPS = 1e-12


def _report_underflow(x: jax.Array, quantized: jax.Array, fp8_dtype) -> None:
    """Emit the fraction of values that quantized to exactly zero from a non-zero input.

    fp8's characteristic failure is UNDERFLOW, not overflow: a scale that is far too large drives
    every value below the format's smallest representable magnitude and they all become exactly
    zero. That is not NaN, not Inf, and not out of range -- it is well-formed garbage, so
    `crash_on_nan` and every other finiteness guard stay silent while the affected term silently
    drops out of the computation. This was not hypothetical: a backward wire running an
    uncalibrated scale eight orders of magnitude too large zeroed every cotangent and raised
    nothing at all. A finiteness guard is the wrong instrument for the actual risk; this is the
    right one.

    Gated on SCALE_A2A_FP8_UNDERFLOW_CHECK=1 because it adds a reduction per quantization. Worth
    paying on the first run of any new configuration or scale.
    """
    if os.environ.get("SCALE_A2A_FP8_UNDERFLOW_CHECK") != "1":
        return
    nonzero_in = jnp.abs(x.astype(jnp.float32)) > 0
    zeroed = jnp.logical_and(nonzero_in, quantized.astype(jnp.float32) == 0)
    jax.debug.print(
        "fp8-underflow dtype={d} zeroed_fraction={f} (of non-zero inputs; >0.5 means the scale is far too large)",
        d=jnp.dtype(fp8_dtype).name,
        f=jnp.sum(zeroed) / jnp.maximum(jnp.sum(nonzero_in), 1),
    )


def _wire_quantize(x: jax.Array, fp8_dtype) -> tuple[jax.Array, jax.Array]:
    """Quantize each row (token slot) with its own current scale over the hidden dim.

    Per-token scaling only: sharing a scale across token rows would couple tokens along
    the sequence axis, which is causally unsafe. Zero padding rows quantize to exact zero.
    """
    xf = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(xf), axis=-1)
    if os.environ.get("SCALE_A2A_FP8_AMAX_PROBE") == "1":
        # Calibration for the supplied-scale variant: a constant scale is only faithful if the
        # real amax is stable across steps and layers, and the e4m3 forward and e5m2 backward
        # magnitudes may differ by orders of magnitude.
        jax.debug.print(
            "fp8-amax-probe dtype={d} tensor_amax={a} p99_row={p} median_row={m}",
            d=jnp.dtype(fp8_dtype).name,
            a=jnp.max(amax),
            p=jnp.quantile(amax, 0.99),
            m=jnp.median(amax),
        )
    scale = jnp.maximum(amax, _WIRE_EPS) / _FP8_WIRE_MAX[jnp.dtype(fp8_dtype).type]
    q = (xf / scale[..., None]).astype(fp8_dtype)
    _report_underflow(x, q, fp8_dtype)
    return q, scale


def _wire_quantize_fixed(x: jax.Array, fp8_dtype, amax: float) -> tuple[jax.Array, jax.Array]:
    """Quantize with a scale supplied from outside instead of reduced from ``x``.

    This is the delayed-scaling arithmetic: the scale depends on no token in the current batch, so
    it is causally safe, and crucially it removes the per-token amax REDUCTION, which measured as
    the entire quantization overhead of the current-scaling variant (~2500 ms/step, two
    `loop_reduce_fusion` passes). Precision cost of a shared tensor-wide scale measured at ~2%
    relative on a hostile fixture and nil on a realistic one, and a scale 30% stale moves the error
    only in the fourth decimal -- which is what makes a supplied scale viable at all.

    Values beyond the format range clip rather than wrap, so an under-estimated amax degrades
    gracefully.
    """
    fp8_max = _FP8_WIRE_MAX[jnp.dtype(fp8_dtype).type]
    scale = jnp.asarray(max(amax, _WIRE_EPS) / fp8_max, dtype=jnp.float32)
    quantized = jnp.clip(x.astype(jnp.float32) / scale, -fp8_max, fp8_max).astype(fp8_dtype)
    _report_underflow(x, quantized, fp8_dtype)
    return quantized, jnp.broadcast_to(scale, x.shape[:-1])


def _pmax_replicated_cotangent(ct, axis_name: str, mesh_size: int):
    """Make a replicated input's cotangent combine with max instead of sum.

    ``shard_map``'s transpose psums the cotangent of a replicated input, which is right for real
    gradients and wrong for amax/scale state, which must combine as a maximum. Dividing a pmax by
    the mesh size makes the implicit outer psum reconstruct exactly the pmax. Mirrors the helper
    already used by the ragged-dot fp8 path on the research branches.
    """
    return jax.lax.pmax(ct, axis_name) / mesh_size


def _fp8_a2a_impl(x: jax.Array, fp8_dtype) -> jax.Array:
    """Fixed-capacity tiled all_to_all with the payload quantized to FP8 on the wire.

    The payload crosses bitcast to uint8 (independent of backend FP8 collective
    support); per-token scales ride a second tiny all_to_all (4 bytes per row vs.
    hidden_dim bytes of payload) with the same routing, so each received row
    dequantizes with its sender's scale.
    """
    q, scale = _wire_quantize(x, fp8_dtype)
    bits = jax.lax.all_to_all(
        jax.lax.bitcast_convert_type(q, jnp.uint8),
        "expert",
        split_axis=0,
        concat_axis=0,
        tiled=True,
    )
    scales = jax.lax.all_to_all(scale[..., None], "expert", split_axis=0, concat_axis=0, tiled=True)
    deq = jax.lax.bitcast_convert_type(bits, fp8_dtype).astype(jnp.float32) * scales
    return deq.astype(x.dtype)


@jax.custom_vjp
def _fp8_wire_all_to_all(x: jax.Array) -> jax.Array:
    """Fixed-capacity a2a carrying E4M3 forward / E5M2 cotangents over the wire.

    Only the permutation legs are quantized (FP8 reductions lose to NCCL hierarchical
    reduce-scatter); the tiled a2a with split_axis=concat_axis=0 is its own transpose,
    so the backward is the same collective with the cotangent quantized to E5M2 and a
    straight-through gradient across the QDQ pair.
    """
    return _fp8_a2a_impl(x, jnp.float8_e4m3fn)


def _fp8_wire_a2a_fwd(x: jax.Array):
    return _fp8_wire_all_to_all(x), None


def _fp8_wire_a2a_bwd(_res, ct: jax.Array):
    return (_fp8_a2a_impl(ct, jnp.float8_e5m2),)


_fp8_wire_all_to_all.defvjp(_fp8_wire_a2a_fwd, _fp8_wire_a2a_bwd)


def _quantize_weight_scalar(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize an expert's weights with ONE scalar scale.

    XLA folds an operand scale into ``__cublas$lt$matmul$f8`` only when the scale is a scalar, and
    weights are not token-indexed so a per-expert scalar shares nothing across tokens. Measured on
    hostile data, a per-column scale buys nothing over this (0.0374 vs 0.0375): the binding limit
    is e4m3's 3-bit mantissa, not scale granularity.
    """
    scale = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32))), _WIRE_EPS) / _FP8_WIRE_MAX[jnp.float8_e4m3fn]
    return (w.astype(jnp.float32) / scale).astype(jnp.float8_e4m3fn), scale


@jax.custom_vjp
def _fp8_dispatch_gemm(send_block: jax.Array, w13: jax.Array) -> jax.Array:
    """Dispatch all_to_all carrying e4m3, fused with the first expert GEMM.

    The per-token scale rides the GEMM *output* rather than its input. Row-linearity makes that
    exact, and it is what keeps the dot on the fp8 custom call: an input-side per-token scale is
    non-scalar, which silently drops the GEMM back to bf16 (and is half of why the earlier
    fp8-wire attempt lost). The collective and the GEMM stay in one differentiated region so the
    fp8 payload never crosses an autodiff boundary.
    """
    return _fp8_dispatch_gemm_fwd(send_block, w13)[0]


def _wire_quantize_dispatch(x: jax.Array, fp8_dtype):
    """Per-token current scaling, or a supplied scale when SCALE_A2A_FP8_AMAX is set.

    A supplied scale is the delayed-scaling arithmetic without the history state: it isolates the
    throughput question (does removing the amax reduction recover the step time?) from the state
    plumbing. Production delayed scaling derives the same scalar from an amax history instead.
    """
    # Separate knobs per direction: measured activation amax is ~4 while cotangent amax is ~1e-7,
    # eight orders of magnitude apart, so one shared constant cannot serve both.
    key = "SCALE_A2A_FP8_AMAX_FWD" if fp8_dtype == jnp.float8_e4m3fn else "SCALE_A2A_FP8_AMAX_BWD"
    fixed = os.environ.get(key)
    if fixed is not None:
        return _wire_quantize_fixed(x, fp8_dtype, float(fixed))
    return _wire_quantize(x, fp8_dtype)


def _fp8_dispatch_gemm_fwd(send_block: jax.Array, w13: jax.Array):
    quantized, scale = _wire_quantize_dispatch(send_block, jnp.float8_e4m3fn)
    bits = jax.lax.all_to_all(
        jax.lax.bitcast_convert_type(quantized, jnp.uint8), "expert", split_axis=0, concat_axis=0, tiled=True
    )
    scales = jax.lax.all_to_all(scale[..., None], "expert", split_axis=0, concat_axis=0, tiled=True)
    received = jax.lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
    row_scale = scales.reshape(-1)
    weight_q, weight_scale = _quantize_weight_scalar(w13)
    hidden = received.reshape(-1, received.shape[-1]).astype(jnp.bfloat16) @ weight_q.astype(jnp.bfloat16)
    hidden = hidden * (row_scale * weight_scale)[:, None].astype(jnp.bfloat16)
    # Residuals must all be JAX types, so dtypes ride as zero-size arrays rather than as
    # dtype objects; the tiled a2a preserves shape, so send_block's shape needs no residual.
    dtypes = (jnp.zeros((), send_block.dtype), jnp.zeros((), w13.dtype))
    return hidden, (received, row_scale, weight_q, weight_scale, dtypes)


def _fp8_dispatch_gemm_bwd(residuals, ct: jax.Array):
    received, row_scale, weight_q, weight_scale, (send_zero, weight_zero) = residuals
    # hidden = (R @ Wq) * row * w_scale, so dL/dR = (ct * row * w_scale) @ Wq^T -- the scale sits on
    # the OUTPUT rows again, so the cotangent GEMM is e5m2 x e4m3 with the scale applied after.
    scaled_ct = ct.astype(jnp.float32) * (row_scale * weight_scale)[:, None]
    ct_q, ct_scale = _wire_quantize_dispatch(scaled_ct, jnp.float8_e5m2)
    d_received = (ct_q.astype(jnp.bfloat16) @ weight_q.T.astype(jnp.bfloat16)).astype(jnp.float32)
    d_received = d_received * ct_scale[:, None]
    # The dequantized value is R * row, so the cotangent w.r.t. it divides the row scale back out.
    d_dequantized = (d_received / row_scale[:, None]).reshape(received.shape)
    d_send = _fp8_a2a_impl(d_dequantized.astype(send_zero.dtype), jnp.float8_e5m2)
    # wgrad stays bf16: its per-token scale varies along the REDUCTION axis and cannot be hoisted
    # out of the sum (hoisting a scalar in its place measures 87% relative error).
    dequantized = received.reshape(-1, received.shape[-1]).astype(jnp.float32) * row_scale[:, None]
    d_w13 = (dequantized.astype(jnp.bfloat16).T @ ct.astype(jnp.bfloat16)).astype(weight_zero.dtype)
    return d_send, d_w13


_fp8_dispatch_gemm.defvjp(_fp8_dispatch_gemm_fwd, _fp8_dispatch_gemm_bwd)


@jax.custom_vjp
def _dispatch_gather(
    x_local: Float[Array, "Tlocal H"],
    token_sources: Int[Array, " send"],
    linear_indices: Int[Array, " assignments"],
    keep: Array,
) -> Float[Array, "send H"]:
    """Gather token rows into the fixed-capacity send buffer.

    Forward is the plain gather ``padded_x[token_sources]``. The custom VJP replaces XLA's
    generic scatter-add transpose with a structured gather-then-segment-sum over the top-k
    assignments, reusing the forward's ``linear_indices`` composition. ``token_sources``,
    ``linear_indices`` and ``keep`` are integer/bool and carry no cotangent.
    """
    hidden_dim = x_local.shape[1]
    padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
    return padded_x[token_sources]


def _dispatch_gather_fwd(x_local, token_sources, linear_indices, keep):
    send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
    # send_x shares x_local's dtype, so the incoming cotangent carries it too; only static shape
    # (tokens_per_shard) and the integer index arrays need to survive into the backward.
    return send_x, (linear_indices, keep, x_local.shape[0])


def _dispatch_gather_bwd(residual, cotangent):
    linear_indices, keep, tokens_per_shard = residual
    send_size = cotangent.shape[0]
    hidden_dim = cotangent.shape[1]
    topk = linear_indices.shape[0] // tokens_per_shard
    # d_x_local[t] = sum over token t's kept assignments of cotangent at their send slot.
    grad_rows = cotangent[jnp.minimum(linear_indices, send_size - 1)]
    grad_rows = jnp.where(keep[:, None], grad_rows, 0).astype(jnp.float32)
    grad_rows = grad_rows.reshape(tokens_per_shard, topk, hidden_dim)
    d_x_local = grad_rows.sum(axis=1).astype(cotangent.dtype)
    return d_x_local, None, None, None


_dispatch_gather.defvjp(_dispatch_gather_fwd, _dispatch_gather_bwd)


@jax.custom_vjp
def _combine_gather(
    send_output: Float[Array, "send H"],
    gather_indices: Int[Array, " assignments"],
    keep: Array,
    assignment_sources: Int[Array, " send"],
) -> Float[Array, "assignments H"]:
    """Gather expert outputs from the send buffer back into assignment order.

    Forward is ``send_output[gather_indices]`` with dropped assignments zeroed. The index map is
    injective on kept assignments, so the true transpose is a gather along the slot->assignment
    inverse (``assignment_sources``) rather than XLA's scatter-add. ``gather_indices``, ``keep``
    and ``assignment_sources`` are integer/bool and carry no cotangent.
    """
    gathered = send_output[gather_indices]
    return jnp.where(keep[:, None], gathered, 0)


def _combine_gather_fwd(send_output, gather_indices, keep, assignment_sources):
    gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
    return gathered, (assignment_sources,)


def _combine_gather_bwd(residual, cotangent):
    (assignment_sources,) = residual
    assignments_per_shard = cotangent.shape[0]
    # slot j was read by assignment assignment_sources[j] (or is unfilled -> assignments_per_shard).
    valid = assignment_sources < assignments_per_shard
    src = jnp.minimum(assignment_sources, assignments_per_shard - 1)
    d_send_output = jnp.where(valid[:, None], cotangent[src], 0).astype(cotangent.dtype)
    return d_send_output, None, None, None


_combine_gather.defvjp(_combine_gather_fwd, _combine_gather_bwd)


def _dispatch_rows(
    x_local: Float[Array, "Tlocal H"],
    linear_indices: Int[Array, "A"],
    send_size: int,
    topk: int,
) -> Float[Array, "S H"]:
    """Build the flat fixed-capacity dispatch buffer from local activations.

    With SCALE_A2A_GATHER_DISPATCH=1, scatters int32 assignment indices and gathers
    activation rows from ``x_local`` (avoids materializing the topk-repeated activation
    and the full-row random scatter). Otherwise scatters repeated bf16 rows directly.
    """
    tokens_per_shard, hidden_dim = x_local.shape
    assignments_per_shard = linear_indices.shape[0]
    if os.environ.get("SCALE_A2A_GATHER_DISPATCH") == "1":
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[linear_indices]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )
        token_sources = jnp.where(
            assignment_sources < assignments_per_shard,
            assignment_sources // topk,
            tokens_per_shard,
        )
        padded_x = jnp.concatenate(
            [x_local, jnp.zeros((1, hidden_dim), dtype=x_local.dtype)],
            axis=0,
        )
        return padded_x[token_sources]
    repeated_x = jnp.repeat(x_local, topk, axis=0)
    return jnp.zeros((send_size, hidden_dim), x_local.dtype).at[linear_indices].set(repeated_x, mode="drop")


def _moe_mlp_ep_fixed_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Expert-parallel MoE with fixed-capacity all-to-all dispatch and combine."""
    chunks = max(int(os.environ.get("SCALE_A2A_CHUNKS", "1")), 1)
    tokens_per_shard = x_local.shape[0]
    if tokens_per_shard % chunks != 0:
        raise ValueError(f"tokens_per_shard={tokens_per_shard} must be divisible by SCALE_A2A_CHUNKS={chunks}")
    if chunks == 1:
        return _fixed_a2a_core(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    tokens_per_chunk = tokens_per_shard // chunks
    chunk_outputs = []
    chunk_dropped = []
    for chunk_index in range(chunks):
        chunk_start = chunk_index * tokens_per_chunk
        chunk_end = (chunk_index + 1) * tokens_per_chunk
        output, dropped = _fixed_a2a_core(
            x_local[chunk_start:chunk_end],
            selected_experts_local[chunk_start:chunk_end],
            combine_weights_local[chunk_start:chunk_end],
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        chunk_outputs.append(output)
        chunk_dropped.append(dropped)
    return jnp.concatenate(chunk_outputs, axis=0), jnp.sum(jnp.stack(chunk_dropped))


def _fixed_a2a_core(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Run one fixed-capacity all-to-all dispatch, expert MLP, and combine."""
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by local expert count={local_experts}")

    tokens_per_shard = x_local.shape[0]
    expert_shards = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    hidden_dim = x_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    capacity = max(int(math.ceil(capacity_factor * assignments_per_shard / num_experts)), 1)

    # Keep XLA's auto-rematerializer from materializing flash-attention scores when
    # estimating memory pressure for the full attention-plus-MoE scan body.
    use_barrier = os.environ.get("SCALE_A2A_NO_BARRIER") != "1"
    if use_barrier:
        x_local, combine_weights_local = jax.lax.optimization_barrier((x_local, combine_weights_local))

    flat_experts = selected_experts_local.reshape(-1).astype(jnp.int32)

    order = jnp.argsort(flat_experts, stable=True)
    inverse_order = jnp.argsort(order)
    expert_counts = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)
    segment_start = jnp.cumsum(expert_counts) - expert_counts
    sorted_rank = jnp.arange(assignments_per_shard, dtype=jnp.int32) - segment_start[flat_experts[order]]
    slot = sorted_rank[inverse_order]
    keep = slot < capacity
    local_expert_indices = (flat_experts % local_experts).astype(jnp.int32)
    destination_shards = (flat_experts // local_experts).astype(jnp.int32)
    bucket_size = expert_shards * capacity
    send_size = local_experts * bucket_size

    rotate_groups = int(os.environ.get("SCALE_A2A_ROTATE", "0") or "0")
    if rotate_groups > 0:
        return _fixed_a2a_rotation(
            x_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            slot=slot,
            keep=keep,
            local_expert_indices=local_expert_indices,
            destination_shards=destination_shards,
            activation_fn=activation_fn,
            capacity=capacity,
            expert_shards=expert_shards,
            groups=rotate_groups,
            use_barrier=use_barrier,
        )

    linear_indices = jnp.where(
        keep,
        local_expert_indices * bucket_size + destination_shards * capacity + slot,
        send_size,
    )

    gather_dispatch = os.environ.get("SCALE_A2A_GATHER_DISPATCH") == "1"
    custom_adjoint = os.environ.get("SCALE_A2A_CUSTOM_ADJOINT") == "1"
    if custom_adjoint and not gather_dispatch:
        raise ValueError("SCALE_A2A_CUSTOM_ADJOINT=1 requires SCALE_A2A_GATHER_DISPATCH=1")
    if custom_adjoint:
        logger.info("fixed-a2a custom adjoint active: structured dispatch/combine gather VJPs")

    # slot -> assignment inverse of linear_indices (assignments_per_shard marks an unfilled slot).
    # Shared by the gather-dispatch forward and the combine gather's structured backward.
    assignment_sources = None
    if gather_dispatch:
        assignment_sources = (
            jnp.full((send_size,), assignments_per_shard, dtype=jnp.int32)
            .at[linear_indices]
            .set(jnp.arange(assignments_per_shard, dtype=jnp.int32), mode="drop")
        )

    moe_dim = moe_w2_local.shape[1]
    with jax.named_scope("dispatch"):
        if gather_dispatch:
            assert assignment_sources is not None
            token_sources = jnp.where(
                assignment_sources < assignments_per_shard,
                assignment_sources // topk,
                tokens_per_shard,
            )
            if custom_adjoint:
                send_x = _dispatch_gather(x_local, token_sources, linear_indices, keep)
            else:
                padded_x = jnp.concatenate([x_local, jnp.zeros((1, hidden_dim), x_local.dtype)], axis=0)
                send_x = padded_x[token_sources]
        else:
            repeated_x = jnp.repeat(x_local, topk, axis=0)
            send_x = jnp.zeros((send_size, hidden_dim), x_local.dtype).at[linear_indices].set(repeated_x, mode="drop")
        send_x = send_x.reshape(local_experts, expert_shards, capacity, hidden_dim)

    # SCALE_A2A_FP8_WIRE=1 quantizes ONLY the two permutation collectives (dispatch and
    # combine a2a) to FP8 on the wire. Routing, keep mask, capacity, and the drop count
    # are computed before quantization, so drop parity with the bf16 wire holds by
    # construction.
    fp8_wire = os.environ.get("SCALE_A2A_FP8_WIRE") == "1"
    if fp8_wire:
        logger.info(
            "fixed-a2a fp8 wire active: e4m3 fwd / e5m2 bwd, per-token scales, local_experts=%d expert_shards=%d",
            local_experts,
            expert_shards,
        )

    def wire_a2a(block: jax.Array) -> jax.Array:
        if fp8_wire:
            return _fp8_wire_all_to_all(block)
        return jax.lax.all_to_all(block, "expert", split_axis=0, concat_axis=0, tiled=True)

    # SCALE_A2A_FP8_GEMM=1 keeps the dispatch payload in e4m3 all the way into the first expert
    # GEMM instead of dequantizing on arrival, with the per-token scale applied to the GEMM output.
    # Requires the fp8 wire; it replaces the dispatch leg's dequantize-then-bf16-dot with a single
    # fp8 dot, which is what makes the wire saving show up as speed rather than as QDQ overhead.
    fp8_gemm = os.environ.get("SCALE_A2A_FP8_GEMM") == "1"
    if fp8_gemm:
        if not fp8_wire:
            raise ValueError("SCALE_A2A_FP8_GEMM=1 requires SCALE_A2A_FP8_WIRE=1")
        logger.info("fixed-a2a fp8 dispatch GEMM active: e4m3 operands, per-token scale on the GEMM output")

    def dispatch_a2a(local_expert_index: int) -> jax.Array:
        with jax.named_scope("dispatch"):
            received = wire_a2a(send_x[local_expert_index])
            return tree_checkpoint_name(received, _CHECKPOINT_DISPATCH_INPUT)

    # SCALE_A2A_PREFETCH=1 issues local expert le+1's dispatch all_to_all before local
    # expert le's GEMM. The collective has no data dependency on the GEMM, so the full
    # a2a kernel for the next round can overlap the current round's compute.
    prefetch = os.environ.get("SCALE_A2A_PREFETCH") == "1"
    if prefetch:
        logger.info("fixed-a2a prefetch active: local_experts=%d expert_shards=%d", local_experts, expert_shards)

    if fp8_gemm and prefetch:
        # The fused op owns its collective, so the prefetch gate cannot hoist the next round's
        # dispatch above this round's GEMM. Harmless now that the parallel-collective budget keeps
        # every a2a async regardless, but it means the two knobs do not compose.
        logger.info("fixed-a2a fp8 dispatch GEMM supersedes the prefetch reorder")

    output_parts = []
    received = dispatch_a2a(0) if prefetch and not fp8_gemm else None
    for local_expert_index in range(local_experts):
        if fp8_gemm:
            pass
        elif prefetch:
            next_received = dispatch_a2a(local_expert_index + 1) if local_expert_index + 1 < local_experts else None
        else:
            received = dispatch_a2a(local_expert_index)
        with jax.named_scope("moe_up_down"):
            if fp8_gemm:
                hidden = _fp8_dispatch_gemm(send_x[local_expert_index], moe_w13_local[local_expert_index])
            else:
                assert received is not None
                expert_input = received.reshape(bucket_size, hidden_dim)
                hidden = expert_input @ moe_w13_local[local_expert_index]
            gate, up = jnp.split(hidden, [moe_dim], axis=-1)
            expert_output = (activation_fn(gate) * up) @ moe_w2_local[local_expert_index]
        with jax.named_scope("combine"):
            returned = wire_a2a(expert_output.reshape(expert_shards, capacity, hidden_dim))
            output_parts.append(returned)
        if prefetch and not fp8_gemm:
            received = next_received

    with jax.named_scope("combine"):
        send_output = jnp.stack(output_parts, axis=0)
        send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
        send_output = send_output.reshape(send_size, hidden_dim)
        gather_indices = jnp.minimum(linear_indices, send_size - 1)
        if custom_adjoint:
            gathered = _combine_gather(send_output, gather_indices, keep, assignment_sources)
        else:
            gathered = jnp.where(keep[:, None], send_output[gather_indices], 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, dropped_total


def _fixed_a2a_rotation(
    x_local: Float[Array, "Tlocal H"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    slot: Int[Array, "A"],
    keep: Array,
    local_expert_indices: Int[Array, "A"],
    destination_shards: Int[Array, "A"],
    activation_fn: Callable[[jax.Array], jax.Array],
    capacity: int,
    expert_shards: int,
    groups: int,
    use_barrier: bool,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    """Fixed-capacity dispatch/combine decomposed into round-robin ppermute rounds.

    The dispatch buffer is laid out by peer *offset* r = (destination - self) mod P
    instead of destination shard id, so round r is a compile-time slice exchanged with
    ppermute perm i -> (i+r) mod P. Rounds are processed in ``groups`` groups: the
    group g+1 permutes are issued before group g's expert GEMM, so the permutes for the
    next group have no data dependency on the current group's compute and can overlap
    it. Combine returns each processed block with the inverted permutation. Bucket
    granularity, capacity, and the overflow policy are identical to the monolithic
    path; only the buffer layout and the collective decomposition differ.
    """
    if expert_shards % groups != 0:
        raise ValueError(f"expert_shards={expert_shards} must be divisible by SCALE_A2A_ROTATE={groups}")
    group_size = expert_shards // groups
    local_experts = moe_w13_local.shape[0]
    tokens_per_shard, hidden_dim = x_local.shape
    topk = combine_weights_local.shape[1]
    moe_dim = moe_w2_local.shape[1]
    block_rows = local_experts * capacity
    send_size = expert_shards * block_rows
    logger.info(
        "fixed-a2a rotation active: groups=%d group_size=%d expert_shards=%d capacity=%d",
        groups,
        group_size,
        expert_shards,
        capacity,
    )

    my_index = jax.lax.axis_index("expert")
    offsets = jnp.mod(destination_shards - my_index, expert_shards).astype(jnp.int32)
    linear_indices = jnp.where(
        keep,
        offsets * block_rows + local_expert_indices * capacity + slot,
        send_size,
    )

    with jax.named_scope("dispatch"):
        send_x = _dispatch_rows(x_local, linear_indices, send_size, topk)
        send_x = send_x.reshape(expert_shards, local_experts, capacity, hidden_dim)

    def permute_group(g: int) -> Float[Array, "S Elocal C H"]:
        parts = []
        for s in range(group_size):
            r = g * group_size + s
            if r == 0:
                parts.append(send_x[0])
                continue
            perm = [(i, (i + r) % expert_shards) for i in range(expert_shards)]
            parts.append(jax.lax.ppermute(send_x[r], "expert", perm))
        return tree_checkpoint_name(jnp.stack(parts, axis=0), _CHECKPOINT_DISPATCH_INPUT)

    def gemm_group(arrived: Float[Array, "S Elocal C H"]) -> Float[Array, "S Elocal C H"]:
        expert_input = arrived.transpose(1, 0, 2, 3).reshape(local_experts, group_size * capacity, hidden_dim)
        hidden = jnp.einsum("eth,ehi->eti", expert_input, moe_w13_local)
        gate, up = jnp.split(hidden, [moe_dim], axis=-1)
        expert_output = jnp.einsum("eti,eih->eth", activation_fn(gate) * up, moe_w2_local)
        return expert_output.reshape(local_experts, group_size, capacity, hidden_dim).transpose(1, 0, 2, 3)

    def combine_group(expert_output: Float[Array, "S Elocal C H"], g: int) -> list[jax.Array]:
        parts = []
        for s in range(group_size):
            r = g * group_size + s
            if r == 0:
                parts.append(expert_output[0])
                continue
            perm = [(i, (i - r) % expert_shards) for i in range(expert_shards)]
            parts.append(jax.lax.ppermute(expert_output[s], "expert", perm))
        return parts

    output_parts: list[jax.Array] = []
    with jax.named_scope("dispatch"):
        arrived = permute_group(0)
    for g in range(groups):
        if g + 1 < groups:
            with jax.named_scope("dispatch"):
                next_arrived = permute_group(g + 1)
        with jax.named_scope("moe_up_down"):
            expert_output = gemm_group(arrived)
        with jax.named_scope("combine"):
            output_parts.extend(combine_group(expert_output, g))
        if g + 1 < groups:
            arrived = next_arrived

    with jax.named_scope("combine"):
        send_output = jnp.stack(output_parts, axis=0)
        send_output = tree_checkpoint_name(send_output, _CHECKPOINT_MOE_OUTPUT)
        send_output = send_output.reshape(send_size, hidden_dim)
        gathered = send_output[jnp.minimum(linear_indices, send_size - 1)]
        gathered = jnp.where(keep[:, None], gathered, 0)
        gathered = gathered.reshape(tokens_per_shard, topk, hidden_dim)
        out_local = jnp.einsum(
            "tkh,tk->th",
            gathered,
            combine_weights_local.astype(gathered.dtype),
            preferred_element_type=jnp.float32,
        ).astype(x_local.dtype)
        assignments_per_shard = tokens_per_shard * topk
        dropped_local = assignments_per_shard - jnp.sum(keep, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    if use_barrier:
        out_local = jax.lax.optimization_barrier(out_local)
    return out_local, dropped_total


def _moe_mlp_ep_ragged_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
) -> tuple[Float[Array, "Tlocal H"], Int[Array, ""]]:
    if os.environ.get("SCALE_A2A_FIXED") == "1":
        return _moe_mlp_ep_fixed_a2a_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            moe_w13_local,
            moe_w2_local,
            activation_fn=activation_fn,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

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
    local_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    local_capacity = max(local_experts, local_capacity)
    recv_capacity = local_capacity

    with jax.named_scope("dispatch"):
        sorted_x, sorted_indices, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        all_group_sizes = jax.lax.all_gather(group_sizes.astype(jnp.int32), "expert")
        clipped_group_sizes = _clip_receiver_group_sizes(
            all_group_sizes,
            local_expert_size=local_experts,
            receiver_capacity=local_capacity,
        )
        sender_group_sizes = clipped_group_sizes[shard_id]
        keep_mask = _expert_prefix_keep_mask(
            group_sizes.astype(jnp.int32),
            sender_group_sizes,
            total_size=assignments_per_shard,
        )
        sorted_x = _compact_by_keep_mask(sorted_x, keep_mask)

        all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
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
        x_dispatch, local_sorted_indices, local_group_sizes = _local_permute_from_counts(
            x_dispatched,
            clipped_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )

    with jax.named_scope("moe_up_down"):
        w13_out = ragged_dot(x_dispatch, moe_w13_local, local_group_sizes)
        moe_dim = moe_w2_local.shape[1]
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
        returned = _expand_from_keep_mask(returned, keep_mask)
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, dropped_total
