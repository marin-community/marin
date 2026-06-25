# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import warnings
from functools import partial

import numpy as np
from jax import custom_jvp, custom_vjp, lax
from jax import numpy as jnp
from jax.typing import DTypeLike

# All of this is copy paste from flax/linen/fp8_ops.py
# (Until we get to the module)

# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def quantize_dequantize(x, q_dtype, scale, compute_dtype):
    qx = quantize(x, q_dtype, scale, compute_dtype)
    return dequantize(qx, x.dtype, scale)


def get_fp8_max(fp8_dtype, out_dtype):
    assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
    return jnp.finfo(fp8_dtype).max.astype(out_dtype)


def quantize(x, q_dtype, scale, compute_dtype):
    # Explicitly cast the max values to the compute dtype to avoid unnecessary
    # casting to FP32 during the subsequent math operations."
    dtype_max = get_fp8_max(q_dtype, compute_dtype)
    scaled_x = x / jnp.broadcast_to(scale.astype(compute_dtype), x.shape)
    clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)
    return clipped_x.astype(q_dtype)


def dequantize(x, dq_dtype, scale):
    return x.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), x.shape)


def compute_scale(amax, scale, fp8_max, margin=0):
    # The algorithm for computing the new scale is sourced from
    #   https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
    # wherein the `original_scale` corresponds to the reciprocal of the `scale`
    # passed in this function.
    scale = 1.0 / scale

    sf = (fp8_max / amax) / (2**margin)
    sf = jnp.where(amax > 0.0, sf, scale)
    sf = jnp.where(jnp.isfinite(amax), sf, scale)

    return 1.0 / sf


def compute_amax_history(x, amax_history):
    amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
    new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)
    return new_history


def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    amax_from_history = jnp.max(amax_history, axis=0)
    new_scale = compute_scale(amax_from_history, scale, dtype_max)

    qx = quantize_dequantize(x, q_dtype, new_scale, compute_dtype)

    new_history = compute_amax_history(x, amax_history)

    return qx, new_scale, new_history


@partial(custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
    qin, _, _ = qdq_and_return(inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
    return qin


def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
    qin, new_scale, new_history = qdq_and_return(inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
    return qin, (new_scale, new_history)


def in_qdq_bwd(compute_dtype, res, g):
    new_scale, new_history = res
    q_g = g
    return q_g, new_scale, new_history


in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
    return out


def out_qdq_fwd(compute_dtype, out, scale, amax_history):
    return out, (scale, amax_history)


def out_qdq_bwd(compute_dtype, res, g):
    scale, amax_history = res
    q_g, new_scale, new_history = qdq_and_return(g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
    return q_g, new_scale, new_history


out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)


@partial(custom_jvp, nondiff_argnums=(2, 3, 4, 5))
def dot_general_with_precision(
    lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None, out_sharding=None, **kwargs
):
    if precision is not None or preferred_element_type is not None:
        # einsum sets preferred_element_type and so this is just noisy
        # warnings.warn(
        #     "The function dot_general_with_precision will set the "
        #     "precision/preferred_element_type and disregard any provided "
        #     "values."
        # )
        pass
    return lax.dot_general(lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT, **kwargs)


@dot_general_with_precision.defjvp
def dot_general_with_precision_jvp(
    dimension_numbers, precision, preferred_element_type, out_sharding, primals, tangents
):
    del preferred_element_type
    del out_sharding
    del precision
    lhs, rhs = primals
    lhs_dot, rhs_dot = tangents

    out = lax.dot_general(lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT)
    grad_out = lax.dot_general(lhs_dot, rhs, dimension_numbers, precision=lax.Precision.HIGHEST) + lax.dot_general(
        lhs, rhs_dot, dimension_numbers, precision=lax.Precision.HIGHEST
    )
    return out, grad_out


# ---------------------------------------------------------------------------
# Direct fp8 quantization path
#
# The functions above implement the legacy "quantize-dequantize" (QDQ) recipe,
# in which the operands handed to `lax.dot_general` are dequantized back to the
# compute dtype, so whether FP8 tensor cores are used at all depends on XLA's
# GemmRewriter folding the QDQ pattern. The functions below implement Flax's
# *direct* recipe, in which genuine FP8 operands are fed to `lax.dot_general`
# (the cast is explicit and the dequantize happens on the output and gradients),
# so FP8 kernels are used without relying on pattern matching.
#
# Faithfully vendored from flax/linen/fp8_ops.py (flax 0.12.4); see
# https://github.com/google/flax/blob/main/flax/linen/fp8_ops.py
#
# Deviation from the Flax source: Flax stores the scale / amax-history state in a
# custom "fm32" extended dtype and converts it back to float32 at each use (its
# `_fm32_to_float32` helper and the `is_fmax32` branch of `update_fp8_meta`).
# Haliax keeps this state as plain float32 arrays (see `Fp8DotGeneralOp`), so the
# fm32 handling is dead code here and is omitted, consistent with the QDQ port
# above.
# ---------------------------------------------------------------------------


def update_fp8_meta(x, q_dtype, scale, amax_history):
    """Compute the next-step scale and rolled amax history (without quantizing `x`)."""
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    amax_from_history = jnp.max(amax_history, axis=0)

    new_scale = compute_scale(amax_from_history, scale, dtype_max)
    new_history = compute_amax_history(x, amax_history)
    return new_scale, new_history


@partial(custom_vjp, nondiff_argnums=(0, 1))
def in_q(compute_dtype, q_dtype, inp, scale, amax_history):
    new_scale, _ = update_fp8_meta(inp, q_dtype, scale, amax_history)
    qin = quantize(inp, q_dtype, new_scale, compute_dtype)
    return qin, new_scale


def in_q_fwd(compute_dtype, q_dtype, inp, scale, amax_history):
    new_scale, new_history = update_fp8_meta(inp, q_dtype, scale, amax_history)
    qin = quantize(inp, q_dtype, new_scale, compute_dtype)
    return (qin, new_scale), (new_scale, new_history)


def in_q_bwd(compute_dtype, q_dtype, res, _):
    new_scale, new_history = res
    # No gradient flows to inp/scale/amax_history; pass the updated scale and
    # history through as the cotangents so that OverwriteWithGradient overwrites
    # the persisted state with them.
    return None, new_scale, new_history


in_q.defvjp(in_q_fwd, in_q_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_dq(dq_type, lhs_scale, rhs_scale, out):
    return dequantize(out, dq_type, lhs_scale * rhs_scale)


def out_dq_fwd(dq_type, lhs_scale, rhs_scale, out):
    return out_dq(dq_type, lhs_scale, rhs_scale, out), None


def out_dq_bwd(dq_type, _, g):
    # The dequantize scaling is folded into `quantized_dot`'s backward instead,
    # so the cotangent passes straight through to the matmul output here.
    return None, None, g


out_dq.defvjp(out_dq_fwd, out_dq_bwd)


def dot_general_transpose_lhs(
    g, x, y, *, dimension_numbers, precision, preferred_element_type: DTypeLike | None, swap_ans=False
):
    def _remaining(original, *removed_lists):
        removed = set(itertools.chain(*removed_lists))
        return [i for i in original if i not in removed]

    def _ranges_like(*xs):
        start = 0
        for x in xs:
            x_len = len(x)
            yield range(start, start + x_len)
            start += x_len

    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    x_ndim = x.aval.ndim
    x_kept = _remaining(range(x_ndim), x_contract, x_batch)
    y_kept = _remaining(range(np.ndim(y)), y_contract, y_batch)
    if swap_ans:
        ans_batch, ans_y, _ = _ranges_like(x_batch, y_kept, x_kept)
    else:
        ans_batch, _, ans_y = _ranges_like(x_batch, x_kept, y_kept)
    dims = ((ans_y, y_kept), (ans_batch, y_batch))
    x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))
    out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)
    x_bar = lax.transpose(
        lax.dot_general(g, y, dims, precision=precision, preferred_element_type=preferred_element_type),
        tuple(out_axes),
    )
    return x_bar


def dot_general_transpose_rhs(g, x, y, *, dimension_numbers, precision, preferred_element_type: DTypeLike | None):
    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
    y_bar = dot_general_transpose_lhs(
        g,
        y,
        x,
        dimension_numbers=swapped_dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        swap_ans=True,
    )
    return y_bar


@partial(custom_vjp, nondiff_argnums=(8, 9))
def quantized_dot(
    lhs,
    q_lhs,
    lhs_scale,  # scale for this step
    rhs,
    q_rhs,
    rhs_scale,  # scale for this step
    out_grad_scale,  # scale from previous step
    out_grad_amax_history,  # amax history from previous step
    dimension_numbers,
    preferred_element_type=None,
):
    return lax.dot_general(
        q_lhs,
        q_rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=lax.Precision.DEFAULT,
    )


def quantized_dot_fwd(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    dimension_numbers,
    preferred_element_type,
):
    out = lax.dot_general(
        q_lhs,
        q_rhs,
        dimension_numbers,
        preferred_element_type=preferred_element_type,
        precision=lax.Precision.DEFAULT,
    )
    res = (lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history)
    return out, res


def quantized_dot_bwd(dimension_numbers, preferred_element_type, res, g):
    (lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history) = res

    new_out_grad_scale, new_out_grad_amax_history = update_fp8_meta(
        g, jnp.float8_e5m2, out_grad_scale, out_grad_amax_history
    )

    q_g = quantize(g, jnp.float8_e5m2, new_out_grad_scale, preferred_element_type)

    grad_lhs = dot_general_transpose_lhs(
        q_g,
        lhs,
        q_rhs,
        dimension_numbers=dimension_numbers,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=preferred_element_type,
    )
    grad_lhs = dequantize(grad_lhs, preferred_element_type, rhs_scale * new_out_grad_scale)

    grad_rhs = dot_general_transpose_rhs(
        q_g,
        q_lhs,
        rhs,
        dimension_numbers=dimension_numbers,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=preferred_element_type,
    )
    grad_rhs = dequantize(grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale)

    return (grad_lhs, None, None, grad_rhs, None, None, new_out_grad_scale, new_out_grad_amax_history)


quantized_dot.defvjp(quantized_dot_fwd, quantized_dot_bwd)


def fp8_scaled_dot_general(
    lhs,
    rhs,
    dimension_numbers,
    precision=None,
    preferred_element_type=None,
    *,
    lhs_scale,
    rhs_scale,
    grad_scale,
    lhs_amax_history,
    rhs_amax_history,
    grad_amax_history,
    quantize_compute_type=jnp.float32,
):
    """Drop-in `dot_general` that quantizes both operands to E4M3, contracts them
    as genuine FP8, and dequantizes the result. Gradients are taken in E5M2 via
    `quantized_dot`'s custom VJP. The scale / amax-history arrays carry the
    delayed-scaling state and are updated through the custom VJP (overwrites)."""
    if precision is not None:
        warnings.warn(
            'fp8_scaled_dot_general will set the "precision" and disregard any provided "precision" argument.'
        )
    q_lhs, new_lhs_scale = in_q(quantize_compute_type, jnp.float8_e4m3fn, lhs, lhs_scale, lhs_amax_history)
    q_rhs, new_rhs_scale = in_q(quantize_compute_type, jnp.float8_e4m3fn, rhs, rhs_scale, rhs_amax_history)
    y = quantized_dot(
        lhs,
        q_lhs,
        new_lhs_scale,
        rhs,
        q_rhs,
        new_rhs_scale,
        grad_scale,
        grad_amax_history,
        dimension_numbers,
        preferred_element_type,
    )
    y = out_dq(dq_type=preferred_element_type, lhs_scale=new_lhs_scale, rhs_scale=new_rhs_scale, out=y)
    return y
