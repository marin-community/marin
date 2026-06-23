# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from functools import partial

import numpy as np
from jax import custom_jvp, custom_vjp, lax
from jax import numpy as jnp

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


def _new_scale_and_history(x, q_dtype, scale, amax_history):
    """Delayed-scaling update: new per-tensor scale from the amax window, and the window
    rolled forward with this step's amax. Shared by the QDQ and direct-quant paths."""
    dtype_max = get_fp8_max(q_dtype, jnp.float32)
    new_scale = compute_scale(jnp.max(amax_history, axis=0), scale, dtype_max)
    new_history = compute_amax_history(x, amax_history)
    return new_scale, new_history


def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
    new_scale, new_history = _new_scale_and_history(x, q_dtype, scale, amax_history)
    qx = quantize_dequantize(x, q_dtype, new_scale, compute_dtype)
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
    # `precision` gates whether XLA's GPU GemmRewriter re-fuses the transient
    # bf16->f8->bf16 operand round-trip into a $f8 cuBLASLt matmul on the FORWARD:
    # HIGHEST fires it, DEFAULT is stripped to bf16 (logbook GFP8-010/012). None
    # keeps the original DEFAULT forward. This primal and the jvp primal-recompute
    # below must use the same precision (custom_jvp contract); `preferred_element_type`
    # is intentionally not forwarded (einsum sets it noisily on the fp8 path).
    fwd_precision = lax.Precision.DEFAULT if precision is None else precision
    return lax.dot_general(lhs, rhs, dimension_numbers, precision=fwd_precision, **kwargs)


@dot_general_with_precision.defjvp
def dot_general_with_precision_jvp(
    dimension_numbers, precision, preferred_element_type, out_sharding, primals, tangents
):
    del preferred_element_type
    del out_sharding
    lhs, rhs = primals
    lhs_dot, rhs_dot = tangents

    # Under value_and_grad this recompute — not the standalone primal — produces the
    # training forward value, so the forward precision must be applied here too
    # (GFP8-012). Tangent (grad) dots are always HIGHEST.
    fwd_precision = lax.Precision.DEFAULT if precision is None else precision
    out = lax.dot_general(lhs, rhs, dimension_numbers, precision=fwd_precision)
    grad_out = lax.dot_general(lhs_dot, rhs, dimension_numbers, precision=lax.Precision.HIGHEST) + lax.dot_general(
        lhs, rhs_dot, dimension_numbers, precision=lax.Precision.HIGHEST
    )
    return out, grad_out


# --- Direct-quantization path (Flax Fp8DirectDotGeneralOp) --------------------
# dq(dot(q(x), q(w))): genuine E4M3 operands flow straight into the forward dot at
# DEFAULT precision and only the output is dequantized. Unlike the operand-QDQ form
# above there is no bf16->f8->bf16 round-trip for `simplify-fp-conversions` to strip,
# so the forward fuses to $f8 without the precision flip (logbook GFP8-014). The
# backward quantizes the output grad to E5M2 and runs the grad dots on f8 operands at
# HIGHEST. Lifted from flax/linen/fp8_ops.py; the `fm32` extended-dtype scale wrapper
# is dropped — our scales are plain float32, for which `_fm32_to_float32` is a no-op.


@partial(custom_vjp, nondiff_argnums=(0,))
def in_q(compute_dtype, inp, scale, amax_history):
    """Quantize an operand to genuine E4M3 (no dequant); also return the live scale."""
    new_scale, _ = _new_scale_and_history(inp, jnp.float8_e4m3fn, scale, amax_history)
    return quantize(inp, jnp.float8_e4m3fn, new_scale, compute_dtype), new_scale


def in_q_fwd(compute_dtype, inp, scale, amax_history):
    new_scale, new_history = _new_scale_and_history(inp, jnp.float8_e4m3fn, scale, amax_history)
    qin = quantize(inp, jnp.float8_e4m3fn, new_scale, compute_dtype)
    return (qin, new_scale), (new_scale, new_history)


def in_q_bwd(compute_dtype, res, _g):
    # inp gets no gradient; the freshly computed scale/history pass through as the
    # (overwrite) gradients of the scale/amax_history inputs (delayed-scaling state).
    new_scale, new_history = res
    return None, new_scale, new_history


in_q.defvjp(in_q_fwd, in_q_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_dq(dq_dtype, lhs_scale, rhs_scale, out):
    """Dequantize the matmul output by the product of the operand scales."""
    return dequantize(out, dq_dtype, lhs_scale * rhs_scale)


def out_dq_fwd(dq_dtype, lhs_scale, rhs_scale, out):
    return out_dq(dq_dtype, lhs_scale, rhs_scale, out), None


def out_dq_bwd(dq_dtype, _res, g):
    # Output cotangent passes straight through to the dot; scales are not differentiated here.
    return None, None, g


out_dq.defvjp(out_dq_fwd, out_dq_bwd)


def dot_general_transpose_lhs(g, x, y, *, dimension_numbers, precision, preferred_element_type, swap_ans=False):
    """LHS gradient of a dot_general (JAX's own transpose rule, exposed so the backward can
    run the grad dot on f8 operands at an explicit precision)."""

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


def dot_general_transpose_rhs(g, x, y, *, dimension_numbers, precision, preferred_element_type):
    """RHS gradient of a dot_general (mirror of the LHS rule with swapped operands)."""
    (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
    swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
    return dot_general_transpose_lhs(
        g,
        y,
        x,
        dimension_numbers=swapped_dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        swap_ans=True,
    )


@partial(custom_vjp, nondiff_argnums=(8, 9))
def quantized_dot(
    lhs,
    q_lhs,
    lhs_scale,
    rhs,
    q_rhs,
    rhs_scale,
    out_grad_scale,
    out_grad_amax_history,
    dimension_numbers,
    preferred_element_type=None,
):
    """Forward f8 dot on already-quantized operands; the custom_vjp carries the E5M2
    output-grad path. Full-precision operands (lhs/rhs) are kept only as backward residuals."""
    return lax.dot_general(
        q_lhs, q_rhs, dimension_numbers, preferred_element_type=preferred_element_type, precision=lax.Precision.DEFAULT
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
        q_lhs, q_rhs, dimension_numbers, preferred_element_type=preferred_element_type, precision=lax.Precision.DEFAULT
    )
    res = (lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history)
    return out, res


def quantized_dot_bwd(dimension_numbers, preferred_element_type, res, g):
    lhs, q_lhs, lhs_scale, rhs, q_rhs, rhs_scale, out_grad_scale, out_grad_amax_history = res

    new_out_grad_scale, new_out_grad_amax_history = _new_scale_and_history(
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
    preferred_element_type,
    *,
    lhs_scale,
    rhs_scale,
    grad_scale,
    lhs_amax_history,
    rhs_amax_history,
    grad_amax_history,
    quantize_compute_type,
):
    """`dot_general` drop-in for per-tensor delayed-scaling FP8 (direct quantization).

    Quantizes each operand to E4M3, runs the f8 dot, and dequantizes the output by the
    operand scales; the backward (in `quantized_dot`) quantizes the output grad to E5M2.
    Scale/amax state flows in as arguments and back out as gradients (overwrite-with-grad).
    """
    q_lhs, new_lhs_scale = in_q(quantize_compute_type, lhs, lhs_scale, lhs_amax_history)
    q_rhs, new_rhs_scale = in_q(quantize_compute_type, rhs, rhs_scale, rhs_amax_history)
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
    return out_dq(preferred_element_type, new_lhs_scale, new_rhs_scale, y)
