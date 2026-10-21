# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 expert MLP for the ragged expert-parallel backend, on the fused SM100 kernels.

The fused kernels need every expert to start on a GROUP_PAD-row boundary, so this MLP takes a
receiver buffer laid out that way: its ``receiver_alignment`` is GROUP_PAD. Each expert's received
rows are a prefix of its span, and the rest of the span, and the buffer's trailing rows, are
unspecified and may be non-finite. The kernels run on the receiver buffer in place, and their row
outputs are already in receiver order. The quantizer reads every row that was not received as
zero, so those rows never reach a scale block, a GEMM, or a weight gradient.

All six grouped GEMMs run in MXFP8:

- forward: dual-quantize the tokens in one Triton pass; the gate/up GEMM, with SwiGLU and the dual
  MXFP8 quantization of the hidden activation ``h`` in its epilogue; the down GEMM.
- backward: dual-quantize the output cotangent the same way; the down data gradient, with the
  SwiGLU backward, the dual quantization of its result, and each row's ``<h, dh>`` in its epilogue;
  the gate/up data gradient; the two weight gradients, from the token-quantized copies.

The residuals are the token-quantized copies the weight gradients read and the bf16 gate/up
pre-activation the SwiGLU backward reads. The bf16 ``h`` is never stored. The weights are kept as
given and quantized again along the axis each GEMM contracts over. ``apply``, the forward of a pass
whose backward recomputes it, runs the same kernels without writing any of the residuals.

The hero compiles with XLA GPU command buffers off (``--xla_gpu_enable_command_buffer=``), and its
launcher refuses MXFP8 with them on. CUDA-graph replays of the forward and backward match the
program without command buffers bitwise once the wgrad kernel acquires its online scale-factor
descriptors. Before that fence, a replay could read a stale descriptor from the other weight
gradient's call at the same workspace address and return a partly NaN down-projection gradient.

Ported from c3cb334f8e:experiments/grug/moe/mxfp8.py (``_forward_pipeline``,
``_backward_pipeline``).
"""

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from levanter.grug._moe.ep_ragged_all_to_all import _RoutingWeightGradient
from levanter.grug._moe.mxfp8.adapter import (
    mxfp8_fused_dswiglu,
    mxfp8_fused_gemm,
    mxfp8_fused_swiglu,
    mxfp8_fused_swiglu_forward,
    mxfp8_fused_wgrad,
)
from levanter.grug._moe.mxfp8.layout import (
    GROUP_PAD,
    PaddedDispatch,
    aligned_dispatch_layout,
    deinterleave_gate_up_blocks,
    interleave_gate_up_blocks,
    wgrad_block_perm,
)
from levanter.grug._moe.mxfp8.quantize import build_sfb, quantize_mxfp8
from levanter.grug._moe.mxfp8.triton_quant import dual_quantize_padded, quantize_rows_padded

# The swizzled scales tile 128 rows by 4 scales of 32 elements, so every GEMM dimension the
# scales cover, H and I, must be a multiple of 128.
_FEATURE_ALIGNMENT = 128


class _Mxfp8Residuals(NamedTuple):
    """What the backward reads. Row-indexed members are in the receiver buffer's row order."""

    x_col: Float[Array, "R H"]
    """The tokens quantized along tokens (E4M3), an operand of the gate/up weight gradient."""
    x_col_scales: Array
    """Their weight-gradient scales, E8M0 ``[H, R / 32]``."""
    h_col: Float[Array, "R I"]
    """The SwiGLU output quantized along tokens (E4M3), an operand of the down weight gradient."""
    h_col_scales: Array
    """Their weight-gradient scales, flat E8M0."""
    c13: Float[Array, "R I2"]
    """The bf16 gate/up pre-activation, with gate and up interleaved in 32-column blocks."""
    moe_w13_local: Float[Array, "Echunk H I2"]
    moe_w2_local: Float[Array, "Echunk I H"]
    physical_group_sizes: Int[Array, "Echunk"]
    active_group_sizes: Int[Array, "Echunk"]


def _aligned_layout(
    physical_group_sizes: Int[Array, "Echunk"], active_group_sizes: Int[Array, "Echunk"], *, rows: int, hidden: int
) -> tuple[PaddedDispatch, Int[Array, "tiles"]]:
    """The receiver buffer's kernel layout, and the weight-gradient scale order of an H-wide operand."""
    layout = aligned_dispatch_layout(physical_group_sizes, active_group_sizes, rows=rows)
    block_perm = wgrad_block_perm(layout.padded_group_sizes, rows=hidden, total_tokens=layout.padded_rows)
    return layout, block_perm


def _quantized_weights(weights: Float[Array, "E N K"]) -> tuple[Float[Array, "E N K"], Array]:
    """Weights quantized along their last axis, the one the GEMM contracts, with per-expert swizzled scales."""
    values, scales = quantize_mxfp8(weights)
    return values, build_sfb(scales)


def _check_feature_sizes(hidden: int, intermediate: int) -> None:
    if hidden % _FEATURE_ALIGNMENT or intermediate % _FEATURE_ALIGNMENT:
        raise ValueError(
            f"the MXFP8 expert MLP needs hidden and intermediate sizes that are multiples of "
            f"{_FEATURE_ALIGNMENT}; got hidden={hidden}, intermediate={intermediate}"
        )


def _forward_weights(
    moe_w13_local: Float[Array, "Echunk H I2"], moe_w2_local: Float[Array, "Echunk I H"]
) -> tuple[Array, Array, Array, Array]:
    """The forward GEMMs' quantized weights and scales: ``(w13, w13_scales, w2, w2_scales)``."""
    # The gate/up GEMM contracts over H, with gate and up interleaved for the SwiGLU epilogue.
    w13, w13_scales = _quantized_weights(jnp.swapaxes(interleave_gate_up_blocks(moe_w13_local), 1, 2))
    # The down GEMM contracts over I.
    w2, w2_scales = _quantized_weights(jnp.swapaxes(moe_w2_local, 1, 2))
    return w13, w13_scales, w2, w2_scales


@dataclasses.dataclass(frozen=True)
class _Mxfp8ExpertMlp:
    """Expert MLP whose six grouped GEMMs run in MXFP8 on the fused SM100 kernels.

    The kernels group rows by the physical sizes, the GROUP_PAD-aligned spans of the receiver
    buffer, and the quantizer reads only the active rows of each span. The down data-gradient
    kernel also writes each row's ``<h, dh>``, which equals the protocol's ``<y, dy>`` up to MXFP8
    rounding, for the expert-side routing-weight gradient. Requires SiLU experts, bfloat16 tokens
    and output cotangents (the Triton quantizer takes only bfloat16), and H and I that are
    multiples of 128. `_select_mxfp8_expert_mlp` checks the activation, the dtype, and that the
    kernels and the quantizer can run.
    """

    routing_weight_gradient: _RoutingWeightGradient

    @property
    def receiver_alignment(self) -> int:
        return GROUP_PAD

    def forward(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        rows, hidden = x_dispatch.shape
        _check_feature_sizes(hidden, moe_w2_local.shape[1])
        layout, block_perm = _aligned_layout(physical_group_sizes, active_group_sizes, rows=rows, hidden=hidden)
        x_rows, x_rows_scales, x_col, x_col_scales = dual_quantize_padded(x_dispatch, layout, block_perm)
        w13, w13_scales, w2, w2_scales = _forward_weights(moe_w13_local, moe_w2_local)
        c13, h, h_col, h_scales, h_col_scales = mxfp8_fused_swiglu(
            x_rows, x_rows_scales, w13, w13_scales, layout.group_ends
        )
        out = mxfp8_fused_gemm(h, h_scales, w2, w2_scales, layout.group_ends)
        residuals = _Mxfp8Residuals(
            x_col=x_col,
            x_col_scales=x_col_scales,
            h_col=h_col,
            h_col_scales=h_col_scales,
            c13=c13,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            physical_group_sizes=physical_group_sizes,
            active_group_sizes=active_group_sizes,
        )
        return out, residuals

    def apply(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        """`forward`'s output, bit for bit, from kernels that skip what only the backward reads.

        The quantizer leaves out the token-quantized tokens and the gate/up kernel the pre-activation
        and the token-quantized ``h``. The returned ``h`` is the gate/up kernel's row-quantized
        output, the stage `forward`'s residuals finish with.
        """
        rows, hidden = x_dispatch.shape
        _check_feature_sizes(hidden, moe_w2_local.shape[1])
        layout = aligned_dispatch_layout(physical_group_sizes, active_group_sizes, rows=rows)
        x_rows, x_rows_scales = quantize_rows_padded(x_dispatch, layout)
        w13, w13_scales, w2, w2_scales = _forward_weights(moe_w13_local, moe_w2_local)
        h, h_scales = mxfp8_fused_swiglu_forward(x_rows, x_rows_scales, w13, w13_scales, layout.group_ends)
        out = mxfp8_fused_gemm(h, h_scales, w2, w2_scales, layout.group_ends)
        return out, h

    def backward(self, residuals, cotangent):
        res = _Mxfp8Residuals(*residuals)
        rows, hidden = cotangent.shape
        intermediate = res.moe_w2_local.shape[1]
        layout, block_perm = _aligned_layout(
            res.physical_group_sizes, res.active_group_sizes, rows=rows, hidden=hidden
        )
        g_rows, g_rows_scales, g_col, g_col_scales = dual_quantize_padded(cotangent, layout, block_perm)
        # The down data gradient contracts over H.
        w2, w2_scales = _quantized_weights(res.moe_w2_local)
        # The gate/up data gradient contracts over the interleaved gate/up axis.
        w13, w13_scales = _quantized_weights(interleave_gate_up_blocks(res.moe_w13_local))
        dc13, dc13_col, dc13_scales, dc13_col_scales, row_dot = mxfp8_fused_dswiglu(
            g_rows, g_rows_scales, w2, w2_scales, res.c13, layout.group_ends
        )
        dx = mxfp8_fused_gemm(dc13, dc13_scales, w13, w13_scales, layout.group_ends)
        dw13_interleaved = mxfp8_fused_wgrad(
            res.x_col, res.x_col_scales, dc13_col, dc13_col_scales.reshape(2 * intermediate, -1), layout.group_ends
        )
        dw2 = mxfp8_fused_wgrad(
            res.h_col, res.h_col_scales.reshape(intermediate, -1), g_col, g_col_scales, layout.group_ends
        )
        return (
            dx.astype(cotangent.dtype),
            deinterleave_gate_up_blocks(dw13_interleaved).astype(res.moe_w13_local.dtype),
            dw2.astype(res.moe_w2_local.dtype),
            row_dot,
        )
