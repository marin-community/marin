# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ungated ReLU^2 expert MLP for the ragged expert-parallel backend, on the cudnn-frontend fused MXFP8 kernels.

The fused kernels need every expert to start on a GROUP_PAD-row boundary, so this MLP takes a
receiver buffer laid out that way: its ``receiver_alignment`` is GROUP_PAD. Each expert's received
rows are a prefix of its span, and the rest of the span, and the buffer's trailing rows, are
unspecified and may be non-finite. The kernels run on the receiver buffer in place, and their row
outputs are already in receiver order. The quantizer reads every row that was not received as
zero, so those rows never reach a scale block, a GEMM, or a weight gradient. Every kernel and the
quantizer stop at the last expert's received rows rounded up to GROUP_PAD, so the buffer's unused
capacity costs nothing; the row outputs past that end are unspecified.

All six grouped GEMMs of ``y = relu(x @ w_up)^2 @ w_down`` run in MXFP8:

- forward: dual-quantize the tokens in one Triton pass; the up GEMM, with ReLU^2 and the row
  quantization of the hidden activation ``h`` in its epilogue; the down GEMM.
- backward: dual-quantize the output cotangent the same way; the down data gradient, with the
  ReLU^2 backward, the dual quantization of its result, ``h`` regenerated from the pre-activation
  and quantized along tokens, and each row's ``<h, dh>`` in its epilogue; the up data gradient;
  the two weight gradients, from the token-quantized copies.

The residuals are the token-quantized tokens and the bf16 pre-activation ``c``; neither copy of
``h`` is kept. The weights are kept as given and quantized again along the axis each GEMM
contracts over, in the layout they are stored in: the forward GEMMs read them N-major and the
backward GEMMs K-major, so no weight is transposed.
"""

import dataclasses
import functools
from typing import NamedTuple

from jaxtyping import Array, Float, Int

from levanter.grug._moe.mxfp8.adapter import mxfp8_fused_gemm, mxfp8_fused_wgrad
from levanter.grug._moe.mxfp8.adapter_srelu import mxfp8_fused_dsrelu, mxfp8_fused_srelu
from levanter.grug._moe.mxfp8.layout import GROUP_PAD, PaddedDispatch, aligned_dispatch_layout, wgrad_block_perm
from levanter.grug._moe.mxfp8.quantize import SF_TILE_ROWS, WeightMajor, quantize_weights
from levanter.grug._moe.mxfp8.triton_quant import dual_quantize_padded

# The swizzled scales tile SF_TILE_ROWS rows by 4 scales of 32 elements, so every GEMM dimension
# the scales cover, H, I and the output width, must be a multiple of the tile height.
_FEATURE_ALIGNMENT = SF_TILE_ROWS


class _Relu2Residuals(NamedTuple):
    """What the backward reads. Row-indexed members are in the receiver buffer's row order."""

    x_col: Float[Array, "R H"]
    """The tokens quantized along tokens (E4M3), an operand of the up weight gradient."""
    x_col_scales: Array
    """Their weight-gradient scales, E8M0 ``[H, R / 32]``."""
    c: Float[Array, "R I"]
    """The bf16 pre-activation ``x @ w_up``."""
    moe_w13_local: Float[Array, "Echunk H I"]
    moe_w2_local: Float[Array, "Echunk I O"]
    physical_group_sizes: Int[Array, "Echunk"]
    active_group_sizes: Int[Array, "Echunk"]


class _WeightGradOperands(NamedTuple):
    """The token-quantized operands of both weight gradients, and the experts' padded row ends, in
    `_weight_grads`' argument order."""

    x_col: Array
    x_col_scales: Array
    dc_col: Array
    dc_col_scales: Array
    h_col: Array
    h_col_scales: Array
    cotangent_col: Array
    cotangent_col_scales: Array
    group_ends: Array


def _token_scale_order(layout: PaddedDispatch, features: int) -> Int[Array, "tiles"]:
    """The weight-gradient scale tile order of a ``features``-wide operand in ``layout``."""
    return wgrad_block_perm(layout.padded_group_sizes, rows=features, total_tokens=layout.padded_rows)


def _check_feature_sizes(*sizes: int) -> None:
    if any(size % _FEATURE_ALIGNMENT for size in sizes):
        raise ValueError(
            f"the MXFP8 ReLU^2 expert MLP needs hidden, intermediate and output sizes that are multiples of "
            f"{_FEATURE_ALIGNMENT}; got {sizes}"
        )


def _weight_grads(
    x_col,
    x_col_scales,
    dc_col,
    dc_col_scales,
    h_col,
    h_col_scales,
    cotangent_col,
    cotangent_col_scales,
    group_ends,
    *,
    w_up_dtype,
    w_down_dtype,
) -> tuple[Array, Array]:
    """``(dw_up, dw_down)`` from `_WeightGradOperands`, in the weights' dtypes."""
    dw_up = mxfp8_fused_wgrad(x_col, x_col_scales, dc_col, dc_col_scales, group_ends)
    dw_down = mxfp8_fused_wgrad(h_col, h_col_scales, cotangent_col, cotangent_col_scales, group_ends)
    return dw_up.astype(w_up_dtype), dw_down.astype(w_down_dtype)


@dataclasses.dataclass(frozen=True)
class _CudnnRelu2Mxfp8ExpertMlp:
    """Ungated ReLU^2 experts whose six grouped GEMMs run in MXFP8 on the cudnn-frontend SM100 kernels.

    ``moe_w13_local`` holds ``w_up`` alone, ``[Echunk, H, I]``. The kernels group rows by the
    physical sizes, the GROUP_PAD-aligned spans of the receiver buffer, up to the last expert's
    received rows, and the quantizer reads only the active rows of each span. The down
    data-gradient kernel also writes each row's ``<h, dh>``, which equals the protocol's
    ``<y, dy>`` up to MXFP8 rounding. Requires bfloat16 tokens and output cotangents (the Triton
    quantizer takes only bfloat16), and H, I and the output width multiples of 128.
    """

    @property
    def receiver_alignment(self) -> int:
        return GROUP_PAD

    def forward(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        rows, hidden = x_dispatch.shape
        _check_feature_sizes(hidden, *moe_w2_local.shape[1:])
        layout = aligned_dispatch_layout(physical_group_sizes, active_group_sizes, rows=rows)
        x_rows, x_rows_scales, x_col, x_col_scales = dual_quantize_padded(
            x_dispatch, layout, _token_scale_order(layout, hidden)
        )
        # Both forward GEMMs contract over their weight's middle axis: H of w_up, I of w_down.
        w_up, w_up_scales = quantize_weights(moe_w13_local, WeightMajor.N)
        w_down, w_down_scales = quantize_weights(moe_w2_local, WeightMajor.N)
        c, h, h_scales = mxfp8_fused_srelu(x_rows, x_rows_scales, w_up, w_up_scales, layout.group_ends)
        out = mxfp8_fused_gemm(h, h_scales, w_down, w_down_scales, layout.group_ends, b_major=WeightMajor.N)
        residuals = _Relu2Residuals(
            x_col=x_col,
            x_col_scales=x_col_scales,
            c=c,
            moe_w13_local=moe_w13_local,
            moe_w2_local=moe_w2_local,
            physical_group_sizes=physical_group_sizes,
            active_group_sizes=active_group_sizes,
        )
        return out, residuals

    def backward(self, residuals, cotangent):
        dx, output_dot, operands, wgrad = self.backward_parts(residuals, cotangent)
        dw_up, dw_down = wgrad(*operands)
        return dx, dw_up, dw_down, output_dot

    def backward_parts(self, residuals, cotangent):
        res = _Relu2Residuals(*residuals)
        rows, out_width = cotangent.shape
        intermediate = res.moe_w2_local.shape[1]
        layout = aligned_dispatch_layout(res.physical_group_sizes, res.active_group_sizes, rows=rows)
        g_rows, g_rows_scales, g_col, g_col_scales = dual_quantize_padded(
            cotangent, layout, _token_scale_order(layout, out_width)
        )
        # Both data-gradient GEMMs contract over their weight's last axis: the output width of
        # w_down, I of w_up.
        w_down, w_down_scales = quantize_weights(res.moe_w2_local, WeightMajor.K)
        w_up, w_up_scales = quantize_weights(res.moe_w13_local, WeightMajor.K)
        dc, dc_scales, dc_col, dc_col_scales, h_col, h_col_scales, row_dot = mxfp8_fused_dsrelu(
            g_rows, g_rows_scales, w_down, w_down_scales, res.c, layout.group_ends
        )
        dx = mxfp8_fused_gemm(dc, dc_scales, w_up, w_up_scales, layout.group_ends, b_major=WeightMajor.K)
        operands = _WeightGradOperands(
            x_col=res.x_col,
            x_col_scales=res.x_col_scales,
            dc_col=dc_col,
            dc_col_scales=dc_col_scales.reshape(intermediate, -1),
            h_col=h_col,
            h_col_scales=h_col_scales.reshape(intermediate, -1),
            cotangent_col=g_col,
            cotangent_col_scales=g_col_scales,
            group_ends=layout.group_ends,
        )
        wgrad = functools.partial(
            _weight_grads, w_up_dtype=res.moe_w13_local.dtype, w_down_dtype=res.moe_w2_local.dtype
        )
        return dx.astype(cotangent.dtype), row_dot, operands, wgrad
