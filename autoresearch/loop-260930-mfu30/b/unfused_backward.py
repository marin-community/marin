# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The QuACK expert-MLP backward before the SwiGLU backward moved into the dh GEMM's epilogue.

Frozen from 6a6bb78853 for control variants: the dh GEMM writes dh, and XLA runs the SwiGLU
backward and the <h, dh> row dot in one fused pass.
"""

import jax
import jax.numpy as jnp
import levanter.grug._moe.sonic_cute as sonic_cute
from levanter.grug._moe.common import _swiglu_gate_up_backward, _unpack_pairs_u32
from levanter.grug._moe.quack_moe_cute import quack_grouped_gemm, quack_grouped_wgrad


def unfused_backward(res, dy):
    x_dispatch, w13_il, moe_w2, gu, h, cu = res
    dh = quack_grouped_gemm(dy, moe_w2, cu, b_major="k", **sonic_cute._QUACK_GROUPED_KW)
    gate, up = _unpack_pairs_u32(gu)
    h_fp32 = jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)
    output_dot_cotangent = jnp.sum(dh.astype(jnp.float32) * h_fp32, axis=-1)
    dw2 = quack_grouped_wgrad(h, dy, cu, **sonic_cute._QUACK_WGRAD_KW)
    d_gu = _swiglu_gate_up_backward(gu, dh)
    dx = quack_grouped_gemm(d_gu, w13_il, cu, b_major="k", **sonic_cute._QUACK_GROUPED_KW)
    dw13_il = quack_grouped_wgrad(x_dispatch, d_gu, cu, **sonic_cute._QUACK_WGRAD_KW)
    return dx, dw13_il, dw2, output_dot_cotangent
