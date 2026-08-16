# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert MLP on the Pallas Blackwell grouped GEMM (``blackwell_ragged_dot_mgpu``).

The activation-path GEMMs (gate/up forward, down forward, and the ``dh``/``dx``
backward contractions) run on jax's warp-specialized tcgen05 ragged-dot kernel,
which measured 2.17x faster than the QuACK/cuDNN path at the hero shard shape
(MEP-040). The two weight gradients stay on cuDNN Frontend grouped Wgrad (a
varlen-k grouping the ragged-dot kernel does not express). Weights keep the
grug ``[E, H, 2I]`` gate/up split layout — no QuACK interleaving.

Rows must be padded to a multiple of ``2 * TILE_M`` (the collective 2-CTA tile);
rows beyond ``sum(group_sizes)`` are never read or written by the kernel, and
callers mask them downstream as usual.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu
from jax.experimental.pallas.ops.gpu import blackwell_ragged_dot_mgpu as brd

from levanter.grug._moe.cudnn_wgrad_cute import cudnn_grouped_wgrad

# Best measured config at hero shard shapes (MEP-040): 66.3 us / 1457 TF/s on
# m=2560 k=6144 n=3072 G=3 bf16. collective=True doubles the effective tile.
_CONFIG = brd.TuningConfig(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    grid_tile_width=12,
    grid_minor_dim=blackwell_matmul_mgpu.MatmulDimension(0),
    max_concurrent_steps=6,
    collective=True,
)
ROW_ALIGN = 2 * _CONFIG.tile_m
# The collective config doubles the effective N tile. Hero dims are not all
# 256-multiples (2I = 6272, I = 3136), so `_grouped` pads the weight's N and
# slices the output.
_N_ALIGN = (2 if _CONFIG.collective else 1) * _CONFIG.tile_n


def _grouped(a: jax.Array, b: jax.Array, group_sizes: jax.Array) -> jax.Array:
    n = b.shape[2]
    n_pad = -n % _N_ALIGN
    if n_pad:
        b = jnp.pad(b, ((0, 0), (0, 0), (0, n_pad)))
    out = brd.ragged_dot_kernel(a, b, group_sizes, config=_CONFIG)
    return out[:, :n] if n_pad else out


@jax.custom_vjp
def brd_expert_mlp(x_dispatch, moe_w13, moe_w2, group_sizes):
    """y = down(swiglu(x @ w13)), grouped by experts, SiLU only."""
    y, _res = _fwd(x_dispatch, moe_w13, moe_w2, group_sizes)
    return y


def _fwd(x_dispatch, moe_w13, moe_w2, group_sizes):
    moe_dim = moe_w2.shape[1]
    gu = _grouped(x_dispatch, moe_w13, group_sizes)
    gate, up = gu[:, :moe_dim], gu[:, moe_dim:]
    act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(x_dispatch.dtype)
    y = _grouped(act, moe_w2, group_sizes)
    return y, (x_dispatch, moe_w13, moe_w2, gu, act, group_sizes)


def _bwd(res, dy):
    x_dispatch, moe_w13, moe_w2, gu, act, group_sizes = res
    moe_dim = moe_w2.shape[1]
    dy = dy.astype(x_dispatch.dtype)
    dact = _grouped(dy, moe_w2.transpose(0, 2, 1), group_sizes)
    dw2 = cudnn_grouped_wgrad(act, dy, group_sizes)
    gate = gu[:, :moe_dim].astype(jnp.float32)
    up = gu[:, moe_dim:].astype(jnp.float32)
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dact_f = dact.astype(jnp.float32)
    dgate = dact_f * up * (sg + silu * (1.0 - sg))
    dup = dact_f * silu
    d_gu = jnp.concatenate([dgate, dup], axis=1).astype(x_dispatch.dtype)
    dx = _grouped(d_gu, moe_w13.transpose(0, 2, 1), group_sizes)
    dw13 = cudnn_grouped_wgrad(x_dispatch, d_gu, group_sizes)
    gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    return dx, dw13, dw2, gs_ct


brd_expert_mlp.defvjp(_fwd, _bwd)


def brd_expert_mlp_padded(
    x_dispatch: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    physical_group_sizes: jax.Array,
    active_group_sizes: jax.Array,
    activation_fn,
) -> jax.Array:
    """`expert_mlp` plug-in for ``marin_ep_moe_local`` (pads rows to the tile)."""
    if activation_fn is not jax.nn.silu:
        raise ValueError("brd_expert_mlp fuses SiLU-gated SwiGLU only")
    del physical_group_sizes
    rows = x_dispatch.shape[0]
    padded = (rows + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN
    if padded != rows:
        x_dispatch = jnp.pad(x_dispatch, ((0, padded - rows), (0, 0)))
    y = brd_expert_mlp(x_dispatch, moe_w13_local, moe_w2_local, active_group_sizes)
    return y[:rows]
