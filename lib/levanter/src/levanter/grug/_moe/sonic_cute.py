# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend using Tri Dao's QuACK SM100 kernels (SonicMoE) on B200.

Dispatch/combine as in ``scatter``, but the expert MLP GEMMs run on QuACK's
``GemmGatedSm100`` / ``GemmDefaultSm100`` via the vendored ``cutlass.jax.cutlass_call``
shim. QuACK does all four activation-path grouped GEMMs (gate/up fwd fused with
SwiGLU, down fwd, and the ``dh``/``dx`` backward matmuls); the SwiGLU backward is
elementwise in JAX; the two weight-gradient GEMMs (``dw13``/``dw2``) stay on XLA
``ragged_dot`` (a different varlen-k grouping). QuACK covers ~2/3 of the MoE FLOPs.
"""

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
)
from levanter.grug._moe.quack_moe_cute import quack_gated_grouped_gemm, quack_grouped_gemm


def _interleave_gate_up(moe_w13: jax.Array, moe_dim: int) -> jax.Array:
    """grug w13 [E,H,2I] gate=[:I], up=[I:] -> interleaved [g0,u0,g1,u1,...] (QuACK layout)."""
    gate = moe_w13[..., :moe_dim]
    up = moe_w13[..., moe_dim:]
    return jnp.stack([gate, up], axis=-1).reshape(moe_w13.shape)


def _quack_tile() -> tuple[int, int]:
    v = os.environ.get("SCALE_QUACK_TILE")  # e.g. "256,128"
    return tuple(int(x) for x in v.split(",")) if v else (256, 128)  # type: ignore[return-value]


def _quack_cluster() -> tuple[int, int, int]:
    v = os.environ.get("SCALE_QUACK_CLUSTER")  # e.g. "1,1,1"
    return tuple(int(x) for x in v.split(",")) if v else (2, 1, 1)  # type: ignore[return-value]


def _quack_swizzle() -> int:
    return int(os.environ.get("SCALE_QUACK_SWIZZLE", "8"))


def _f8(x):
    """Cast a grouped-GEMM operand to e4m3 when SCALE_QUACK_FP8=1 (Blackwell FP8 tensor
    cores ~2x BF16). Values are O(1) from init/silu, well within e4m3 range, so a direct
    cast suffices for a throughput (MFU) measurement — no scaling needed for correctness here."""
    return x.astype(jnp.float8_e4m3fn) if os.environ.get("SCALE_QUACK_FP8") == "1" else x


@jax.custom_vjp
def _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu):
    """y = down( swiglu( x @ w13_il ) ), grouped by experts. Activation-path GEMMs on QuACK.

    ``group_sizes``/``cu`` are traced int arrays passed as explicit args (not closed
    over — that leaks under shard_map; not nondiff_argnums — that rejects tracers).

    The QuACK tile/cluster/swizzle are env-overridable (``SCALE_QUACK_TILE`` /
    ``SCALE_QUACK_CLUSTER`` / ``SCALE_QUACK_SWIZZLE``) to work around the SM100 grouped
    GEMM's dim-dependent launch failures at some hidden widths.
    """
    tm, cm, sw = _quack_tile(), _quack_cluster(), _quack_swizzle()
    _gu, h = quack_gated_grouped_gemm(_f8(x_dispatch), _f8(w13_il), cu, return_preact=True, tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)
    return quack_grouped_gemm(_f8(h), _f8(moe_w2), cu, b_major="n", tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)


def _expert_mlp_fwd(x_dispatch, w13_il, moe_w2, group_sizes, cu):
    tm, cm, sw = _quack_tile(), _quack_cluster(), _quack_swizzle()
    gu, h = quack_gated_grouped_gemm(_f8(x_dispatch), _f8(w13_il), cu, return_preact=True, tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)
    y = quack_grouped_gemm(_f8(h), _f8(moe_w2), cu, b_major="n", tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)
    return y, (x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu)


def _expert_mlp_bwd(res, dy):
    x_dispatch, w13_il, moe_w2, gu, h, group_sizes, cu = res
    tm, cm, sw = _quack_tile(), _quack_cluster(), _quack_swizzle()
    # down backward: dh via QuACK (transposed contraction), dw2 via XLA weight-grad
    dh = quack_grouped_gemm(_f8(dy), _f8(moe_w2), cu, b_major="k", tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)
    (dw2,) = jax.vjp(lambda w: ragged_dot(h, w, group_sizes), moe_w2)[1](dy)
    # SwiGLU backward (interleaved gate/up), elementwise
    gate, up = gu[:, 0::2], gu[:, 1::2]
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dgate = dh * up * (sg + silu * (1.0 - sg))
    dup = dh * silu
    d_gu = jnp.stack([dgate, dup], axis=-1).reshape(gu.shape)
    # gate/up backward: dx via QuACK, dw13 via XLA weight-grad
    dx = quack_grouped_gemm(_f8(d_gu), _f8(w13_il), cu, b_major="k", tile_mn=tm, cluster_mnk=cm, max_swizzle=sw)
    (dw13_il,) = jax.vjp(lambda w: ragged_dot(x_dispatch, w, group_sizes), w13_il)[1](d_gu)
    # int-typed routing args get float0 zero cotangents
    gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    cu_ct = np.zeros(cu.shape, dtype=jax.dtypes.float0)
    return dx, dw13_il, dw2, gs_ct, cu_ct


_expert_mlp.defvjp(_expert_mlp_fwd, _expert_mlp_bwd)


def _moe_mlp_local_sonic_cute(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)
    moe_dim = moe_w2.shape[1]
    w13_il = _interleave_gate_up(moe_w13, moe_dim)
    cu = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(group_sizes).astype(jnp.int32)])

    with jax.named_scope("moe_up_down_quack"):
        out_dispatch = tree_checkpoint_name(
            _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu), _CHECKPOINT_DISPATCH_OUTPUT
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
