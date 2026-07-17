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

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jax import P
from jax.sharding import AxisType, get_abstract_mesh, reshard
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


# Residual storage for the expert weights: shard the hidden dim over the FSDP axis
# (mirroring the param layout) and re-gather in backward — the same all-gather
# whole-block remat pays — so a remat split that leaves the MoE live does not pin
# 26 layers of gathered expert weights. The local MoE usually runs inside a
# shard_map (the FSDP axis is Manual there → slice/all_gather); under explicit
# GSPMD sharding use reshard; with no mesh (unit tests) store the full weight.
_FSDP_AXIS = "data"


def _fsdp_axis_kind() -> tuple[AxisType | None, int]:
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty or _FSDP_AXIS not in mesh.shape:
        return None, 1
    kinds = dict(zip(mesh.axis_names, mesh.axis_types))
    return kinds[_FSDP_AXIS], int(mesh.shape[_FSDP_AXIS])


def _store_sharded(w: jax.Array, dim: int) -> jax.Array:
    """Store a (gathered) weight residual sharded along ``dim`` over the FSDP axis."""
    kind, n = _fsdp_axis_kind()
    if n <= 1 or w.shape[dim] % n != 0:
        return w
    if kind == AxisType.Explicit:
        spec: list[str | None] = [None] * w.ndim
        spec[dim] = _FSDP_AXIS
        return reshard(w, P(*spec))
    if kind == AxisType.Manual:
        shard = w.shape[dim] // n
        start = jax.lax.axis_index(_FSDP_AXIS) * shard
        return jax.lax.dynamic_slice_in_dim(w, start, shard, axis=dim)
    return w


def _restore_full(w: jax.Array, dim: int, full_size: int) -> jax.Array:
    """Re-gather a weight residual stored by ``_store_sharded``."""
    if w.shape[dim] == full_size:
        return w
    kind, _ = _fsdp_axis_kind()
    if kind == AxisType.Explicit:
        return reshard(w, P(*(None for _ in w.shape)))
    if kind == AxisType.Manual:
        return jax.lax.all_gather(w, _FSDP_AXIS, axis=dim, tiled=True)
    raise AssertionError(f"weight residual sharded along dim {dim} but FSDP axis kind is {kind}")


@jax.custom_vjp
def _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu, x, token_dispatch):
    """y = down( swiglu( x @ w13_il ) ), grouped by experts. Activation-path GEMMs on QuACK.

    ``group_sizes``/``cu`` are traced int arrays passed as explicit args (not closed
    over — that leaks under shard_map; not nondiff_argnums — that rejects tracers).
    ``x``/``token_dispatch`` are backward hints: ``x_dispatch`` must equal
    ``x[token_dispatch]``, so backward re-gathers it instead of pinning the [T*K, H]
    dispatch tensor as a residual. ``x`` gets a zero cotangent — the real gradient
    flows through ``x_dispatch`` into the caller's gather transpose.
    """
    _gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True)
    return quack_grouped_gemm(h, moe_w2, cu, b_major="n")


def _expert_mlp_fwd(x_dispatch, w13_il, moe_w2, group_sizes, cu, x, token_dispatch):
    gu, h = quack_gated_grouped_gemm(x_dispatch, w13_il, cu, return_preact=True)
    y = quack_grouped_gemm(h, moe_w2, cu, b_major="n")
    # Slim residuals: backward re-gathers x_dispatch from (x, token_dispatch) and
    # recomputes h elementwise from gu; weights are stored FSDP-sharded.
    res = (
        x,
        token_dispatch,
        _store_sharded(w13_il, 1),  # [E, H, 2I] sharded along H
        _store_sharded(moe_w2, 2),  # [E, I, H] sharded along H
        gu,
        group_sizes,
        cu,
    )
    return y, res


def _expert_mlp_bwd(res, dy):
    x, token_dispatch, w13_stored, w2_stored, gu, group_sizes, cu = res
    hidden = x.shape[-1]
    x_dispatch = x[token_dispatch]
    w13_il = _restore_full(w13_stored, 1, hidden)
    moe_w2 = _restore_full(w2_stored, 2, hidden)
    # h = swiglu(gu), recomputed elementwise (fp32 internally, matching kernel accum)
    gate32 = gu[:, 0::2].astype(jnp.float32)
    up32 = gu[:, 1::2].astype(jnp.float32)
    h = (gate32 * jax.nn.sigmoid(gate32) * up32).astype(gu.dtype)
    # down backward: dh via QuACK (transposed contraction), dw2 via XLA weight-grad
    dh = quack_grouped_gemm(dy, moe_w2, cu, b_major="k")
    (dw2,) = jax.vjp(lambda w: ragged_dot(h, w, group_sizes), moe_w2)[1](dy)
    # SwiGLU backward (interleaved gate/up), elementwise
    gate, up = gu[:, 0::2], gu[:, 1::2]
    sg = jax.nn.sigmoid(gate)
    silu = gate * sg
    dgate = dh * up * (sg + silu * (1.0 - sg))
    dup = dh * silu
    d_gu = jnp.stack([dgate, dup], axis=-1).reshape(gu.shape)
    # gate/up backward: dx via QuACK, dw13 via XLA weight-grad
    dx = quack_grouped_gemm(d_gu, w13_il, cu, b_major="k")
    (dw13_il,) = jax.vjp(lambda w: ragged_dot(x_dispatch, w, group_sizes), w13_il)[1](d_gu)
    # int-typed routing args get float0 zero cotangents
    gs_ct = np.zeros(group_sizes.shape, dtype=jax.dtypes.float0)
    cu_ct = np.zeros(cu.shape, dtype=jax.dtypes.float0)
    td_ct = np.zeros(token_dispatch.shape, dtype=jax.dtypes.float0)
    # x is a backward hint only; its gradient flows through x_dispatch (see _expert_mlp)
    x_ct = jnp.zeros_like(x)
    return dx, dw13_il, dw2, gs_ct, cu_ct, x_ct, td_ct


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
            _expert_mlp(x_dispatch, w13_il, moe_w2, group_sizes, cu, x, token_dispatch),
            _CHECKPOINT_DISPATCH_OUTPUT,
        )

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
