# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend using Tri Dao's QuACK SM100 kernels (SonicMoE) on B200.

Dispatch/combine as in ``scatter``, but the expert MLP GEMMs run on QuACK's
``GemmGatedSm100`` / ``GemmDefaultSm100`` via the vendored ``cutlass.jax.cutlass_call``
shim. QuACK does all four activation-path grouped GEMMs (gate/up fwd fused with
SwiGLU, down fwd, and the ``dh``/``dx`` backward matmuls, and — via the varlen-k
mode — the two weight-gradient GEMMs ``dw13``/``dw2``); the SwiGLU backward is
elementwise in JAX. Set ``SONIC_CUTE_WGRAD=xla`` to fall back to the XLA
``ragged_dot`` weight grads for A/B comparison.
"""

import os
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jax import P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
)
from levanter.grug._moe.quack_moe_cute import (
    quack_gated_grouped_gemm,
    quack_grouped_gemm,
    quack_grouped_wgrad_gemm,
)

# Weight-grad GEMM backend: "quack" (varlen-k grouped GEMM on SM100) or "xla" (ragged_dot).
_WGRAD_IMPL = os.environ.get("SONIC_CUTE_WGRAD", "quack")
if _WGRAD_IMPL not in ("quack", "xla"):
    raise ValueError(f"SONIC_CUTE_WGRAD={_WGRAD_IMPL!r} must be 'quack' or 'xla'")


def _interleave_gate_up(moe_w13: jax.Array, moe_dim: int) -> jax.Array:
    """grug w13 [E,H,2I] gate=[:I], up=[I:] -> interleaved [g0,u0,g1,u1,...] (QuACK layout)."""
    gate = moe_w13[..., :moe_dim]
    up = moe_w13[..., moe_dim:]
    return jnp.stack([gate, up], axis=-1).reshape(moe_w13.shape)


def _fsdp_spec(spec: P) -> P | None:
    """Return ``spec`` if every named axis exists in the current mesh, else None."""
    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return None
    for entry in spec:
        names = entry if isinstance(entry, tuple) else (entry,)
        for name in names:
            if name is not None and name not in mesh.shape:
                return None
    return spec


def _store_sharded(w: jax.Array, spec: P) -> jax.Array:
    """Reshard a (gathered) weight back to its FSDP layout for residual storage."""
    resolved = _fsdp_spec(spec)
    return w if resolved is None else reshard(w, resolved)


def _restore_full(w: jax.Array, spec: P) -> jax.Array:
    """All-gather a residual weight stored in FSDP layout."""
    resolved = _fsdp_spec(spec)
    return w if resolved is None else reshard(w, P(*(None for _ in w.shape)))


# Residual-storage layouts for the expert weights (mirror the FSDP param sharding:
# hidden dim over "data"). Residuals are stored sharded and re-gathered in backward —
# the same all-gather whole-block remat pays — so a remat split that leaves the MoE
# live does not pin 26 layers of gathered expert weights.
_W13_STORE_SPEC = P(None, "data", None)
_W2_STORE_SPEC = P(None, None, "data")


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
        _store_sharded(w13_il, _W13_STORE_SPEC),
        _store_sharded(moe_w2, _W2_STORE_SPEC),
        gu,
        group_sizes,
        cu,
    )
    return y, res


def _expert_mlp_bwd(res, dy):
    x, token_dispatch, w13_stored, w2_stored, gu, group_sizes, cu = res
    x_dispatch = x[token_dispatch]
    w13_il = _restore_full(w13_stored, _W13_STORE_SPEC)
    moe_w2 = _restore_full(w2_stored, _W2_STORE_SPEC)
    # h = swiglu(gu), recomputed elementwise (fp32 internally, matching kernel accum)
    gate32 = gu[:, 0::2].astype(jnp.float32)
    up32 = gu[:, 1::2].astype(jnp.float32)
    h = (gate32 * jax.nn.sigmoid(gate32) * up32).astype(gu.dtype)
    # down backward: dh via QuACK (transposed contraction), dw2 via XLA weight-grad
    dh = quack_grouped_gemm(dy, moe_w2, cu, b_major="k")
    num_experts = moe_w2.shape[0]
    if _WGRAD_IMPL == "quack":
        dw2 = quack_grouped_wgrad_gemm(h, dy, cu, num_experts)
    else:
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
    if _WGRAD_IMPL == "quack":
        dw13_il = quack_grouped_wgrad_gemm(x_dispatch, d_gu, cu, num_experts)
    else:
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
