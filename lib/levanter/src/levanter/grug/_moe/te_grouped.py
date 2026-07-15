# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""TransformerEngine grouped-GEMM local Grug MoE backend (Fallback A).

Keeps grug's own token dispatch (argsort -> grouped layout -> scatter-add
combine, identical to ``scatter.py`` and free of any per-rank token cap) but
runs the two expert GEMMs through TE's ``grouped_dense`` (cuBLAS grouped GEMM)
instead of ``haliax.nn.ragged_dot``. This unlocks fp8/mxfp8 expert compute via
TE's ``grouped_quantize`` and sidesteps the XLA/cuDNN ragged_dot path -- WITHOUT
needing NCCL-EP, ``ep_bootstrap``, multi-process topology, or the 8192
tokens/rank dispatch cap.

``grouped_dense`` is a ``jax.custom_vjp`` so backward is handled by TE. It takes
``x [M, K]``, ``kernel [G, K, N]``, ``group_sizes [G]`` with
``contracting_dims=((1,), (1,))`` -- a direct analog of grug's
``ragged_dot(x, w, group_sizes)``.
"""

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _prepare_moe_dispatch,
    _zero_dropped_assignments,
    split_moe_w13_output,
)

try:
    from transformer_engine.jax.dense import grouped_dense
    from transformer_engine.jax.quantize import noop_quantizer_set

    _TE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only without the TE build
    grouped_dense = None  # type: ignore[assignment]
    noop_quantizer_set = None  # type: ignore[assignment]
    _TE_IMPORT_ERROR = exc


_CONTRACT = ((1,), (1,))

# Option 1: seal the two TE grouped_dense calls inside their own remat boundary so
# the outer block checkpoint (recompute_all) differentiates the expert-GEMM region
# as a single unit instead of re-tracing TE's grouped_dense custom_vjp/FFI, which
# otherwise explodes XLA compile. everything_saveable keeps the inner region's
# intermediates so the inner backward recomputes nothing.
_TE_REMAT_BOUNDARY = os.environ.get("GRUG_TE_REMAT_BOUNDARY") == "1"


def te_grouped_available() -> bool:
    return _TE_IMPORT_ERROR is None


def _expert_two_gemm(x_dispatch, moe_w13, moe_w2, group_sizes, moe_dim, qs1, qs2, activation_fn):
    w13_out = tree_checkpoint_name(
        grouped_dense(x_dispatch, moe_w13, group_sizes, contracting_dims=_CONTRACT, quantizer_set=qs1),
        _CHECKPOINT_EXPERT_HIDDEN,
    )
    gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
    return tree_checkpoint_name(
        grouped_dense(activation_fn(gate) * up, moe_w2, group_sizes, contracting_dims=_CONTRACT, quantizer_set=qs2),
        _CHECKPOINT_DISPATCH_OUTPUT,
    )


def _moe_mlp_local_te_grouped(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    fc1_quantizer_set=None,
    fc2_quantizer_set=None,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """grug dispatch + TE grouped_dense expert GEMMs + scatter-add combine."""
    if _TE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "transformer_engine.jax grouped_dense is unavailable. Build TE from source "
            f"(NVTE_WITH_NCCL_EP may be 0). Underlying import error: {_TE_IMPORT_ERROR!r}"
        )
    qs1 = fc1_quantizer_set if fc1_quantizer_set is not None else noop_quantizer_set
    qs2 = fc2_quantizer_set if fc2_quantizer_set is not None else noop_quantizer_set

    x_dispatch, w_dispatch, token_dispatch, group_sizes = _prepare_moe_dispatch(
        x, selected_experts, combine_weights, num_experts=num_experts
    )
    x_dispatch = tree_checkpoint_name(x_dispatch, _CHECKPOINT_DISPATCH_INPUT)

    moe_dim = moe_w2.shape[1]
    with jax.named_scope("moe_up_down_te_grouped"):
        if _TE_REMAT_BOUNDARY:

            def _sealed(x_disp, w13, w2, gs):
                return _expert_two_gemm(x_disp, w13, w2, gs, moe_dim, qs1, qs2, activation_fn)

            out_dispatch = jax.checkpoint(
                _sealed, policy=jax.checkpoint_policies.everything_saveable, prevent_cse=False
            )(x_dispatch, moe_w13, moe_w2, group_sizes)
        else:
            out_dispatch = _expert_two_gemm(x_dispatch, moe_w13, moe_w2, group_sizes, moe_dim, qs1, qs2, activation_fn)

    with jax.named_scope("scatter"):
        out = jnp.zeros_like(x).at[token_dispatch].add(out_dispatch * w_dispatch[:, None], mode="drop")
    return out, _zero_dropped_assignments()
