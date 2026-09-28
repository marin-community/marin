# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged (grouped) ungated ReLU² expert MLP ``relu(x W_up_g)^2 W_down_g`` with fused-epilogue GEMMs.

The ragged counterpart of ``relu2_mlp``: rows of ``x`` are sorted by expert and ``group_sizes`` gives each
expert's row count, as in ``jax.lax.ragged_dot``. ``relu2_ragged_mlp`` is a ``custom_vjp``: the forward keeps
only ``post = relu(x W_up)^2``, and the backward forms ``d pre = (g W_down^T) 2 sqrt(post)`` in one GEMM
epilogue, so neither ``pre`` nor ``d post`` reaches HBM. ``implementation`` picks the GEMMs: ``"pallas_gpu"``
(Triton kernels), ``"pallas_interpret"`` (the same kernels in the Pallas interpreter), or ``"reference"``
(``jax.lax.ragged_dot``, same math).
"""

from functools import partial
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .pallas_gpu import BlockSizes, Epilogue, gmm, tgmm
from .reference import gmm_reference, tgmm_reference

Implementation: TypeAlias = Literal["reference", "pallas_gpu", "pallas_interpret"]
IMPLEMENTATIONS: tuple[Implementation, ...] = ("reference", "pallas_gpu", "pallas_interpret")


def _gmm(a, b, group_sizes, post, *, trans_b: bool, epilogue: Epilogue, implementation, block_sizes: BlockSizes):
    if implementation != "reference":
        return gmm(
            a,
            b,
            group_sizes,
            post,
            trans_b=trans_b,
            epilogue=epilogue,
            block_sizes=block_sizes.gmm,
            interpret=implementation == "pallas_interpret",
        )
    acc = gmm_reference(a, b.mT if trans_b else b, group_sizes)
    if epilogue == Epilogue.RELU2:
        acc = jnp.square(jax.nn.relu(acc))
    elif epilogue == Epilogue.RELU2_DPRE:
        acc = acc * 2.0 * jnp.sqrt(post.astype(jnp.float32))
    return acc.astype(a.dtype)


def _tgmm(a, b, group_sizes, implementation, block_sizes: BlockSizes):
    if implementation == "reference":
        return tgmm_reference(a, b, group_sizes)
    return tgmm(a, b, group_sizes, block_sizes=block_sizes.tgmm, interpret=implementation == "pallas_interpret")


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _relu2_ragged_mlp(x, w_up, w_down, group_sizes, implementation: Implementation, block_sizes: BlockSizes):
    out, _ = _relu2_ragged_mlp_fwd(x, w_up, w_down, group_sizes, implementation, block_sizes)
    return out


def _relu2_ragged_mlp_fwd(x, w_up, w_down, group_sizes, implementation, block_sizes):
    gemm = partial(_gmm, trans_b=False, implementation=implementation, block_sizes=block_sizes)
    post = gemm(x, w_up, group_sizes, None, epilogue=Epilogue.RELU2)
    out = gemm(post, w_down, group_sizes, None, epilogue=Epilogue.NONE)
    return out, (x, w_up, w_down, group_sizes, post)


def _relu2_ragged_mlp_bwd(implementation, block_sizes, residuals, g):
    x, w_up, w_down, group_sizes, post = residuals
    g = g.astype(x.dtype)
    gemm_t = partial(_gmm, trans_b=True, implementation=implementation, block_sizes=block_sizes)
    d_w_down = _tgmm(post, g, group_sizes, implementation, block_sizes).astype(w_down.dtype)
    dpre = gemm_t(g, w_down, group_sizes, post, epilogue=Epilogue.RELU2_DPRE)
    d_w_up = _tgmm(x, dpre, group_sizes, implementation, block_sizes).astype(w_up.dtype)
    dx = gemm_t(dpre, w_up, group_sizes, None, epilogue=Epilogue.NONE)
    return dx, d_w_up, d_w_down, None


_relu2_ragged_mlp.defvjp(_relu2_ragged_mlp_fwd, _relu2_ragged_mlp_bwd)


def relu2_ragged_mlp(
    x: Float[Array, "M K"],
    w_up: Float[Array, "G K N"],
    w_down: Float[Array, "G N Mout"],
    group_sizes: Int[Array, "G"],
    *,
    implementation: Implementation,
    block_sizes: BlockSizes | None = None,
) -> Float[Array, "M Mout"]:
    """``relu(x_g @ w_up[g])^2 @ w_down[g]`` for each group's rows. Rows past ``sum(group_sizes)`` come out
    zero and get zero gradient."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"relu2_ragged_mlp implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    return _relu2_ragged_mlp(x, w_up, w_down, group_sizes, implementation, block_sizes or BlockSizes.get_default())
