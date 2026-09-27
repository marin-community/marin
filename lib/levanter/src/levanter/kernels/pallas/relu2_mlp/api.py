# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Batched ungated ReLU² expert MLP ``relu(x W_up)^2 W_down`` with a fused-epilogue backward.

``relu2_mlp`` is a ``custom_vjp``: the forward keeps only ``post = relu(x W_up)^2``, and the backward
forms ``d pre = (g W_down^T) 2 sqrt(post)`` in one GEMM epilogue. The two weight gradients and ``dx`` stay
XLA einsums. ``implementation`` picks the epilogue GEMMs: ``"pallas_gpu"`` (the Triton kernels) or
``"reference"`` (plain JAX, same math).
"""

from functools import partial
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .pallas_gpu import BlockSizes, relu2_dpre_pallas, relu2_up_pallas
from .reference import relu2_dpre_reference, relu2_up_reference

Implementation: TypeAlias = Literal["reference", "pallas_gpu", "pallas_interpret"]
IMPLEMENTATIONS: tuple[Implementation, ...] = ("reference", "pallas_gpu", "pallas_interpret")


def _up(x, w_up, implementation: Implementation, block_sizes: BlockSizes):
    if implementation == "reference":
        return relu2_up_reference(x, w_up)
    return relu2_up_pallas(x, w_up, block_sizes=block_sizes, interpret=implementation == "pallas_interpret")


def _dpre(g, w_down, post, implementation: Implementation, block_sizes: BlockSizes):
    if implementation == "reference":
        return relu2_dpre_reference(g, w_down, post)
    return relu2_dpre_pallas(g, w_down, post, block_sizes=block_sizes, interpret=implementation == "pallas_interpret")


def _pad_rows(a: jax.Array, rows: int) -> jax.Array:
    pad = rows - a.shape[1]
    return a if pad == 0 else jnp.pad(a, ((0, 0), (0, pad), (0, 0)))


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _relu2_mlp(x, w_up, w_down, implementation: Implementation, block_sizes: BlockSizes):
    post = _up(x, w_up, implementation, block_sizes)
    return jnp.einsum("ern,enm->erm", post, w_down, preferred_element_type=jnp.float32).astype(x.dtype)


def _relu2_mlp_fwd(x, w_up, w_down, implementation, block_sizes):
    post = _up(x, w_up, implementation, block_sizes)
    out = jnp.einsum("ern,enm->erm", post, w_down, preferred_element_type=jnp.float32).astype(x.dtype)
    return out, (x, w_up, w_down, post)


def _relu2_mlp_bwd(implementation, block_sizes, residuals, g):
    x, w_up, w_down, post = residuals
    d_w_down = jnp.einsum("ern,erm->enm", post, g, preferred_element_type=jnp.float32).astype(w_down.dtype)
    dpre = _dpre(g.astype(x.dtype), w_down, post, implementation, block_sizes)
    d_w_up = jnp.einsum("erk,ern->ekn", x, dpre, preferred_element_type=jnp.float32).astype(w_up.dtype)
    dx = jnp.einsum("ern,ekn->erk", dpre, w_up, preferred_element_type=jnp.float32).astype(x.dtype)
    return dx, d_w_up, d_w_down


_relu2_mlp.defvjp(_relu2_mlp_fwd, _relu2_mlp_bwd)


def fused_relu2(x: jax.Array) -> jax.Array:
    """``relu(x)^2`` elementwise. As a MoE ``activation`` with ungated experts, the pooled-wave EP backend
    recognizes it and runs the expert MLP through ``relu2_mlp`` instead of separate GEMMs and elementwise ops."""
    return jnp.square(jax.nn.relu(x))


def relu2_mlp(
    x: Float[Array, "E R K"],
    w_up: Float[Array, "E K N"],
    w_down: Float[Array, "E N M"],
    *,
    implementation: Implementation,
    block_sizes: BlockSizes | None = None,
) -> Float[Array, "E R M"]:
    """``relu(x @ w_up)^2 @ w_down`` per expert. Rows are zero-padded to the row tile (``relu^2(0) = 0``,
    so padding adds nothing and gets no gradient) and sliced back."""
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"relu2_mlp implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    bs = block_sizes or BlockSizes.get_default()
    rows = x.shape[1]
    padded = -(-rows // bs.bm) * bs.bm
    out = _relu2_mlp(_pad_rows(x, padded), w_up, w_down, implementation, bs)
    return out[:, :rows]
