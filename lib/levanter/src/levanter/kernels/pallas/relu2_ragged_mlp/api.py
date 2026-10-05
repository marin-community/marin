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
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .pallas_gpu import BlockSizes, Epilogue, gmm, tgmm
from .reference import gmm_reference, tgmm_reference

Implementation: TypeAlias = Literal["reference", "pallas_gpu", "pallas_interpret"]
IMPLEMENTATIONS: tuple[Implementation, ...] = ("reference", "pallas_gpu", "pallas_interpret")
_DEFAULT_BLOCKS = BlockSizes.get_default()


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


class Relu2RaggedResiduals(NamedTuple):
    """What the backward reads: the inputs and ``post = relu(x W_up)^2``, never ``pre`` or the output."""

    x: Float[Array, "M K"]
    w_up: Float[Array, "G K N"]
    w_down: Float[Array, "G N Mout"]
    group_sizes: Int[Array, "G"]
    post: Float[Array, "M N"]


def relu2_ragged_mlp_forward(
    x: Float[Array, "M K"],
    w_up: Float[Array, "G K N"],
    w_down: Float[Array, "G N Mout"],
    group_sizes: Int[Array, "G"],
    *,
    implementation: Implementation,
    block_sizes: BlockSizes | None = None,
) -> tuple[Float[Array, "M Mout"], Relu2RaggedResiduals]:
    """``relu2_ragged_mlp``'s forward and the residuals ``relu2_ragged_mlp_backward`` reads."""
    _check_implementation(implementation)
    gemm = partial(_gmm, trans_b=False, implementation=implementation, block_sizes=block_sizes or _DEFAULT_BLOCKS)
    post = gemm(x, w_up, group_sizes, None, epilogue=Epilogue.RELU2)
    out = gemm(post, w_down, group_sizes, None, epilogue=Epilogue.NONE)
    return out, Relu2RaggedResiduals(x, w_up, w_down, group_sizes, post)


def relu2_ragged_mlp_backward(
    residuals: Relu2RaggedResiduals,
    g: Float[Array, "M Mout"],
    *,
    implementation: Implementation,
    block_sizes: BlockSizes | None = None,
) -> tuple[Float[Array, "M K"], Float[Array, "G K N"], Float[Array, "G N Mout"], Float[Array, "M"]]:
    """The backward of ``relu2_ragged_mlp_forward``, plus each row's ``<y, g>`` in fp32.

    ``y = post W_down`` row by row, so ``<y, g> = <post, g W_down^T>``, and with
    ``d pre = (g W_down^T) 2 sqrt(post)`` that is ``<d pre, sqrt(post)> / 2``: the row dot needs
    neither ``y`` nor ``g W_down^T``, only the two tensors the backward already holds. It is the
    gradient of a per-row scale applied to the output. Rows past ``sum(group_sizes)`` of ``g`` are
    never read, and those rows of ``dx`` and of the row dot are zero.

    Returns ``(dx, dw_up, dw_down, output_dot_cotangent)``.
    """
    _check_implementation(implementation)
    block_sizes = block_sizes or _DEFAULT_BLOCKS
    x, w_up, w_down, group_sizes, post = residuals
    g = g.astype(x.dtype)
    gemm_t = partial(_gmm, trans_b=True, implementation=implementation, block_sizes=block_sizes)
    d_w_down = _tgmm(post, g, group_sizes, implementation, block_sizes).astype(w_down.dtype)
    dpre = gemm_t(g, w_down, group_sizes, post, epilogue=Epilogue.RELU2_DPRE)
    output_dot_cotangent = 0.5 * jnp.sum(dpre.astype(jnp.float32) * jnp.sqrt(post.astype(jnp.float32)), axis=-1)
    d_w_up = _tgmm(x, dpre, group_sizes, implementation, block_sizes).astype(w_up.dtype)
    dx = gemm_t(dpre, w_up, group_sizes, None, epilogue=Epilogue.NONE)
    return dx, d_w_up, d_w_down, output_dot_cotangent


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _relu2_ragged_mlp(x, w_up, w_down, group_sizes, implementation: Implementation, block_sizes: BlockSizes):
    out, _ = _relu2_ragged_mlp_fwd(x, w_up, w_down, group_sizes, implementation, block_sizes)
    return out


def _relu2_ragged_mlp_fwd(x, w_up, w_down, group_sizes, implementation, block_sizes):
    return relu2_ragged_mlp_forward(
        x, w_up, w_down, group_sizes, implementation=implementation, block_sizes=block_sizes
    )


def _relu2_ragged_mlp_bwd(implementation, block_sizes, residuals, g):
    dx, d_w_up, d_w_down, _ = relu2_ragged_mlp_backward(
        residuals, g, implementation=implementation, block_sizes=block_sizes
    )
    return dx, d_w_up, d_w_down, None


_relu2_ragged_mlp.defvjp(_relu2_ragged_mlp_fwd, _relu2_ragged_mlp_bwd)


def _check_implementation(implementation: str) -> None:
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"relu2_ragged_mlp implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")


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
    _check_implementation(implementation)
    return _relu2_ragged_mlp(x, w_up, w_down, group_sizes, implementation, block_sizes or _DEFAULT_BLOCKS)
