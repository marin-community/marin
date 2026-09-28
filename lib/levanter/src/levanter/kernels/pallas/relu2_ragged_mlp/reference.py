# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Plain-JAX versions of the ragged ReLU² expert MLP: the per-group oracle and the XLA ragged-dot GEMMs."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

_WEIGHT_GRAD_DIM_NUMS = jax.lax.RaggedDotDimensionNumbers(
    dot_dimension_numbers=(((0,), (0,)), ((), ())),
    lhs_ragged_dimensions=(0,),
    rhs_group_dimensions=[],
)


def relu2_ragged_mlp_reference(
    x: Float[Array, "M K"],
    w_up: Float[Array, "G K N"],
    w_down: Float[Array, "G N Mout"],
    group_sizes: Int[Array, "G"],
) -> Float[Array, "M Mout"]:
    """``relu(x_g W_up_g)^2 W_down_g`` one group at a time; rows past ``sum(group_sizes)`` are zero."""
    ends = jnp.cumsum(group_sizes)
    rows = jnp.arange(x.shape[0])
    out = jnp.zeros((x.shape[0], w_down.shape[-1]), jnp.float32)
    for g in range(w_up.shape[0]):
        in_group = ((rows >= ends[g] - group_sizes[g]) & (rows < ends[g]))[:, None]
        pre = jnp.dot(x, w_up[g], preferred_element_type=jnp.float32)
        post = jnp.square(jax.nn.relu(pre)).astype(x.dtype)
        out = out + jnp.where(in_group, jnp.dot(post, w_down[g], preferred_element_type=jnp.float32), 0.0)
    return out.astype(x.dtype)


def gmm_reference(
    a: Float[Array, "M Kc"], b: Float[Array, "G Kc N"], group_sizes: Int[Array, "G"]
) -> Float[Array, "M N"]:
    return jax.lax.ragged_dot(a, b, group_sizes, preferred_element_type=jnp.float32)


def tgmm_reference(
    a: Float[Array, "M Ka"], b: Float[Array, "M Nb"], group_sizes: Int[Array, "G"]
) -> Float[Array, "G Ka Nb"]:
    out = jax.lax.ragged_dot_general(a, b, group_sizes, _WEIGHT_GRAD_DIM_NUMS, preferred_element_type=jnp.float32)
    return out.astype(a.dtype)
