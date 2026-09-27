# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Vanilla JAX oracle for the batched ungated ReLU² expert MLP ``relu(x W_up)^2 W_down``."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def relu2_up_reference(x: Float[Array, "E R K"], w_up: Float[Array, "E K N"]) -> Float[Array, "E R N"]:
    """``relu(x @ w_up)^2`` per expert, accumulated in fp32 and cast to ``x.dtype``."""
    pre = jnp.einsum("erk,ekn->ern", x, w_up, preferred_element_type=jnp.float32)
    return jnp.square(jax.nn.relu(pre)).astype(x.dtype)


def relu2_dpre_reference(
    g: Float[Array, "E R M"], w_down: Float[Array, "E N M"], post: Float[Array, "E R N"]
) -> Float[Array, "E R N"]:
    """Cotangent of the pre-activation, ``(g @ w_down^T) * 2 sqrt(post)`` (``relu(pre) = sqrt(post)``)."""
    dpost = jnp.einsum("erm,enm->ern", g, w_down, preferred_element_type=jnp.float32)
    return (dpost * 2.0 * jnp.sqrt(post.astype(jnp.float32))).astype(g.dtype)


def relu2_mlp_reference(
    x: Float[Array, "E R K"], w_up: Float[Array, "E K N"], w_down: Float[Array, "E N M"]
) -> Float[Array, "E R M"]:
    post = relu2_up_reference(x, w_up)
    return jnp.einsum("ern,enm->erm", post, w_down, preferred_element_type=jnp.float32).astype(x.dtype)
