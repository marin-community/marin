# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sigmoid top-k gates (SwitchHead-style) with optional load balancing, shared by the mixture variants."""

import functools

import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float


def switchhead_weights(
    x: Float[Array, "B S D"], gate: Float[Array, "D GE"], experts: int, topk: int
) -> Float[Array, "B S G E"]:
    """SwitchHead's non-competitive expert weights: ``sigmoid(x W_gate)`` on each head's top-k logits, else 0."""
    logits = rearrange(jnp.einsum("bsd,dg->bsg", x, gate).astype(jnp.float32), "b s (g e) -> b s g e", e=experts)
    weights = jax.nn.sigmoid(logits)
    if topk >= experts:
        return weights
    threshold = jax.lax.top_k(logits, topk)[0][..., -1:]
    return jnp.where(logits >= threshold, weights, 0.0)


def mixture_weights(
    x: Float[Array, "... D"],
    gate: Float[Array, "D E"],
    experts: int,
    topk: int,
    selection_bias: Float[Array, " E"] | None = None,
    entropy_weight: float = 0.0,
    renorm: bool = False,
) -> Float[Array, "... E"]:
    """SwitchHead's sigmoid top-k weights for one group of ``experts`` blocks, over any leading axes of ``x``
    (computed in place, so the weights keep ``x``'s batch sharding). ``selection_bias`` (``LatentMixBalance.BIAS``)
    shifts only which blocks are picked; ``entropy_weight`` (``LatentMixBalance.ENTROPY``) adds the balance
    regularizer's gradient to the logits; ``renorm`` scales each token's kept weights to sum to 1."""
    logits = jnp.einsum("...d,de->...e", x, gate).astype(jnp.float32)
    if entropy_weight:
        logits = _entropy_balanced(logits, entropy_weight)
    weights = jax.nn.sigmoid(logits)
    if topk >= experts:
        return weights
    select = logits if selection_bias is None else logits + jax.lax.stop_gradient(selection_bias)
    threshold = jax.lax.top_k(select, topk)[0][..., -1:]
    chosen = select >= threshold
    if selection_bias is not None:
        # The bias gets the load error as its gradient; the 0 * keeps its custom VJP on the backward path.
        load = jnp.mean(chosen.astype(jnp.float32), axis=tuple(range(chosen.ndim - 1)))
        weights = weights + 0.0 * _load_error_grad(selection_bias, jax.lax.stop_gradient(load))
    kept = jnp.where(chosen, weights, 0.0)
    if renorm:
        kept = kept / jnp.sum(kept, axis=-1, keepdims=True)
    return kept


@jax.custom_vjp
def _load_error_grad(bias: Float[Array, " E"], load: Float[Array, " E"]) -> Float[Array, " E"]:
    """Identity on ``bias`` whose backward returns ``load - mean(load)``: a sign-SGD step on it is the
    auxiliary-loss-free balance update (overloaded blocks' biases fall)."""
    return bias


def _load_error_grad_fwd(bias, load):
    return bias, load


def _load_error_grad_bwd(load, g):
    del g
    return load - jnp.mean(load), jnp.zeros_like(load)


_load_error_grad.defvjp(_load_error_grad_fwd, _load_error_grad_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _entropy_balanced(logits: Float[Array, "... E"], weight: float) -> Float[Array, "... E"]:
    """Identity on ``logits`` whose backward adds ``d(weight * -H(mean_t softmax(logits_t))) / d logits``."""
    return logits


def _entropy_balanced_fwd(logits, weight):
    return logits, logits


def _entropy_balanced_bwd(weight, logits, g):
    def neg_entropy(lg):
        mean_p = jnp.mean(jax.nn.softmax(lg, axis=-1).reshape(-1, lg.shape[-1]), axis=0)
        return weight * jnp.sum(mean_p * jnp.log(mean_p + 1e-9))

    return (g + jax.grad(neg_entropy)(logits),)


_entropy_balanced.defvjp(_entropy_balanced_fwd, _entropy_balanced_bwd)
