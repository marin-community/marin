# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-sided EMA gradient-Gram preconditioning of the momentum polar factor."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig
from levanter.optim.util import flatten_linear_layers, unflatten_linear_layers, zeropower_via_newtonschulz5


NEWTON_SCHULZ = "newton_schulz"
SVD = "svd"


class GramState(NamedTuple):
    momentum: jax.Array
    left: jax.Array
    right: jax.Array


class EmaGramState(NamedTuple):
    matrices: tuple[GramState, ...]


def init_gram_state(matrix: jax.Array) -> GramState:
    """Allocate FP32 buffers; leading dimensions index independent matrices."""
    batch, rows, cols = matrix.shape[:-2], matrix.shape[-2], matrix.shape[-1]
    return GramState(
        jnp.zeros(matrix.shape, jnp.float32),
        jnp.zeros((*batch, rows, rows), jnp.float32),
        jnp.zeros((*batch, cols, cols), jnp.float32),
    )


@jax.named_call
def update_gram_state(gradient: jax.Array, state: GramState, beta1: float, beta2: float) -> GramState:
    """Accumulate the already globally reduced, clipped gradient before taking powers."""
    g = gradient.astype(jnp.float32)
    gt = jnp.swapaxes(g, -1, -2)
    return GramState(
        beta1 * state.momentum + (1 - beta1) * g,
        beta2 * state.left + (1 - beta2) * jnp.matmul(g, gt, precision=jax.lax.Precision.HIGHEST),
        beta2 * state.right + (1 - beta2) * jnp.matmul(gt, g, precision=jax.lax.Precision.HIGHEST),
    )


@jax.named_call
def relative_gram_power(gram: jax.Array, alpha: float, ridge: float) -> jax.Array:
    """Return (Gram / mean eigenvalue + ridge I)^(-alpha), or I for a zero Gram."""
    gram = (gram + jnp.swapaxes(gram, -1, -2)) * 0.5
    mean = jnp.trace(gram, axis1=-2, axis2=-1) / gram.shape[-1]
    normalized = gram / jnp.where(mean > 0, mean, 1)[..., None, None]
    values, vectors = jnp.linalg.eigh(normalized)
    powers = (jnp.maximum(values, 0) + ridge) ** (-alpha)
    power = jnp.matmul(
        vectors * powers[..., None, :], jnp.swapaxes(vectors, -1, -2), precision=jax.lax.Precision.HIGHEST
    )
    return jnp.where((mean > 0)[..., None, None], power, jnp.eye(gram.shape[-1], dtype=gram.dtype))


def exact_polar(matrix: jax.Array) -> jax.Array:
    """SVD partial isometry with numerical zero modes removed (offline reference)."""
    u, singular, vh = jnp.linalg.svd(matrix, full_matrices=False)
    threshold = jnp.finfo(matrix.dtype).eps * max(matrix.shape[-2:]) * singular[..., :1]
    return jnp.matmul(u * (singular > threshold)[..., None, :], vh, precision=jax.lax.Precision.HIGHEST)


@jax.named_call
def gram_direction(
    state: GramState, *, alpha: float, ridge: float, polar_steps: int, polar_method: str = NEWTON_SCHULZ
) -> jax.Array:
    """Return a positive descent direction without advancing the supplied EMA state."""
    m = state.momentum
    if alpha:
        left = relative_gram_power(state.left, alpha, ridge)
        right = relative_gram_power(state.right, alpha, ridge)
        inner = jnp.matmul(
            jnp.matmul(left, m, precision=jax.lax.Precision.HIGHEST), right, precision=jax.lax.Precision.HIGHEST
        )
    else:
        inner = m
    if polar_method == SVD:
        polar = exact_polar(inner)
    elif polar_method == NEWTON_SCHULZ:
        flat = inner.reshape((-1, *inner.shape[-2:]))
        polar = jax.vmap(partial(zeropower_via_newtonschulz5, steps=polar_steps))(flat).reshape(inner.shape)
    else:
        raise ValueError(f"Unknown polar method: {polar_method}")
    if alpha:
        return jnp.matmul(
            jnp.matmul(left, polar, precision=jax.lax.Precision.HIGHEST), right, precision=jax.lax.Precision.HIGHEST
        )
    return polar


def scale_by_ema_gram(
    *, alpha: float, ridge: float, beta1: float = 0.95, beta2: float = 0.99, polar_steps: int = 5
) -> optax.GradientTransformation:
    """Transform matrix-only trees, preserving Haliax Linear layouts and scan axes.

    Every step refreshes both full (unblocked) Grams, including for alpha=0.
    Call after gradient accumulation/reduction and global clipping. No Nesterov,
    bias correction, aspect-ratio scaling, or update-norm constraint is applied.
    """
    if alpha < 0 or ridge <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or polar_steps < 1:
        raise ValueError("Invalid EMA Gram hyperparameters")

    def init_fn(params):
        leaves = jax.tree.leaves(flatten_linear_layers(params))
        if any(p.ndim < 2 for p in leaves):
            raise ValueError("EMA Gram accepts matrix parameters only; route vectors to Adam")
        return EmaGramState(tuple(init_gram_state(p) for p in leaves))

    def update_fn(updates, state, params=None):
        del params  # Optax permits callers to pass this keyword.
        leaves, treedef = jax.tree.flatten(flatten_linear_layers(updates))
        states = tuple(update_gram_state(g, s, beta1, beta2) for g, s in zip(leaves, state.matrices, strict=True))
        directions = [
            gram_direction(s, alpha=alpha, ridge=ridge, polar_steps=polar_steps).astype(g.dtype)
            for g, s in zip(leaves, states, strict=True)
        ]
        return unflatten_linear_layers(updates, treedef.unflatten(directions)), EmaGramState(states)

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("ema_gram_polar")
@dataclass(frozen=True)
class EmaGramConfig(OptimizerConfig):
    """EMA-Gram matrices and fixed AdamW nonmatrices, with one global clip first."""

    alpha: float = 0.25
    ridge: float = 1e-4
    momentum: float = 0.95
    gram_beta: float = 0.99
    polar_steps: int = 5
    adam_lr: float = 3e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.1

    def build(self, num_train_steps):
        def optimizer(learning_rate, adam_lr):
            matrices = optax.chain(
                scale_by_ema_gram(
                    alpha=self.alpha,
                    ridge=self.ridge,
                    beta1=self.momentum,
                    beta2=self.gram_beta,
                    polar_steps=self.polar_steps,
                ),
                optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()),
                optax.scale(-learning_rate),
            )
            nonmatrices = optax.adamw(
                adam_lr,
                b1=self.adam_beta1,
                b2=self.adam_beta2,
                eps=self.adam_epsilon,
                weight_decay=self.weight_decay,
                mask=self.build_weight_decay_mask(),
            )
            # Reuse Muon's embedding/head/bias routing, without its aspect-ratio scaling.
            mask = partial(MuonConfig.create_mask, self, use_kimi_scaling=True)
            return optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.multi_transform({"muon": matrices, "adamw": nonmatrices}, mask),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps),
            adam_lr=self.lr_scheduler(num_train_steps, override_lr=self.adam_lr),
        )
