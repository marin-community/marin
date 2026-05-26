# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AMUSE: Anytime Muon with Stable Gradient Evaluation (arxiv 2605.22432).

AMUSE is a time-varying Schedule-Free (SF) wrapper around any base optimizer
(Muon for matrix-valued parameters, AdamW-without-β1 for the rest). It keeps
three sequences:

    Y_t = (1 - β_t) Z_t + β_t X_t          # gradient-evaluation point
    Z_{t+1} = (1 - η_t λ) Z_t + Δz         # base sequence (Δz comes from base opt)
    X_{t+1} = (1 - c_{t+1}) X_t + c_{t+1} Z_{t+1}    # averaged sequence (used for inference)

with a time-varying β_t schedule (Eq. 1 of the paper):

    β_t = β_1                                  for t ≤ T_0
    β_t = 1 - ((T_0 - 1)/(t - 1))^ρ * (1 - β_1) for t > T_0

The LR-weighted averaging coefficient (matching schedule_free in
``optax.contrib`` / Defazio et al. 2024) is

    c_{t+1} = η_t² / Σ_{i=1}^{t} η_i²

Convention: the model's stored parameters are kept at Y (the gradient
evaluation point) so the trainer's grad lands at the right place. Z is
stored in optimizer state. For inference, ``amuse_eval_params(state, params)``
returns X via inverse interpolation X = (Y - (1-β_t) Z) / β_t.
"""

from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree as optax_tree


class AmuseState(NamedTuple):
    """State for the AMUSE wrapper.

    z: base sequence (same pytree structure as params).
    base_state: state of the wrapped base optimizer (e.g., Muon momentum, Adam V).
    weight_sum: running Σ η_i^weight_lr_power, for the c_t averaging coefficient.
    max_lr: max learning rate seen so far (matches optax.contrib.schedule_free).
    step: step counter (1-indexed, advanced before computing β_t / c_t).
    """

    z: optax.Params
    base_state: optax.OptState
    weight_sum: chex.Array
    max_lr: chex.Array
    step: chex.Array


def _amuse_beta_t(
    step: chex.Array,
    warmup_steps: int,
    beta1: float,
    rho: float,
) -> chex.Array:
    """β_t schedule from AMUSE (Eq. 1).

    During warmup (step ≤ T_0) β_t = β_1. Afterwards β_t = 1 - ((T_0-1)/(t-1))^ρ * (1 - β_1).
    """
    t = step.astype(jnp.float32)
    t0 = jnp.asarray(float(warmup_steps), dtype=jnp.float32)
    # ratio = (T_0 - 1) / (t - 1), clipped to [0, 1] for t > T_0.
    denom = jnp.maximum(t - 1.0, 1.0)
    ratio = (t0 - 1.0) / denom
    decay = jnp.power(ratio, jnp.asarray(float(rho), dtype=jnp.float32))
    beta_post = 1.0 - decay * (1.0 - float(beta1))
    return jnp.where(t <= t0, jnp.asarray(float(beta1), dtype=jnp.float32), beta_post)


def amuse(
    base_optimizer: optax.GradientTransformation,
    learning_rate: optax.ScalarOrSchedule,
    beta1: float = 0.6,
    rho: float = 0.8,
    warmup_steps: int = 2000,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
) -> optax.GradientTransformation:
    """AMUSE wrapper.

    The model's parameters represent Y_t (the gradient-evaluation point); grads
    received by ``update_fn`` are ∇L(Y_t). The base_optimizer is expected to
    return updates Δz that include the ``-η_t`` scaling (so applying them to
    Z gives the base sequence step). Weight decay (``λ``) is applied to Z via
    the coupled form ``Z ← (1 - η_t λ) Z + Δz``.

    Args:
        base_optimizer: Any optax transform that maps grad → -η_t * direction
            (e.g., scale_with_grug_muonh, or a chain ending in scale_by_lr).
            It should NOT have its own SF-style β_1 momentum (AMUSE replaces
            that with the time-varying β_t interpolation). Muon's μ momentum
            is fine — it operates pre-orthogonalization, distinct from SF.
        learning_rate: schedule callable; used for the c_{t+1} averaging
            coefficient and for weight decay scaling.
        beta1: initial SF interpolation coefficient. Defaults to 0.6 per the
            paper's d512 LLM recipe (β_1 ∈ {0.4, 0.6}).
        rho: rate of β_t growth toward 1. Defaults to 0.8 per the paper
            (ρ ∈ {0.6, 0.8}). ρ=0 reduces AMUSE to fixed-β SF.
        warmup_steps: T_0. β_t = β_1 below this, schedule kicks in after.
        weight_decay: λ applied to Z.
        weight_lr_power: exponent on η in the LR-weighted average. Defaults
            to 2.0, matching optax.contrib.schedule_free / Defazio 2024.
    """

    def _lr(step: chex.Array) -> chex.Array:
        if callable(learning_rate):
            return jnp.asarray(learning_rate(step), dtype=jnp.float32)
        return jnp.asarray(float(learning_rate), dtype=jnp.float32)

    def init_fn(params: optax.Params) -> AmuseState:
        z = jax.tree.map(lambda t: t.copy(), params)
        return AmuseState(
            z=z,
            base_state=base_optimizer.init(params),
            weight_sum=jnp.zeros([], jnp.float32),
            max_lr=jnp.zeros([], jnp.float32),
            step=jnp.zeros([], jnp.int32),
        )

    def update_fn(grads, state: AmuseState, params=None):
        if params is None:
            raise ValueError("AMUSE requires params for the Y→X→Y reconstruction")

        # params here is Y_t. We don't need to extract X_t to compute the base
        # step (the base optimizer operates on Z), but we DO need X_t when
        # computing the new Y_{t+1}. We recover X_t from the SF identity:
        #     Y_t = (1 - β_t) Z_t + β_t X_t  =>  X_t = (Y_t - (1-β_t) Z_t) / β_t
        next_step = state.step + 1
        beta_t = _amuse_beta_t(state.step, warmup_steps, beta1, rho)
        beta_t = jnp.maximum(beta_t, 1e-8)  # avoid div-by-zero in X recovery
        x_t = jax.tree.map(lambda y, z: (y - (1.0 - beta_t) * z) / beta_t, params, state.z)

        # Base optimizer step: Δz_grad = base_optimizer(grad, base_state, params=z)
        # We pass z (not params=Y) so the base optimizer's internal state sees the
        # right operating point; for stateless base optimizers this doesn't matter.
        base_update, new_base_state = base_optimizer.update(grads, state.base_state, params=state.z)

        # Decoupled weight decay on Z: Z ← (1 - η λ) Z + Δz_grad.
        lr_t = _lr(next_step)
        if weight_decay and weight_decay > 0.0:
            wd = lr_t * float(weight_decay)
            new_z = jax.tree.map(lambda z, du: (1.0 - wd) * z + du, state.z, base_update)
        else:
            new_z = jax.tree.map(lambda z, du: z + du, state.z, base_update)

        # LR-weighted online average (matches optax.contrib.schedule_free): the
        # weight for step t is max_lr^p, normalized by Σ over the trajectory.
        new_max_lr = jnp.maximum(state.max_lr, lr_t)
        weight = new_max_lr ** float(weight_lr_power)
        new_weight_sum = state.weight_sum + weight
        c_next = jnp.where(new_weight_sum > 0.0, weight / new_weight_sum, 0.0)
        new_x = jax.tree.map(lambda x, z: (1.0 - c_next) * x + c_next * z, x_t, new_z)

        # New Y_{t+1} = (1 - β_{t+1}) Z_{t+1} + β_{t+1} X_{t+1}.
        beta_next = _amuse_beta_t(next_step, warmup_steps, beta1, rho)
        new_y = jax.tree.map(lambda z, x: (1.0 - beta_next) * z + beta_next * x, new_z, new_x)

        # Return update = new_Y - Y_t so optax.apply_updates(Y_t, update) = Y_{t+1}.
        update = jax.tree.map(lambda y_new, y_old: y_new - y_old, new_y, params)
        new_state = AmuseState(
            z=new_z,
            base_state=new_base_state,
            weight_sum=new_weight_sum,
            max_lr=new_max_lr,
            step=next_step,
        )
        return update, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def amuse_eval_params(state: AmuseState, params, beta1: float = 0.6, rho: float = 0.8, warmup_steps: int = 2000):
    """Recover X_t (the averaged sequence used for inference) from Y_t (params) and Z_t (state.z).

    X_t = (Y_t - (1 - β_t) Z_t) / β_t. Uses β_t evaluated at ``state.step``
    (which is the count of completed steps).
    """
    beta_t = _amuse_beta_t(state.step, warmup_steps, beta1, rho)
    beta_t = jnp.maximum(beta_t, 1e-8)
    return jax.tree.map(lambda y, z: (y - (1.0 - beta_t) * z) / beta_t, params, state.z)


__all__ = ["AmuseState", "amuse", "amuse_eval_params"]
