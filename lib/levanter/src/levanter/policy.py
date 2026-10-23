# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Clipped policy objectives with explicit token and sequence weights."""

from dataclasses import dataclass
from enum import StrEnum

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray

from levanter.metrics import Metric, ReductionType

LOG_PROB_DELTA_CLIP = 20.0


class KlGradient(StrEnum):
    DETACHED = "detached"
    DIFFERENTIABLE = "differentiable"


@dataclass(frozen=True)
class ClippedPolicyConfig:
    eps_clip_low: float
    eps_clip_high: float
    kl_loss_coef: float
    kl_gradient: KlGradient


def _clamp(value: NamedArray, low: float, high: float) -> NamedArray:
    # Torch clamp passes the full gradient at either boundary; JAX clip halves it.
    return hax.where(value < low, low, hax.where(value > high, high, value))


def clipped_policy_loss(
    log_probs: NamedArray,
    old_log_probs: NamedArray,
    reference_log_probs: NamedArray | None,
    advantages: NamedArray,
    policy_weights: NamedArray,
    kl_weights: NamedArray,
    *,
    config: ClippedPolicyConfig,
    accumulation_steps: int,
) -> tuple[jax.Array, dict[str, Metric]]:
    """Return loss and metrics for Levanter gradient accumulation.

    Pass the number of execution microbatches as accumulation_steps to cancel
    Levanter's gradient averaging. All other inputs are sliced from the full
    batch; old/reference logprobs, advantages and weights are detached.
    Detached KL affects reported loss but contributes no gradient. Choose that
    behavior explicitly in config.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    old_log_probs = jax.tree_util.tree_map(jax.lax.stop_gradient, old_log_probs)
    advantages = jax.tree_util.tree_map(jax.lax.stop_gradient, advantages)
    policy_weights = jax.tree_util.tree_map(jax.lax.stop_gradient, policy_weights)
    kl_weights = jax.tree_util.tree_map(jax.lax.stop_gradient, kl_weights)
    ratio = hax.exp(
        _clamp((log_probs - old_log_probs).astype(jnp.float32), -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
    ).astype(log_probs.dtype)
    surrogate = ratio * advantages
    clipped = _clamp(ratio, 1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    policy_loss = -(hax.minimum(surrogate, clipped) * policy_weights).sum().scalar()
    kl_loss = jnp.zeros((), dtype=log_probs.dtype)
    if config.kl_loss_coef != 0:
        if reference_log_probs is None:
            raise ValueError("reference_log_probs are required for nonzero KL coefficient")
        reference_log_probs = jax.tree_util.tree_map(jax.lax.stop_gradient, reference_log_probs)
        delta = _clamp(reference_log_probs - log_probs, -LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
        kl = _clamp(hax.exp(delta) - delta - 1, -10.0, 10.0)
        if config.kl_gradient == KlGradient.DETACHED:
            kl = jax.tree_util.tree_map(jax.lax.stop_gradient, kl)
        kl_loss = (kl * kl_weights).sum().scalar()
    loss = policy_loss + config.kl_loss_coef * kl_loss
    metrics = {
        "loss": Metric.from_value(loss, ReductionType.SUM),
        "policy_loss": Metric.from_value(policy_loss, ReductionType.SUM),
        "policy_kl": Metric.from_value(kl_loss, ReductionType.SUM),
        "ppo_clip_ratio": Metric.from_value(
            ((clipped < surrogate) * policy_weights).sum().scalar(), ReductionType.SUM
        ),
    }
    selected = clipped < surrogate
    low_pressure = ratio < 1 - config.eps_clip_low
    high_pressure = ratio > 1 + config.eps_clip_high
    for name, condition in (
        ("ppo_clip_ratio_low", selected & low_pressure),
        ("ppo_clip_ratio_high", selected & high_pressure),
        ("ppo_clip_pressure_low", low_pressure),
        ("ppo_clip_pressure_high", high_pressure),
        ("ppo_ratio_exact_unit_fraction", ratio == 1),
    ):
        metrics[name] = Metric.from_value((condition * policy_weights).sum().scalar(), ReductionType.SUM)
    metrics["log_ratio_abs_max"] = Metric.from_value(
        hax.where(policy_weights > 0, hax.abs(log_probs - old_log_probs), 0).max().scalar(),
        ReductionType.MAX,
    )
    return loss * accumulation_steps, metrics
