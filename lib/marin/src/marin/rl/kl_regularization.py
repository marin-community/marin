# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KL regularization helpers for RL objectives.

The ``k2``/``k3`` names follow the sampled-KL estimator shorthand introduced in
John Schulman's "Approximating KL Divergence" and reused in recent RLHF
analyses such as "A Comedy of Estimators" and "Rethinking KL Regularization in
RLHF".
"""

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp


class KLMode(StrEnum):
    """Supported trainer-side KL surrogates for the RL loss.

    The ``k2``/``k3`` labels follow Schulman's sampled-KL estimator shorthand.
    """

    NONE = "none"
    K3_LOSS = "k3_loss"
    K2_LOSS = "k2_loss"


@dataclass(frozen=True)
class KLConfig:
    """Configuration for trainer-owned KL regularization."""

    mode: KLMode = KLMode.NONE
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if self.mode == KLMode.NONE and self.beta != 0.0:
            raise ValueError("beta must be 0.0 when KL mode is NONE")

    def enabled(self) -> bool:
        """Return whether KL regularization is active."""
        return self.mode != KLMode.NONE and self.beta > 0.0


@dataclass(frozen=True)
class KLStatistics:
    """Reported KL estimates for the sampled rollout tokens."""

    k1_mean: jax.Array
    k2_mean: jax.Array
    k3_mean: jax.Array


def token_log_ratio(current_logprobs: jax.Array, reference_logprobs: jax.Array) -> jax.Array:
    """Return per-token log ratio between current and reference policies."""
    return current_logprobs - reference_logprobs


def k2_from_log_ratio(log_ratio: jax.Array) -> jax.Array:
    """Return the quadratic ``k2`` KL surrogate for sampled tokens."""
    return 0.5 * jnp.square(log_ratio)


def k3_from_log_ratio(log_ratio: jax.Array) -> jax.Array:
    """Return the numerically stable ``k3`` KL surrogate for sampled tokens."""
    return jnp.expm1(-log_ratio) + log_ratio


def masked_response_mean(values: jax.Array, loss_masks: jax.Array) -> jax.Array:
    """Average per-example masked response values across the batch."""
    return jnp.mean(jnp.sum(values * loss_masks, axis=1) / jnp.sum(loss_masks, axis=1))


def kl_penalty_from_log_ratio(log_ratio: jax.Array, mode: KLMode) -> jax.Array:
    """Return the optimization penalty for the selected KL mode."""
    if mode == KLMode.K2_LOSS:
        return k2_from_log_ratio(log_ratio)
    if mode == KLMode.K3_LOSS:
        return k3_from_log_ratio(log_ratio)

    raise ValueError(f"KL penalty is undefined for mode {mode}")


def kl_statistics_from_log_ratio(log_ratio: jax.Array, loss_masks: jax.Array) -> KLStatistics:
    """Return reported KL estimates over masked response tokens."""
    return KLStatistics(
        k1_mean=masked_response_mean(log_ratio, loss_masks),
        k2_mean=masked_response_mean(k2_from_log_ratio(log_ratio), loss_masks),
        k3_mean=masked_response_mean(k3_from_log_ratio(log_ratio), loss_masks),
    )
