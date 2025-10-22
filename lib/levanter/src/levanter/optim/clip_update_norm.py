# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import optax
from jax import numpy as jnp
from jax.tree_util import register_dataclass

import levanter.tracker


@register_dataclass
@dataclass(frozen=True)
class ClipUpdateNormState:
    """State for the ClipUpdateNorm gradient transformation."""

    update_norms: jax.Array  # Stores last 'rolling_interval_length' update_norms
    valid_mask: jax.Array  # Boolean mask for valid entries in update_norms
    current_idx: jax.Array  # Current index in the rolling window (circular buffer)
    count: jax.Array  # Number of valid entries currently in the buffer (up to buffer_size)


@dataclass(frozen=True)
class ClipUpdateNormConfig:
    """Configuration for the ClipUpdateNorm gradient transformation."""

    rolling_interval_length: int = 128
    sigma_factor: float = 3.0

    def build(self) -> optax.GradientTransformation:
        return clip_update_by_historical_norm(self.rolling_interval_length, self.sigma_factor)


def clip_update_by_historical_norm(
    rolling_interval_length: int = 128, sigma_factor: float = 3.0
) -> optax.GradientTransformation:
    """
    An optax gradient transformation that clips the norm of the update.

    The clipping threshold is dynamic, based on the historical mean and standard deviation of update norms.
    Specifically, the maximum allowed norm is `historical_mean + sigma_factor * historical_stddev`.

    The decision is made based on a rolling window of the last `rolling_interval_length` steps.
    If there are fewer than `rolling_interval_length // 2` steps in the history, no clipping is applied.
    """

    def init_fn(params):
        del params
        buffer_size = rolling_interval_length
        update_norms = jnp.zeros((buffer_size,), dtype=jnp.float32)
        valid_mask = jnp.full((buffer_size,), False, dtype=jnp.bool_)
        current_idx = jnp.zeros((), dtype=jnp.int32)
        count = jnp.zeros((), dtype=jnp.int32)

        return ClipUpdateNormState(
            update_norms=update_norms,
            current_idx=current_idx,
            count=count,
            valid_mask=valid_mask,
        )

    def update_fn(updates, state: ClipUpdateNormState, params=None):
        del params
        global_norm = optax.global_norm(updates)

        min_data_points = jnp.maximum(2, rolling_interval_length // 2)
        can_clip_based_on_history = state.count >= min_data_points

        update_norm_mean_history = jnp.mean(state.update_norms, where=state.valid_mask)
        update_norm_std_history = jnp.std(state.update_norms, where=state.valid_mask)
        update_norm_std_safe_history = jnp.maximum(update_norm_std_history, 1e-6)

        update_norm_threshold = update_norm_mean_history + sigma_factor * update_norm_std_safe_history

        # don't clip if we don't have enough data by setting max_norm to infinity
        max_norm = jnp.where(can_clip_based_on_history, update_norm_threshold, jnp.inf)

        # similar to optax.clip_by_global_norm
        g_norm = global_norm
        trigger = g_norm > max_norm

        scaling_factor = jnp.where(trigger, max_norm / jnp.maximum(g_norm, 1e-6), 1.0)

        clipped_updates = jax.tree_util.tree_map(lambda t: t * scaling_factor, updates)

        levanter.tracker.jit_log(
            {
                "optim/clip_update/did_clip": (scaling_factor < 1.0).astype(jnp.float32),
                "optim/clip_update/pre_clip_norm": global_norm,
                "optim/clip_update/update_norm_threshold": update_norm_threshold,
                "optim/clip_update/post_clip_norm": jnp.minimum(max_norm, global_norm),
            }
        )

        # Update state with the current update norm
        buffer_size = state.update_norms.shape[0]

        # update the circular buffer. we use the *unclipped* global norm as the history to prevent collapse
        new_update_norms = state.update_norms.at[state.current_idx].set(global_norm)
        new_valid_mask = state.valid_mask.at[state.current_idx].set(True)
        new_current_idx = (state.current_idx + 1) % buffer_size
        new_count = jnp.minimum(state.count + 1, buffer_size)

        new_state = ClipUpdateNormState(
            update_norms=new_update_norms,
            valid_mask=new_valid_mask,
            current_idx=new_current_idx,
            count=new_count,
        )

        return clipped_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
