# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union

import jax
import optax
from jax import numpy as jnp
from jax.tree_util import GetAttrKey, register_pytree_with_keys_class

from haliax.jax_utils import is_jax_array_like

import levanter.tracker
from levanter.utils.tree_utils import tree_flatten_one_level_with_keys


# Define the state structure for the optimizer
# We use a custom pytree so that we can load a skipstep state from a non-skipstep state.
@register_pytree_with_keys_class
@dataclass(frozen=True)
class SkipStepState:
    inner_opt_state: optax.OptState
    losses: jax.Array  # Stores last 'rolling_interval_length' losses
    grad_norms: jax.Array  # Stores last 'rolling_interval_length' grad_norms
    valid_mask: jax.Array  # Boolean mask for valid entries in losses/grad_norms
    current_idx: jax.Array  # Current index in the rolling window (circular buffer)
    count: jax.Array  # Number of valid entries currently in the buffer (up to buffer_size)

    def tree_flatten_with_keys(self):
        inner_pairs, inner_def = tree_flatten_one_level_with_keys(self.inner_opt_state)
        pairs = None
        if len(inner_pairs) == 1:
            # make sure it's not a singleton
            if inner_pairs[0][0] is None:
                pairs = [GetAttrKey("inner_opt_state"), self.inner_opt_state]

        if pairs is None:
            pairs = [
                *inner_pairs,
                (GetAttrKey("_skipstep_losses"), self.losses),
                (GetAttrKey("_skipstep_grad_norms"), self.grad_norms),
                (GetAttrKey("_skipstep_valid_mask"), self.valid_mask),
                (GetAttrKey("_skipstep_current_idx"), self.current_idx),
                (GetAttrKey("_skipstep_count"), self.count),
            ]

        return pairs, inner_def

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        inner_state = jax.tree.unflatten(aux_data, children[:-5])

        return cls(
            inner_opt_state=inner_state,
            losses=children[-5],
            grad_norms=children[-4],
            valid_mask=children[-3],
            current_idx=children[-2],
            count=children[-1],
        )


@dataclass(frozen=True)
class SkipStepConfig:
    """
    Configuration for "skip step" logic in an optimizer.

    This optimizer skips steps based on the history of loss and gradient norms.

    If the current loss or gradient norm is significantly higher than the historical mean plus a multiple of the
    standard deviation, the step is skipped. The decision is made based on a rolling window of the last
    `rolling_interval_length` steps.

    IF there are fewer than `rolling_interval_length // 2` steps in the history, the step is always taken.

    See https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/optim/skip_step_optimizer.py
    """

    rolling_interval_length: int = 128
    sigma_factor: float = 6.0

    @staticmethod
    def from_bool_int_or_config(config: Union[bool, int, "SkipStepConfig"]) -> Optional["SkipStepConfig"]:
        """
        Converts a boolean, integer, or SkipStepConfig to a SkipStepConfig.
        If the input is True, it returns a default SkipStepConfig.
        If the input is False, it returns None.
        If the input is an integer, it sets rolling_interval_length to that value.
        """
        if isinstance(config, SkipStepConfig):
            return config
        elif config is True:
            return SkipStepConfig()
        elif config is False:
            return None
        elif isinstance(config, int):
            return SkipStepConfig(rolling_interval_length=config)
        else:
            raise ValueError(f"Invalid type for SkipStepConfig: {type(config)}")

    def wrap(self, inner_optimizer: optax.GradientTransformation) -> optax.GradientTransformation:
        def init_fn(params):
            inner_opt_state = inner_optimizer.init(params)
            buffer_size = self.rolling_interval_length

            # Initialize losses and grad_norms with zeros.
            losses = jnp.zeros((buffer_size,), dtype=jnp.float32)
            grad_norms = jnp.zeros((buffer_size,), dtype=jnp.float32)

            # Initialize valid_mask to all False, as no data is valid yet.
            valid_mask = jnp.full((buffer_size,), False, dtype=jnp.bool_)
            current_idx = jnp.zeros((), dtype=jnp.int32)
            count = jnp.zeros((), dtype=jnp.int32)

            return SkipStepState(
                inner_opt_state=inner_opt_state,
                losses=losses,
                grad_norms=grad_norms,
                current_idx=current_idx,
                count=count,
                valid_mask=valid_mask,
            )

        def update_fn(updates, state: SkipStepState, params, *, loss: Optional[jax.Array] = None, **extra_args):
            if loss is None:
                raise ValueError(
                    "Loss must be provided as a keyword argument to the update function when using SkipStepOptimizer."
                )

            global_norm = optax.global_norm(updates)

            # Decision Logic (step_factor) based on PAST data
            min_data_points = jnp.maximum(2, self.rolling_interval_length // 2)
            # can_skip_based_on_history uses state.count (count before adding current data point)
            can_skip_based_on_history = state.count >= min_data_points

            loss_mean_history = jnp.mean(state.losses, where=state.valid_mask)
            loss_std_history = jnp.std(state.losses, where=state.valid_mask)
            loss_std_safe_history = jnp.maximum(loss_std_history, 1e-6)

            grad_norm_mean_history = jnp.mean(state.grad_norms, where=state.valid_mask)
            grad_norm_std_history = jnp.std(state.grad_norms, where=state.valid_mask)
            grad_norm_std_safe_history = jnp.maximum(grad_norm_std_history, 1e-6)

            loss_threshold = loss_mean_history + self.sigma_factor * loss_std_safe_history
            grad_norm_threshold = grad_norm_mean_history + self.sigma_factor * grad_norm_std_safe_history

            # Compare current loss and grad_norm with thresholds derived from history
            loss_ok = loss <= loss_threshold
            grad_norm_ok = global_norm <= grad_norm_threshold

            # we skip if loss OR grad_norm are NOT ok.
            # so should_skip is true if NOT (loss_ok AND grad_norm_ok)
            # which is (NOT loss_ok) OR (NOT grad_norm_ok)
            should_skip = jnp.logical_not(jnp.logical_and(loss_ok, grad_norm_ok))

            step_factor = jnp.where(jnp.logical_and(can_skip_based_on_history, should_skip), 0.0, 1.0)

            buffer_size = state.losses.shape[0]

            new_losses = state.losses.at[state.current_idx].set(loss)
            new_grad_norms = state.grad_norms.at[state.current_idx].set(global_norm)
            new_valid_mask = state.valid_mask.at[state.current_idx].set(True)
            new_current_idx = (state.current_idx + 1) % buffer_size
            new_count = jnp.minimum(state.count + 1, buffer_size)

            levanter.tracker.jit_log(
                {
                    "optim/skipped_step": (step_factor == 0.0).astype(jnp.float32),
                    "optim/skip_step/loss_threshold": loss_threshold,
                    "optim/skip_step/loss": loss,
                    "optim/skip_step/loss_std": loss_std_safe_history,
                    "optim/skip_step/grad_norm_threshold": grad_norm_threshold,
                    "optim/skip_step/grad_norm": global_norm,
                    "optim/skip_step/grad_norm_std": grad_norm_std_safe_history,
                }
            )

            inner_opt_state = state.inner_opt_state

            inner_updates, new_inner_opt_state = inner_optimizer.update(updates, inner_opt_state, params, **extra_args)
            actual_updates = jax.tree.map(
                lambda x: step_factor * x if is_jax_array_like(x) else x,
                inner_updates,  # type: ignore
            )

            new_inner_opt_state = jax.tree.map(
                lambda x, y: jnp.where(step_factor == 0.0, x, y), inner_opt_state, new_inner_opt_state  # type: ignore
            )

            return actual_updates, SkipStepState(
                inner_opt_state=new_inner_opt_state,
                losses=new_losses,
                grad_norms=new_grad_norms,
                current_idx=new_current_idx,
                count=new_count,
                valid_mask=new_valid_mask,
            )

        return optax.GradientTransformationExtraArgs(init_fn, update_fn)
