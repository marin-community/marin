# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded, replay-enabled Snowball learner update diagnostic.

This module deliberately implements a weighted cross-entropy (CE) update, not an online RL or
PPO objective.  The route capture is made once from the public Snowball API; the optimizer then
updates the current model while replay keeps expert identities fixed and recomputes current
combine weights and gradients.
"""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax

import haliax as hax
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballLMHeadModel, SnowballRouteCaptureBatch
from levanter.optim.config import AdamConfig


@dataclass(frozen=True)
class SnowballReplayUpdateResult:
    """Outputs from one replay-enabled weighted-CE learner update."""

    model: SnowballLMHeadModel
    route_captures: SnowballRouteCaptureBatch
    gradients: Any
    pre_update_loss: jax.Array
    post_update_loss: jax.Array
    pre_update_logits: hax.NamedArray
    post_update_logits: hax.NamedArray
    post_update_replay_logits: hax.NamedArray
    used_jit: bool


def _scalar_loss(value: jax.Array | hax.NamedArray) -> jax.Array:
    """Return a scalar loss without changing the model's public loss contract."""
    return value.array if isinstance(value, hax.NamedArray) else value


def run_snowball_replay_update(
    model: SnowballLMHeadModel,
    example: LmExample,
    *,
    learning_rate: float = 1e-3,
    use_jit: bool = False,
) -> SnowballReplayUpdateResult:
    """Run one real replay-enabled weighted-CE update.

    The returned logits make it possible for a caller to compare teacher-forced, fresh, and
    replayed full-vocabulary rows after the update.  This helper does not claim an RL objective;
    integrating sampled rewards, advantages, or PPO clipping remains a separate learner concern.
    """
    route_captures = model.capture_route_captures(example.tokens, example.attn_mask)

    def loss(current: SnowballLMHeadModel) -> jax.Array:
        return _scalar_loss(current.compute_next_token_loss(example, route_captures=route_captures))

    value_and_grad = eqx.filter_value_and_grad(loss)
    if use_jit:
        value_and_grad = eqx.filter_jit(value_and_grad)
    pre_update_loss, gradients = value_and_grad(model)

    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        warmup=0.0,
        lr_schedule="constant",
    ).build(num_train_steps=1)
    parameters = eqx.filter(model, eqx.is_array)
    parameter_gradients = eqx.filter(gradients, eqx.is_array)
    optimizer_state = optimizer.init(parameters)
    updates, _ = optimizer.update(parameter_gradients, optimizer_state, params=parameters)
    updated_model = eqx.apply_updates(model, eqx.combine(updates, jax.tree.map(lambda _: None, model)))

    post_update_logits = updated_model(example.tokens, example.attn_mask)
    post_update_replay_logits = updated_model(
        example.tokens,
        example.attn_mask,
        route_captures=route_captures,
    )
    post_update_loss = _scalar_loss(updated_model.compute_next_token_loss(example, route_captures=route_captures))
    return SnowballReplayUpdateResult(
        model=updated_model,
        route_captures=route_captures,
        gradients=gradients,
        pre_update_loss=pre_update_loss,
        post_update_loss=post_update_loss,
        pre_update_logits=model(example.tokens, example.attn_mask),
        post_update_logits=post_update_logits,
        post_update_replay_logits=post_update_replay_logits,
        used_jit=use_jit,
    )
