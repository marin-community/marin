# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .training_config import OptimizerConfig

# Adapted from:
# https://github.com/young-geng/EasyLM/blob/main/EasyLM/optimizers.py


def warmup_linear_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Linear warmup followed by linear decay.

    Args:
      init_value: Initial value for the scalar to be annealed.
      peak_value: Peak value for scalar to be annealed at end of warmup.
      warmup_steps: Positive integer, the length of the linear warmup.
      decay_steps: Positive integer, the total length of the schedule. Note that
        this includes the warmup time, so the number of steps during which cosine
        annealing is applied is `decay_steps - warmup_steps`.
      end_value: End value of the scalar to be annealed.
    Returns:
      schedule: A function that maps step counts to values.
    """
    schedules = [
        optax.linear_schedule(init_value=init_value, end_value=peak_value, transition_steps=warmup_steps),
        optax.linear_schedule(init_value=peak_value, end_value=end_value, transition_steps=decay_steps - warmup_steps),
    ]
    return optax.join_schedules(schedules, [warmup_steps])


schedule_by_name = dict(
    cos=optax.warmup_cosine_decay_schedule,
    linear=warmup_linear_decay_schedule,
)


def load_adamw_optimizer(
    config: OptimizerConfig,
    weight_decay_mask: Callable | None = None,
) -> tuple[optax.GradientTransformation, dict[str, Any]]:
    learning_rate_schedule = schedule_by_name[config.schedule](
        init_value=config.init_lr,
        peak_value=config.lr,
        warmup_steps=config.lr_warmup_steps,
        decay_steps=config.lr_decay_steps,
        end_value=config.end_lr,
    )

    optimizer_info = dict(
        learning_rate_schedule=learning_rate_schedule,
    )

    if config.multiply_by_parameter_scale:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=config.b1,
                decay_rate=config.b2,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
            optax_add_scheduled_weight_decay(
                lambda step: -learning_rate_schedule(step) * config.weight_decay,
                weight_decay_mask,
            ),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=config.weight_decay,
                b1=config.b1,
                b2=config.b2,
                mask=weight_decay_mask,
                mu_dtype=jnp.bfloat16 if config.bf16_momentum else jnp.float32,
            ),
        )

    return optimizer, optimizer_info


class OptaxScheduledWeightDecayState(NamedTuple):
    count: jnp.ndarray


def optax_add_scheduled_weight_decay(schedule_fn, mask=None):
    """Apply weight decay with schedule."""

    def init_fn(params):
        del params
        return OptaxScheduledWeightDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("Params cannot be None for weight decay!")

        weight_decay = schedule_fn(state.count)
        updates = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        return updates, OptaxScheduledWeightDecayState(count=optax.safe_int32_increment(state.count))

    if mask is not None:
        return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)
