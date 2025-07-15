from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

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
    init_lr: float = 0.0,
    end_lr: float = 3e-5,
    lr: float = 3e-4,
    lr_warmup_steps: int = 3000,
    lr_decay_steps: int = 300000,
    b1: float = 0.9,
    b2: float = 0.95,
    clip_gradient: float = 1.0,
    weight_decay: float = 0.1,
    bf16_momentum: bool = False,
    multiply_by_parameter_scale: bool = False,
    weight_decay_mask: Callable | None = None,
    schedule: str = "cos",
) -> tuple[optax.GradientTransformation, dict[str, Any]]:
    learning_rate_schedule = schedule_by_name[schedule](
        init_value=init_lr,
        peak_value=lr,
        warmup_steps=lr_warmup_steps,
        decay_steps=lr_decay_steps,
        end_value=end_lr,
    )

    optimizer_info = dict(
        learning_rate_schedule=learning_rate_schedule,
    )

    if multiply_by_parameter_scale:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_gradient),
            optax.adafactor(
                learning_rate=learning_rate_schedule,
                multiply_by_parameter_scale=True,
                momentum=b1,
                decay_rate=b2,
                factored=False,
                clipping_threshold=None,
                dtype_momentum=jnp.bfloat16 if bf16_momentum else jnp.float32,
            ),
            optax_add_scheduled_weight_decay(
                lambda step: -learning_rate_schedule(step) * weight_decay,
                weight_decay_mask,
            ),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(clip_gradient),
            optax.adamw(
                learning_rate=learning_rate_schedule,
                weight_decay=weight_decay,
                b1=b1,
                b2=b2,
                mask=weight_decay_mask,
                mu_dtype=jnp.bfloat16 if bf16_momentum else jnp.float32,
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
