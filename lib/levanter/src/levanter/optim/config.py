# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import re
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import draccus
import equinox as eqx
import jax
import numpy as np
import optax
from jax import numpy as jnp

import haliax

import levanter.tracker
from levanter.optim.clip_update_norm import ClipUpdateNormConfig
from levanter.optim.skipstep import SkipStepConfig
from levanter.optim.util import log_norm_passthrough, scan_aware_clip_by_block_rms
from levanter.utils.jax_utils import leaf_key_paths


@dataclass(frozen=True)
class LrScheduleContext:
    warmup_steps: int
    decay_steps: int
    learning_rate: float
    min_lr_ratio: float
    min_lr: float


class LrSchedule(draccus.ChoiceRegistry, abc.ABC):
    @abc.abstractmethod
    def build(self, ctx: LrScheduleContext) -> Callable:
        raise NotImplementedError


@LrSchedule.register_subclass("constant")
@dataclass(frozen=True)
class ConstantLrSchedule(LrSchedule):
    def build(self, ctx: LrScheduleContext):
        return optax.constant_schedule(ctx.learning_rate)


@LrSchedule.register_subclass("cosine")
@dataclass(frozen=True)
class CosineLrSchedule(LrSchedule):
    exponent: float = 1.0

    def build(self, ctx: LrScheduleContext):
        return optax.cosine_decay_schedule(ctx.learning_rate, ctx.decay_steps, ctx.min_lr_ratio, self.exponent)


@LrSchedule.register_subclass("linear")
@dataclass(frozen=True)
class LinearLrSchedule(LrSchedule):
    def build(self, ctx: LrScheduleContext):
        return optax.linear_schedule(ctx.learning_rate, ctx.min_lr, ctx.decay_steps)


@LrSchedule.register_subclass("inv_sqrt")
@dataclass(frozen=True)
class InvSqrtLrSchedule(LrSchedule):
    timescale: float = 10000

    def build(self, ctx: LrScheduleContext):
        return _inv_sqrt_decay_schedule(ctx.learning_rate, ctx.min_lr, ctx.warmup_steps, self.timescale)


@LrSchedule.register_subclass("inv")
@dataclass(frozen=True)
class InvLrSchedule(LrSchedule):
    def build(self, ctx: LrScheduleContext):
        return _inv_decay_schedule(ctx.learning_rate, ctx.min_lr, ctx.decay_steps)


@LrSchedule.register_subclass("power")
@dataclass(frozen=True)
class PowerLrSchedule(LrSchedule):
    # Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler (Shen et al., 2024)
    # https://arxiv.org/abs/2408.13359
    # The scheduler and default hyperparameters are intended for use with maximal update parametrization
    # (mup), as described in the paper. The scheduler may work without mup.

    batch_size: int
    """Required for the power learning rate schedule to determine number of tokens trained on so far."""
    seq_length: int
    """Required for the power learning rate schedule to determine number of tokens trained on so far."""
    a: float = 4.6
    """Learning rate amplitude for the power learning rate schedule. Must be a positive number."""
    b: float = -0.51
    """Power-law exponent for the power learning rate schedule. Must be a negative number."""

    def __post_init__(self):
        if self.a <= 0 or self.b >= 0:
            raise ValueError("Power schedule expects a > 0 and b < 0")

    def build(self, ctx: LrScheduleContext):
        def schedule(step):
            tokens_trained = step * self.batch_size * self.seq_length
            return jnp.minimum(ctx.learning_rate, self.batch_size * self.a * tokens_trained**self.b)

        return schedule


@dataclass(frozen=True)
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4
    weight_decay: float = 0.1

    min_lr_ratio: float = 0.1
    """The lr scheduler operates on 4 stages: [warmup] - {[stable] - [decay]} x haps - [cooldown]"""
    warmup: int | float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    decay: int | float | None = None
    """fraction of training steps to use as decay, or steps to use. None means full decay"""
    rewarmup: int | float = 0.0
    "If using a cycle, how much of the cycle to use as re-warmup. 0.0 means no re-warmup."
    cooldown: Optional[float] = None
    """Deprecated, as its semantics are confusing."""
    cycle_length: int | float | None | list[int] = None
    """ Length of cycle. If <= 1, it is treated as a fraction of the total number of steps. None is equivalent to 1.0."""
    cycles: int | list[int] | None = None
    """Number of cycles or a list of cycle endpoints. Can use at most one of cycle_length, cycles, or haps."""

    lr_schedule: LrSchedule | str = "cosine"  # constant, cosine, linear
    haps: Optional[list[int]] = None
    """Deprecated."""
    weight_decay_modules: Optional[list[str] | str] = None
    """A regex or a list of strings to identify where to mask weight.
    For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""
    default_weight_decay_mask: Optional[bool] = None
    """Whether to apply a default reasonable weight decay to modules not explicitly masked. None means it will if
    no weight_decay_modules are set. False means it will not. True means it will regardless of weight_decay_modules."""

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "adam"

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def build_weight_decay_mask(self):
        def reasonable_default(module, path):
            # TODO: gross
            if "LayerNorm" in path:
                return False
            if "RMSNorm" in path:
                return False
            if "RmsNorm" in path:
                return False
            if "Embedding" in path:
                return False
            if path.endswith("bias"):
                return False
            return None

        if self.weight_decay_modules is None and self.default_weight_decay_mask is False:
            return None
        else:
            should_use_default = self.default_weight_decay_mask is True or (
                self.default_weight_decay_mask is None and self.weight_decay_modules is None
            )

            def is_leaf(x):
                return eqx.is_array(x) or isinstance(x, eqx.Module) or haliax.is_named_array(x)

            # mask based on regex or module path
            def _apply_on(decayed_paths, x, from_root_key_path, from_class_keypath):
                if isinstance(x, eqx.Module):
                    is_leaf_here = lambda y: x is not y and is_leaf(y)  # noqa: E731
                    # we want to support both Linear.weight and transformer.encoder.layers.0.mlp.dense.weight
                    class_name = x.__class__.__name__
                    # recursively apply to submodules.
                    from_root_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=from_root_key_path)
                    from_class_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=class_name)
                    this_mask = jax.tree_util.tree_map(
                        partial(_apply_on, decayed_paths),
                        x,
                        from_root_key_paths,
                        from_class_key_paths,
                        is_leaf=lambda y: x is not y and is_leaf(y),
                    )
                    return this_mask
                elif not haliax.util.is_jax_or_hax_array_like(x):
                    return x

                should_decay = None
                for key_path in [from_root_key_path, from_class_keypath]:
                    if key_path is None:
                        continue

                    if isinstance(self.weight_decay_modules, str):
                        compiled_regex = re.compile(self.weight_decay_modules)
                        should_decay = should_decay or compiled_regex.match(key_path) is not None
                    elif isinstance(self.weight_decay_modules, list):
                        should_decay = should_decay or any(
                            key_path.__contains__(target) for target in self.weight_decay_modules
                        )

                    if should_use_default and not should_decay:
                        should_decay = reasonable_default(x, key_path)

                    if should_decay:
                        break

                if should_decay is None:
                    if should_use_default:
                        should_decay = True
                    else:
                        should_decay = False

                if should_decay:
                    decayed_paths.append(from_root_key_path)

                return should_decay

            def mask_fn(model):
                decayed_paths = []
                mask = jax.tree_util.tree_map(
                    partial(_apply_on, decayed_paths, from_class_keypath=None),
                    model,
                    leaf_key_paths(model, is_leaf=is_leaf),
                    is_leaf=is_leaf,
                )

                # log all decayed weights
                levanter.tracker.log_hyperparameters({"decayed_weights": sorted(decayed_paths)})

                return mask

            return mask_fn

    def lr_scheduler(self, num_train_steps, override_lr=None):
        if self.cooldown is not None:
            warnings.warn("cooldown is deprecated. Just use the normal schedule.", DeprecationWarning)
            cooldown_steps = _convert_frac_or_steps(self.cooldown, num_train_steps)
        else:
            cooldown_steps = 0

        total_main_steps = num_train_steps - cooldown_steps
        cooldown_points = self._get_cycle_minima(total_main_steps)

        learning_rate = self.learning_rate
        if override_lr is not None:
            learning_rate = override_lr

        min_lr = learning_rate * self.min_lr_ratio

        schedules = []
        boundaries = []

        previous_end = 0.0

        for cycle, (start, end) in enumerate(zip(cooldown_points[:-1], cooldown_points[1:])):
            cycle_steps = end - start
            if cycle == 0:  # warmup
                warmup_steps = _convert_frac_or_steps(self.warmup, cycle_steps)
            else:
                warmup_steps = _convert_frac_or_steps(self.rewarmup, cycle_steps)

            if warmup_steps != 0:
                warmup = optax.linear_schedule(previous_end, learning_rate, warmup_steps)
                schedules.append(warmup)
                boundaries.append(start + warmup_steps)

            lr_decay_steps = (
                _convert_frac_or_steps(self.decay, cycle_steps)
                if self.decay is not None
                else cycle_steps - warmup_steps
            )
            stable_steps = cycle_steps - warmup_steps - lr_decay_steps

            if stable_steps != 0:
                stable = optax.constant_schedule(learning_rate)
                schedules.append(stable)
                boundaries.append(start + warmup_steps + stable_steps)

            if isinstance(self.lr_schedule, str):
                match self.lr_schedule:
                    case "constant":
                        schedule = optax.constant_schedule(learning_rate)
                    case "cosine":
                        schedule = optax.cosine_decay_schedule(learning_rate, lr_decay_steps, self.min_lr_ratio)
                    case "linear":
                        schedule = optax.linear_schedule(learning_rate, min_lr, lr_decay_steps)
                    case "inv_sqrt":
                        schedule = _inv_sqrt_decay_schedule(learning_rate, min_lr, warmup_steps, 10000)
                    case "inv":
                        schedule = _inv_decay_schedule(learning_rate, min_lr, lr_decay_steps)
                    case _:
                        raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
            elif isinstance(self.lr_schedule, LrSchedule):
                schedule = self.lr_schedule.build(
                    LrScheduleContext(
                        warmup_steps=warmup_steps,
                        decay_steps=lr_decay_steps,
                        learning_rate=learning_rate,
                        min_lr_ratio=self.min_lr_ratio,
                        min_lr=min_lr,
                    )
                )
            else:
                raise ValueError(f"lr_schedule must be a string or an instance of LrSchedule, got {self.lr_schedule}")

            previous_end = schedule(lr_decay_steps)

            schedules.append(schedule)
            boundaries.append(end)

        if cooldown_steps != 0:
            final_main_lr = schedule(lr_decay_steps)
            cooldown = optax.linear_schedule(final_main_lr, min_lr, cooldown_steps)
            schedules.append(cooldown)

        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedules[0]

        return schedule

    def _get_cycle_minima(self, total_main_steps):
        if self.cycle_length is not None:
            if self.cycles is not None:
                raise ValueError("Can't use both cycle_length and cycles.")
            if self.haps is not None:
                warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
                raise ValueError("Can't use both cycle_length and haps.")

            if isinstance(self.cycle_length, int | float):
                cycle_length = _convert_frac_or_steps(self.cycle_length, total_main_steps)
                cooldown_points = [i * cycle_length for i in range(1, total_main_steps // cycle_length)]
                if total_main_steps % cycle_length != 0:
                    warnings.warn(
                        "Cycle length does not divide total number of steps. The last cycle will be shorter."
                    )

            elif isinstance(self.cycle_length, list):
                lengths = np.array(self.cycle_length)
                steps = np.cumsum(lengths)
                if steps[-1] > total_main_steps:
                    raise ValueError(f"Cycle lengths exceed total number of steps: {steps[-1]} > {total_main_steps}")
                cooldown_points = steps.tolist()
            else:
                raise ValueError("Invalid cycle_length. Must be a fraction, number of steps, or a list of steps.")

        elif self.haps is not None:
            warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
            cooldown_points = list(self.haps)
        elif isinstance(self.cycles, int):
            # insert a warmup then the rest of the steps
            cooldown_points = [int(total_main_steps / self.cycles * (i + 1)) for i in range(self.cycles - 1)]
        elif isinstance(self.cycles, list):
            cooldown_points = list(self.cycles)
        else:
            cooldown_points = []

        cooldown_points.insert(0, 0)
        if cooldown_points[-1] != total_main_steps:
            cooldown_points.append(total_main_steps)
        return cooldown_points


def _inv_sqrt_decay_schedule(lr: float, min_lr: float, warmup_steps: int, timescale: float = 10000):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _inv_decay_schedule(lr: float, min_lr: float, decay_steps: int):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / ((lr / min_lr - 1) * jnp.maximum(count, 1) / decay_steps + 1))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _convert_frac_or_steps(frac_or_steps: float | int, num_train_steps: int):
    # if it's greater than 1, it must be a whole number of steps
    if frac_or_steps < 0.0 or (frac_or_steps > 1.0 and frac_or_steps % 1 != 0):
        raise ValueError(f"Invalid fraction {frac_or_steps}. Must be between 0 and 1. You can also use (whole) steps.")
    if frac_or_steps <= 1.0:
        return int(frac_or_steps * num_train_steps)

    return int(frac_or_steps)


@dataclass(frozen=True)
class HessianOptConfig(OptimizerConfig, abc.ABC):
    update_interval: int = 10
    """How often to update the hessian approximation."""


@OptimizerConfig.register_subclass("adam")
@dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    nesterov: bool = False
    update_rms_clipping: Optional[float] = None
    """
    If set, this will use RMS clipping on the update, a la Adafactor or StableAdamW (https://arxiv.org/pdf/2304.13013)

    (Note that this is distinct from StableAdamW b/c we clip on RMS(m/sqrt(v)) rather than RMS(g/sqrt(v)).)

    A value of 1.0 is recommended for most models, but you can set it to None to disable RMS clipping.
    """

    clip_update_norm: Optional[ClipUpdateNormConfig] = None
    """
    If set, this will clip the update norm based on the historical mean and standard deviation of update norms. A less extreme version of skip_bad_steps.
    """

    skip_bad_steps: SkipStepConfig | int | bool = False
    """
    If set, defines the configuration for skipping steps when gradients are too large.

    int means history length, bool means True for default config, False for no skipping.

    "Bad" here means either the loss or grad norm is much much larger than the average of the last
    `rolling_interval_length` steps. (Default is 128 steps, with a sigma factor of 6.0)

    See https://github.com/allenai/OLMo-core/blob/main/src/olmo_core/optim/skip_step_optimizer.py
    """

    adamc_weight_decay: bool = False
    """
    If set, use the AdamC corrected weight decay, which keeps
    ``weight_decay / lr`` constant across training.

    This follows Defazio, *On the Correct Treatment of Weight Decay in Adam*
    (2025, https://arxiv.org/abs/2506.02285v2).
    """

    def __post_init__(self):
        if self.update_rms_clipping is not None and self.update_rms_clipping <= 0:
            raise ValueError("update_rms_clipping must be a positive number or None.")

        if self.clip_update_norm is not None and self.update_rms_clipping is not None:
            raise ValueError("Cannot use both update_rms_clipping and clip_update_norm at the same time.")

    def build(self, num_train_steps):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon, nesterov=self.nesterov))

            if self.weight_decay > 0:
                if self.adamc_weight_decay:
                    max_lr = self.learning_rate
                    weight_decay = self.weight_decay * (learning_rate / max_lr)
                else:
                    weight_decay = self.weight_decay
                components.append(optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()))

            if self.update_rms_clipping is not None:
                components.append(log_norm_passthrough("optim/pre_clip_update_norm"))
                components.append(scan_aware_clip_by_block_rms(self.update_rms_clipping))
                components.append(log_norm_passthrough("optim/post_clip_update_norm"))

            if self.clip_update_norm is not None:
                components.append(log_norm_passthrough("optim/pre_clip_update_norm"))
                components.append(self.clip_update_norm.build())
                components.append(log_norm_passthrough("optim/post_clip_update_norm"))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            if self.skip_bad_steps:
                optimizer = SkipStepConfig.from_bool_int_or_config(self.skip_bad_steps).wrap(optimizer)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


@OptimizerConfig.register_subclass("lion")
@dataclass(frozen=True)
class LionConfig(OptimizerConfig):
    """
    Lion optimizer configuration
    cf:
    Original Paper: https://arxiv.org/abs/2302.06675
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer"""

        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_lion(self.beta1, self.beta2))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        optimizer_instance = optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))

        return optimizer_instance
