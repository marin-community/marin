# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.june_tpu_67b_a2b.moe.adamh import scale_by_adamh
from experiments.june_tpu_67b_a2b.moe.model import Transformer


class TiedExpertLrScale(StrEnum):
    """Learning-rate divisor applied to expert banks reused across layers."""

    UNSCALED = "unscaled"
    SQRT = "sqrt"
    LINEAR = "linear"


def _tied_expert_lr_divisor(group_size: int, scale: TiedExpertLrScale) -> float:
    if group_size <= 0:
        raise ValueError(f"expert bank group size must be positive, got {group_size}")
    if scale is TiedExpertLrScale.UNSCALED:
        return 1.0
    if scale is TiedExpertLrScale.SQRT:
        return math.sqrt(group_size)
    if scale is TiedExpertLrScale.LINEAR:
        return float(group_size)
    raise ValueError(f"unknown tied expert LR scale: {scale}")


def _expert_bank_index(path: object) -> int | None:
    path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
    parts = path_str.lower().split(".")
    if len(parts) < 3 or parts[0] != "expert_banks" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _is_stacked_expert_bank_path(path: object) -> bool:
    path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
    return path_str.lower().startswith("expert_banks.stacked.")


def _tied_expert_group_name(group_size: int) -> str:
    return f"muonh_expert_g{group_size}"


_STACKED_EXPERT_GROUP = "muonh_expert_stacked"


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(
    params,
    direction_updates,
    learning_rate: float,
    leading_axis_lr_divisors: tuple[float, ...] | None = None,
):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        parameter_learning_rate = learning_rate
        if leading_axis_lr_divisors is not None:
            if param.ndim < 2 or param.shape[0] != len(leading_axis_lr_divisors):
                raise ValueError(
                    "stacked expert parameter leading axis must match the number of expert banks; "
                    f"got shape {param.shape} for divisors {leading_axis_lr_divisors}"
                )
            parameter_learning_rate = learning_rate / jnp.asarray(leading_axis_lr_divisors).reshape(
                (-1,) + (1,) * (param.ndim - 1)
            )
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - parameter_learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - parameter_learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(
        scale_invariant_update,
        params,
        direction_updates,
        is_leaf=lambda x: x is None,
    )


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    leading_axis_lr_divisors: tuple[float, ...] | None = None,
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        muon_updates, next_state = muon_transform.update(updates, state, params)
        muonh_updates = _scale_invariant_hyperball_updates(
            params,
            muon_updates,
            learning_rate,
            leading_axis_lr_divisors,
        )
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate_up,
      mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer with bank-aware tied-expert learning rates.

    Three LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all GatedNorms.
      Newton-Schulz orthogonalisation + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate``
      / 1-D norm weights.

    ``max_grad_norm`` defaults to ``None`` here (no clipping) for the 1pct-noclip
    schedule used by the May Recipe baseline.
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"
    rmsnorm_to_adam: bool = False
    """When True, force stacked-block ``rms_attn.weight`` / ``rms_mlp.weight``
    leaves into the 'adam' group instead of muonh. Stacking promotes those
    semantically-1D scale vectors to ndim=2, and the muonh fallback then runs
    NS + a single-Frobenius hyperball over the whole ``(num_layers, hidden_dim)``
    slab — which doesn't correspond to a meaningful update for RMSNorm scales.
    Defaults to False to keep behaviour bit-equivalent to the unstacked path."""
    schedule_num_train_steps_override: int | None = None
    """When set, the LR scheduler (warmup + decay span + min_lr_ratio anchor)
    is parameterized by this value instead of the trainer's ``num_train_steps``.
    Used for partial-schedule resumes where you want to stop training early
    (trainer's num_train_steps) while still following the original full
    schedule's LR trajectory up to the stop step. ``None`` preserves the
    default behavior (schedule tracks trainer)."""
    expert_bank_for_layer: tuple[int, ...] | None = None
    """Expert-bank topology used to derive each bank's layer reuse count."""
    tied_expert_lr_scale: TiedExpertLrScale = TiedExpertLrScale.SQRT

    @property
    def expert_bank_group_sizes(self) -> tuple[int, ...]:
        if self.expert_bank_for_layer is None:
            return ()
        if not self.expert_bank_for_layer:
            raise ValueError("expert_bank_for_layer must not be empty")
        if any(bank_id < 0 for bank_id in self.expert_bank_for_layer):
            raise ValueError("expert_bank_for_layer bank IDs must be non-negative")
        bank_ids = set(self.expert_bank_for_layer)
        if bank_ids != set(range(len(bank_ids))):
            raise ValueError(
                "expert_bank_for_layer must use contiguous bank IDs starting at zero; "
                f"got {self.expert_bank_for_layer}"
            )
        return tuple(self.expert_bank_for_layer.count(bank_id) for bank_id in range(max(self.expert_bank_for_layer) + 1))

    @property
    def expert_bank_lr_divisors(self) -> tuple[float, ...]:
        return tuple(
            _tied_expert_lr_divisor(group_size, self.tied_expert_lr_scale) for group_size in self.expert_bank_group_sizes
        )

    def build(self, num_train_steps):
        n = self.schedule_num_train_steps_override or num_train_steps
        learning_rate_schedule = self.lr_scheduler(n)
        adam_lr_schedule = self.lr_scheduler(n, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform_at(lr, *, leading_axis_lr_divisors: tuple[float, ...] | None = None):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=lr,
                        coefficient_type=self.coefficient_type,
                        leading_axis_lr_divisors=leading_axis_lr_divisors,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform_at(learning_rate),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            group_sizes = self.expert_bank_group_sizes
            if group_sizes:
                transforms[_STACKED_EXPERT_GROUP] = muonh_transform_at(
                    learning_rate,
                    leading_axis_lr_divisors=self.expert_bank_lr_divisors,
                )
                for group_size in set(group_sizes):
                    if group_size <= 1:
                        continue
                    divisor = _tied_expert_lr_divisor(group_size, self.tied_expert_lr_scale)
                    transforms[_tied_expert_group_name(group_size)] = muonh_transform_at(learning_rate / divisor)
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        group_sizes = self.expert_bank_group_sizes
        if group_sizes and isinstance(params, Transformer):
            model_mapping = params.config.resolved_expert_bank_for_layer
            if self.expert_bank_for_layer != model_mapping:
                raise ValueError(
                    "optimizer expert_bank_for_layer must match the model topology; "
                    f"got {self.expert_bank_for_layer}, expected {model_mapping}"
                )
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "router_bias" in path_lower
                or path_lower.endswith(".attn_gate")
                or ".router" in path_lower
            ):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if group_sizes and _is_stacked_expert_bank_path(path):
                return _STACKED_EXPERT_GROUP
            bank_index = _expert_bank_index(path)
            if bank_index is not None and group_sizes:
                if bank_index >= len(group_sizes):
                    raise ValueError(
                        "expert_bank_for_layer does not cover expert bank path "
                        f"{path_str}: {self.expert_bank_for_layer}"
                    )
                group_size = group_sizes[bank_index]
                if group_size > 1:
                    return _tied_expert_group_name(group_size)
            # Optionally route stacked RMSNorm scales to adam instead of letting
            # them fall into the matrix bucket below.
            if self.rmsnorm_to_adam and ("rms_attn" in path_lower or "rms_mlp" in path_lower):
                return "adam"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            # Route any matrix-shaped param to muonh: 2D and 3D from the unstacked
            # path, plus 4D when use_array_stacked_blocks=True adds a leading
            # num_layers axis. Without including 4D, stacked expert weights fall
            # through to adam, whose update does not reshard the (unsharded)
            # gradient from shard_map transposition — XLA then aligns mu / nu
            # buffers to the unsharded shape and per-chip opt_state explodes.
            if hasattr(param, "ndim") and param.ndim in (2, 3, 4):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "TiedExpertLrScale",
    "scale_with_grug_muonh",
]
