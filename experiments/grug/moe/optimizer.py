# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe.klsoaph import scale_by_klsoaph


def _uses_adamh_baseline_adam_group(path_lower: str) -> bool:
    return (
        "token_embed" in path_lower
        or "router_bias" in path_lower
        or "attn_gate" in path_lower
        or ".router" in path_lower
    )


def _uses_klsoaph_baseline_adam_group(path_lower: str) -> bool:
    """KL Soap H variant override: route attn_gate into the matrix group.

    Same as _uses_adamh_baseline_adam_group but drops "attn_gate". The
    attention gate parameter is a small 2-D tensor (hidden_dim, num_heads);
    the KL Soap H sweep keeps it under the SOAP preconditioner.
    """
    return "token_embed" in path_lower or "router_bias" in path_lower or ".router" in path_lower


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


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            param_norm = jnp.linalg.norm(param)
            update_norm = jnp.linalg.norm(update)
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param_norm = jnp.linalg.norm(new_param)
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
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
        muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate)
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByGrugNorMuonHState(NamedTuple):
    muon_state: optax.OptState
    row_nu: optax.Updates


def _output_axis_second_moment(param):
    if not hasattr(param, "ndim") or param.ndim < 2:
        return param
    return jnp.zeros(param.shape[:-2] + param.shape[-1:], dtype=param.dtype)


def _output_axis_mean_square(update):
    return jnp.mean(jnp.square(update), axis=-2)


def _expand_output_axis_stat(stat):
    return jnp.expand_dims(stat, axis=-2)


def scale_with_grug_normuonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    beta2: float = 0.95,
    steps: int = 5,
    muon_eps: float = 1e-8,
    normuon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
) -> optax.GradientTransformation:
    """NorMuon direction inside the Grug hyperball update.

    Grug stores dense arrays as (fan_in, fan_out), so the paper's row/neuron
    statistic maps to the trailing output axis here.
    """
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
    )

    def init_fn(params):
        row_nu = jax.tree.map(_output_axis_second_moment, params)
        return ScaleByGrugNorMuonHState(muon_state=muon_transform.init(params), row_nu=row_nu)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_normuonh requires params for norm-preserving updates")

        muon_updates, next_muon_state = muon_transform.update(updates, state.muon_state, params)

        def update_second_moment(prev, update):
            if update is None or not hasattr(update, "ndim") or update.ndim < 2:
                return prev
            return beta2 * prev + (1 - beta2) * _output_axis_mean_square(update)

        row_nu = jax.tree.map(
            update_second_moment,
            state.row_nu,
            muon_updates,
            is_leaf=lambda x: x is None,
        )

        def normalize_update(update, nu):
            if update is None or not hasattr(update, "ndim") or update.ndim < 2:
                return update
            return update / (jnp.sqrt(_expand_output_axis_stat(nu)) + normuon_eps)

        normuon_updates = jax.tree.map(
            normalize_update,
            muon_updates,
            row_nu,
            is_leaf=lambda x: x is None,
        )
        normuonh_updates = _scale_invariant_hyperball_updates(params, normuon_updates, learning_rate)
        return normuonh_updates, ScaleByGrugNorMuonHState(muon_state=next_muon_state, row_nu=row_nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_grug_klsoaph(
    beta1: float = 0.95,
    beta2: float = 0.9,
    shampoo_beta: float = 0.9,
    eps: float = 1e-8,
    precond_freq: int = 5,
    init_factor: float = 0.1,
    block_size: int = 128,
    learning_rate: float = 0.018,
) -> optax.GradientTransformation:
    """KL Soap H transform: block-wise SOAP-eigenbasis Adam direction + hyperball post-step.

    Reproduces KLSOAPH from KellerJordan/modded-nanogpt PR #290 *within
    each ``block_size x block_size`` block*; the full-shape direction is
    reassembled before the hyperball step so normalization sees the
    original (unblocked) parameter and update. Default
    ``(beta1, beta2, shampoo_beta)`` matches upstream's "passing" tuple
    (0.95, 0.9, 0.9). ``precond_freq`` defaults to 5 to amortize the
    warm-started QR-iteration refresh on TPU.
    """
    soap_transform = scale_by_klsoaph(
        beta1=beta1,
        beta2=beta2,
        shampoo_beta=shampoo_beta,
        eps=eps,
        precond_freq=precond_freq,
        init_factor=init_factor,
        block_size=block_size,
    )

    def init_fn(params):
        return soap_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_klsoaph requires params for norm-preserving updates")
        direction, next_state = soap_transform.update(updates, state, params)
        klsoaph_updates = _scale_invariant_hyperball_updates(params, direction, learning_rate)
        return klsoaph_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.w_gate_up, mlp.w_down, shared.w_*)
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
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """MuonH for Grug MoE matrices outside the AdamH baseline Adam group.

    - muonh: matrix leaves that the AdamH baseline routes to AdamH or AdamH-expert
    - adamh: lm head / output projection matrix
    - adam: leaves that the AdamH baseline routes to Adam
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    coefficient_type: CoefficientType = "quintic"

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
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
                    "muonh": muonh_transform(),
                    "adamh": adamh_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_normuonh_v1")
@dataclass(frozen=True)
class GrugMoeNorMuonHConfig(OptimizerConfig):
    """NorMuon inside the Grug MoE hyperball update.

    - normuonh: matrix leaves that the AdamH baseline routes to AdamH or AdamH-expert
    - adamh: lm head / output projection matrix
    - adam: leaves that the AdamH baseline routes to Adam
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    normuon_epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    coefficient_type: CoefficientType = "quintic"

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def normuonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_normuonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        beta2=self.beta2,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        normuon_eps=self.normuon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
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
                    "normuonh": normuonh_transform(),
                    "adamh": adamh_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "normuonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_klsoaph_v1")
@dataclass(frozen=True)
class GrugMoeKLSoapHConfig(OptimizerConfig):
    """KL Soap H for Grug MoE.

    Reproduces KLSOAPH from KellerJordan/modded-nanogpt PR #290. The "KL"
    qualifier names the scale-invariant ("hyperball") post-step, not a
    KL-divergence projection.

    Variant override versus MuonH/NorMuonH: the attention gate parameter
    (`attn_gate`) is routed to KL Soap H rather than the baseline Adam
    group, since it is a 2-D matrix and benefits from the preconditioner.

    - klsoaph: matrix leaves outside the variant baseline Adam group
    - adamh: lm head / output projection matrix
    - adam: leaves that the variant routes to plain Adam
    """

    adam_lr: float = 6e-4
    # Default (beta1, beta2, shampoo_beta) matches upstream "passing" tuple from
    # KellerJordan/modded-nanogpt PR #290 (beta1=0.95, beta2=0.9, shampoo_beta=0.9).
    beta1: float = 0.95
    beta2: float = 0.9
    shampoo_beta: float = 0.9
    epsilon: float = 1e-8
    precond_freq: int = 5
    init_factor: float = 0.1
    # Block size for the block-wise SOAP preconditioner. Each (rows, cols) weight
    # is tiled into (R, C) blocks of (block_size, block_size); SOAP runs per
    # block. Direction is reassembled to full shape before the hyperball post-step
    # so normalization sees the original parameter/update.
    block_size: int = 128
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def klsoaph_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_klsoaph(
                        beta1=self.beta1,
                        beta2=self.beta2,
                        shampoo_beta=self.shampoo_beta,
                        eps=self.epsilon,
                        precond_freq=self.precond_freq,
                        init_factor=self.init_factor,
                        block_size=self.block_size,
                        learning_rate=learning_rate,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
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
                    "klsoaph": klsoaph_transform(),
                    "adamh": adamh_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_klsoaph_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "klsoaph"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeKLSoapHConfig",
    "GrugMoeMuonHConfig",
    "GrugMoeNorMuonHConfig",
    "ScaleByGrugNorMuonHState",
    "scale_with_grug_klsoaph",
    "scale_with_grug_muonh",
    "scale_with_grug_normuonh",
]
