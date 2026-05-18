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


def _uses_adamh_baseline_adam_group(path_lower: str) -> bool:
    # Use endswith for ``attn_gate`` so only the actual attention-gate leaf
    # matches (path ``...attn.attn_gate``), NOT ``attn_gated_norm.w_{up,down}``.
    return (
        "token_embed" in path_lower
        or "router_bias" in path_lower
        or path_lower.endswith(".attn_gate")
        or ".router" in path_lower
    )


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
            # Route every GatedNorm instance (per-block attn / mlp + model-level
            # embed / final) to AdamH at ``learning_rate``. Without this, the
            # baseline routing was asymmetric: attn_gated_norm hit the small-LR
            # Adam group via substring bug, the other three hit muonh.
            if "gated_norm" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_may_arch_gn_lr_v1")
@dataclass(frozen=True)
class GrugMoeMuonHMayArchGNLrConfig(OptimizerConfig):
    """MuonH config used by the may_arch GN-LR sensitivity sweep.

    Five LR groups:

    | group           | params                                  | base LR           |
    |-----------------|-----------------------------------------|-------------------|
    | ``muonh``       | matrices (attn, MoE MLPs, shared)       | ``learning_rate`` |
    | ``adamh_embed`` | ``token_embed``                         | ``learning_rate`` |
    | ``adamh``       | ``lm_head`` / ``output_proj``           | ``learning_rate`` |
    | ``adam_gn_wdown`` | GatedNorm ``.w_down`` (all 4 instances) | ``adam_lr``       |
    | ``adam_gn_wup`` | GatedNorm ``.w_up`` (all 4 instances)   | ``adam_lr``       |
    | ``adam``        | router / router_bias / attn_gate / 1-D  | ``adam_lr``       |

    GatedNorms are routed to two sub-groups so the sweep can perturb the
    LR of just the output projection (``w_up``) independently from the
    rest of the GatedNorm parameters. Setting both ``gn_lr_scale`` and
    ``gn_wup_lr_scale`` to 1.0 makes the two groups behave identically
    to a single "GN -> adam at adam_lr" group.

    Used by ``muonh_may_arch_gn_lr_sweep.py`` to test +-30% perturbations
    on the GatedNorm group with the may_arch architecture (256 experts,
    PKO, partial rope, last_layer_pko, no long-window).
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
    # Per-group scale multipliers (1.0 = no perturbation).
    muonh_lr_scale: float = 1.0
    adamh_embed_lr_scale: float = 1.0
    adamh_lmhead_lr_scale: float = 1.0
    adam_lr_scale: float = 1.0
    # Scales the whole GatedNorm group (both w_up and w_down) relative to adam_lr.
    gn_lr_scale: float = 1.0
    # Multiplicative extra scale applied only to GatedNorm ``.w_up`` on top of gn_lr_scale.
    gn_wup_lr_scale: float = 1.0

    def build(self, num_train_steps):
        muonh_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.muonh_lr_scale)
        adamh_embed_lr_schedule = self.lr_scheduler(
            num_train_steps, override_lr=self.learning_rate * self.adamh_embed_lr_scale
        )
        adamh_lmhead_lr_schedule = self.lr_scheduler(
            num_train_steps, override_lr=self.learning_rate * self.adamh_lmhead_lr_scale
        )
        gn_wdown_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr * self.gn_lr_scale)
        gn_wup_lr_schedule = self.lr_scheduler(
            num_train_steps,
            override_lr=self.adam_lr * self.gn_lr_scale * self.gn_wup_lr_scale,
        )
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr * self.adam_lr_scale)

        def optimizer(muonh_lr, adamh_embed_lr, adamh_lmhead_lr, gn_wdown_lr, gn_wup_lr, adam_lr):
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
                        learning_rate=muonh_lr,
                        coefficient_type=self.coefficient_type,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_embed_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_embed_lr))
                return optax.chain(*components)

            def adamh_lmhead_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_lmhead_lr))
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muonh": muonh_transform(),
                    "adamh_embed": adamh_embed_transform(),
                    "adamh": adamh_lmhead_transform(),
                    "adam_gn_wdown": adam_transform_at(gn_wdown_lr),
                    "adam_gn_wup": adam_transform_at(gn_wup_lr),
                    "adam": adam_transform_at(adam_lr),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            muonh_lr=muonh_lr_schedule,
            adamh_embed_lr=adamh_embed_lr_schedule,
            adamh_lmhead_lr=adamh_lmhead_lr_schedule,
            gn_wdown_lr=gn_wdown_lr_schedule,
            gn_wup_lr=gn_wup_lr_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            # token_embed -> AdamH (Frobenius hyperball).
            if "token_embed" in path_lower:
                return "adamh_embed"
            # GatedNorm: split into separate w_up / w_down sub-groups so the
            # sweep can perturb the output projection alone.
            if "gated_norm" in path_lower:
                if path_lower.endswith(".w_up"):
                    return "adam_gn_wup"
                if path_lower.endswith(".w_down"):
                    return "adam_gn_wdown"
                # Any unexpected GatedNorm leaf falls through to plain adam.
                return "adam"
            # Plain Adam group (small-LR).
            if "router_bias" in path_lower or path_lower.endswith(".attn_gate") or ".router" in path_lower:
                return "adam"
            # AdamH lm-head group.
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_may_arch_gn_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHMayArchGNMuonHConfig(OptimizerConfig):
    """may_arch MuonH variant: route GatedNorms to the muonh group (no col-norm).

    Four LR groups:
    - ``muonh``: matrices (attn, MoE MLP, shared) **and** all 4 GatedNorms.
    - ``adamh_embed``: ``token_embed``.
    - ``adamh``: ``lm_head`` / ``output_proj``.
    - ``adam``: ``router`` / ``router_bias`` / ``attn_gate`` / 1-D norm weights.

    Sibling to :class:`GrugMoeMuonHMayArchGNMuonHColNormConfig` but
    without the per-row/col norm equalization step. Defaults to LR scale
    1.0x everywhere. ``max_grad_norm`` defaults to ``None`` here (no
    clipping) for this variant.
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
    # If non-None, route ``token_embed`` to a plain-Adam group with LR
    # ``embed_adam_lr_scale * adam_lr``. Default ``None`` keeps the
    # baseline AdamH-on-embed routing.
    embed_adam_lr_scale: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        embed_adam_lr_schedule = None
        if self.embed_adam_lr_scale is not None:
            embed_adam_lr_schedule = self.lr_scheduler(
                num_train_steps, override_lr=self.adam_lr * self.embed_adam_lr_scale
            )

        def optimizer(learning_rate, adam_lr, embed_adam_lr):
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
                "muonh": muonh_transform(),
                "adamh_embed": adamh_transform_at(learning_rate),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            if self.embed_adam_lr_scale is not None:
                transforms["adam_embed"] = adam_transform_at(embed_adam_lr)
            return optax.multi_transform(transforms, self.create_mask)

        # ``optax.inject_hyperparams`` requires every kwarg to be a real
        # value -- pass a sentinel zero when the embed-on-adam group is
        # disabled. The transform itself isn't registered, so the value
        # never affects training.
        embed_adam_lr_kw = embed_adam_lr_schedule if embed_adam_lr_schedule is not None else 0.0
        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            embed_adam_lr=embed_adam_lr_kw,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam_embed" if self.embed_adam_lr_scale is not None else "adamh_embed"
            if "router_bias" in path_lower or path_lower.endswith(".attn_gate") or ".router" in path_lower:
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_may_arch_1pct_gn_split_lr_v1")
@dataclass(frozen=True)
class GrugMoeMuonHMayArch1pctGnSplitLrConfig(OptimizerConfig):
    """1pct-noclip with GatedNorm w_up vs w_down on separate MuonH LR scales.

    Splits the single ``muonh`` group of :class:`GrugMoeMuonHMayArchGNMuonHConfig`
    (the 1pct-noclip baseline) into three MuonH sub-groups:

    | sub-group           | matches                                  | LR                                    |
    |---------------------|------------------------------------------|---------------------------------------|
    | ``muonh``           | everything matrix-shaped except GN       | ``learning_rate``                     |
    | ``muonh_gn_wup``    | GatedNorm ``.w_up`` (all 4 instances)    | ``learning_rate * gn_wup_lr_scale``   |
    | ``muonh_gn_wdown``  | GatedNorm ``.w_down`` (all 4 instances)  | ``learning_rate * gn_wdown_lr_scale`` |

    Setting both ``gn_wup_lr_scale`` and ``gn_wdown_lr_scale`` to 1.0 reduces
    to :class:`GrugMoeMuonHMayArchGNMuonHConfig` (1pct-noclip baseline).

    Motivated by #5746's gnwup-0.7x finding (in the 2pct+clip recipe): GN w_up
    wants ~30% less LR than GN w_down at all 3 scales. This config lets us test
    the same direction on the 1pct-noclip recipe.
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
    gn_wup_lr_scale: float = 1.0
    gn_wdown_lr_scale: float = 1.0

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        gn_wup_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.gn_wup_lr_scale)
        gn_wdown_lr_schedule = self.lr_scheduler(
            num_train_steps, override_lr=self.learning_rate * self.gn_wdown_lr_scale
        )
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, gn_wup_lr, gn_wdown_lr, adam_lr):
            def muonh_transform_at(lr):
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

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muonh": muonh_transform_at(learning_rate),
                    "muonh_gn_wup": muonh_transform_at(gn_wup_lr),
                    "muonh_gn_wdown": muonh_transform_at(gn_wdown_lr),
                    "adamh_embed": adamh_transform_at(learning_rate),
                    "adamh": adamh_transform_at(learning_rate),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            gn_wup_lr=gn_wup_lr_schedule,
            gn_wdown_lr=gn_wdown_lr_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adamh_embed"
            if "router_bias" in path_lower or path_lower.endswith(".attn_gate") or ".router" in path_lower:
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorm split into w_up and w_down sub-groups.
            if "gated_norm" in path_lower:
                if path_lower.endswith(".w_up"):
                    return "muonh_gn_wup"
                if path_lower.endswith(".w_down"):
                    return "muonh_gn_wdown"
                return "muonh"
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


__all__ = [
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "GrugMoeNorMuonHConfig",
    "ScaleByGrugNorMuonHState",
    "scale_with_grug_muonh",
    "scale_with_grug_normuonh",
]
