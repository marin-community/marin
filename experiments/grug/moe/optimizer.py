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


@OptimizerConfig.register_subclass("grug_moe_muonh_per_group_lr_v1")
@dataclass(frozen=True)
class GrugMoeMuonHPerGroupLrConfig(OptimizerConfig):
    """MuonH config with independent LR multipliers per optimizer group.

    Same routing as :class:`GrugMoeMuonHConfig` (matrices -> muonh,
    lm_head / output_proj / GatedNorms -> adamh, biases / embeds /
    router / attn_gate -> adam) but each of the three transforms gets
    its own LR scaling factor:

    - ``muonh_lr_scale``: multiplies ``learning_rate`` for the muonh group.
    - ``adamh_lr_scale``: multiplies ``learning_rate`` for the adamh group.
    - ``adam_lr_scale``: multiplies ``adam_lr`` for the adam group.

    With all three scales = 1.0 the optimizer is identical to
    ``GrugMoeMuonHConfig``. Used by ``muonh_drop_gn_attngate_lr_sweep`` to
    sensitivity-test ±30% deviations of each group's LR around the
    baseline heuristic.
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
    muonh_lr_scale: float = 1.0
    adamh_lr_scale: float = 1.0
    adam_lr_scale: float = 1.0

    def build(self, num_train_steps):
        # Each group runs its own LR schedule via ``override_lr`` so that
        # warmup / decay shape is shared but the peak value is scaled.
        muonh_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.muonh_lr_scale)
        adamh_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.adamh_lr_scale)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr * self.adam_lr_scale)

        def optimizer(muonh_lr, adamh_lr, adam_lr):
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

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_lr))
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
            muonh_lr=muonh_lr_schedule,
            adamh_lr=adamh_lr_schedule,
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
            if "gated_norm" in path_lower:
                return "adamh"
            if hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_four_group_lr_v1")
@dataclass(frozen=True)
class GrugMoeMuonHFourGroupLrConfig(OptimizerConfig):
    """MuonH config with four independent LR groups.

    Splits the AdamH-style large-matrix group into two: ``adamh_lmhead``
    (just ``output_proj`` / ``lm_head``) and ``adamh_embed`` (just
    ``token_embed`` — moved out of the small-LR ``adam`` group). Each
    group has its own scale field multiplying either ``learning_rate``
    (muonh / adamh_lmhead / adamh_embed) or ``adam_lr`` (adam).

    Default scales bake in the new "tuned" point: lm_head and embed get
    0.7x, the adam (router / norms / biases) group gets 1.3x, muonh
    stays at 1.0x. Used by the second LR sensitivity sweep on top of
    the muonh-drop-gn-attngate recipe.
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
    muonh_lr_scale: float = 1.0
    adamh_lmhead_lr_scale: float = 0.7
    adamh_embed_lr_scale: float = 0.7
    adam_lr_scale: float = 1.3

    def build(self, num_train_steps):
        muonh_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.learning_rate * self.muonh_lr_scale)
        adamh_lmhead_lr_schedule = self.lr_scheduler(
            num_train_steps, override_lr=self.learning_rate * self.adamh_lmhead_lr_scale
        )
        adamh_embed_lr_schedule = self.lr_scheduler(
            num_train_steps, override_lr=self.learning_rate * self.adamh_embed_lr_scale
        )
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr * self.adam_lr_scale)

        def optimizer(muonh_lr, adamh_lmhead_lr, adamh_embed_lr, adam_lr):
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

            def adamh_lmhead_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_lmhead_lr))
                return optax.chain(*components)

            def adamh_embed_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, adamh_embed_lr))
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
                    "adamh_lmhead": adamh_lmhead_transform(),
                    "adamh_embed": adamh_embed_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            muonh_lr=muonh_lr_schedule,
            adamh_lmhead_lr=adamh_lmhead_lr_schedule,
            adamh_embed_lr=adamh_embed_lr_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            # token_embed gets its own AdamH group (moved out of small-LR adam).
            if "token_embed" in path_lower:
                return "adamh_embed"
            # adam group: router / router_bias / attn_gate (no token_embed!).
            if "router_bias" in path_lower or path_lower.endswith(".attn_gate") or ".router" in path_lower:
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh_lmhead"
            if "gated_norm" in path_lower:
                # No GatedNorms in the drop-GN model, but keep the clause so the
                # config is reusable. Route to adamh_lmhead since they share
                # the same default LR.
                return "adamh_lmhead"
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
