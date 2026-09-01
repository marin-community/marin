# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe_hero_ep.adamh import scale_by_adamh
from experiments.grug.moe_hero_ep.grugmuon_hero import _grug_scale_with_muon_hero, _target_named_sharding


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding without touching single-device arrays."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state
        updates = _match_named_sharding_to_params(updates, params)
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


def _pin_sharding(x, ref):
    """Reshard ``x`` to ``ref``'s named sharding so a following norm reduces correctly.

    ``new_param`` is a computed intermediate; leaving it with an SPMD-inferred sharding lets the sharded
    ``norm(new_param)`` over-count and collapse the tensor (issue #8073). This reshard is a same-layout
    no-op at runtime.
    """
    sharding = _target_named_sharding(ref)
    return jax.sharding.reshard(x, sharding) if sharding is not None else x


def _scale_invariant_hyperball_updates(params, direction_updates, learning_rate: float):
    """MuonH hyperball step: move along the orthogonalized direction, then project back to the
    parameter's Frobenius sphere (scale-invariant update)."""
    direction_updates = _match_named_sharding_to_params(direction_updates, params)

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            # jnp.linalg.norm over a sharded matrix mis-lowers under SPMD and over-counts (issue #8073);
            # sum-of-squares in float32 plus a same-layout reshard of the intermediate reduces correctly.
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param.astype(jnp.float32))))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update.astype(jnp.float32))))
            new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
            new_param = _pin_sharding(new_param, param)
            new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param.astype(jnp.float32))))
            return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

        axes = tuple(range(1, param.ndim))
        param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
        new_param = param - learning_rate * update * param_norm / jnp.maximum(update_norm, 1e-10)
        new_param = _pin_sharding(new_param, param)  # correct the sharded norm reduction (issue #8073)
        new_param_norm = jnp.sqrt(jnp.sum(jnp.square(new_param), axis=axes, keepdims=True))
        return new_param / jnp.maximum(new_param_norm, 1e-10) * param_norm - param

    return jax.tree.map(scale_invariant_update, params, direction_updates, is_leaf=lambda x: x is None)


def _is_gate_or_router_weight(path_lower: str) -> bool:
    """True for the ``attn_gate`` and ``router`` weight leaves (``router_bias`` excluded)."""
    return "router_bias" not in path_lower and (path_lower.endswith(".attn_gate") or ".router" in path_lower)


def _gate_router_decay_mask(params):
    """Boolean pytree that is True on the ``attn_gate`` and ``router`` weight leaves -- the ones that
    receive decoupled weight decay -- and False everywhere else."""
    paths = leaf_key_paths(params)

    def is_target(_, path):
        path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
        return _is_gate_or_router_weight(path_str.lower())

    return jax.tree.map(is_target, params, paths)


def _scale_by_adam_gate_router_decay(
    b1: float, b2: float, eps: float, weight_decay: float, total_steps: int
) -> optax.GradientTransformation:
    """``optax.scale_by_adam`` plus decoupled (AdamW-style) weight decay on ``attn_gate`` and the
    ``router`` weight, annealed linearly from ``weight_decay`` to 0 across ``total_steps``.

    The decay coefficient is read from the Adam step ``count`` rather than a separate schedule, so on
    a checkpoint resume it evaluates at the restored global step. The state stays
    ``optax.ScaleByAdamState`` -- byte-for-byte the plain ``scale_by_adam`` structure -- so a
    checkpoint written without weight decay restores unchanged (moments and count preserved), and the
    decay simply switches on from the resumed step.
    """
    adam = optax.scale_by_adam(b1, b2, eps)

    def init_fn(params):
        return adam.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("_scale_by_adam_gate_router_decay requires params for decoupled decay")
        step = state.count  # incremented once per update, so it holds the current global step
        updates, next_state = adam.update(updates, state, params)
        wd = weight_decay * jnp.clip(1.0 - step / total_steps, 0.0, None)
        mask = _gate_router_decay_mask(params)
        updates = jax.tree.map(lambda u, p, keep: u + wd * p if keep else u, updates, params, mask)
        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    use_syrk: bool = True,
) -> optax.GradientTransformation:
    """MuonH transform for the stacked hero model: Newton-Schulz direction + Frobenius hyperball step."""
    muon_transform = _grug_scale_with_muon_hero(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        coefficient_type=coefficient_type,
        use_syrk=use_syrk,
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


@OptimizerConfig.register_subclass("grug_moe_hero_ep_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """MuonH optimizer for the EP hero model. Three LR groups (muonh / adamh / adam):

    - ``muonh``: matrix leaves (attn, MoE MLP, shared) and GatedNorms -- Newton-Schulz
      orthogonalization + Frobenius hyperball scale-invariant step.
    - ``adamh``: ``output_proj`` / ``lm_head``.
    - ``adam``: ``token_embed`` / ``router`` / ``router_bias`` / ``attn_gate`` / 1-D norm gains
      and the tiny SConv kernels.

    ``use_syrk`` routes the 4D expert-stack Newton-Schulz through QuACK's symmetric GEMM.

    ``gate_router_weight_decay`` (0 disables) applies decoupled weight decay to ``attn_gate`` and the
    ``router`` weight only, annealed linearly from the given value to 0 over training. It is folded
    into the ``adam`` group's transform (state unchanged), so it can be switched on when continuing
    from a checkpoint trained without it.
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
    use_syrk: bool = True
    gate_router_weight_decay: float = 0.0

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
                        use_syrk=self.use_syrk,
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
                if self.gate_router_weight_decay > 0.0:
                    components.append(
                        _scale_by_adam_gate_router_decay(
                            self.beta1, self.beta2, self.epsilon, self.gate_router_weight_decay, num_train_steps
                        )
                    )
                else:
                    components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                return optax.chain(*components)

            transforms = {
                "muonh": muonh_transform(),
                "adamh": adamh_transform_at(learning_rate),
                "adam": adam_transform_at(adam_lr),
            }
            return optax.multi_transform(transforms, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower or "router_bias" in path_lower or _is_gate_or_router_weight(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            # GatedNorms route to muonh (NS + Frobenius hyperball), same as matrices.
            if "gated_norm" in path_lower:
                return "muonh"
            # Scanning prepends a layer axis, so norm gains / SConv kernels stay named ``.weight``
            # (route to Adam) while expert matrices become 4D and other matmuls 3D (route to MuonH).
            if path_lower.endswith(".weight"):
                return "adam"
            if hasattr(param, "ndim") and param.ndim in (2, 3, 4):
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


__all__ = [
    "GrugMoeMuonHConfig",
    "scale_with_grug_muonh",
]
