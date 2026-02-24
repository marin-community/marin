# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Muon optimizer for models using raw JAX arrays with (fan_in, fan_out) layout,
such as Grug models.

2D arrays and 3D arrays (where the first dim is a batch/expert dim) are routed
to Muon, except those whose path contains 'embed', 'lm_head', or 'output'
(case-insensitive), which use AdamW.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig, ScaleByMuonState
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, CoefficientType
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("grug_muon")
@dataclass(frozen=True)
class GrugMuonConfig(MuonConfig):
    """
    Muon optimizer for models that use raw JAX arrays in (fan_in, fan_out) layout.

    Routing rules:
    - 2D and 3D arrays whose path does NOT contain 'embed', 'lm_head', or 'output' -> Muon
    - Everything else -> AdamW
    """

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_transform():
                components = []
                components.append(
                    _grug_scale_with_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.coefficient_type,
                    )
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embed" in path_lower or "lm_head" in path_lower or "output" in path_lower:
                return "adamw"
            elif hasattr(param, "ndim") and param.ndim in (2, 3):
                return "muon"
            else:
                return "adamw"

        return jax.tree.map(mask_fn, params, paths)


def _grug_scale_with_muon(
    momentum=0.95, nesterov=True, steps=5, muon_eps=1e-8, use_kimi_scaling=False, coefficient_type="quintic"
):
    """Muon gradient transformation for raw 2D JAX arrays in (fan_in, fan_out) layout."""
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = buf

        def transform_array(x):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3):
                return x
            if x.ndim == 3:
                from jax.sharding import PartitionSpec as P, reshard

                # Replicate then vmap the Newton-Schulz core over the batch/expert dim
                x = reshard(x, P(None, None, None))
                updated = jax.vmap(
                    lambda m: _newtonschulz_core(m, steps=steps, eps=muon_eps, coefficient_type=coefficient_type)
                )(x)
                # Layout per slice is (fan_in, fan_out)
                fan_in, fan_out = updated.shape[1], updated.shape[2]
            else:
                updated = _zeropower_via_newtonschulz(x, steps=steps, eps=muon_eps, coefficient_type=coefficient_type)
                # Layout is (fan_in, fan_out)
                fan_in, fan_out = updated.shape
            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(fan_in, fan_out))
            updated *= scale
            return updated

        updates = jax.tree.map(transform_array, updates)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _newtonschulz_core(X, steps: int = 5, eps: float = 1e-7, coefficient_type: CoefficientType = "quintic"):
    """Pure Newton-Schulz iteration on a 2D matrix.

    No resharding logic so this is safe to call under vmap. Callers are
    responsible for replicating X before calling this function.
    """
    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    X /= jnp.linalg.norm(X) + eps

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transpose:
        X = X.T

    return X


def _zeropower_via_newtonschulz(X, steps: int = 5, eps: float = 1e-7, coefficient_type: CoefficientType = "quintic"):
    """Newton-Schulz iteration to orthogonalize a 2D matrix.

    Replicates X before iterating to avoid sharding ambiguities in the
    X @ X.T contractions.
    """
    from jax.sharding import PartitionSpec as P, reshard

    assert X.ndim == 2
    X = reshard(X, P(None, None))
    return _newtonschulz_core(X, steps=steps, eps=eps, coefficient_type=coefficient_type)
