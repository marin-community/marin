# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""AdamHR: Riemannian Adam on the fixed-radius manifold.

Instead of the step-then-project approach of AdamH, AdamHR:
1. Projects gradients into the tangent space (removing the radial component)
2. Maintains Adam state (mu, nu) in the tangent space
3. Computes Adam update from tangent-space state
4. Retracts back to the sphere
5. Parallel-transports optimizer state to the new tangent space

Input embeddings are routed through the constrained path rowwise,
so each embedding row maintains its initialization norm.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from levanter.optim.constrained_geometry import parallel_transport, project_tangent, retract
from levanter.optim.util import is_linear_like_module, label_linear_like_module
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("adamHR")
@dataclass(frozen=True)
class AdamHRConfig(OptimizerConfig):
    """Riemannian AdamH optimizer.

    Projects gradients to the tangent space of the fixed-radius sphere,
    runs Adam in the tangent space, retracts, and transports state.

    Input embeddings are routed through the constrained path rowwise.
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    adam_lr: float = 6e-4

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def adamhr_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_by_adamhr(self.beta1, self.beta2, self.epsilon, learning_rate)
                )
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "adamhr": adamhr_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule
        )

    def create_mask(self, params):
        """Route linear weights AND input embeddings to adamhr; biases and norms to adam."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                # Rowwise-constrained embeddings: route weight to adamhr
                return "adamhr"
            elif is_linear_like_module(param):
                return label_linear_like_module(param, weight_label="adamhr", bias_label="adam")
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=is_linear_like_module)


class ScaleByAdamHRState(NamedTuple):
    step_count: chex.Array
    mu: optax.Updates
    nu: optax.Updates


def scale_by_adamhr(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    learning_rate: float = 0.02,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """Riemannian Adam on the fixed-radius sphere.

    Maintains first and second moment estimates in the tangent space.
    After each update, retracts to the manifold and transports state.
    """

    mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        nu = otu.tree_zeros_like(params)
        return ScaleByAdamHRState(step_count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params):
        # 1. Project gradients to tangent space
        tangent_grads = jax.tree.map(
            lambda g, p: None if g is None else project_tangent(g, p),
            updates,
            params,
            is_leaf=lambda x: x is None,
        )

        # 2. Update Adam state in tangent space
        mu = otu.tree_update_moment(tangent_grads, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(tangent_grads, state.nu, b2, 2)
        count_inc = optax.safe_increment(state.step_count)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)

        # 3. Compute tangent-space Adam direction
        tangent_updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )

        # 4. Scale by -lr and retract to manifold
        adamhr_updates = jax.tree.map(
            lambda p, u: None if p is None else retract(p, -u, learning_rate),
            params,
            tangent_updates,
            is_leaf=lambda x: x is None,
        )

        # 5. Compute new params for state transport
        new_params = jax.tree.map(
            lambda p, delta: None if p is None else p + delta,
            params,
            adamhr_updates,
            is_leaf=lambda x: x is None,
        )

        # 6. Parallel-transport mu to new tangent space
        mu_transported = jax.tree.map(
            lambda m, p_old, p_new: None if m is None else parallel_transport(m, p_old, p_new),
            mu,
            params,
            new_params,
            is_leaf=lambda x: x is None,
        )

        mu_transported = otu.tree_cast(mu_transported, mu_dtype)

        return adamhr_updates, ScaleByAdamHRState(step_count=count_inc, mu=mu_transported, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)
