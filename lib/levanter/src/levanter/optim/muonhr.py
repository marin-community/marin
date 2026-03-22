# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MuonHR: Riemannian Muon on the fixed-radius manifold.

Same Riemannian approach as AdamHR, but uses Muon (Newton-Schulz
orthogonalization + momentum) for linear weights instead of Adam.
Three-way routing: MuonHR for linear weights, AdamHR for lm_head
and embeddings, Adam for biases and norms.
"""

import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import jax
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.adamhr import scale_by_adamhr
from levanter.optim.config import OptimizerConfig
from levanter.optim.constrained_geometry import parallel_transport, project_tangent, retract
from levanter.optim.util import (
    CoefficientType,
    label_linear_like_module,
    map_flattened_linear_layers,
    zeropower_via_newtonschulz5,
)
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("muonHR")
@dataclass(frozen=True)
class MuonHRConfig(OptimizerConfig):
    """Riemannian MuonH optimizer.

    Uses Muon (Newton-Schulz orthogonalization + momentum) for linear weights
    with Riemannian constrained updates. AdamHR for lm_head and embeddings.
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    coefficient_type: CoefficientType = "quintic"

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonhr_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_muonhr(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        learning_rate,
                        self.coefficient_type,
                    )
                )
                return optax.chain(*components)

            def adamhr_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamhr(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muonhr": muonhr_transform(),
                "adamhr": adamhr_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """Route linear weights to muonhr, lm_head and embeddings to adamhr, rest to adam."""
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adamhr"
            elif "lm_head" in path_str:
                return "adamhr"
            elif isinstance(param, haliax.nn.Linear):
                return label_linear_like_module(param, weight_label="muonhr", bias_label="adam")
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


class ScaleByMuonHRState(NamedTuple):
    momentum_buffer: optax.Updates


def scale_with_muonhr(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    learning_rate=0.02,
    coefficient_type="quintic",
):
    """Riemannian Muon: project to tangent space, orthogonalize, retract, transport."""
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByMuonHRState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        # 1. Project gradients to tangent space
        tangent_grads = jax.tree.map(
            lambda g, p: None if g is None else project_tangent(g, p),
            updates,
            params,
            is_leaf=lambda x: x is None,
        )

        # 2. Momentum in tangent space
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            tangent_grads,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            tangent_updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                tangent_grads,
                is_leaf=lambda x: x is None,
            )
        else:
            tangent_updates = buf

        # 3. Newton-Schulz orthogonalization on linear layers
        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2
            array = layer.weight.array
            updated_weight_array = zeropower_via_newtonschulz5(
                array, steps=steps, eps=muon_eps, coefficient_type=coefficient_type
            )
            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)
            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        muon_updates = map_flattened_linear_layers(transform_linear_layer, tangent_updates)

        # 4. Retract to manifold
        muonhr_updates = jax.tree.map(
            lambda p, u: None if p is None else retract(p, -u, learning_rate),
            params,
            muon_updates,
            is_leaf=lambda x: x is None,
        )

        # 5. Transport momentum buffer to new tangent space
        new_params = jax.tree.map(
            lambda p, delta: None if p is None else p + delta,
            params,
            muonhr_updates,
            is_leaf=lambda x: x is None,
        )
        buf_transported = jax.tree.map(
            lambda m, p_old, p_new: None if m is None else parallel_transport(m, p_old, p_new),
            buf,
            params,
            new_params,
            is_leaf=lambda x: x is None,
        )

        return muonhr_updates, ScaleByMuonHRState(momentum_buffer=buf_transported)

    return optax.GradientTransformation(init_fn, update_fn)
