# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu

import haliax

from levanter.optim.config import OptimizerConfig
from experiments.speedrun.prism_berkeley_qwen3_scaling.optimizer_helpers import (
    flatten_linear_layers,
    label_linear_like_module,
    map_flattened_linear_layers,
    normalize_2d_update_fro_norm,
)
from levanter.utils.jax_utils import leaf_key_paths


def _weight_decay_hyperparam(
    base_weight_decay: float,
    *,
    learning_rate_schedule,
    peak_learning_rate: float,
    adamc_weight_decay: bool,
):
    if not adamc_weight_decay:
        return base_weight_decay
    if peak_learning_rate <= 0:
        raise ValueError(f"peak_learning_rate must be positive, got {peak_learning_rate}.")

    def schedule(count):
        return base_weight_decay * (learning_rate_schedule(count) / peak_learning_rate)

    return schedule


PrismBerkeleyOrder = Literal[3, 5]


def _quartic_objective(alpha: jax.Array, c1: jax.Array, c2: jax.Array, c3: jax.Array, c4: jax.Array) -> jax.Array:
    return (((c4 * alpha + c3) * alpha + c2) * alpha + c1) * alpha


def _grid_search_quartic_argmin(
    c1: jax.Array,
    c2: jax.Array,
    c3: jax.Array,
    c4: jax.Array,
    *,
    alpha_min: float,
    alpha_max: float,
    grid_points: int,
) -> jax.Array:
    candidates = jnp.linspace(alpha_min, alpha_max, grid_points, dtype=jnp.float32)
    values = _quartic_objective(candidates, c1, c2, c3, c4)
    return candidates[jnp.argmin(values)]


def _trace_sketch_powers(residual: jax.Array, sketch: jax.Array, max_power: int) -> dict[int, jax.Array]:
    sketch_t = sketch.T.astype(jnp.float32)
    current = sketch_t
    traces = {}
    for power in range(1, max_power + 1):
        current = residual @ current
        traces[power] = jnp.sum(sketch_t * current)
    return traces


def _prism_berkeley_alpha(
    residual: jax.Array,
    sketch: jax.Array,
    *,
    order: PrismBerkeleyOrder,
    alpha_grid_points: int,
) -> jax.Array:
    traces = _trace_sketch_powers(residual, sketch, 6 if order == 3 else 10)

    if order == 3:
        c1 = 4.0 * traces[3] - 4.0 * traces[2]
        c2 = 6.0 * traces[4] - 10.0 * traces[3] + 4.0 * traces[2]
        c3 = 4.0 * traces[5] - 8.0 * traces[4] + 4.0 * traces[3]
        c4 = traces[6] - 2.0 * traces[5] + traces[4]
        alpha_min, alpha_max = 0.5, 1.0
    elif order == 5:
        c1 = 0.5 * traces[7] + 2.0 * traces[6] + 0.5 * traces[5] - 3.0 * traces[4]
        c2 = 1.5 * traces[8] + 3.0 * traces[7] - 4.5 * traces[6] - 4.0 * traces[5] + 4.0 * traces[4]
        c3 = 2.0 * traces[9] - 6.0 * traces[7] + 4.0 * traces[6]
        c4 = traces[10] - 2.0 * traces[9] + traces[8]
        alpha_min, alpha_max = 3.0 / 8.0, 29.0 / 20.0
    else:
        raise ValueError(f"Unsupported PRISM-Berkeley order: {order}")

    return _grid_search_quartic_argmin(
        c1.astype(jnp.float32),
        c2.astype(jnp.float32),
        c3.astype(jnp.float32),
        c4.astype(jnp.float32),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        grid_points=alpha_grid_points,
    )


def _make_default_sketch(weight_shape: tuple[int, ...], sketch_size: int, key: jax.Array) -> jax.Array:
    if len(weight_shape) < 2:
        raise ValueError(f"Expected a matrix-like weight shape, got {weight_shape}.")

    min_dim = min(weight_shape[-2:])
    effective_sketch = min(sketch_size, min_dim)
    sketch_shape = (*weight_shape[:-2], effective_sketch, min_dim)
    return jax.random.normal(key, sketch_shape, dtype=jnp.float32) / jnp.sqrt(effective_sketch)


def _init_sketch_cache(params, *, sketch_size: int, seed: int):
    flattened_params = flatten_linear_layers(params)
    leaves, treedef = jax.tree_util.tree_flatten(
        flattened_params, is_leaf=lambda x: isinstance(x, haliax.nn.Linear) or x is None
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))

    sketches = []
    for leaf, key in zip(leaves, keys, strict=True):
        if isinstance(leaf, haliax.nn.Linear) and leaf.weight is not None and leaf.weight.array is not None:
            sketches.append(_make_default_sketch(tuple(leaf.weight.array.shape), sketch_size, key))
        else:
            sketches.append(None)

    return jax.tree_util.tree_unflatten(treedef, sketches)


def _prism_berkeley_polar(
    matrix: jax.Array,
    sketch: jax.Array | None,
    *,
    steps: int,
    order: PrismBerkeleyOrder,
    eps: float,
    alpha_grid_points: int,
    fixed_alpha_steps: int,
) -> jax.Array:
    chex.assert_rank(matrix, 2)

    working = matrix.astype(jnp.float32)
    working = working / (jnp.linalg.norm(working) + eps)

    transposed = False
    if working.shape[0] < working.shape[1]:
        working = working.T
        transposed = True

    min_dim = working.shape[1]
    if sketch is None:
        raise ValueError("PRISM-Berkeley expected a cached sketch for a linear layer, but received None.")
    if sketch.ndim != 2:
        raise ValueError(f"PRISM-Berkeley expected a rank-2 sketch, got shape {sketch.shape}.")
    if sketch.shape[1] != min_dim:
        raise ValueError(f"PRISM-Berkeley sketch shape {sketch.shape} is incompatible with min_dim={min_dim}.")
    sketch = sketch.astype(jnp.float32)

    identity = jnp.eye(min_dim, dtype=jnp.float32)
    fixed_alpha = 1.0 if order == 3 else 29.0 / 20.0

    for step_idx in range(steps):
        residual = identity - working.T @ working
        alpha = lax_cond_fixed_or_adaptive(
            step_idx < fixed_alpha_steps,
            fixed_alpha,
            residual,
            sketch,
            order,
            alpha_grid_points,
        )
        if order == 3:
            transform = identity + alpha * residual
        else:
            residual_sq = residual @ residual
            transform = identity + 0.5 * residual + alpha * residual_sq
        working = working @ transform

    if transposed:
        working = working.T

    return working.astype(matrix.dtype)


def lax_cond_fixed_or_adaptive(
    use_fixed: bool,
    fixed_alpha: float,
    residual: jax.Array,
    sketch: jax.Array,
    order: PrismBerkeleyOrder,
    alpha_grid_points: int,
) -> jax.Array:
    return jax.lax.cond(
        jnp.asarray(use_fixed),
        lambda _: jnp.asarray(fixed_alpha, dtype=jnp.float32),
        lambda _: _prism_berkeley_alpha(
            residual,
            sketch,
            order=order,
            alpha_grid_points=alpha_grid_points,
        ),
        operand=None,
    )


class ScaleByPrismBerkeleyState(NamedTuple):
    momentum_buffer: optax.Updates
    sketch_cache: optax.Updates


def scale_with_prism_berkeley(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    order: PrismBerkeleyOrder = 3,
    alpha_grid_points: int = 65,
    fixed_alpha_steps: int = 0,
    sketch_size: int = 5,
    sketch_seed: int = 0,
):
    steps = int(steps)
    alpha_grid_points = int(alpha_grid_points)
    fixed_alpha_steps = int(fixed_alpha_steps)
    sketch_size = int(sketch_size)

    if order not in (3, 5):
        raise ValueError(f"PRISM-Berkeley order must be 3 or 5, got {order}.")
    if alpha_grid_points < 3:
        raise ValueError(f"alpha_grid_points must be >= 3, got {alpha_grid_points}.")
    if sketch_size <= 0:
        raise ValueError(f"sketch_size must be positive, got {sketch_size}.")

    def init_fn(params):
        return ScaleByPrismBerkeleyState(
            momentum_buffer=otu.tree_zeros_like(params),
            sketch_cache=_init_sketch_cache(params, sketch_size=sketch_size, seed=sketch_seed),
        )

    def update_fn(updates, state, params=None):
        del params

        grad_coeff = 1.0 - momentum
        momentum_buffer = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + grad_coeff * g,
            state.momentum_buffer,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + grad_coeff * g,
                momentum_buffer,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates = momentum_buffer

        def transform_linear_layer(layer: haliax.nn.Linear, sketch: jax.Array | None):
            assert layer.weight.ndim == 2
            transformed_weight_array = _prism_berkeley_polar(
                layer.weight.array,
                sketch,
                steps=steps,
                order=order,
                eps=muon_eps,
                alpha_grid_points=alpha_grid_points,
                fixed_alpha_steps=fixed_alpha_steps,
            )
            transformed_weight_array = normalize_2d_update_fro_norm(transformed_weight_array)

            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, transformed_weight_array.shape[0] / transformed_weight_array.shape[1]))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(transformed_weight_array.shape[0], transformed_weight_array.shape[1]))
            transformed_weight_array = transformed_weight_array * scale

            return dataclasses.replace(
                layer,
                weight=dataclasses.replace(
                    layer.weight, array=transformed_weight_array.astype(layer.weight.array.dtype)
                ),
            )

        transformed_updates = map_flattened_linear_layers(transform_linear_layer, updates, state.sketch_cache)
        return transformed_updates, ScaleByPrismBerkeleyState(
            momentum_buffer=momentum_buffer,
            sketch_cache=state.sketch_cache,
        )

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("prism_berkeley")
@dataclass(frozen=True)
class PrismBerkeleyConfig(OptimizerConfig):
    lr: float = 0.02
    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    weight_decay: float = 0.0
    adam_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    adamc_weight_decay: bool = False
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False
    order: PrismBerkeleyOrder = 3
    alpha_grid_points: int = 65
    fixed_alpha_steps: int = 0
    sketch_size: int = 5
    sketch_seed: int = 0

    def build(self, num_train_steps):
        if self.order not in (3, 5):
            raise ValueError(f"PrismBerkeleyConfig.order must be 3 or 5, got {self.order}.")
        if self.alpha_grid_points < 3:
            raise ValueError(f"PrismBerkeleyConfig.alpha_grid_points must be >= 3, got {self.alpha_grid_points}.")
        if self.sketch_size <= 0:
            raise ValueError(f"PrismBerkeleyConfig.sketch_size must be positive, got {self.sketch_size}.")

        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        weight_decay_hyperparam = _weight_decay_hyperparam(
            self.weight_decay,
            learning_rate_schedule=learning_rate_schedule,
            peak_learning_rate=self.learning_rate,
            adamc_weight_decay=self.adamc_weight_decay,
        )
        adam_base_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
        adam_weight_decay_hyperparam = _weight_decay_hyperparam(
            adam_base_weight_decay,
            learning_rate_schedule=adam_lr_schedule,
            peak_learning_rate=self.adam_lr,
            adamc_weight_decay=self.adamc_weight_decay,
        )

        def optimizer(learning_rate, adam_lr, weight_decay, adam_weight_decay):
            def prism_berkeley_transform():
                components = [
                    scale_with_prism_berkeley(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.order,
                        self.alpha_grid_points,
                        self.fixed_alpha_steps,
                        self.sketch_size,
                        self.sketch_seed,
                    ),
                    optax.add_decayed_weights(weight_decay, self.build_weight_decay_mask()),
                    optax.scale(-learning_rate),
                ]
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "prism_berkeley": prism_berkeley_transform(),
                    "adamw": adamw_transform(),
                },
                partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            weight_decay=weight_decay_hyperparam,
            adam_weight_decay=adam_weight_decay_hyperparam,
        )

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            if isinstance(param, haliax.nn.Linear):
                assert param._out_first or use_kimi_scaling
                return label_linear_like_module(param, weight_label="prism_berkeley", bias_label="adamw")
            return "adamw"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, haliax.nn.Linear))


OptimizerConfig.register_subclass("prism-berkeley", PrismBerkeleyConfig)
