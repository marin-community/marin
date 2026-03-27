# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Muon optimizer for models using raw JAX arrays with (fan_in, fan_out) layout,
such as Grug models.

All 2D arrays are routed to Muon, except those whose path contains
'embed', 'lm_head', or 'output' (case-insensitive), which use AdamW.
"""

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig, ScaleByMuonState
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

VMAP_REPLICATED = "vmap_replicated"
STACK_BATCH_SHARDED = "stack_batch_sharded"
ORTHOGONALIZATION_LAYOUTS = (VMAP_REPLICATED, STACK_BATCH_SHARDED)


def _target_sharding(array) -> jax.sharding.Sharding | None:
    if array is None or not hasattr(array, "shape"):
        return None

    sharding = getattr(array, "sharding", None)
    if sharding is not None:
        return sharding

    aval = jax.typeof(array)
    return getattr(aval, "sharding", None)


def _partition_spec(array) -> PartitionSpec | None:
    sharding = _target_sharding(array)
    if sharding is None:
        return None
    return getattr(sharding, "spec", None)


def _batch_sharded_stack_target_pspec(array) -> PartitionSpec | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 3:
        return None

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return None

    mesh_shape = tuple((axis_name, axis_size) for axis_name, axis_size in mesh.shape.items() if axis_size > 1)
    if not mesh_shape:
        return None

    batch_axis = tuple(axis_name for axis_name, _ in mesh_shape)
    batch_shards = math.prod(axis_size for _, axis_size in mesh_shape)
    if array.shape[0] % batch_shards != 0:
        return None

    if len(batch_axis) == 1:
        return PartitionSpec(batch_axis[0], None, None)
    return PartitionSpec(batch_axis, None, None)


@OptimizerConfig.register_subclass("grug_muon")
@dataclass(frozen=True)
class GrugMuonConfig(MuonConfig):
    """
    Muon optimizer for models that use raw JAX arrays in (fan_in, fan_out) layout.

    Routing rules:
    - 2D arrays whose path does NOT contain 'embed', 'lm_head', or 'output' -> Muon
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
                components.append(_match_update_sharding())
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
            elif hasattr(param, "ndim") and param.ndim == 2:
                return "muon"
            elif hasattr(param, "ndim") and param.ndim == 3 and ("w_up_gate" in path_lower or "w_down" in path_lower):
                return "muon"
            else:
                return "adamw"

        return jax.tree.map(mask_fn, params, paths)


def _grug_scale_with_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    coefficient_type="quintic",
    orthogonalization_layout: str = STACK_BATCH_SHARDED,
):
    """Muon gradient transformation for raw arrays with matrix-shaped trailing dimensions."""
    steps = int(steps)
    if orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
        raise ValueError(
            f"Unknown orthogonalization_layout={orthogonalization_layout!r}. "
            f"Expected one of {ORTHOGONALIZATION_LAYOUTS!r}."
        )

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

        def transform_array(x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3):
                return x
            if x.ndim == 2:
                updated = _zeropower_via_newtonschulz_replicated(
                    x,
                    steps,
                    muon_eps,
                    coefficient_type,
                    None,
                )
            else:
                if orthogonalization_layout == VMAP_REPLICATED:
                    updated = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps,
                            muon_eps,
                            coefficient_type,
                            None,
                        )
                    )(x)
                else:
                    stack_target_pspec = _batch_sharded_stack_target_pspec(param)
                    if stack_target_pspec is None:
                        updated = jax.vmap(
                            lambda matrix: _zeropower_via_newtonschulz_replicated(
                                matrix,
                                steps,
                                muon_eps,
                                coefficient_type,
                                None,
                            )
                        )(x)
                    else:
                        updated = _zeropower_via_newtonschulz_batched_stack_sharded(
                            x,
                            steps,
                            muon_eps,
                            coefficient_type,
                            stack_target_pspec,
                        )

            fan_in, fan_out = updated.shape[-2:]
            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(fan_in, fan_out))
            updated *= scale
            return updated

        if params is None:
            updates = jax.tree.map(lambda x: transform_array(x, None), updates)
        else:
            updates = jax.tree.map(transform_array, updates, params)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _match_update_sharding():
    """Ensure updates inherit the parameter sharding expected by apply_updates."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _zeropower_via_newtonschulz_replicated(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Legacy Grug Muon orthogonalization that fully replicates each matrix.

    Replicates the array across devices before iterating to avoid sharding
    ambiguities in the X @ X.T contractions. The caller is responsible for
    restoring the final parameter layout. Kept for A/B benchmarking.
    """
    from jax.sharding import PartitionSpec as P, reshard

    assert X.ndim == 2
    del target_pspec  # Kept for signature parity with the other Newton-Schulz helpers.

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    if has_mesh:
        X = reshard(X, P(None, None))
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        out_sharding = P(None, None) if has_mesh else None
        A = jnp.einsum("ik,jk->ij", X, X, out_sharding=out_sharding)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A, out_sharding=out_sharding)
        X = a * X + jnp.einsum("ik,kj->ij", B, X, out_sharding=out_sharding)

    if transpose:
        X = X.T

    return X


def _zeropower_via_newtonschulz_batched_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Run Newton-Schulz on a stacked batch of matrices with only the batch axis sharded."""
    from jax.sharding import reshard

    assert X.ndim == 3

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps)

    transpose = False
    if X.shape[-2] > X.shape[-1]:
        X = jnp.swapaxes(X, -1, -2)
        transpose = True

    if target_pspec is None:
        target_pspec = _batch_sharded_stack_target_pspec(X)

    if has_mesh and target_pspec is not None:
        X = reshard(X, target_pspec)

    X_out_sharding = target_pspec if (has_mesh and target_pspec is not None) else None
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("...ik,...jk->...ij", X, X, out_sharding=X_out_sharding)
        B = b * A + c * jnp.einsum("...ik,...kj->...ij", A, A, out_sharding=X_out_sharding)
        X = a * X + jnp.einsum("...ik,...kj->...ij", B, X, out_sharding=X_out_sharding)

    if transpose:
        X = jnp.swapaxes(X, -1, -2)

    return X
