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
from jax.sharding import reshard
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
            elif (
                hasattr(param, "ndim")
                and param.ndim == 3
                and ("w_up_gate" in path_lower or "w_gate_up" in path_lower or "w_down" in path_lower)
            ):
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

        def _is_fused_gate_up_path(path) -> bool:
            # Detect a fused MoE gate+up leaf by its pytree path attribute name.
            # When MoEExpertMlp has split_w_gate_up=False, the (E, D, 2I) leaf
            # is stored as w_gate_up; we want per-half NS by reshaping to
            # (E, D, 2, I) and vmapping over both leading dims so the math
            # matches what split_w_gate_up=True would do.
            for entry in path:
                name = getattr(entry, "name", None)
                if name == "w_gate_up":
                    return True
            return False

        ns_fn = lambda m: _zeropower_via_newtonschulz_replicated(m, steps, muon_eps, coefficient_type, None)

        def transform_array(path, x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3, 4):
                return x
            if x.ndim == 2:
                updated = _zeropower_via_newtonschulz_replicated(
                    x,
                    steps,
                    muon_eps,
                    coefficient_type,
                    None,
                )
            elif x.ndim == 4:
                # Stacked + MoE 4D leaf, shape (L, E, D, I) for w_gate/w_up
                # or (L, E, I, D) for w_down. The "one matrix per chip"
                # plan: bf16 cast, free-merge (L, E) into LE (replicated),
                # explicit all-to-all reshard to put LE on data axis (each
                # chip ends up with merged/num_chips full matrices), local
                # NS, then reverse. Splits the cross-axis sharding migration
                # from the axis merge so XLA does each cheaply rather than
                # together (which has been triggering 87 GB materializations).
                mesh = jax.sharding.get_abstract_mesh()
                if mesh.empty:
                    return x
                mesh_shape_items = [(n, s) for n, s in mesh.shape.items() if s > 1]
                if not mesh_shape_items:
                    return x
                batch_shards = 1
                for _, s in mesh_shape_items:
                    batch_shards *= s
                layers, expert_count, d, last = x.shape
                merged = layers * expert_count
                if merged % batch_shards != 0:
                    return x
                batch_axes = tuple(n for n, _ in mesh_shape_items)

                # Detect whether this leaf is w_down (path attribute) so the
                # intermediate 3D / 4D specs reflect the actual axis order.
                is_w_down = any(getattr(entry, "name", None) == "w_down" for entry in path)
                if is_w_down:
                    # (L, E, I, D) → merged (LE, I, D), data axis stays on D.
                    intermediate_3d_spec = PartitionSpec(None, "model", "data")
                    orig_4d_spec = PartitionSpec(None, "expert", "model", "data")
                else:
                    # (L, E, D, I) → merged (LE, D, I), data axis stays on D.
                    intermediate_3d_spec = PartitionSpec(None, "data", "model")
                    orig_4d_spec = PartitionSpec(None, "expert", "data", "model")

                target_3d_spec = (
                    PartitionSpec(batch_axes[0], None, None)
                    if len(batch_axes) == 1
                    else PartitionSpec(batch_axes, None, None)
                )

                # 1) bf16 cast — preserves sharding, halves the bytes for
                #    the all-to-all that follows.
                x_bf16 = x.astype(jnp.bfloat16)

                # 2) Free reshape: only merges axes that are size-1-mapped
                #    on the mesh (L=replicated, E=on expert axis size 1).
                #    Sharding stays on the data axis. No data movement.
                x_flat = jax.lax.reshape(x_bf16, (merged, d, last), out_sharding=intermediate_3d_spec)

                # 3) Explicit all-to-all reshard: move sharding from the
                #    data-side axis to the leading batch axis. After this,
                #    each chip holds merged/batch_shards full matrices.
                x_distributed = jax.sharding.reshard(x_flat, target_3d_spec)

                # 4) Local NS: each chip processes its matrices without any
                #    further resharding. vmap over the leading axis runs NS
                #    on each (D, I) (or (I, D)) matrix the chip owns.
                local_ns = lambda matrix: _zeropower_via_newtonschulz_local(matrix, steps, muon_eps, coefficient_type)
                updated_distributed = jax.vmap(local_ns)(x_distributed)

                # 5) Reverse all-to-all: put sharding back on the data axis.
                updated_flat = jax.sharding.reshard(updated_distributed, intermediate_3d_spec)

                # 6) Free reverse reshape back to 4D.
                updated_bf16 = jax.lax.reshape(
                    updated_flat,
                    (layers, expert_count, d, last),
                    out_sharding=orig_4d_spec,
                )

                # 7) Cast back to the input dtype so the downstream optax
                #    chain (kimi scaling, etc.) sees the same dtype.
                updated = updated_bf16.astype(x.dtype)
            elif _is_fused_gate_up_path(path):
                # Per-half NS on a fused (E, D, 2I) tensor: reshape to
                # (E, D, 2, I), double-vmap NS, reshape back. Reshapes are
                # free stride changes on TPU.
                e, d, two_i = x.shape
                assert two_i % 2 == 0, f"fused gate_up trailing dim must be even, got {two_i}"
                i_half = two_i // 2
                x_split = x.reshape(e, d, 2, i_half)
                updated_split = jax.vmap(
                    jax.vmap(ns_fn, in_axes=2, out_axes=2),
                    in_axes=0,
                    out_axes=0,
                )(x_split)
                updated = updated_split.reshape(e, d, two_i)
            else:
                if orthogonalization_layout == VMAP_REPLICATED:
                    updated = jax.vmap(ns_fn)(x)
                else:
                    stack_target_pspec = _batch_sharded_stack_target_pspec(param)
                    if stack_target_pspec is None:
                        updated = jax.vmap(ns_fn)(x)
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
            updates = jax.tree_util.tree_map_with_path(lambda path, x: transform_array(path, x, None), updates)
        else:
            updates = jax.tree_util.tree_map_with_path(transform_array, updates, params)

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


def _zeropower_via_newtonschulz_local(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
) -> jax.Array:
    """Newton-Schulz that assumes X is already fully local to one chip.

    Unlike ``_zeropower_via_newtonschulz_replicated``, this variant does NOT
    do an internal ``reshard(X, P(None, None))`` to gather the matrix to all
    chips. Callers are responsible for setting up sharding so that each chip
    already holds the matrices it needs to process locally — useful when
    NS is being run on per-chip-owned matrices (one or many) via vmap over
    a leading batch axis sharded on the data axis.
    """
    assert X.ndim == 2

    # Run NS in bf16 to halve memory and double matmul throughput on TPU.
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("ik,jk->ij", X, X)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A)
        X = a * X + jnp.einsum("ik,kj->ij", B, X)

    if transpose:
        X = X.T

    return X.astype(orig_dtype)


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
    P = PartitionSpec
    assert X.ndim == 2
    del target_pspec  # Kept for signature parity with the other Newton-Schulz helpers.

    # Run NS in bf16 to halve all-gather bytes and double matmul throughput;
    # cast back to the param dtype on exit so optimizer state stays fp32.
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

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

    return X.astype(orig_dtype)


def _zeropower_via_newtonschulz_batched_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Run Newton-Schulz on a stacked batch of matrices with only the batch axis sharded."""
    assert X.ndim == 3

    # Run NS in bf16 to halve all-gather bytes and double matmul throughput;
    # cast back to the param dtype on exit so optimizer state stays fp32.
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

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

    return X.astype(orig_dtype)
