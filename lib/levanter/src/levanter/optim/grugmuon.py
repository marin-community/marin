# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Muon optimizer for models using raw JAX arrays with (fan_in, fan_out) layout,
such as Grug models.

All 2D arrays are routed to Muon, except those whose path contains
'embed', 'lm_head', or 'output' (case-insensitive), which use AdamW.
"""

import math
import os
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


def _effective_ns_steps(steps: int) -> int:
    """Newton-Schulz iteration count, forced to 0 when SCALE_MUON_DROP_NS_MATMULS=1.

    Dropping the iterations skips the XX^T / A@A / B@X matmuls while keeping the bf16 cast,
    Frobenius normalization, and any reshard/distribution the caller performs -- isolating the
    NS matmul cost without the fp32/undistributed confound of a full no-op. Read at trace time."""
    return 0 if os.environ.get("SCALE_MUON_DROP_NS_MATMULS") == "1" else steps


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
            # Route by role, not raw ndim: stacked-block scanning prepends a layer axis
            # (attn/MLP 2D->3D, MoE experts 3D->4D), so an ndim test would misroute the
            # scanned weights. Muon gets the weight matrices (whose orthogonalizable matrix
            # is the trailing two dims); the embedding, LM head, router gate, and any
            # bias/scalar (<2D) use AdamW.
            if not hasattr(param, "ndim") or param.ndim < 2:
                return "adamw"
            # RMSNorm/LayerNorm gains are per-dimension scales, not orthogonalizable matrices
            # (they stack to 2D under scan), so keep them on AdamW alongside embed/head/router.
            if any(k in path_lower for k in ("embed", "lm_head", "output", "router", "norm")):
                return "adamw"
            return "muon"

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

        def transform_array(path, x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3, 4):
                return x
            if os.environ.get("SCALE_MUON_NO_NS") == "1":
                # No-op the Newton-Schulz orthogonalization (momentum-only update): skips
                # the all-gather/reshard transient. Not real Muon; for memory/fit probes.
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
                # Stacked MoE expert leaf (L, E, D, I) / (L, E, I, D): distribute whole
                # matrices across chips (data-parallel over L*E) and run NS locally, never
                # gathering D/I to full-replicated.
                updated = _newtonschulz_4d_distributed(path, x, steps, muon_eps, coefficient_type)
            else:
                # 3D non-expert leaf (attn q/k/v/o + gated norms + dense, stacked [L, d_in, d_out]
                # under scan). SCALE_MUON_DIST_NONEXPERT=1 forces the stack-sharded distributed NS
                # (each chip orthogonalizes its ~L/shards matrices) instead of replicating the whole
                # stack to P(None,None) and running NS redundantly on every device. Read at trace
                # time so it can be toggled in-process.
                effective_layout = orthogonalization_layout
                if os.environ.get("SCALE_MUON_DIST_NONEXPERT") == "1":
                    effective_layout = STACK_BATCH_SHARDED
                if effective_layout == VMAP_REPLICATED:
                    updated = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps,
                            muon_eps,
                            coefficient_type,
                            None,
                        )
                    )(x)
                elif os.environ.get("SCALE_MUON_PAD_NONEXPERT") == "1":
                    # Pad the L-stack up to a multiple of the (intra-rack) shard count so a stack
                    # whose length doesn't divide the mesh (e.g. 48 layers over 64 data GPUs) still
                    # distributes one matrix per chip instead of replicating NS on every device.
                    updated = _newtonschulz_padded_stack_sharded(x, steps, muon_eps, coefficient_type)
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
    """Newton-Schulz that assumes ``X`` is already fully local to one device.

    Unlike :func:`_zeropower_via_newtonschulz_replicated`, this does NOT reshard ``X`` to
    ``P(None, None)`` to gather it across devices. The caller must arrange sharding so each
    device already holds the matrices it processes locally (e.g. vmapped over a leading axis
    that is sharded on the batch/data axis).
    """
    assert X.ndim == 2
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True

    for i in range(_effective_ns_steps(steps)):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("ik,jk->ij", X, X)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A)
        X = a * X + jnp.einsum("ik,kj->ij", B, X)

    if transpose:
        X = X.T

    return X.astype(orig_dtype)


def _newtonschulz_4d_distributed(
    path,
    x: jax.Array,
    steps: int,
    eps: float,
    coefficient_type: CoefficientType,
) -> jax.Array:
    """Newton-Schulz on a stacked 4D MoE expert leaf without gathering D/I to replicated.

    The leaf is ``(L, E, D, I)`` for ``w_gate``/``w_up`` or ``(L, E, I, D)`` for ``w_down``,
    sharded ``P(None, "expert", "data", "model")``. The "one matrix per chip" plan: bf16
    cast, free-merge ``(L, E) -> LE`` keeping the sharding on the ``data`` axis, an explicit
    all-to-all reshard to move ``LE`` onto the batch axis (each chip ends up owning
    ``LE / shards`` *full* matrices), local NS, then reverse. Splitting the axis merge from
    the cross-axis migration lets XLA do each cheaply instead of materializing the full stack.
    """
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return x
    mesh_shape_items = [(name, size) for name, size in mesh.shape.items() if size > 1]
    if not mesh_shape_items:
        return x

    # SCALE_MUON_INTRA_RACK=1: keep the stack-assembly all-to-all off the cross-rack DCN axis.
    # Distribute NS only over intra-rack axes (drop replica_dcn) so the reshard stays on NVLink;
    # each DP replica then redundantly orthogonalizes its full stack from the already-all-reduced
    # gradient -- trading cheap duplicated NS matmuls for zero cross-rack all-to-all. Read at trace
    # time. Only drops the axis if intra-rack axes remain to distribute over.
    if os.environ.get("SCALE_MUON_INTRA_RACK") == "1":
        intra_axes = [(name, size) for name, size in mesh_shape_items if name != "replica_dcn"]
        if intra_axes:
            mesh_shape_items = intra_axes

    layers, expert_count, d, last = x.shape
    merged = layers * expert_count

    # Largest subset of batch mesh axes whose product divides ``merged``; NS replicates
    # across any axes that don't divide it rather than silently skipping orthogonalization.
    best_axes: tuple[str, ...] = ()
    best_shards = 0
    for mask in range(1, 1 << len(mesh_shape_items)):
        subset = [mesh_shape_items[i] for i in range(len(mesh_shape_items)) if mask & (1 << i)]
        prod = 1
        for _, size in subset:
            prod *= size
        if merged % prod == 0 and prod > best_shards:
            best_axes = tuple(name for name, _ in subset)
            best_shards = prod
    if not best_axes:
        raise ValueError(
            f"4D NS: no subset of batch mesh axes {dict(mesh.shape)} divides "
            f"merged={merged} (layers={layers} * experts={expert_count}) for "
            f"{jax.tree_util.keystr(path)}."
        )

    is_w_down = any(getattr(entry, "name", None) == "w_down" for entry in path)
    if is_w_down:
        intermediate_3d_spec = PartitionSpec(None, "model", "data")
        orig_4d_spec = PartitionSpec(None, "expert", "model", "data")
    else:
        intermediate_3d_spec = PartitionSpec(None, "data", "model")
        orig_4d_spec = PartitionSpec(None, "expert", "data", "model")
    target_3d_spec = (
        PartitionSpec(best_axes[0], None, None) if len(best_axes) == 1 else PartitionSpec(best_axes, None, None)
    )

    x_bf16 = x.astype(jnp.bfloat16)
    x_flat = jax.lax.reshape(x_bf16, (merged, d, last), out_sharding=intermediate_3d_spec)
    x_distributed = reshard(x_flat, target_3d_spec)
    local_ns = lambda matrix: _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)
    updated_distributed = jax.vmap(local_ns)(x_distributed)
    updated_flat = reshard(updated_distributed, intermediate_3d_spec)
    updated_bf16 = jax.lax.reshape(updated_flat, (layers, expert_count, d, last), out_sharding=orig_4d_spec)
    return updated_bf16.astype(x.dtype)


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

    for i in range(_effective_ns_steps(steps)):
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
    for i in range(_effective_ns_steps(steps)):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("...ik,...jk->...ij", X, X, out_sharding=X_out_sharding)
        B = b * A + c * jnp.einsum("...ik,...kj->...ij", A, A, out_sharding=X_out_sharding)
        X = a * X + jnp.einsum("...ik,...kj->...ij", B, X, out_sharding=X_out_sharding)

    if transpose:
        X = jnp.swapaxes(X, -1, -2)

    return X.astype(orig_dtype)


def _newtonschulz_padded_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
) -> jax.Array:
    """Distribute NS over a 3D stack whose length does not divide the mesh, via zero-padding.

    The non-expert stack is ``[L, d_in, d_out]`` (e.g. L=48 attn/dense matrices). When L does not
    divide the intra-rack shard count (48 over 64 data GPUs), the plain stack-sharded path can't
    distribute and falls back to replicating NS on every device. Here we pad L up to the next
    multiple of the shard count with zero matrices (``NS(0) == 0``), shard one matrix per chip,
    run local NS, gather, and slice the padding off. Honors ``SCALE_MUON_INTRA_RACK`` by keeping
    the distribution (and hence the scatter/gather) off the cross-rack ``replica_dcn`` axis.
    """
    P = PartitionSpec
    assert X.ndim == 3
    local = lambda matrix: _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return jax.vmap(local)(X)
    axes = [(name, size) for name, size in mesh.shape.items() if size > 1]
    if os.environ.get("SCALE_MUON_INTRA_RACK") == "1":
        intra_axes = [(name, size) for name, size in axes if name != "replica_dcn"]
        if intra_axes:
            axes = intra_axes
    if not axes:
        return jax.vmap(local)(X)

    batch_axis = tuple(name for name, _ in axes)
    batch_shards = math.prod(size for _, size in axes)
    layers = X.shape[0]
    pad = (-layers) % batch_shards

    Xp = jnp.pad(X, ((0, pad), (0, 0), (0, 0))) if pad else X
    target = P(batch_axis[0], None, None) if len(batch_axis) == 1 else P(batch_axis, None, None)
    Xd = reshard(Xp, target)
    updated = jax.vmap(local)(Xd)
    updated = reshard(updated, P(None, None, None))
    return updated[:layers] if pad else updated
