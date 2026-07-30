# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Newton-Schulz orthogonalization for the always-stacked FSDP grug MoE hero model.

Opinionated companion to ``levanter.optim.grugmuon`` for the hero, whose layers are always
array-stacked. The transform orthogonalizes each Muon leaf:

- 2D matrices: replicated Newton-Schulz.
- 3D matrix stacks (stacked non-expert layers): distributed over the intra-rack batch mesh axes,
  zero-padding the leading axis so it divides the shard count.
- 4D expert stacks ``[layers, experts, fan_in, fan_out]``: reshaped to a matrix stack and
  distributed over the intra-rack batch axes without gathering the matrix dimensions, optionally
  using QuACK's symmetric GEMM for ``X @ X.T``.

The optimizer config (routing, LR groups, the MuonH hyperball step) lives in ``optimizer.py``.
"""

import math
from importlib import import_module

import jax
import jax.numpy as jnp
import optax
from jax import shard_map
from jax.sharding import PartitionSpec, reshard
from levanter.optim.muon import ScaleByMuonState
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, CoefficientType
from optax import tree_utils as otu


def _intra_rack_axes(mesh) -> list[tuple[str, int]]:
    """Batch mesh axes with size > 1, excluding the cross-rack DCN axis.

    Orthogonalization stays within a rack: distributing the Newton-Schulz stack over
    ``replica_dcn`` would run the reshards across the slow inter-rack link.
    """
    axes = [(name, size) for name, size in mesh.shape.items() if size > 1]
    intra = [(name, size) for name, size in axes if name != "replica_dcn"]
    return intra if intra else axes


def _grug_scale_with_muon_hero(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    coefficient_type="quintic",
    *,
    use_syrk: bool = True,
):
    """Muon gradient transformation for the stacked hero model (2D/3D/4D leaves)."""
    steps = int(steps)

    def init_fn(params):
        return ScaleByMuonState(momentum_buffer=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            state.momentum_buffer,
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

        def transform_array(path, x):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3, 4):
                return x
            if x.ndim == 2:
                updated = _zeropower_via_newtonschulz_replicated(x, steps, muon_eps, coefficient_type)
            elif x.ndim == 3:
                updated = _newtonschulz_padded_stack_sharded(x, steps, muon_eps, coefficient_type)
            else:
                updated = _newtonschulz_4d_distributed(path, x, steps, muon_eps, coefficient_type, use_syrk)

            fan_in, fan_out = updated.shape[-2:]
            if not use_kimi_scaling:
                scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
            else:
                scale = 0.2 * jnp.sqrt(jnp.maximum(fan_in, fan_out))
            return updated * scale

        updates = jax.tree_util.tree_map_with_path(transform_array, updates)
        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _zeropower_via_newtonschulz_replicated(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
) -> jax.Array:
    """Newton-Schulz on a single matrix, fully replicated across devices.

    Replicates before iterating to avoid sharding ambiguity in the X @ X.T contractions. Runs in
    bf16 to halve all-gather bytes and double matmul throughput; casts back to the param dtype.
    """
    P = PartitionSpec
    assert X.ndim == 2
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    if has_mesh:
        X = reshard(X, P(None, None))
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        out_sharding = P(None, None) if has_mesh else None
        A = jnp.einsum("ik,jk->ij", X, X, out_sharding=out_sharding)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A, out_sharding=out_sharding)
        X = a * X + jnp.einsum("ik,kj->ij", B, X, out_sharding=out_sharding)

    if transpose:
        X = X.T
    return X.astype(orig_dtype)


def _zeropower_via_newtonschulz_local(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
) -> jax.Array:
    """Run Newton-Schulz on a matrix that is already local to one device."""
    assert X.ndim == 2
    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    X = X / (jnp.linalg.norm(X) + eps)

    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = jnp.einsum("ik,jk->ij", X, X)
        B = b * A + c * jnp.einsum("ik,kj->ij", A, A)
        X = a * X + jnp.einsum("ik,kj->ij", B, X)

    if transpose:
        X = X.T
    return X.astype(orig_dtype)


def _newtonschulz_batched_syrk(
    X: jax.Array,
    steps: int,
    eps: float,
    coefficient_type: CoefficientType,
) -> jax.Array:
    """Run batched Newton-Schulz with QuACK symmetric products (X @ X.T)."""
    quack_symmetric_gemm = import_module("levanter.grug._moe.quack_symmetric_cute").quack_symmetric_gemm

    orig_dtype = X.dtype
    X = X.astype(jnp.bfloat16)
    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps)

    transpose = X.shape[-2] > X.shape[-1]
    if transpose:
        X = jnp.swapaxes(X, -1, -2)

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = quack_symmetric_gemm(X)
        B = b * A + c * quack_symmetric_gemm(A)
        X = a * X + jnp.matmul(B, X)

    if transpose:
        X = jnp.swapaxes(X, -1, -2)
    return X.astype(orig_dtype)


def _newtonschulz_4d_distributed(
    path,
    x: jax.Array,
    steps: int,
    eps: float,
    coefficient_type: CoefficientType,
    use_syrk: bool,
) -> jax.Array:
    """Run Newton-Schulz on a stacked 4D expert leaf without gathering matrix dims."""

    def local_ns(matrix):
        return _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return jax.vmap(jax.vmap(local_ns))(x)
    mesh_shape_items = _intra_rack_axes(mesh)
    if not mesh_shape_items:
        return jax.vmap(jax.vmap(local_ns))(x)

    layers, expert_count, d, last = x.shape
    merged = layers * expert_count

    best_axes: tuple[str, ...] = ()
    best_shards = 0
    for mask in range(1, 1 << len(mesh_shape_items)):
        subset = [mesh_shape_items[i] for i in range(len(mesh_shape_items)) if mask & (1 << i)]
        prod = math.prod(size for _, size in subset)
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
    if use_syrk:
        updated_distributed = shard_map(
            lambda stack: _newtonschulz_batched_syrk(stack, steps, eps, coefficient_type),
            mesh=mesh,
            in_specs=target_3d_spec,
            out_specs=target_3d_spec,
            check_vma=False,
        )(x_distributed)
    else:
        updated_distributed = jax.vmap(local_ns)(x_distributed)
    updated_flat = reshard(updated_distributed, intermediate_3d_spec)
    updated_bf16 = jax.lax.reshape(updated_flat, (layers, expert_count, d, last), out_sharding=orig_4d_spec)
    return updated_bf16.astype(x.dtype)


def _newtonschulz_padded_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
) -> jax.Array:
    """Distribute a matrix stack over the intra-rack batch axes, zero-padding the leading axis."""
    P = PartitionSpec
    assert X.ndim == 3

    def local(matrix):
        return _zeropower_via_newtonschulz_local(matrix, steps, eps, coefficient_type)

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return jax.vmap(local)(X)
    axes = _intra_rack_axes(mesh)
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
