# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Newton-Schulz orthogonalization for the always-stacked EP grug MoE hero model.

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
from jax.sharding import NamedSharding, PartitionSpec, reshard
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


def _target_named_sharding(array) -> NamedSharding | None:
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        sharding = getattr(jax.typeof(array), "sharding", None)
    return sharding if isinstance(sharding, NamedSharding) else None


def _grug_scale_with_muon_hero(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
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

        def transform_array(path, x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3, 4):
                return x
            if x.ndim == 2:
                updated = _zeropower_via_newtonschulz_replicated(x, steps, muon_eps, coefficient_type)
            elif x.ndim == 3:
                updated = _newtonschulz_padded_stack_sharded(
                    x,
                    steps,
                    muon_eps,
                    coefficient_type,
                    target_sharding=_target_named_sharding(param),
                )
            else:
                updated = _newtonschulz_4d_distributed(path, x, steps, muon_eps, coefficient_type, use_syrk)

            fan_in, fan_out = updated.shape[-2:]
            scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
            return updated * scale

        updates = jax.tree_util.tree_map_with_path(
            transform_array,
            updates,
            params,
            is_leaf=lambda x: x is None,
        )
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
    is_w_down = any(getattr(entry, "name", None) == "w_down" for entry in path)
    trailing = ("model", "data") if is_w_down else ("data", "model")
    orig_4d_spec = PartitionSpec(None, "expert", *trailing)

    if int(mesh.shape.get("expert", 1)) > 1:
        candidate_axes = [(name, size) for name, size in mesh_shape_items if name != "expert"]
        best_axes: tuple[str, ...] = ()
        best_shards = 1
        for mask in range(1, 1 << len(candidate_axes)):
            subset = [candidate_axes[i] for i in range(len(candidate_axes)) if mask & (1 << i)]
            prod = math.prod(size for _, size in subset)
            if layers % prod == 0 and prod > best_shards:
                best_axes = tuple(name for name, _ in subset)
                best_shards = prod

        layer_spec = best_axes[0] if len(best_axes) == 1 else best_axes or None
        distributed_4d_spec = PartitionSpec(layer_spec, "expert", None, None)
        x_distributed = reshard(x.astype(jnp.bfloat16), distributed_4d_spec)
        if use_syrk:

            def local_syrk(stack):
                local_layers, local_experts, local_d, local_last = stack.shape
                flat = jax.lax.reshape(stack, (local_layers * local_experts, local_d, local_last))
                updated = _newtonschulz_batched_syrk(flat, steps, eps, coefficient_type)
                return jax.lax.reshape(updated, stack.shape)

            updated_distributed = shard_map(
                local_syrk,
                mesh=mesh,
                in_specs=distributed_4d_spec,
                out_specs=distributed_4d_spec,
                check_vma=False,
            )(x_distributed)
        else:
            updated_distributed = jax.vmap(jax.vmap(local_ns))(x_distributed)
        return reshard(updated_distributed, orig_4d_spec).astype(x.dtype)

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

    intermediate_3d_spec = PartitionSpec(None, *trailing)
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
    *,
    target_sharding: NamedSharding | None = None,
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
    Xd = reshard(Xp, P(batch_axis[0], None, None))
    if len(batch_axis) > 1:
        Xd = reshard(Xd, P(batch_axis, None, None))
    updated = jax.vmap(local)(Xd)
    if target_sharding is not None:
        target_spec = target_sharding.spec
        if target_spec and target_spec[0] is not None:
            raise ValueError("padded stack Newton-Schulz requires a replicated parameter layer axis")
        updated = reshard(updated, target_sharding)
        return updated[:layers] if pad else updated
    updated = reshard(updated, P(None, None, None))
    return updated[:layers] if pad else updated
