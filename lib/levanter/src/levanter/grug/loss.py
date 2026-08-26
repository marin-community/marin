# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax.sharding import Mesh, NamedSharding, get_abstract_mesh, get_mesh, reshard
from jax.sharding import PartitionSpec as P

from levanter.kernels.pallas.fused_cross_entropy_loss import (
    BlockSizes,
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


def _leading_axis_specs(x: jax.Array) -> tuple[object, ...]:
    x_type = jax.typeof(x)
    sharding = getattr(x_type, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is None:
        sharding = getattr(x, "sharding", None)
        spec = getattr(sharding, "spec", None)
    if spec is not None:
        normalized = tuple(spec) + (None,) * (x.ndim - len(spec))
        return normalized[:-1]
    return (("data",),) + (None,) * (x.ndim - 2)


def _axis_names_from_specs(axis_specs: tuple[object, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for axis_spec in axis_specs:
        axes = axis_spec if isinstance(axis_spec, tuple) else (axis_spec,)
        for axis in axes:
            if axis is not None and axis not in names:
                names.append(str(axis))
    return tuple(names)


def _psum_over_axes(x: jax.Array, axis_names: tuple[str, ...]) -> jax.Array:
    if len(axis_names) == 0:
        return x
    if len(axis_names) == 1:
        return jax.lax.psum(x, axis_names[0])
    return jax.lax.psum(x, axis_names)


def _current_mesh() -> Mesh | jax.sharding.AbstractMesh:
    try:
        mesh = get_mesh()
    except ValueError:
        mesh = None
    if mesh is not None and not mesh.empty:
        return mesh
    return get_abstract_mesh()


def _reshard_for_shard_map(
    x: jax.Array,
    mesh: Mesh | jax.sharding.AbstractMesh | None,
    spec: P,
) -> jax.Array:
    if mesh is not None and not mesh.empty:
        return reshard(x, NamedSharding(mesh, spec))
    return x


@named_call
def fused_linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    weight: jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    dtype: jnp.dtype = jnp.float32,
    precision: jax.lax.PrecisionLike = None,
    implementation: str | tuple[str, ...] | None = None,
    block_sizes: BlockSizes | None = None,
    batch_chunk_size: int | None = None,
) -> jax.Array:
    """Compute cross-entropy loss via the fused kernel path.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        weight: Optional per-example weights with shape matching labels.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        dtype: Accumulator dtype for logits/logsumexp.
        precision: Optional matmul precision override for XLA/reference paths.
        implementation: Optional fused CE backend selection override.
        block_sizes: Optional kernel block-size override.
        batch_chunk_size: Optional number of flattened tokens scored by each fused-kernel call.

    Returns:
        If reduction=="none": array with shape labels.shape.
        Else: scalar array.
    """
    if lm_head.ndim != 2:
        raise ValueError(f"lm_head must be 2D (hidden_dim, vocab), got shape={lm_head.shape}")
    hidden_dim = hidden.shape[-1]
    if lm_head.shape[0] != hidden_dim:
        raise ValueError(f"hidden_dim mismatch: hidden={hidden_dim}, lm_head={lm_head.shape[0]}")

    reduction_mode: str | None
    if reduction == "none":
        reduction_mode = None
    elif reduction in ("sum", "mean"):
        reduction_mode = reduction
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    mesh = _current_mesh()
    has_mesh = mesh is not None and not mesh.empty
    weight_array = weight if weight is not None else jnp.ones_like(labels, dtype=dtype)
    leading_axis_specs = _leading_axis_specs(hidden) if has_mesh else ()
    batch_axis_names = _axis_names_from_specs(leading_axis_specs) if has_mesh else ()

    def _loss_shard(
        shard_hidden: jax.Array,
        shard_lm_head: jax.Array,
        shard_labels: jax.Array,
        shard_weight: jax.Array,
    ) -> jax.Array:
        flat_hidden = shard_hidden.reshape((-1, hidden_dim))
        flat_labels = shard_labels.reshape((-1,)).astype(jnp.int32)
        flat_weight = shard_weight.reshape((-1,))

        def _flat_loss(hidden_chunk, label_chunk, weight_chunk):
            return fused_cross_entropy_loss_and_logsumexp_penalty(
                hidden_chunk,
                label_chunk,
                shard_lm_head,
                reduction=None,
                weight=weight_chunk,
                logsumexp_weight=logsumexp_weight,
                dtype=dtype,
                logit_soft_cap=None,
                precision=precision,
                implementation=implementation,
                block_sizes=block_sizes,
            )

        if batch_chunk_size is None:
            loss = _flat_loss(flat_hidden, flat_labels, flat_weight)
        else:
            if batch_chunk_size <= 0:
                raise ValueError(f"batch_chunk_size must be positive, got {batch_chunk_size}")
            if flat_hidden.shape[0] % batch_chunk_size != 0:
                raise ValueError(
                    f"flattened token count {flat_hidden.shape[0]} must be divisible by "
                    f"batch_chunk_size {batch_chunk_size}"
                )
            chunk_count = flat_hidden.shape[0] // batch_chunk_size
            hidden_chunks = flat_hidden.reshape(chunk_count, batch_chunk_size, hidden_dim)
            label_chunks = flat_labels.reshape(chunk_count, batch_chunk_size)
            weight_chunks = flat_weight.reshape(chunk_count, batch_chunk_size)
            loss = jax.lax.map(
                lambda chunks: _flat_loss(*chunks),
                (hidden_chunks, label_chunks, weight_chunks),
            ).reshape(-1)

        if reduction_mode is None:
            return loss.reshape(shard_labels.shape)

        local_sum = jnp.sum(loss)
        local_denom = jnp.sum(flat_weight)
        total_sum = _psum_over_axes(local_sum, batch_axis_names)
        if reduction_mode == "sum":
            return total_sum
        total_denom = _psum_over_axes(local_denom, batch_axis_names)
        return jnp.where(total_denom != 0, total_sum / total_denom, jnp.zeros_like(total_denom))

    if not has_mesh:
        return _loss_shard(hidden, lm_head, labels, weight_array)

    hidden_spec = P(*leading_axis_specs, None)
    lm_head_spec = P(None, None)
    label_spec = P(*leading_axis_specs)
    hidden = _reshard_for_shard_map(hidden, mesh, hidden_spec)
    lm_head = _reshard_for_shard_map(lm_head, mesh, lm_head_spec)
    labels = _reshard_for_shard_map(labels, mesh, label_spec)
    weight_array = _reshard_for_shard_map(weight_array, mesh, label_spec)

    out_specs = label_spec if reduction_mode is None else P()
    return jax.shard_map(
        _loss_shard,
        mesh=mesh,
        in_specs=(hidden_spec, lm_head_spec, label_spec, label_spec),
        out_specs=out_specs,
        check_vma=False,
    )(hidden, lm_head, labels, weight_array)


__all__ = [
    "fused_linear_softmax_cross_entropy_loss",
]
