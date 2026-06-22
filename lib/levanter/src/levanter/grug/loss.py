# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard

from haliax.jax_utils import named_call
from levanter.grug.sharding import _mesh_axis_size
from levanter.kernels.pallas.fused_cross_entropy_loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.kernels.pallas.fused_cross_entropy_loss.api import default_implementations


def _batch_axis_spec(x: jax.Array):
    x_type = jax.typeof(x)
    sharding = getattr(x_type, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return spec[0]
    sharding = getattr(x, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        return spec[0]
    return ("data",)


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


def _ce_implementation_order(implementation: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if implementation is None:
        implementation = default_implementations()
    if isinstance(implementation, str):
        return (implementation,)
    if isinstance(implementation, tuple):
        return implementation
    return ()


def _uses_vocab_sharded_ce(implementation_order: tuple[str, ...]) -> bool:
    return any(impl in ("xla", "pallas_gpu") for impl in implementation_order)


def _lm_head_spec_for_ce(
    lm_head: jax.Array,
    mesh: Mesh | jax.sharding.AbstractMesh | None,
    implementation_order: tuple[str, ...],
) -> P:
    if _uses_vocab_sharded_ce(implementation_order) and lm_head.shape[1] % _mesh_axis_size(mesh, "model") == 0:
        if _mesh_axis_size(mesh, "model") > 1:
            return P(None, "model")
    return P(None, None)


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
    implementation_order = _ce_implementation_order(implementation)

    mesh = _current_mesh()
    has_mesh = mesh is not None and not mesh.empty
    weight_array = weight if weight is not None else jnp.ones_like(labels, dtype=dtype)
    batch_axis_spec = _batch_axis_spec(hidden) if has_mesh else None

    flat_hidden = hidden.reshape((-1, hidden_dim))
    flat_labels = labels.reshape((-1,)).astype(jnp.int32)
    flat_weight = weight_array.reshape((-1,))

    if has_mesh:
        hidden_spec = P(batch_axis_spec, None)
        lm_head_spec = _lm_head_spec_for_ce(lm_head, mesh, implementation_order)
        label_spec = P(batch_axis_spec)
        flat_hidden = _reshard_for_shard_map(flat_hidden, mesh, hidden_spec)
        lm_head = _reshard_for_shard_map(lm_head, mesh, lm_head_spec)
        flat_labels = _reshard_for_shard_map(flat_labels, mesh, label_spec)
        flat_weight = _reshard_for_shard_map(flat_weight, mesh, label_spec)

    loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        flat_hidden,
        flat_labels,
        lm_head,
        reduction=reduction_mode,
        weight=flat_weight,
        logsumexp_weight=logsumexp_weight,
        dtype=dtype,
        logit_soft_cap=None,
        precision=precision,
        implementation=implementation,
    )
    if reduction_mode is None:
        return loss.reshape(labels.shape)
    return loss


__all__ = [
    "fused_linear_softmax_cross_entropy_loss",
]
