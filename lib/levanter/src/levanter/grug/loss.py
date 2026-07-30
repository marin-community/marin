# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused linear softmax cross-entropy for grug.

This wraps the shared fused kernel API for TPU and falls back to a full-logits
reference implementation on non-TPU backends.
"""

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, get_abstract_mesh, get_mesh, reshard

from haliax.jax_utils import named_call
from levanter.kernels.pallas.fused_cross_entropy_loss import (
    fused_cross_entropy_loss_and_logsumexp_penalty,
)


class _ChunkLayout(NamedTuple):
    chunk_size: int
    padded_num_tokens: int
    num_chunks: int


def _chunk_layout(num_tokens: int, chunk_size: int) -> _ChunkLayout:
    effective_chunk_size = min(chunk_size, max(num_tokens, 1))
    padded_num_tokens = -(-num_tokens // effective_chunk_size) * effective_chunk_size
    num_chunks = padded_num_tokens // effective_chunk_size
    return _ChunkLayout(effective_chunk_size, padded_num_tokens, num_chunks)


def _cross_entropy_and_log_normalizers(
    logits: jax.Array,
    labels: jax.Array,
    logsumexp_weight: float,
) -> tuple[jax.Array, jax.Array]:
    log_normalizers = jax.scipy.special.logsumexp(logits, axis=-1)
    target_logits = jnp.take_along_axis(logits, labels[:, None], axis=1)[:, 0]
    loss = log_normalizers - target_logits
    if logsumexp_weight:
        loss = loss + logsumexp_weight * log_normalizers * log_normalizers
    return loss, log_normalizers


def _chunked_cross_entropy_forward(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
    logsumexp_weight: float,
    chunk_size: int,
) -> jax.Array:
    num_tokens, hidden_dim = hidden.shape
    layout = _chunk_layout(num_tokens, chunk_size)
    padded_hidden = jnp.pad(hidden, ((0, layout.padded_num_tokens - num_tokens), (0, 0)))
    padded_labels = jnp.pad(labels.astype(jnp.int32), (0, layout.padded_num_tokens - num_tokens))
    padded_weight = jnp.pad(weight, (0, layout.padded_num_tokens - num_tokens))

    def chunk_loss(chunk: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        chunk_hidden, chunk_labels, chunk_weight = chunk
        logits = (chunk_hidden @ lm_head).astype(jnp.float32)
        loss, _ = _cross_entropy_and_log_normalizers(logits, chunk_labels, logsumexp_weight)
        return chunk_weight * loss

    chunked_loss = jax.lax.map(
        chunk_loss,
        (
            padded_hidden.reshape(layout.num_chunks, layout.chunk_size, hidden_dim),
            padded_labels.reshape(layout.num_chunks, layout.chunk_size),
            padded_weight.reshape(layout.num_chunks, layout.chunk_size),
        ),
    )
    return chunked_loss.reshape(layout.padded_num_tokens)[:num_tokens]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _chunked_weighted_cross_entropy(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
    logsumexp_weight: float,
    chunk_size: int,
    backward_pass_unroll: int,
) -> jax.Array:
    del backward_pass_unroll
    return _chunked_cross_entropy_forward(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size)


def _chunked_cross_entropy_vjp_forward(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
    logsumexp_weight: float,
    chunk_size: int,
    backward_pass_unroll: int,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    del backward_pass_unroll
    loss = _chunked_cross_entropy_forward(hidden, lm_head, labels, weight, logsumexp_weight, chunk_size)
    return loss, (hidden, lm_head, labels, weight)


def _chunked_cross_entropy_vjp_backward(
    logsumexp_weight: float,
    chunk_size: int,
    backward_pass_unroll: int,
    residual: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    output_cotangent: jax.Array,
) -> tuple[jax.Array, jax.Array, None, jax.Array]:
    hidden, lm_head, labels, weight = residual
    num_tokens, hidden_dim = hidden.shape
    vocab_size = lm_head.shape[1]
    layout = _chunk_layout(num_tokens, chunk_size)
    padded_hidden = jnp.pad(hidden, ((0, layout.padded_num_tokens - num_tokens), (0, 0)))
    padded_labels = jnp.pad(labels.astype(jnp.int32), (0, layout.padded_num_tokens - num_tokens))
    padded_weight = jnp.pad(weight, (0, layout.padded_num_tokens - num_tokens))
    padded_cotangent = jnp.pad(output_cotangent.astype(jnp.float32), (0, layout.padded_num_tokens - num_tokens))

    def backward_chunk(
        lm_head_cotangent: jax.Array,
        chunk: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        chunk_hidden, chunk_labels, chunk_weight, chunk_cotangent = chunk
        logits = (chunk_hidden @ lm_head).astype(jnp.float32)
        unweighted_loss, log_normalizers = _cross_entropy_and_log_normalizers(logits, chunk_labels, logsumexp_weight)
        probabilities = jnp.exp(logits - log_normalizers[:, None])
        one_hot_labels = jax.nn.one_hot(chunk_labels, vocab_size, dtype=jnp.float32)
        if logsumexp_weight:
            probability_scale = (1.0 + 2.0 * logsumexp_weight * log_normalizers)[:, None]
        else:
            probability_scale = 1.0
        loss_scale = (chunk_weight * chunk_cotangent)[:, None]
        logits_cotangent = (loss_scale * (probabilities * probability_scale - one_hot_labels)).astype(hidden.dtype)
        hidden_cotangent = logits_cotangent @ lm_head.T
        chunk_lm_head_cotangent = (chunk_hidden.T @ logits_cotangent).astype(jnp.float32)
        weight_cotangent = chunk_cotangent * unweighted_loss
        return lm_head_cotangent + chunk_lm_head_cotangent, (hidden_cotangent, weight_cotangent)

    lm_head_cotangent, (hidden_cotangent, weight_cotangent) = jax.lax.scan(
        backward_chunk,
        jnp.zeros((hidden_dim, vocab_size), jnp.float32),
        (
            padded_hidden.reshape(layout.num_chunks, layout.chunk_size, hidden_dim),
            padded_labels.reshape(layout.num_chunks, layout.chunk_size),
            padded_weight.reshape(layout.num_chunks, layout.chunk_size),
            padded_cotangent.reshape(layout.num_chunks, layout.chunk_size),
        ),
        unroll=backward_pass_unroll,
    )
    return (
        hidden_cotangent.reshape(layout.padded_num_tokens, hidden_dim)[:num_tokens].astype(hidden.dtype),
        lm_head_cotangent.astype(lm_head.dtype),
        None,
        weight_cotangent.reshape(layout.padded_num_tokens)[:num_tokens].astype(weight.dtype),
    )


_chunked_weighted_cross_entropy.defvjp(
    _chunked_cross_entropy_vjp_forward,
    _chunked_cross_entropy_vjp_backward,
)


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


def _axis_names_from_spec(axis_spec) -> tuple[str, ...]:
    if axis_spec is None:
        return ()
    if isinstance(axis_spec, tuple):
        return tuple(str(name) for name in axis_spec)
    return (str(axis_spec),)


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


def _linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    weight: jax.Array | None,
    reduction: str,
    dtype: jnp.dtype,
    per_token_loss: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
) -> jax.Array:
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
    batch_axis_spec = _batch_axis_spec(hidden) if has_mesh else None
    batch_axis_names = _axis_names_from_spec(batch_axis_spec) if has_mesh else ()

    def _loss_shard(
        shard_hidden: jax.Array,
        shard_lm_head: jax.Array,
        shard_labels: jax.Array,
        shard_weight: jax.Array,
    ) -> jax.Array:
        flat_hidden = shard_hidden.reshape((-1, hidden_dim))
        flat_labels = shard_labels.reshape((-1,)).astype(jnp.int32)
        flat_weight = shard_weight.reshape((-1,))

        loss = per_token_loss(flat_hidden, shard_lm_head, flat_labels, flat_weight)

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

    hidden_spec = P(batch_axis_spec)
    lm_head_spec = P(None, None)
    label_spec = P(batch_axis_spec)
    hidden = _reshard_for_shard_map(hidden, mesh, hidden_spec)
    lm_head = _reshard_for_shard_map(lm_head, mesh, lm_head_spec)
    labels = _reshard_for_shard_map(labels, mesh, label_spec)
    weight_array = _reshard_for_shard_map(weight_array, mesh, label_spec)

    out_specs = hidden_spec if reduction_mode is None else P()
    return jax.shard_map(
        _loss_shard,
        mesh=mesh,
        in_specs=(hidden_spec, lm_head_spec, label_spec, label_spec),
        out_specs=out_specs,
        check_vma=False,
    )(hidden, lm_head, labels, weight_array)


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

    def per_token_loss(
        flat_hidden: jax.Array,
        shard_lm_head: jax.Array,
        flat_labels: jax.Array,
        flat_weight: jax.Array,
    ) -> jax.Array:
        return fused_cross_entropy_loss_and_logsumexp_penalty(
            flat_hidden,
            flat_labels,
            shard_lm_head,
            reduction=None,
            weight=flat_weight,
            logsumexp_weight=logsumexp_weight,
            dtype=dtype,
            logit_soft_cap=None,
            precision=precision,
            implementation=implementation,
        )

    return _linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction=reduction,
        dtype=dtype,
        per_token_loss=per_token_loss,
    )


@named_call
def chunked_linear_softmax_cross_entropy_loss(
    hidden: jax.Array,
    lm_head: jax.Array,
    labels: jax.Array,
    *,
    chunk_size: int,
    backward_pass_unroll: int,
    weight: jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Compute cross-entropy in token chunks and recompute logits during the backward pass.

    Args:
        hidden: Array with shape (..., hidden_dim).
        lm_head: Array with shape (hidden_dim, vocab_size).
        labels: Integer array with shape (...,).
        chunk_size: Maximum number of tokens in each per-device logit tile.
        backward_pass_unroll: Number of backward scan iterations to unroll.
        weight: Optional per-example weights with shape matching labels.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        dtype: Dtype used to create or cast per-token weights. Logits and loss are computed in float32.

    Returns:
        If reduction=="none": array with shape labels.shape.
        Else: scalar array.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if backward_pass_unroll <= 0:
        raise ValueError(f"backward_pass_unroll must be positive, got {backward_pass_unroll}")

    def per_token_loss(
        flat_hidden: jax.Array,
        shard_lm_head: jax.Array,
        flat_labels: jax.Array,
        flat_weight: jax.Array,
    ) -> jax.Array:
        return _chunked_weighted_cross_entropy(
            flat_hidden,
            shard_lm_head,
            flat_labels,
            flat_weight.astype(dtype),
            float(logsumexp_weight or 0.0),
            chunk_size,
            backward_pass_unroll,
        )

    return _linear_softmax_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        weight=weight,
        reduction=reduction,
        dtype=dtype,
        per_token_loss=per_token_loss,
    )


__all__ = [
    "chunked_linear_softmax_cross_entropy_loss",
    "fused_linear_softmax_cross_entropy_loss",
]
