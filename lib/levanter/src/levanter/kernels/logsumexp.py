# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-row log-normalization for selected-token and full-distribution scores."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P, reshard

from levanter.sharding import partition_spec_of

_ROW_TILE = 8


class _ShiftedNormalizer(NamedTuple):
    pivot_index: jax.Array
    pivot: jax.Array
    correction: jax.Array

    @property
    def logsumexp(self) -> jax.Array:
        return self.pivot + self.correction


def accurate_logsumexp(logits: jax.Array) -> jax.Array:
    """Reduce the final axis with TPU's highest accuracy exp and log."""
    maximum = jnp.max(logits, axis=-1)
    maximum = jax.lax.stop_gradient(jnp.where(jnp.isfinite(maximum), maximum, 0))
    total = jnp.sum(jax.lax.exp(logits - maximum[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST), axis=-1)
    return maximum + jax.lax.log(total, accuracy=jax.lax.AccuracyMode.HIGHEST)


def _shifted_normalizer(logits: jax.Array) -> _ShiftedNormalizer:
    pivot_index = jnp.argmax(logits, axis=-1)
    maximum = jnp.max(logits, axis=-1)
    finite = jnp.isfinite(maximum)
    pivot_index = jnp.where(finite, pivot_index, -1)
    pivot = jnp.where(finite, maximum, 0)
    # Exclude exactly one maximum so tied logits still contribute to the tail.
    is_pivot = jnp.arange(logits.shape[-1]) == pivot_index[..., None]
    exponentials = jax.lax.exp(logits - pivot[..., None], accuracy=jax.lax.AccuracyMode.HIGHEST)
    tail = jnp.sum(jnp.where(is_pivot, 0, exponentials), axis=-1)
    correction = jax.lax.log1p(tail, accuracy=jax.lax.AccuracyMode.HIGHEST)
    correction = jnp.where(jnp.isneginf(maximum), -jnp.inf, correction)
    return _ShiftedNormalizer(pivot_index, pivot, correction)


def _local_shifted_normalizer(logits: jax.Array) -> _ShiftedNormalizer:
    rows = math.prod(logits.shape[:-1])
    flat = logits.reshape((rows, logits.shape[-1]))
    padded = jnp.pad(flat, ((0, (-rows) % _ROW_TILE), (0, 0)))

    reduced = jax.lax.map(_shifted_normalizer, padded.reshape((-1, _ROW_TILE, flat.shape[-1])))
    return jax.tree.map(lambda value: value.reshape((-1,))[:rows].reshape(logits.shape[:-1]), reduced)


@custom_vmap
def _mapped_shifted_normalizer(logits: jax.Array) -> _ShiftedNormalizer:
    sharding = logits.sharding if not isinstance(logits, jax.core.Tracer) else jax.typeof(logits).sharding
    if isinstance(sharding, NamedSharding) and AxisType.Auto in sharding.mesh.axis_types:
        # Auto axes are absent from a traced PartitionSpec. Let the compiler
        # partition the reduction instead of declaring the input replicated.
        reduced = _shifted_normalizer(logits)
    elif isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
        spec = tuple(sharding.spec) + (None,) * (logits.ndim - len(sharding.spec))
        local_spec = P(*spec[:-1], None)
        reduced = jax.shard_map(
            _local_shifted_normalizer,
            mesh=sharding.mesh,
            in_specs=local_spec,
            out_specs=P(*spec[:-1]),
            check_vma=False,
        )(reshard(logits, local_spec))
    else:
        reduced = _local_shifted_normalizer(logits)
    # Keep the row reduction separate from downstream scoring.
    return jax.lax.optimization_barrier(reduced)


@_mapped_shifted_normalizer.def_vmap
def _shifted_normalizer_vmap(_axis_size, in_batched, logits):
    # Re-enter with the physical batch axes visible before choosing row tiles.
    # The engine vmaps its sampler; a default rule would add axes outside them.
    normalizer = _mapped_shifted_normalizer(logits)
    return normalizer, jax.tree.map(lambda _: in_batched[0], normalizer)


@jax.custom_jvp
def _batch_invariant_shifted_normalizer(logits: jax.Array) -> _ShiftedNormalizer:
    return _mapped_shifted_normalizer(logits)


@_batch_invariant_shifted_normalizer.defjvp
def _shifted_normalizer_jvp(primals, tangents):
    (logits,), (tangent,) = primals, tangents
    normalizer = _batch_invariant_shifted_normalizer(logits)
    pivot_index, pivot, correction = normalizer
    is_pivot = jnp.arange(logits.shape[-1]) == pivot_index[..., None]
    pivot_dot = jnp.sum(jnp.where(is_pivot, tangent, 0), axis=-1)
    log_probs = (logits - pivot[..., None]) - correction[..., None]
    probabilities = jax.lax.exp(log_probs, accuracy=jax.lax.AccuracyMode.HIGHEST)
    # Differentiate the tail in pivot-relative coordinates. At high confidence,
    # the pivot derivative is the negative tail mass, which survives rounding.
    tail_probs = jnp.where(is_pivot, 0, probabilities)
    correction_dot = jnp.sum(tail_probs * (tangent - pivot_dot[..., None]), axis=-1)
    index_dot = jnp.zeros_like(pivot_index, dtype=jax.dtypes.float0)
    return normalizer, _ShiftedNormalizer(index_dot, pivot_dot, correction_dot)


def batch_invariant_logsumexp(logits: jax.Array) -> jax.Array:
    """Return the FP32 log-sum-exp over the final vocabulary axis.

    Explicit meshes fix the reduction order across batch shapes and preserve
    leading-axis sharding. Auto meshes delegate partitioning to JAX.
    """
    return _batch_invariant_shifted_normalizer(logits.astype(jnp.float32)).logsumexp


def selected_token_logits(logits: jax.Array, token_ids: jax.Array) -> jax.Array:
    """Gather one logit per row along the final vocabulary axis."""
    return jnp.take_along_axis(logits, token_ids[..., None], axis=-1).squeeze(-1)


def batch_invariant_selected_logprobs(logits: jax.Array, token_ids: jax.Array) -> jax.Array:
    """Return FP32 log probabilities for token IDs shaped like the leading axes."""
    logits = logits.astype(jnp.float32)
    spec = partition_spec_of(logits)
    if spec is not None and len(spec) == logits.ndim and spec[-1] is not None:
        # The lookup needs the same vocabulary shards as the row normalization.
        logits = reshard(logits, P(*spec[:-1], None))
    pivot_index, pivot, correction = _batch_invariant_shifted_normalizer(logits)
    selected = selected_token_logits(logits, token_ids)
    shifted = jnp.where(token_ids == pivot_index, 0, selected - pivot)
    return shifted - correction


def batch_invariant_log_softmax(logits: jax.Array) -> jax.Array:
    """Return FP32 policy log probabilities over the final vocabulary axis."""
    logits = logits.astype(jnp.float32)
    normalizer = _batch_invariant_shifted_normalizer(logits)
    pivot_index, pivot, correction = normalizer
    is_pivot = jnp.arange(logits.shape[-1]) == pivot_index[..., None]
    shifted = jnp.where(is_pivot, 0, logits - pivot[..., None])
    return shifted - correction[..., None]
